"""
https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py
"""

from entsoe import EntsoePandasClient
from entsoe.mappings import NEIGHBOURS

import pandas as pd
import time, os
from datetime import timedelta, datetime

from data_collection_modules.eu_locations import (
    countries_metadata, country_code_name_mapping
)
from data_collection_modules.utils import compare_columns

from logger import get_logger


logger = get_logger(__name__)

entsoe_generation_type_mapping = {
    # non-renewables
    "hard_coal": ["Fossil Hard coal Actual Aggregated"],
    "lignite": ["Fossil Brown coal/Lignite Actual Aggregated"],
    "gas": ["Fossil Gas Actual Aggregated"],
    "coal_derived_gas": ["Fossil Coal-derived gas Actual Aggregated"], # Fossil Coal-derived gas Actual Aggregated
    "oil": ["Fossil Oil Actual Aggregated"],
    "other_fossil" : ["Other Actual Aggregated"], # Other renewable Actual Aggregated
    "nuclear": ["Nuclear Actual Aggregated"],
    # renewables (stable)
    "biomass": ["Biomass Actual Aggregated"],
    "waste": ["Waste Actual Aggregated"],
    "geothermal": ["Geothermal Actual Aggregated"],
    "other_renewables": ["Other renewable Actual Aggregated"],
    "pumped_storage": ["Hydro Pumped Storage Actual Aggregated"],
    "run_of_river": ["Hydro Run-of-river and poundage Actual Aggregated"],
    "water_reservoir": ["Hydro Water Reservoir Actual Aggregated"],
    # renewables (highly volatile)
    "solar": ["Solar Actual Aggregated"],
    "wind_onshore": ["Wind Onshore Actual Aggregated"],
    "wind_offshore": ["Wind Offshore Actual Aggregated"],
}

# hourly_flow_data = ['france', 'sweden4']

# entsoe_neighbors = ["AT","BE","CH","CZ","DK_1","DK_2","FR","NO_2","NL","PL","SE_4"]
# flow_mapping = {
#     'AT':'austria', 'BE':'belgium', 'CH':'switzerland', 'CZ':'czechia',
#     'DK_1':'denmark1', 'DK_2':'denmark2', 'FR':'france', 'NO_2':'norway2',
#     'NL':'netherlands', 'PL':'poland', 'SE_4':'sweden4'
# }
#
# country_code_to_flows = 'DE_LU' # Germany-Luxembourg

def preprocess_generation(df_gen:pd.DataFrame, drop_consumption:bool, freq:str)->pd.DataFrame:

    df_gen.columns = [" ".join(a) for a in df_gen.columns.to_flat_index()]

    # df_final = pd.concat( [df_load, df_gen], axis=1 )  # Concatenate dataframes in columns dimension.
    if "Nuclear Actual Aggregated" in df_gen.columns.tolist():
        df_gen["Nuclear Actual Aggregated"] = df_gen["Nuclear Actual Aggregated"].fillna(0)

    df_gen.index = pd.to_datetime(df_gen.index, utc=True).tz_convert(tz="UTC")
    df_gen.index.name = 'date'

    if drop_consumption:  # Drop columns containing actual consumption.
        df_gen.drop(list(df_gen.filter(regex="Consumption")), axis=1, inplace=True)

    df_gen.interpolate(method="time", axis=0, inplace=True)
    for joint_category, old_categories in entsoe_generation_type_mapping.items():
        existing_columns = [col for col in old_categories if col in df_gen.columns]
        if existing_columns:
            # Sum up existing columns and drop them
            df_gen[joint_category] = df_gen[existing_columns].sum(axis=1, skipna=False)
            df_gen.drop(columns=existing_columns, inplace=True)

    if freq == 'hourly':
        df_gen = df_gen.resample("1h").mean() # average power in a given hour
    elif freq == 'minutely_15': # assuming original data is in 15 min intervals
        pass
    else:
        raise NotImplementedError("Frequency ressampling not implemented for {}".format(freq))
    return df_gen

def fetch_entsoe_data_from_api(country_dict:dict, working_dir:str, start_date:pd.Timestamp or None,
                               today:pd.Timestamp,api_key:str,freq:str,cols:list or None,verbose:bool)->pd.DataFrame:

    if freq not in ['hourly', 'minutely_15']:
        raise KeyError(f"Frequency must be 'hourly', 'minutely_15'. Given: {freq}")

    c_code:str = country_dict['code'] # DE, FR ...
    bidding_zone = country_dict['bidding_zone'] # DE_LU, FR ....
    tso_regions:list[dict] = country_dict['regions'] # [{TSO:name,suffix}] ....


    client = EntsoePandasClient(api_key=api_key)

    ''' ----------- DATA PER TSO -------------- '''

    df = pd.DataFrame()
    cach_fnames = []
    for i, region in enumerate(tso_regions):
        if verbose: logger.info(
            f"Requesting ENTSO-E data for region {region['name']} for freq {freq} from {start_date} till {today}"
        )

        ''' ------------ GENERATION ENERGY MIX ---------------- '''
        df_gen = None
        fname1 = f"tmp_gen{region['suffix']}_hist_{freq}.parquet"
        cach_fnames.append(fname1)
        if os.path.isfile(working_dir + fname1):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname1}")
            df_gen = pd.read_parquet(working_dir + fname1)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(
                        f"Collecting generation for freq {freq} for {region['name']} from {start_date} to {today}"
                    )
                    df_gen = client.query_generation(
                        country_code=region['name'], start=start_date, end=today,psr_type=None
                    )
                    df_gen = preprocess_generation(df_gen, drop_consumption=True, freq=freq)
                    # drop nuclear for Germany as they phased it out
                    if ('nuclear' in df_gen.columns) and (c_code=='DE'):
                        logger.info(f"Dropping nuclear columns from dataframe for country {c_code}")
                        df_gen.drop(columns=['nuclear'], inplace=True)
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(
                        f"Failed to fetch generation from ENTSOE API ({i}/{5}) for region "
                        f"{region['name']} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(5)
                    continue
                break
            if df_gen is None:
                raise ConnectionAbortedError(
                    f"Failed to fetch generation from ENTSOE API for region "
                    f"{region['name']} for freq {freq} from {start_date} till {today}."
                )
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname1}")
            df_gen.to_parquet(working_dir + fname1)

        ''' ------------- TOTAL GENERATION FORECAST ------------------ '''
        df_gen_f = None
        fname2 = f"tmp_gen{region['suffix']}_total_forecast_{freq}.parquet"
        cach_fnames.append(fname2)
        if os.path.isfile(working_dir + fname2):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname2}")
            df_gen_f = pd.read_parquet(working_dir + fname2)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- generation forecast aggregated
                    if verbose: logger.info(
                        f"Collecting generation forecast for {region['name']} for freq {freq} from {start_date} to {today}"
                    )
                    df_gen_f = client.query_generation_forecast(
                        country_code=region['name'], start=start_date, end=today
                    )
                    df_gen_f = df_gen_f.rename(
                        f"generation_forecast"
                    )
                    df_gen_f = pd.DataFrame(df_gen_f)
                    df_gen_f.index = pd.to_datetime(df_gen_f.index, utc=True).tz_convert(tz="UTC")
                    df_gen_f.index.name = 'date'
                    df_gen_f.interpolate(method="time", axis=0, inplace=True)
                    if freq == 'hourly': df_gen_f = df_gen_f.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(
                        f"Failed to fetch generation forecast from ENTSOE API ({i}/{5}) for region "
                          f"{region['name']} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(5)
                    continue
                break
            if df_gen_f is None:
                raise ConnectionAbortedError(
                    f"Failed to fetch generation forecast from ENTSOE API for region "
                    f"{region['name']} for freq {freq} from {start_date} till {today}."
                )
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname2}")
            df_gen_f.to_parquet(working_dir + fname2)

        ''' ------------- SOLAR & WIND GENERATION FORECAST ------------- '''
        df_gen_sw_f = None
        fname3 = f"tmp_gen{region['suffix']}_solarwind_forecast_{freq}.parquet"
        cach_fnames.append(fname3)
        if os.path.isfile(working_dir + fname3):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname3}")
            df_gen_sw_f = pd.read_parquet(working_dir + fname3)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(
                        f"Collecting solar & wind generation forecast for {region['name']} "
                        f"for freq {freq} from {start_date} to {today}"
                    )
                    df_gen_sw_f = client.query_wind_and_solar_forecast(
                        country_code=region['name'], start=start_date, end=today
                    )
                    df_gen_sw_f.rename(columns={
                        "Solar":"solar_forecast",
                        "Wind Offshore":"wind_offshore_forecast",
                        "Wind Onshore":"wind_onshore_forecast"
                    }, inplace=True)
                    df_gen_sw_f.index = pd.to_datetime(df_gen_sw_f.index, utc=True).tz_convert(tz="UTC")
                    df_gen_sw_f.index.name = 'date'
                    df_gen_sw_f.interpolate(method="time", axis=0, inplace=True)
                    if freq == 'hourly': df_gen_sw_f = df_gen_sw_f.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(
                        f"Failed to fetch solar & wind generation forecast from ENTSOE API ({i}/{5}) for region "
                        f"{region['name']} for freq {freq} from {start_date} till {today}: \n\t{e}")
                    time.sleep(5)
                    continue
                break
            if df_gen_sw_f is None:
                raise ConnectionAbortedError(
                    f"Failed to fetch solar & wind generation forecast from ENTSOE API for region "
                    f"{region['name']} for freq {freq} from {start_date} till {today}."
                )
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname3}")
            df_gen_sw_f.to_parquet(working_dir + fname3)

        ''' ---------- LOAD ------------ '''
        df_load = None
        fname4 = f"tmp_load{region['suffix']}_forecast_{freq}.parquet"
        cach_fnames.append(fname4)
        if os.path.isfile(working_dir + fname4):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname4}")
            df_load = pd.read_parquet(working_dir + fname4)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(
                        f"Collecting load for {region['name']} for freq {freq} from {start_date} to {today}"
                    )
                    df_load = client.query_load_and_forecast(country_code=region['name'], start=start_date, end=today)
                    df_load.rename(columns={
                        "Forecasted Load": "load_forecast",
                        "Actual Load": "load",
                    }, inplace=True)
                    df_load.index = pd.to_datetime(df_load.index, utc=True).tz_convert(tz="UTC")
                    df_load.index.name = 'date'
                    df_load.interpolate(method="time", axis=0, inplace=True)
                    if freq == 'hourly': df_load = df_load.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(
                        f"Failed to fetch load from ENTSOE API ({i}/{5}) for region "
                        f"{region['name']} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(5)
                    continue
                break
            if df_load is None:
                raise ConnectionAbortedError(
                    f"Failed to fetch load from ENTSOE API for region "
                    f"{region['name']} for freq {freq} from {start_date} till {today}."
                )
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname4}")
            df_load.to_parquet(working_dir + fname4)

        # --- Combine dataframes (assume indexes match)
        df_tot = pd.merge(df_gen, df_gen_sw_f, left_index=True, right_index=True, how="left")
        df_tot = pd.merge(df_tot, df_gen_f, left_index=True, right_index=True, how="left")
        df_tot = pd.merge(df_tot, df_load, left_index=True, right_index=True, how="left")

        # adding suffixes (TSO-specific)
        df_tot.columns = [col + region['suffix'] for col in df_tot.columns]

        # appending to the main dataframe
        if df.empty:  df = df_tot
        else: df = pd.merge(df, df_tot, left_index=True, right_index=True, how="left")


    ''' --------- CROSS-BORDER FLOWS (15min) -------- '''

    neighborhood = NEIGHBOURS[bidding_zone]

    # df_flows = pd.DataFrame()
    # fname5 = f"tmp_flows_{bidding_zone}_{freq}.parquet"
    # cach_fnames.append(fname5)
    # if os.path.isfile(working_dir + fname5):
    #     if verbose: logger.info(f"Loading temporary file: {working_dir + fname5}")
    #     df_flows = pd.read_parquet(working_dir + fname5)
    # else:
    #     # query generation
    #     for i in range(5):
    #         try:
    #             df_flows = pd.DataFrame()
    #             for i, country in enumerate(neighborhood):
    #                 logger.info(f"Processing flows for country {country} ({i}/{len(neighborhood)})")
    #                 if not country in country_code_name_mapping:
    #                     raise KeyError(f"No mapping found for {country} for country code {c_code}")
    #                 time.sleep(10)
    #                 df_export = pd.DataFrame(client.query_crossborder_flows(
    #                     country, bidding_zone, start=start_date, end=today),
    #                     columns=[country_code_name_mapping[country]+'_flow_export']
    #                 )
    #                 time.sleep(10)
    #                 df_import = pd.DataFrame(client.query_crossborder_flows(
    #                     bidding_zone, country, start=start_date, end=today),
    #                     columns=[country_code_name_mapping[country]+'_flow_import']
    #                 )
    #                 df_ = pd.merge(df_export,df_import,left_index=True,right_index=True)
    #                 if df_flows.empty: df_flows = df_.copy()
    #                 else: df_flows = pd.merge(df_flows, df_.copy(), left_index=True, right_index=True, how='left')
    #
    #             if freq == 'hourly': df_flows = df_flows.resample('h').mean()
    #         except Exception as e:
    #             logger.error(
    #                 f"Failed to fetch cross-border flows from ENTSOE API ({i}/{5}) for bidding zone"
    #                 f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
    #             )
    #             time.sleep(5)
    #             continue
    #         break
    #     if df_flows.empty:
    #         raise ConnectionAbortedError(
    #             f"Failed to fetch cross-border flows from ENTSOE API for bidding_zone  "
    #             f"{bidding_zone} (empty df) for freq {freq} from {start_date} till {today}."
    #         )
    #     if verbose: logger.info(f"Saving temporary file: {working_dir + fname5}")
    #     df_flows.to_parquet(working_dir + fname5)
    #
    # # combine
    # df = pd.merge(df, df_flows, left_index=True, right_index=True, how="left")

    df_flows = pd.DataFrame()
    fname6 = f"tmp_flows_{bidding_zone}_{freq}.parquet"
    cach_fnames.append(fname6)
    if os.path.isfile(working_dir + fname6):
        if verbose: logger.info(f"Loading temporary file: {working_dir + fname6}")
        df_flows = pd.read_parquet(working_dir + fname6)
    else:
        for i, country in enumerate(neighborhood):
            if not country in country_code_name_mapping:
                raise KeyError("No mapping found for {}".format(country))

            # export
            df_export = pd.DataFrame()
            for i in range(3):
                logger.info(f"Processing flow export for {c_code} from country {country} ({i}/{len(neighborhood)})")
                time.sleep(3)

                try:
                    df_export = pd.DataFrame(client.query_crossborder_flows(
                        country, bidding_zone, start=start_date, end=today),
                        columns=[country_code_name_mapping[country]+'_flow_export']
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch flows from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                        f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(3)
                    continue # repeat
                break # success
            if df_export.empty:
                logger.error(
                    f"Failed to fetch flows export from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                    f"{bidding_zone} for freq {freq} from {start_date} till {today} after 3 attempts"
                )


            # import
            df_import = pd.DataFrame()
            for i in range(3):
                logger.info(f"Processing flow import for {c_code} from country {country} ({i}/{len(neighborhood)})")
                time.sleep(3)

                try:
                    df_import = pd.DataFrame(client.query_crossborder_flows(
                        bidding_zone, country, start=start_date, end=today),
                        columns=[country_code_name_mapping[country]+'_flow_import']
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch flows from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                        f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(3)
                    continue # repeat
                break # success
            if df_import.empty:
                logger.error(
                    f"Failed to fetch flows import from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                    f"{bidding_zone} for freq {freq} from {start_date} till {today} after 3 attempts"
                )


            # combine
            df_ = pd.merge(df_export,df_import,left_index=True,right_index=True)
            if df_flows.empty: df_flows = df_.copy()
            else: df_flows = pd.merge(df_flows, df_,left_index=True,right_index=True,how='left')
        # check
        if freq == 'hourly': df_flows = df_flows.resample('h').mean()
        if df_flows.empty:
            raise ConnectionAbortedError(
                f"Failed to fetch cross-border flows from ENTSOE API for   "
                f"{bidding_zone} (empty df) for freq {freq} from {start_date} till {today}."
            )
        if verbose: logger.info(f"Saving temporary file: {working_dir + fname6}")
        df_flows.to_parquet(working_dir + fname6)
    # combine
    df = pd.merge(df, df_flows, left_index=True, right_index=True, how="left")



    ''' --------- CROSS-BORDER EXCHANGES (hourly) -------- '''
    df_exchanges = pd.DataFrame()
    fname6 = f"tmp_exchanges_{bidding_zone}_{freq}.parquet"
    cach_fnames.append(fname6)
    if os.path.isfile(working_dir + fname6):
        if verbose: logger.info(f"Loading temporary file: {working_dir + fname6}")
        df_exchanges = pd.read_parquet(working_dir + fname6)
    else:
        for i, country in enumerate(neighborhood):
            if not country in country_code_name_mapping:
                raise KeyError("No mapping found for {}".format(country))

            # export
            df_export = pd.DataFrame()
            for i in range(3):
                logger.info(f"Processing exchange export for {c_code} from country {country} ({i}/{len(neighborhood)})")
                time.sleep(3)

                try:
                    df_export = pd.DataFrame(client.query_scheduled_exchanges(
                        country, bidding_zone, start=start_date, end=today),
                        columns=[country_code_name_mapping[country]+'_exchange_export']
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch exchanges from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                        f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(3)
                    continue # repeat
                break # success
            if df_export.empty:
                logger.error(
                    f"Failed to fetch exchanges export from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                    f"{bidding_zone} for freq {freq} from {start_date} till {today} after 3 attempts"
                )

            # import
            df_import = pd.DataFrame()
            for i in range(3):
                logger.info(f"Processing exchange import for {c_code} from country {country} ({i}/{len(neighborhood)})")
                time.sleep(3)

                try:
                    df_import = pd.DataFrame(client.query_scheduled_exchanges(
                        bidding_zone, country, start=start_date, end=today),
                        columns=[country_code_name_mapping[country]+'_exchange_import']
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch exchanges from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                        f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
                    )
                    time.sleep(3)
                    continue # repeat
                break # success
            if df_import.empty:
                logger.error(
                    f"Failed to fetch flows import from ENTSOE API ({i}/{5}) for country {country} for bidding_zone "
                    f"{bidding_zone} for freq {freq} from {start_date} till {today} after 3 attempts"
                )

            # combine
            df_ = pd.merge(df_export,df_import,left_index=True,right_index=True)
            if df_exchanges.empty: df_exchanges = df_.copy()
            else: df_exchanges = pd.merge(df_exchanges, df_,left_index=True,right_index=True,how='left')
        # check
        if freq == 'hourly': df_exchanges = df_exchanges.resample('h').mean()
        if df_exchanges.empty:
            raise ConnectionAbortedError(
                f"Failed to fetch cross-border exchanges from ENTSOE API for   "
                f"{bidding_zone} (empty df) for freq {freq} from {start_date} till {today}."
            )
        if verbose: logger.info(f"Saving temporary file: {working_dir + fname6}")
        df_exchanges.to_parquet(working_dir + fname6)
    # combine
    df = pd.merge(df, df_exchanges, left_index=True, right_index=True, how="left")




    # df_exchanges = pd.DataFrame()
    # fname6 = f"tmp_exchanges_{bidding_zone}_{freq}.parquet"
    # cach_fnames.append(fname6)
    # if os.path.isfile(working_dir + fname6):
    #     if verbose: logger.info(f"Loading temporary file: {working_dir + fname6}")
    #     df_exchanges = pd.read_parquet(working_dir + fname6)
    # else:
    #     for i in range(5):
    #         try:
    #             df_exchanges = pd.DataFrame()
    #             for i, country in enumerate(neighborhood):
    #                 logger.info(f"Processing exchanges for country {country} ({i}/{len(neighborhood)})")
    #                 if not country in country_code_name_mapping:
    #                     raise KeyError("No mapping found for {}".format(country))
    #                 time.sleep(10)
    #                 df_export = pd.DataFrame(client.query_scheduled_exchanges(
    #                     country, bidding_zone, start=start_date, end=today),
    #                     columns=[country_code_name_mapping[country]+'_exchange_export']
    #                 )
    #                 time.sleep(10)
    #                 df_import = pd.DataFrame(client.query_scheduled_exchanges(
    #                     bidding_zone, country, start=start_date, end=today),
    #                     columns=[country_code_name_mapping[country]+'_exchange_import']
    #                 )
    #                 df_ = pd.merge(df_export,df_import,left_index=True,right_index=True)
    #                 if df_exchanges.empty: df_exchanges = df_.copy()
    #                 else: df_exchanges = pd.merge(df_exchanges, df_.copy(),left_index=True,right_index=True,how='left')
    #
    #             if freq == 'hourly': df_exchanges = df_exchanges.resample('h').mean()
    #         except Exception as e:
    #             logger.error(
    #                 f"Failed to fetch exchanges from ENTSOE API ({i}/{5}) for bidding_zone "
    #                 f"{bidding_zone} for freq {freq} from {start_date} till {today}: \n\t{e}"
    #             )
    #             time.sleep(5)
    #             continue
    #         break
    #     if df_exchanges.empty:
    #         raise ConnectionAbortedError(
    #             f"Failed to fetch cross-border exchanges from ENTSOE API for   "
    #             f"{bidding_zone} (empty df) for freq {freq} from {start_date} till {today}."
    #         )
    #     if verbose: logger.info(f"Saving temporary file: {working_dir + fname6}")
    #     df_exchanges.to_parquet(working_dir + fname6)
    #
    # # combine
    # df = pd.merge(df, df_exchanges, left_index=True, right_index=True, how="left")

    # check that columns match before removing temporary files
    if cols:
        for col_name in cols:
            if not col_name in df.columns.tolist():
                logger.error(
                    f"Expected column name '{col_name}' is not in the collected ENTSOE data. Creating column of zeroes"
                )
                df[col_name] = 0.
        for col_name in df.columns.tolist():
            if not col_name in cols:
                raise KeyError("Column name '{}' from collected data is not expected".format(col_name))

    # remove temporary files
    if verbose: logger.info(
        f"Successfully collected ENTSO-E data for freq {freq} with (df={df.shape}) from {start_date} till {today}. "
        f"Removing {len(cach_fnames)} temporary files..."
    )
    # fname7 = f"tmp_flows_{bidding_zone}.parquet"
    # if os.path.isfile(working_dir + fname7):
    #     os.remove(working_dir + fname7)
    # fname8 = f"tmp_exchanges_{bidding_zone}.parquet"
    # if os.path.isfile(working_dir + fname8):
    #     os.remove(working_dir + fname8)
    for fname in cach_fnames:
        if os.path.isfile(working_dir + fname):
            os.remove(working_dir + fname)

    return df

def create_entsoe_from_api(country_dict:dict, start_date:pd.Timestamp or None, today:pd.Timestamp,data_dir:str,
                           api_key:str,freq:str,verbose:bool):
    if not os.path.isdir(data_dir):
        logger.info(f"Directory {data_dir} does not exist, creating it now...")
        os.mkdir(data_dir)
    df = fetch_entsoe_data_from_api(country_dict, data_dir, start_date, today, api_key, freq, None, verbose)
    fname = data_dir + f'history_{freq}.parquet'
    if not country_dict["code"] in data_dir:
        raise KeyError(f"Expected country code {country_dict['code']} in outdir {data_dir}")
    if verbose: logger.info(
        f"ENTSOE data is successfully collected for country {country_dict['code']} "
        f"for freq {freq}. Shape={df.shape}. Saving into {fname}"
    )
    df.to_parquet(fname)

def update_entsoe_from_api(country_dict:dict,today:pd.Timestamp,data_dir:str,api_key:str,freq:str,verbose:bool):

    fname = data_dir + f'history_{freq}.parquet'
    df_hist:pd.DataFrame = pd.read_parquet(fname)
    df_hist.index = pd.to_datetime(df_hist.index, utc=True)
    start_date = pd.Timestamp(df_hist.index[-1]) - timedelta(days=3) # to override previously incorrect last values
    if verbose: logger.info(
        f"Updating ENTSOE data for freq {freq} from {start_date} till {today}. Current shape={df_hist.shape}"
    )

    df = fetch_entsoe_data_from_api(
        country_dict, data_dir, start_date, today, api_key, freq, df_hist.columns.tolist(), verbose
    )

    compare_columns(df, df_hist)
    if len(df.columns) != len(df_hist.columns):
        raise ValueError(
            f"Historic dataframe has {len(df_hist.columns)} columns for freq {freq}, updated one has {len(df.columns)}"
        )

    combined = df.combine_first(df_hist)
    combined.sort_index(inplace=True)

    if verbose: logger.info(
        f"ENTSOE data is successfully updated for country {country_dict['code']} "
        f"freq {freq}. Shape={df.shape}. Saving into {fname}"
    )
    combined.to_parquet(fname)

if __name__ == '__main__':
    from dotenv import load_dotenv
    # Load .env file
    load_dotenv()
    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")
    start_date = pd.Timestamp(datetime(year=2024, month=2, day=1), tz='UTC')
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    end_date = today
    create_entsoe_from_api(start_date,today,api_key=entsoe_api_key,freq='minutely_15',verbose=True,data_dir='../database_15min/entsoe/')

#
# class DataENTSOE:
#     def __init__(self):
#         pass
#
# def update_entsoe_from_api(today:pd.Timestamp,data_dir:str,verbose):
#     fname = data_dir + 'history_hourly.parquet'
#     df_hist = pd.read_parquet(fname)
#
#     first_timestamp = pd.Timestamp(df_hist.dropna(how='any', inplace=False).first_valid_index())
#     last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
#
#     # ---------- SET UPDATE TIMES ------------
#     start_date = last_timestamp-timedelta(hours=24)
#     end_date = today+timedelta(hours=24)
#
#
# def create_hist(today:pd.Timestamp,data_dir:str,verbose,drop_consumption:bool=True,downsample:bool=True):
#
#     generation_type_mapping = {
#         "actual_load": ["Actual Load"],
#         "hard_coal": ["Fossil Hard coal Actual Aggregated"],
#         "lignite": ["Fossil Brown coal/Lignite Actual Aggregated"],
#         "gas": ["Fossil Gas Actual Aggregated"],
#         "other_fossil": [
#             "Fossil Coal-derived gas Actual Aggregated",
#             "Fossil Oil Actual Aggregated",
#             "Other Actual Aggregated",
#         ],
#         "nuclear": ["Nuclear Actual Aggregated"],
#         "biomass": ["Biomass Actual Aggregated"],
#         "waste": ["Waste Actual Aggregated"],
#         "other_renewable": [
#             "Geothermal Actual Aggregated",
#             "Other renewable Actual Aggregated",
#         ],
#         "hydro": [
#             "Hydro Pumped Storage Actual Aggregated",
#             "Hydro Run-of-river and poundage Actual Aggregated",
#             "Hydro Water Reservoir Actual Aggregated",
#         ],
#         "solar": [
#             "Solar Actual Aggregated",
#         ],
#         "wind_onshore": ["Wind Onshore Actual Aggregated"],
#         "wind_offshore": ["Wind Offshore Actual Aggregated"],
#     }
#
#     start_date = pd.Timestamp(datetime(year=2024, month=11, day=1),tz='UTC')
#     end_date = today
#     client = EntsoePandasClient(api_key="94aa148a-330b-4eee-ba0c-8a5eb0b17825")
#     df_flows = client.query_crossborder_flows(
#         country_code_from='DE', country_code_to='FR', start=start_date, end=end_date
#     )
#     df_load = client.query_load(country_code='DE', start=start_date, end=end_date)
#     df_gen = client.query_generation(country_code='DE', start=start_date, end=end_date, psr_type=None)
#     df_gen.columns = [" ".join(a) for a in df_gen.columns.to_flat_index()]
#
#     df_final = pd.concat( [df_load, df_gen], axis=1 )  # Concatenate dataframes in columns dimension.
#     if "Nuclear Actual Aggregated" in df_final.columns.tolist():
#         df_final["Nuclear Actual Aggregated"] = df_final["Nuclear Actual Aggregated"].fillna(0)
#
#     df_final.index = pd.to_datetime(df_final.index, utc=True).tz_convert(tz="UTC")
#
#     if drop_consumption:  # Drop columns containing actual consumption.
#         df_final.drop(list(df_final.filter(regex="Consumption")), axis=1, inplace=True)
#
#     df_final.interpolate(method="time", axis=0, inplace=True)
#     for joint_category, old_categories in generation_type_mapping.items():
#         existing_columns = [col for col in old_categories if col in df_final.columns]
#         if existing_columns:
#             # Sum up existing columns and drop them
#             df_final[joint_category] = df_final[existing_columns].sum(axis=1, skipna=False)
#             df_final.drop(columns=existing_columns, inplace=True)
#
#     if downsample:
#         df_final = df_final.resample("1h").mean()
#
#     print(df_final.head())
#     print(df_final.columns)
#
# if __name__ == '__main__':
#     today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
#     today = today.normalize() + pd.DateOffset(hours=today.hour)
#     # TODO add tests
#     create_hist(today,data_dir='./database/entsoe',verbose=True)