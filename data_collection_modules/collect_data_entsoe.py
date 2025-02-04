"""
https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py
"""

from entsoe import EntsoePandasClient
import pandas as pd
import time, os
from datetime import timedelta, datetime

from data_collection_modules.german_locations import de_regions
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

def preprocess_generation(df_gen:pd.DataFrame, drop_consumption:bool, verbose:bool)->pd.DataFrame:

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

    df_gen = df_gen.resample("1h").mean() # average power in a given hour
    if 'nuclear' in df_gen.columns: df_gen.drop(columns=['nuclear'], inplace=True)
    return df_gen

def fetch_entsoe_data_from_api(working_dir:str, start_date:pd.Timestamp or None, today:pd.Timestamp,api_key:str,verbose:bool)->pd.DataFrame:

    client = EntsoePandasClient(api_key=api_key)

    df = pd.DataFrame()
    for i, region in enumerate(de_regions):
        if verbose: logger.info(f"Requesting ENTSO-E data for region {region['name']} from {start_date} till {today}")

        ''' ------------ GENERATION ENERGY MIX ---------------- '''
        df_gen = None
        fname = f"tmp_gen{region['suffix']}_hist.parquet"
        if os.path.isfile(working_dir + fname):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname}")
            df_gen = pd.read_parquet(working_dir + fname)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(f"Collecting generation for {region['name']} from {start_date} to {today}")
                    df_gen = client.query_generation(country_code=region['name'], start=start_date, end=today,psr_type=None)
                    df_gen = preprocess_generation(df_gen, drop_consumption=True, verbose=verbose)
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(f"Failed to fetch generation from ENTSOE API ({i}/{5}) for region "
                          f"{region['name']} from {start_date} till {today}: \n\t{e}")
                    time.sleep(5)
                    continue
                break
            if df_gen is None:
                raise ConnectionAbortedError(f"Failed to fetch generation from ENTSOE API for region "
                                             f"{region['name']} from {start_date} till {today}.")
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname}")
            df_gen.to_parquet(working_dir + fname)

        ''' ------------- TOTAL GENERATION FORECAST ------------------ '''
        df_gen_f = None
        fname = f"tmp_gen{region['suffix']}_total_forecast.parquet"
        if os.path.isfile(working_dir + fname):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname}")
            df_gen_f = pd.read_parquet(working_dir + fname)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- generation forecast aggregated
                    if verbose: logger.info(f"Collecting generation forecast for {region['name']} from {start_date} to {today}")
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
                    df_gen_f = df_gen_f.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(f"Failed to fetch generation forecast from ENTSOE API ({i}/{5}) for region "
                          f"{region['name']} from {start_date} till {today}: \n\t{e}")
                    time.sleep(5)
                    continue
                break
            if df_gen_f is None:
                raise ConnectionAbortedError(f"Failed to fetch generation forecast from ENTSOE API for region "
                                             f"{region['name']} from {start_date} till {today}.")
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname}")
            df_gen_f.to_parquet(working_dir + fname)

        ''' ------------- SOLAR & WIND GENERATION FORECAST ------------- '''
        df_gen_sw_f = None
        fname = f"tmp_gen{region['suffix']}_solarwind_forecast.parquet"
        if os.path.isfile(working_dir + fname):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname}")
            df_gen_sw_f = pd.read_parquet(working_dir + fname)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(
                        f"Collecting solar & wind generation forecast for {region['name']} from {start_date} to {today}"
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
                    df_gen_sw_f = df_gen_sw_f.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(f"Failed to fetch solar & wind generation forecast from ENTSOE API ({i}/{5}) for region "
                          f"{region['name']} from {start_date} till {today}: \n\t{e}")
                    time.sleep(5)
                    continue
                break
            if df_gen_sw_f is None:
                raise ConnectionAbortedError(
                    f"Failed to fetch solar & wind generation forecast from ENTSOE API for region "
                    f"{region['name']} from {start_date} till {today}."
                )
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname}")
            df_gen_sw_f.to_parquet(working_dir + fname)

        ''' ---------- LOAD ------------ '''
        df_load = None
        fname = f"tmp_load{region['suffix']}_forecast.parquet"
        if os.path.isfile(working_dir + fname):
            if verbose: logger.info(f"Loading temporary file: {working_dir + fname}")
            df_load = pd.read_parquet(working_dir + fname)
        else:
            # query generation
            for i in range(5):
                try:
                    # --- REALIZED GENERATION ---
                    if verbose: logger.info(f"Collecting load for {region['name']} from {start_date} to {today}")
                    df_load = client.query_load_and_forecast(country_code=region['name'], start=start_date, end=today)
                    df_load.rename(columns={
                        "Forecasted Load": "load_forecast",
                        "Actual Load": "load",
                    }, inplace=True)
                    df_load.index = pd.to_datetime(df_load.index, utc=True).tz_convert(tz="UTC")
                    df_load.index.name = 'date'
                    df_load.interpolate(method="time", axis=0, inplace=True)
                    df_load = df_load.resample('h').mean()
                    time.sleep(5) # not to trigger ENTSO-E API abort
                except Exception as e:
                    logger.error(f"Failed to fetch load from ENTSOE API ({i}/{5}) for region "
                          f"{region['name']} from {start_date} till {today}: \n\t{e}")
                    time.sleep(5)
                    continue
                break
            if df_load is None:
                raise ConnectionAbortedError(f"Failed to fetch load from ENTSOE API for region "
                                             f"{region['name']} from {start_date} till {today}.")
            if verbose: logger.info(f"Saving temporary file: {working_dir + fname}")
            df_load.to_parquet(working_dir + fname)

        # --- Combine dataframes (assume indexes match)
        df_tot = pd.merge(df_gen, df_gen_sw_f, left_index=True, right_index=True, how="left")
        df_tot = pd.merge(df_tot, df_gen_f, left_index=True, right_index=True, how="left")
        df_tot = pd.merge(df_tot, df_load, left_index=True, right_index=True, how="left")

        # adding suffixes (TSO-specific)
        df_tot.columns = [col + region['suffix'] for col in df_tot.columns]

        # appending to the main dataframe
        if df.empty:  df = df_tot
        else: df = pd.merge(df, df_tot, left_index=True, right_index=True, how="left")

    if verbose: logger.info(f"Successfully collected ENTSO-E data (df={df.shape}) from {start_date} till {today}. "
                      f"Removing temporary files...")

    for i, region in enumerate(de_regions):
        fnames = [
            f"tmp_gen{region['suffix']}_hist.parquet",
            f"tmp_gen{region['suffix']}_total_forecast.parquet",
            f"tmp_gen{region['suffix']}_solarwind_forecast.parquet",
            f"tmp_load{region['suffix']}_forecast.parquet"
        ]
        for fname in fnames:
            if os.path.isfile(working_dir + fname):
                os.remove(working_dir + fname)

    return df

def create_entsoe_from_api(start_date:pd.Timestamp or None, today:pd.Timestamp,data_dir:str,api_key:str,verbose:bool):
    df = fetch_entsoe_data_from_api(data_dir, start_date, today, api_key, verbose)
    fname = data_dir + 'history.parquet'
    if verbose: logger.info(f"ENTSOE data is successfully collected. Shape={df.shape}. Saving into {fname}")
    df.to_parquet(fname)

def update_entsoe_from_api(today:pd.Timestamp,data_dir:str,api_key:str,verbose:bool):

    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)
    df_hist.index = pd.to_datetime(df_hist.index, utc=True)
    start_date = pd.Timestamp(df_hist.index[-1]) - timedelta(days=3) # to override previously incorrect last values
    if verbose: logger.info(f"Updating ENTSOE data from {start_date} till {today}. Current shape={df_hist.shape}")
    df = fetch_entsoe_data_from_api(data_dir, start_date, today, api_key, verbose)
    compare_columns(df, df_hist)
    if len(df.columns) != len(df_hist.columns):
        raise ValueError(f"Historic dataframe has {len(df_hist.columns)} columns, updated one has {len(df.columns)}")
    combined = df.combine_first(df_hist)
    combined.sort_index(inplace=True)
    if verbose: logger.info(f"ENTSOE data is successfully updated. Shape={df.shape}. Saving into {fname}")
    combined.to_parquet(fname)

if __name__ == '__main__':
    pass

#
# class DataENTSOE:
#     def __init__(self):
#         pass
#
# def update_entsoe_from_api(today:pd.Timestamp,data_dir:str,verbose):
#     fname = data_dir + 'history.parquet'
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