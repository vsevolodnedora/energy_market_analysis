"""
https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py
"""

from entsoe import EntsoePandasClient
import pandas as pd
from datetime import timedelta, datetime

from data_collection_modules.german_locations import de_regions
from data_collection_modules.utils import compare_columns

def preprocess_generation(df_gen:pd.DataFrame, downsample:bool, drop_consumption:bool, verbose:bool)->pd.DataFrame:

    generation_type_mapping = {
        "actual_load": ["Actual Load"],
        "hard_coal": ["Fossil Hard coal Actual Aggregated"],
        "lignite": ["Fossil Brown coal/Lignite Actual Aggregated"],
        "gas": ["Fossil Gas Actual Aggregated"],
        "other_fossil": [
            "Fossil Coal-derived gas Actual Aggregated",
            "Fossil Oil Actual Aggregated",
            "Other Actual Aggregated",
        ],
        "nuclear": ["Nuclear Actual Aggregated"],
        "biomass": ["Biomass Actual Aggregated"],
        "waste": ["Waste Actual Aggregated"],
        "other_renewable": [
            "Geothermal Actual Aggregated",
            "Other renewable Actual Aggregated",
        ],
        "hydro": [
            "Hydro Pumped Storage Actual Aggregated",
            "Hydro Run-of-river and poundage Actual Aggregated",
            "Hydro Water Reservoir Actual Aggregated",
        ],
        "solar": [
            "Solar Actual Aggregated",
        ],
        "wind_onshore": ["Wind Onshore Actual Aggregated"],
        "wind_offshore": ["Wind Offshore Actual Aggregated"],
    }

    df_gen.columns = [" ".join(a) for a in df_gen.columns.to_flat_index()]

    # df_final = pd.concat( [df_load, df_gen], axis=1 )  # Concatenate dataframes in columns dimension.
    if "Nuclear Actual Aggregated" in df_gen.columns.tolist():
        df_gen["Nuclear Actual Aggregated"] = df_gen["Nuclear Actual Aggregated"].fillna(0)

    df_gen.index = pd.to_datetime(df_gen.index, utc=True).tz_convert(tz="UTC")
    df_gen.index.name = 'date'

    if drop_consumption:  # Drop columns containing actual consumption.
        df_gen.drop(list(df_gen.filter(regex="Consumption")), axis=1, inplace=True)

    df_gen.interpolate(method="time", axis=0, inplace=True)
    for joint_category, old_categories in generation_type_mapping.items():
        existing_columns = [col for col in old_categories if col in df_gen.columns]
        if existing_columns:
            # Sum up existing columns and drop them
            df_gen[joint_category] = df_gen[existing_columns].sum(axis=1, skipna=False)
            df_gen.drop(columns=existing_columns, inplace=True)

    if downsample:
        df_gen = df_gen.resample("1h").mean()

    return df_gen

def fetch_entsoe_data_from_api(start_date:pd.Timestamp or None, today:pd.Timestamp,api_key:str,verbose:bool)->pd.DataFrame:
    client = EntsoePandasClient(api_key=api_key)
    # collect solar and wind generation per TSO controlled region
    df = pd.DataFrame()
    for i, region in enumerate(de_regions):
        if verbose: print(f"Requesting generation data for region {region['name']} from {start_date} till {today}")

        df_gen = None
        for i in range(5):
            try:
                df_gen = client.query_generation(country_code=region['name'], start=start_date, end=today, psr_type=None)
                df_gen = preprocess_generation(df_gen, drop_consumption=True, downsample=True, verbose=verbose)
            except Exception as e:
                print(f"Failed to fetch generation from ENTSOE API ({i}/{5}) for region "
                      f"{region['name']} from {start_date} till {today}: \n\t{e}")
                continue
            break
        if df_gen is None:
            raise ConnectionResetError(
                f"Failed to fetch generation data from ENTSOE API for region "
                f"{region['name']} from {start_date} till {today}"
            )

        df_gen.columns = [col + region['suffix'] for col in df_gen.columns]

        if df.empty:  df = df_gen
        else: df = pd.merge(df, df_gen, left_index=True, right_index=True, how="left")

    return df

def create_entsoe_from_api(start_date:pd.Timestamp or None, today:pd.Timestamp,data_dir:str,api_key:str,verbose:bool):
    df = fetch_entsoe_data_from_api(start_date, today, api_key, verbose)
    fname = data_dir + 'history.parquet'
    if verbose: print(f"ENTSOE data is successfully collected. Shape={df.shape}. Saving into {fname}")
    df.to_parquet(fname)

def update_entsoe_from_api(today:pd.Timestamp,data_dir:str,api_key:str,verbose:bool):

    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)
    df_hist.index = pd.to_datetime(df_hist.index, utc=True)
    start_date = pd.Timestamp(df_hist.index[-1]) - timedelta(days=3) # to override previously incorrect last values
    if verbose: print(f"Updating ENTSOE data from {start_date} till {today}. Current shape={df_hist.shape}")
    df = fetch_entsoe_data_from_api(start_date, today, api_key, verbose)
    compare_columns(df, df_hist)
    if len(df.columns) != len(df_hist.columns):
        raise ValueError(f"Historic dataframe has {len(df_hist.columns)} columns, updated one has {len(df.columns)}")
    combined = df.combine_first(df_hist)
    combined.sort_index(inplace=True)
    if verbose: print(f"ENTSOE data is successfully updated. Shape={df.shape}. Saving into {fname}")
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