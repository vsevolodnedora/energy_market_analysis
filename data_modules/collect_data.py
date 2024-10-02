'''
    Main file for data collection
    See notebooks/SAME_FILE_NAME.ipynb for intercative / manual option
'''
import os.path
from datetime import datetime, timedelta
from glob import glob
import pandas as pd

from .collect_data_smard import DataEnergySMARD
from .collect_data_openmeteo import (
    get_weather_data_from_api_forecast,get_weather_data_from_api,process_weather_quantities
)
from .locations import locations

def collect(today:pd.Timestamp,update:bool,crop_original:pd.Timestamp or None,horizon_size:int,data_dir:str):

    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    crop_original = crop_original.normalize() + pd.DateOffset(hours=crop_original.hour)
    if update:
        # load the file to update
        if os.path.isfile(data_dir+'latest.parquet'):
            df_original = pd.read_parquet(data_dir+'latest.parquet')
        elif os.path.isfile(data_dir+'original.parquet'):
            df_original = pd.read_parquet(data_dir+'original.parquet')
        else:
            raise FileNotFoundError(f"File in {data_dir}latest.parquet "
                                    f"or {data_dir}original.parquet does not exist")

        # check frequency
        time_diffs = df_original.index.to_series().diff().dropna().unique()
        if (time_diffs == '1 hours').all():
            df_original = df_original.asfreq('h')

        # get start and end of the dataframe
        first_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).first_valid_index())
        last_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).last_valid_index())
        print(f"Original data loaded: Data from {first_timestamp} to {last_timestamp}")

        # crop for speed
        if crop_original:
            df_original = df_original.loc[crop_original:]
            print(f"Crop original data to start from {crop_original}")
        # last data is expected to have forecast, so move 2 horizon_sizes back
        start_date = last_timestamp - timedelta(hours=2*horizon_size) # to override previous forecasts
    else:
        start_date = today - timedelta(days=365) #  new data for 1 yesr
    end_date = today + timedelta(hours=horizon_size)
    print(f"Start_date={start_date} today={today} end_date={end_date}")

    # --------- COLLECT ENERGY GENERATION & LOAD DATA ---------------------
    o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
    df_smard_flow = o_smard.get_international_flow()
    df_smard_gen_forecasted = o_smard.get_forecasted_generation()
    df_smard_con_forecasted = o_smard.get_forecasted_consumption()
    df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_on='date',right_on='date',how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_on='date',right_on='date',how='outer')
    df_smard.to_parquet(data_dir+'upd_smard_energy.parquet')
    print(f"Smard data is successfully collected in {data_dir}upd_smard_energy.parquet")

    # --------- COLLECT ELECTRICITY PRICES DATA ----------------------------

    # --------- COLLECT WEATHERDATA ----------------------------------------
    df_om_hist = get_weather_data_from_api(start_date, today-timedelta(hours=12), locations)
    df_om_forecast = get_weather_data_from_api_forecast(locations=locations)
    if not df_om_forecast.columns.equals(df_om_hist.columns):
        print("! Error. Column mismatch between historical and forecasted weather!")
    df_om = pd.concat([df_om_hist, df_om_forecast[df_om_hist.columns]], ignore_index=True)
    df_om.drop_duplicates(subset='date', keep='last', inplace=True)
    df_om=process_weather_quantities(df_om,locations)
    df_om.set_index('date', inplace=True)
    df_om.to_parquet(data_dir+'upd_openweather.parquet')
    print(f"Openweather data is successfully collected in {data_dir}upd_openweather.parquet")


def parse_epexspot(raw_datadir:str, datadir:str):
    # updated data from epex-spot (date is in ECT)
    # path_to_epex_spot_scraped_data = '/home/vsevolod/Work_DS/GIT/GitHub/epex_de_collector/data/DE-LU/DayAhead_MRC/'
    files = glob(raw_datadir + '*.csv')
    df_da_upd = pd.DataFrame()
    for file in files:
        df_i = pd.read_csv(file)
        df_da_upd = pd.concat([df_da_upd, df_i])
    if len(files) == 0:
        raise FileNotFoundError(f"File in {raw_datadir} does not exist")
    df_da_upd['date'] = pd.to_datetime(df_da_upd['date'])
    df_da_upd.sort_values(by='date', inplace=True)
    df_da_upd.drop_duplicates(subset='date', keep='first', inplace=True)
    # for agreement with energy-charts
    df_da_upd['date'] = df_da_upd['date'].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC') #
    df_da_upd.set_index('date',inplace=True)
    # df_da_upd = df_da_upd[start_date:end_date]
    # df_da_upd.reset_index('date',inplace=True)
    # df_da_upd.head()
    # return df_da_upd
    df_da_upd.to_parquet(datadir+'upd_epexspot.parquet')
    print(f"Epexspot data is successfully saved to {datadir}upd_epexspot.parquet")

if __name__ == '__main__':
    # TODO add tests
    pass