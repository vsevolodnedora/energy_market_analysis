'''
    Main file for data collection
    See notebooks/SAME_FILE_NAME.ipynb for intercative / manual option
'''
import os.path
from datetime import datetime, timedelta
from glob import glob
import pandas as pd
try: pd.set_option('future.no_silent_downcasting', True)
except: print("Failed to set future.no_silent_downcasting=True")
import numpy as np
import gc

from .collect_data_smard import DataEnergySMARD
from .collect_data_openmeteo import (
    get_weather_data_from_api_forecast,get_weather_data_from_api
)
from .locations import locations

def validate_dataframe(df: pd.DataFrame, text:str = '') -> bool:
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"DataFrame index is not sorted in ascending order | {text}")

    if df.isna().any().any():
        # Find columns with NaN values and their counts
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]

        # Print results
        if nan_columns.empty:
            print(f"No NaN values found in the DataFrame.")
        else:
            print(f"In {text} | Columns with NaN values and their counts:")
            for col, count in nan_columns.items():
                print(f"{col}: {count} NaN values")
        return False

    # Check for infinite values and handle similarly to NaN check
    if np.isinf(df.values).any():
        # Find columns with infinite values and their counts
        inf_counts = np.isinf(df).sum()
        inf_columns = inf_counts[inf_counts > 0]

        if inf_columns.empty:
            print("No infinite values found in the DataFrame.")
        else:
            print(f"In {text} | Columns with infinite values and their counts:")
            for col, count in inf_columns.items():
                print(f"{col}: {count} infinite values")
        return False

    return True

class DataCollector:
    def __init__(self,today:pd.Timestamp,data_dir:str):
        self.data_dir = data_dir
        self.today = today
        self.df_hist = pd.read_parquet(data_dir+'history.parquet')
        self.df_forecast = pd.read_parquet(data_dir+'forecast.parquet')
        print(f"Existing historic dataframe is loaded successfully. Shape {self.df_hist.shape}")
        # ---
        self.first_timestamp = pd.Timestamp(self.df_hist.dropna(how='any', inplace=False).first_valid_index())
        self.last_timestamp = pd.Timestamp(self.df_hist.dropna(how='all', inplace=False).last_valid_index())
        # --- file names for historic data to be updated
        self.fname_smard_upd = 'upd_smard_energy.parquet'
        self.fname_om_hist_upd = 'upd_openmeteo_hist.parquet'
        self.fname_es_hist_upd = 'upd_epexspot.parquet'
        # --- file names for forecasts to be updated
        self.fname_om_forecast_upd = 'upd_openmeteo_forecast.parquet'

    def update(self, force_update:bool)->None:

        if (not force_update) and (self.today <= self.last_timestamp):
            print("Data is up to date")
            return None

        print(f"Data ends on {self.last_timestamp}. Today is {self.today}. Updating... ")

        self._update_smard_from_api(self.last_timestamp-timedelta(hours=24), self.today+timedelta(hours=24))

        self._update_epexspot_from_files(self.last_timestamp-timedelta(hours=24), self.today+timedelta(hours=24))

        self._update_openmeteo_from_api(self.last_timestamp-timedelta(hours=24), self.today-timedelta(hours=12))

        self._update_historical_dataframe()

        self._update_forecast_dataframe()

    def get_hist(self)->pd.DataFrame:
        return self.df_hist

    def get_forecast(self)->pd.DataFrame:
        return self.df_forecast

    # ------------------------------------------------------------------------ #

    def _update_smard_from_api(self, start_date:pd.Timestamp, end_date:pd.Timestamp):
        # ---------- UPDATE SMARD -------------
        print(f"Updating SMARD data from {start_date} to {end_date}")
        o_smard = DataEnergySMARD( start_date=start_date,  end_date=end_date  )
        df_smard_flow = o_smard.get_international_flow()
        df_smard_gen_forecasted = o_smard.get_forecasted_generation()
        df_smard_con_forecasted = o_smard.get_forecasted_consumption()
        df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_on='date',right_on='date',how='outer')
        df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_on='date',right_on='date',how='outer')
        df_smard.set_index('date',inplace=True)
        df_smard.to_parquet(self.data_dir+self.fname_smard_upd,engine='pyarrow')
        print(f'Update for SMARD data is successfully created at '
              f'{self.data_dir+self.fname_smard_upd} with shape {df_smard.shape}')

        gc.collect()

    def _update_epexspot_from_files(self, start_date:pd.Timestamp, end_date:pd.Timestamp):
        # ----------- UPDATE EPEXTSPOT --------------
        epex_spot_fpath = './data/DE-LU/DayAhead_MRC/'
        files = glob(epex_spot_fpath + '*.csv')
        print(f"Updating EPEX SPOT data ({epex_spot_fpath})")
        df_da_upd = pd.DataFrame()
        for file in files:
            df_i = pd.read_csv(file)
            df_da_upd = pd.concat([df_da_upd, df_i])
        if len(files) == 0:
            raise FileNotFoundError(f"File in {epex_spot_fpath} does not exist")
        df_da_upd['date'] = pd.to_datetime(df_da_upd['date'])
        df_da_upd.sort_values(by='date', inplace=True)
        df_da_upd.drop_duplicates(subset='date', keep='first', inplace=True)
        # for agreement with energy-charts
        df_da_upd['date'] = df_da_upd['date'].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC') #
        df_da_upd.rename(columns={'Price':'DA_auction_price'},inplace=True)
        # we do not need other data for now
        df_da_upd = df_da_upd[['date','DA_auction_price']]
        df_da_upd.set_index('date',inplace=True)
        df_da_upd = df_da_upd[start_date:end_date] # for simplicity in concatenating all data
        # df_da_upd.reset_index('date',inplace=True)
        # df_da_upd.head()
        # return df_da_upd
        # df_da_upd.fillna(value=nan_parquet,inplace=True)
        df_da_upd.to_parquet(self.fname_es_hist_upd,engine='pyarrow')
        print(f"Epexspot data is successfully saved to {self.fname_es_hist_upd} with shape {df_da_upd.shape}")

        gc.collect()

    def _update_openmeteo_from_api(self, start_date:pd.Timestamp, end_date:pd.Timestamp):
        # ------- UPDATE WEATHER DATA -------------
        print(f"Updating SMARD data from {self.last_timestamp} to {self.today}")
        df_om_hist = get_weather_data_from_api(
            start_date=start_date,
            today=end_date,
            locations=locations
        )
        df_om_hist.set_index('date',inplace=True)
        df_om_hist.to_parquet(self.data_dir+self.fname_om_hist_upd,engine='pyarrow')
        print(f'Update for openmeteo historical data is successfully created at '
              f'{self.data_dir+self.fname_om_hist_upd} with shape {df_om_hist.shape}')

        gc.collect()

        df_om_forecast = get_weather_data_from_api_forecast(locations=locations)
        df_om_forecast.set_index('date',inplace=True)
        if not df_om_forecast.columns.equals(df_om_hist.columns):
            unique_to_df1 = set(df_om_hist.columns) - set(df_om_forecast.columns)
            unique_to_df2 = set(df_om_forecast.columns) - set(df_om_hist.columns)
            print(f"! Error. Column mismatch between historical and forecasted weather! unique to "
                  f"df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")
        df_om_forecast.to_parquet(self.data_dir+self.fname_om_forecast_upd,engine='pyarrow')
        print(f'Update for openmeteo forecast data is successfully created at '
              f'{self.data_dir+self.fname_om_forecast_upd} with shape {df_om_forecast.shape}')

        gc.collect()

    def _update_historical_dataframe(self):

        # ------- CONCATENATE HISTORIC DATA UPDATES ------------
        # load updates to the data
        df_upd_smard = pd.read_parquet(self.data_dir+self.fname_smard_upd)
        df_upd_om = pd.read_parquet(self.data_dir+self.fname_om_hist_upd)
        df_upd_es = pd.read_parquet(self.data_dir+self.fname_es_hist_upd)
        df_hist_upd = pd.DataFrame(index=df_upd_smard[:self.today].index)
        for df in [df_upd_smard, df_upd_om, df_upd_es]:
            df.fillna(value=np.nan, inplace=True)
            df.infer_objects()
            df = df[:self.today] # hard limit to today for all datasets
            df_hist_upd = pd.merge(left=df_hist_upd, right=df, left_index=True, right_index=True, how='outer')
        # check columns
        for col in self.df_hist.columns:
            if not col in df_hist_upd.columns:
                raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")

        self.df_hist = self.df_hist.combine_first(df_hist_upd)

        validate_dataframe(self.df_hist, text="Updated df_hist")

        self.df_hist.to_parquet(self.data_dir+'history.parquet', engine='pyarrow')
        print(f"History dataframe is successfully updated till {self.today}. N={len(df_hist_upd)} rows are added. "
              f"New shape={self.df_hist.shape}")

        gc.collect()

    def _update_forecast_dataframe(self):
        # ------- CONCATENATE FORECASTS DATA UPDATES ------------
        df_om_forecast = pd.read_parquet(self.data_dir+self.fname_om_forecast_upd)
        df_om_forecast = df_om_forecast[self.today+timedelta(hours=1):] # shift to next hour and limit to
        self.df_forecast = pd.DataFrame(columns=self.df_hist.columns, index=df_om_forecast.index)
        # we want same columns in historical and forecasted data, so we fill them with nans except where we have data
        for col in df_om_forecast.columns:
            self.df_forecast[col] = df_om_forecast[col]
        self.df_forecast.to_parquet(self.data_dir+'forecast.parquet', engine='pyarrow')
        print("Forecast dataframe is successfully updated")

        gc.collect()

    # ------------------------------------------------------------------------ #


if __name__ == '__main__':
    # TODO add tests
    pass