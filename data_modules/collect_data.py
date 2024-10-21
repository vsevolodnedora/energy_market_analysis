'''
    Main file for data collection
    See notebooks/SAME_FILE_NAME.ipynb for intercative / manual option
'''
import os.path
from datetime import datetime, timedelta
from glob import glob
import pandas as pd; pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from .collect_data_smard import DataEnergySMARD
from .collect_data_openmeteo import (
    get_weather_data_from_api_forecast,get_weather_data_from_api
)
from .pslp import calculate_pslps
from .locations import locations

def merge_original_and_updates(df_original:pd.DataFrame, data_dir:str):
    # load updates to the data
    df_upd_smard = pd.read_parquet(data_dir+'upd_smard_energy.parquet')
    df_upd_om = pd.read_parquet(data_dir+'upd_openweather.parquet')
    df_upd_es = pd.read_parquet(data_dir+'upd_epexspot.parquet')
    for df in [df_upd_smard,df_upd_om,df_upd_es,df_upd_smard]:
        df.fillna(value=np.nan,inplace=True)
        df.infer_objects()
    # merge updates to a single dataframe (combine columns, essencially)
    # df_upd = df_upd_smard.join(df_upd_om, how='left')
    # df_upd = df_upd.join(df_upd_es, how='left')

    # df_upd = pd.merge(left=df_upd_smard, right=df_upd_es, on='date', how='outer')
    # df_upd = pd.merge(left=df_upd, right=df_upd_om, on='date', how='outer')

    df_upd = pd.merge(left=df_upd_smard, right=df_upd_es, left_index=True, right_index=True, how='outer')
    df_upd = pd.merge(left=df_upd, right=df_upd_om, left_index=True, right_index=True, how='outer')

    # df_upd.dropna(inplace=True,how='all'

    # check columns
    for col in df_original.columns:
        if not col in df_upd.columns:
            raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")
    for col in df_upd.columns:
        if not col in df_original.columns:
            print(f"! Warning col={col} is not in the original dataframe")

    # combine dataframes preferring updated data over original
    df = df_original.combine_first(df_upd)
    # Where df2 has non-NaN values, override with df2's values
    df.update(df_upd)

    return df

def collect_from_api(today:pd.Timestamp,start_date:pd.Timestamp, end_date:pd.Timestamp, data_dir:str):

    # --------- COLLECT ENERGY GENERATION & LOAD DATA ---------------------
    o_smard = DataEnergySMARD(start_date=start_date, end_date=end_date)
    df_smard_flow = o_smard.get_international_flow()
    df_smard_gen_forecasted = o_smard.get_forecasted_generation()
    df_smard_con_forecasted = o_smard.get_forecasted_consumption()
    df_smard = pd.merge(left=df_smard_flow,right=df_smard_gen_forecasted,left_on='date',right_on='date',how='outer')
    df_smard = pd.merge(left=df_smard,right=df_smard_con_forecasted,left_on='date',right_on='date',how='outer')
    df_smard.set_index('date',inplace=True)
    df_smard = df_smard[start_date:end_date]
    # df_smard.fillna(value=nan_parquet,inplace=True)
    df_smard.to_parquet(data_dir+'upd_smard_energy.parquet',engine='pyarrow')
    print(f"Smard data is successfully collected in {data_dir}upd_smard_energy.parquet")

    # --------- COLLECT ELECTRICITY PRICES DATA ----------------------------
    pass

    # --------- COLLECT WEATHERDATA ----------------------------------------
    df_om_hist = get_weather_data_from_api(start_date, today-timedelta(hours=12), locations)
    df_om_forecast = get_weather_data_from_api_forecast(locations=locations)
    if not df_om_forecast.columns.equals(df_om_hist.columns):
        print("! Error. Column mismatch between historical and forecasted weather!")
    df_om = pd.concat([df_om_hist, df_om_forecast[df_om_hist.columns]], ignore_index=True)
    df_om.drop_duplicates(subset='date', keep='last', inplace=True)
    # df_om = process_weather_quantities(df_om, locations)
    df_om.set_index('date',inplace=True)
    df_om = df_om[start_date:end_date]
    # df_om.fillna(value=nan_parquet,inplace=True)
    df_om.to_parquet(data_dir+'upd_openweather.parquet',engine='pyarrow')
    print(f"Openweather data is successfully collected in {data_dir}upd_openweather.parquet")

def parse_epexspot(raw_datadir:str, datadir:str, start_date:pd.Timestamp, end_date:pd.Timestamp):
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
    df_da_upd.rename(columns={'Price':'DA_auction_price'},inplace=True)
    # we do not need other data for now
    df_da_upd = df_da_upd[['date','DA_auction_price']]
    df_da_upd.set_index('date',inplace=True)
    df_da_upd = df_da_upd[start_date:end_date]
    # df_da_upd.reset_index('date',inplace=True)
    # df_da_upd.head()
    # return df_da_upd
    # df_da_upd.fillna(value=nan_parquet,inplace=True)
    df_da_upd.to_parquet(datadir+'upd_epexspot.parquet',engine='pyarrow')
    print(f"Epexspot data is successfully saved to {datadir}upd_epexspot.parquet")


def performance_metrics_pslp(df:pd.DataFrame, df_pslp:pd.DataFrame, output_dir:str):
    # Create a dictionary to store metrics for each column
    evaluation = {col: {} for col in df_pslp.columns}  # Predefine the structure for each column
    for col in df_pslp.columns:
        actual_values = df[col].fillna(0.).infer_objects(copy=False).to_numpy()
        predicted_values = df_pslp[col].fillna(0.).infer_objects(copy=False).to_numpy()

        evaluation[col]['MAE'] = mean_absolute_error(actual_values, predicted_values)
        evaluation[col]['MSE'] = mean_squared_error(actual_values, predicted_values)
        evaluation[col]['R2'] = r2_score(actual_values, predicted_values)
        evaluation[col]['MAPE'] = mean_absolute_percentage_error(actual_values, predicted_values)

    # Convert the dictionary into a DataFrame
    df_metrics = pd.DataFrame.from_dict(evaluation, orient='index')  # 'index' makes the column names as index
    df_metrics.index.name = 'Column'
    df_metrics.reset_index(inplace=True)  # If you prefer the columns to be one of the dataframe columns rather than the index
    df_metrics.to_csv(output_dir+'/pslp_metrics.csv', index=False)

def visualize_pslp_vs_actual(df:pd.DataFrame, df_pslp:pd.DataFrame,  output_dir:str):
    ''' plot two  '''
    # Ensure both dataframes are indexed by datetime
    # df.index = pd.to_datetime(df.index)
    # df_pslp.index = pd.to_datetime(df_pslp.index)

    # Number of columns to plot
    num_columns = len(df.columns)

    # Create a figure and set of subplots
    fig, axs = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, num_columns * 3), sharex=True)

    # If there's only one column, axs might not be an array, handle this case
    if num_columns == 1:
        axs = [axs]

    # Loop over each column and plot the respective data
    for i, column in enumerate(df.columns):
        # Original data
        axs[i].plot(df.index, df[column], linestyle='-', color='gray',linewidth=0.5,alpha=0.5, label='Original',)
        axs[i].plot(df_pslp.index, df_pslp[column], linestyle=':', color='gray',linewidth=0.5,alpha=0.5, label='PSLP')

        # Residuals
        residual = df[column] - df_pslp[column]
        axs[i].plot(df.index, residual, linestyle='-', color='black', linewidth=0.8, label='Residual')

        # Set title to column name
        axs[i].set_ylabel(column)

        # Add grid, legend, and labels
        axs[i].grid(True)
        axs[i].legend()

    # Label the x-axis with dates more neatly
    plt.xticks(rotation=45)
    plt.xlabel('Date')

    # Adjust layout so plots don't overlap
    plt.tight_layout()
    plt.savefig(output_dir+'/pslp_vs_actual.png')
    # Show the plot
    # plt.show()


def collate_and_update(df_original:pd.DataFrame, today:pd.Timestamp, test_start_date:pd.Timestamp,
                       start_date:pd.Timestamp, data_dir:str, output_dir:str):

    # --------- COMBINE DATAFRAMES ------------------
    df_updated = merge_original_and_updates(df_original=df_original,data_dir=data_dir)

    if df_updated.isna().sum().any():
        print(f"Completing dataset using PSPL starting at {test_start_date}")

        # impute nans at the end of the dataframe
        df_pslp = calculate_pslps(
            df=df_updated.copy(deep=True),#crop_dataframe_to_last_full_day(df_updated).copy(deep=True),
            start_date=test_start_date,
            lookback=14,
            country_code="DE"
        )
        df_pslp.dropna(how='all', inplace=True)
        df_pslp.set_index('date',inplace=True)

        # estimate performance metrics in approximating new values with PSLP
        performance_metrics_pslp(
            df=df_updated[test_start_date:today], df_pslp=df_pslp[test_start_date:today], output_dir=output_dir)

        visualize_pslp_vs_actual(
            df=df_updated[test_start_date:today], df_pslp=df_pslp[test_start_date:today], output_dir=output_dir)

        df_pslp = df_pslp[start_date:]

        # Combine the dataframes, preferring values from df_updated
        result_df = df_updated.combine_first(df_pslp)

        # Overwrite NaNs in df1 with values from df2 where overlapping
        for col in df_updated.columns:
            result_df[col] = df_updated[col].combine_first(df_pslp[col])
        if (result_df.isna().sum() > 0).any():
            raise ValueError("There are NaN values in final dataframe")

        df_updated = result_df

    df_updated.to_parquet(data_dir+'latest.parquet', engine='pyarrow')
    print(f"Latest dataset is successfully saved to {data_dir}latest.parquet")

if __name__ == '__main__':
    # TODO add tests
    pass