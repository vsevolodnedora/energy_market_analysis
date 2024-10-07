import pandas as pd
import os

from datetime import datetime, timedelta

from data_modules.collect_data import collect_from_api, parse_epexspot, collate_and_update
from ml_modules.lstm_window_stateless_torch import train_predict

if __name__ == '__main__':

    update: bool = True
    start_date = None # if update -- none, infer from last dataset
    data_dir='./database/'
    output_dir='./output/'
    horizon_size = 3*24 # update and forecast window
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    end_date = today + timedelta(hours=horizon_size)
    limit_train_df_past_to = today - timedelta(days=365) # use only one year of dataset (for now)
    train_test_ratio = 0.8
    model_name = 'lstm_fLoadFlow_wPCAtdp1_h72_f3'

    # load dataset and check if it is up-to-date
    df_original = pd.read_parquet(data_dir+'latest.parquet')
    first_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).first_valid_index())
    last_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).last_valid_index())
    print(f"Latest data loaded: Data from {first_timestamp} to {last_timestamp}")
    if last_timestamp >= end_date:
        print(f"Original data is up to date (>={end_date})")
        update = False
    start_date = last_timestamp - timedelta(hours=horizon_size) # to override previous forecasts
    print(f"Start_date={start_date} today={today} end_date={end_date}")

    train_start_date = first_timestamp if limit_train_df_past_to is None else limit_train_df_past_to
    test_start_date = df_original[train_start_date:today].index[
        int(len(df_original[train_start_date:today]) * train_test_ratio)
    ]

    # update dataset if needed and generate new forecasts
    if True:
        print("Updating data")
        parse_epexspot(raw_datadir='./data/DE-LU/DayAhead_MRC/', datadir=data_dir,
                       start_date=start_date, end_date=end_date)

        collect_from_api(today=today, start_date=start_date, end_date=end_date, data_dir=data_dir)

        collate_and_update(
            df_original=df_original, today=today, test_start_date=test_start_date, start_date=start_date,
            data_dir=data_dir, output_dir=output_dir
        )

        # load updated dataset and perform forecast
        df_latest = pd.read_parquet(data_dir+'latest.parquet')

        # train_predict(
        #     df=df_latest[train_start_date:today],today=today,output_dir=output_dir+'lstm_base/',
        #     train=True
        # )


    # forecasting
    # df = pd.read_parquet(data_dir+'latest.parquet')
    # start_date = pd.Timestamp(df.dropna(how='any',inplace=False).first_valid_index())
    # end_date = pd.Timestamp(df.dropna(how='any',inplace=False).last_valid_index())
    # if not limit_df_past_to is None: start_date = limit_df_past_to
    # print(f"Forecast dataset from {start_date} to {end_date}")
    # test_start_date = df[start_date:today].index[ int(len(df[start_date:today]) * train_test_ratio) ]
    # pspl_metrics( df[:today], start_date=test_start_date, out_dir=output_dir )


    # # ----------------------- LOAD DATA & SET TIMES ---------------------
    # today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    # # crop_original = crop_original.normalize() + pd.DateOffset(hours=crop_original.hour)
    # if update:
    #     # load the file to update
    #     if os.path.isfile(data_dir+'latest.parquet'):
    #         df_original = pd.read_parquet(data_dir+'latest.parquet')
    #     elif os.path.isfile(data_dir+'original.parquet'):
    #         df_original = pd.read_parquet(data_dir+'original.parquet')
    #     else:
    #         raise FileNotFoundError(f"File in {data_dir}latest.parquet "
    #                                 f"or {data_dir}original.parquet does not exist")
    #
    #     # remove previous forecasts
    #     df_original = df_original[:(today-timedelta(hours=horizon_size))]
    #
    #     # check frequency
    #     time_diffs = df_original.index.to_series().diff().dropna().unique()
    #     if (time_diffs == '1 hours').all():
    #         df_original = df_original.asfreq('h')
    #
    #     # get start and end of the dataframe
    #     first_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).first_valid_index())
    #     last_timestamp = pd.Timestamp(df_original.dropna(how='any',inplace=False).last_valid_index())
    #     print(f"Original data loaded: Data from {first_timestamp} to {last_timestamp}")
    #     if (last_timestamp.normalize() + pd.DateOffset(hours=today.hour)) >= today - timedelta(hours=horizon_size):
    #         print("Original data is up to date. No updates required")
    #         update = False
    #     # crop for speed
    #     # if crop_original:
    #     #     df_original = df_original.loc[crop_original:]
    #     #     print(f"Crop original data to start from {crop_original}")
    #     # last data is expected to have forecast, so move 2 horizon_sizes back
    #     start_date = last_timestamp - timedelta(hours=horizon_size) # to override previous forecasts
    # else:
    #     start_date = today - timedelta(days=365) #  new data for 1 yesr
    # end_date = today + timedelta(hours=horizon_size)
    # print(f"Start_date={start_date} today={today} end_date={end_date}")
    #
    # # ------------- COLLECT | COLLATE | IMPUTE ----------------
    # if update:
    #     parse_epexspot(raw_datadir='./data/DE-LU/DayAhead_MRC/', datadir=data_dir,
    #                    start_date=start_date, end_date=end_date)
    #     collect_from_api(today=today, start_date=start_date, end_date=end_date, data_dir=data_dir)
    #
    #     # read collected .parquet files and collate with latest full dataset, clean, impute
    #     collate_data(df_original=df_original,start_date=start_date,data_dir='./database/')


    # ------------- MACHINE LEARNING MODEL ----------------
    z =1
