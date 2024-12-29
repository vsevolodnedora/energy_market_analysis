from datetime import datetime, timedelta
from glob import glob
import pandas as pd
import gc

from .utils import validate_dataframe_simple

def update_epexspot_from_files(today:pd.Timestamp,data_dir:str,verbose:bool,raw_data_dir='./data/DE-LU/DayAhead_MRC/'):

    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)

    first_timestamp = pd.Timestamp(df_hist.dropna(how='any', inplace=False).first_valid_index())
    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())

    start_date = last_timestamp - timedelta(hours=24)
    end_date = today + timedelta(hours=24)

    # ----------- UPDATE EPEXTSPOT --------------

    files = glob(raw_data_dir + '*.csv')
    if verbose: print(f"Updating EPEX SPOT data ({raw_data_dir})")

    df_da_upd = pd.DataFrame()
    for file in files:
        df_i = pd.read_csv(file)
        df_da_upd = pd.concat([df_da_upd, df_i])

    if len(files) == 0:
        raise FileNotFoundError(f"File in {raw_data_dir} does not exist")

    df_da_upd['date'] = pd.to_datetime(df_da_upd['date'])
    df_da_upd.sort_values(by='date', inplace=True)
    df_da_upd.drop_duplicates(subset='date', keep='first', inplace=True)

    # for agreement with energy-charts
    df_da_upd['date'] = df_da_upd['date'].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC') #
    df_da_upd.rename(columns={'Price':'DA_auction_price'},inplace=True)

    # we do not need other data for now
    df_da_upd = df_da_upd[['date','DA_auction_price']]
    df_da_upd.set_index('date',inplace=True)
    df_da_upd = df_da_upd[start_date:today]

    for col in df_hist.columns:
        if not col in df_da_upd.columns:
            raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")

        # combine
    df_hist = df_hist.combine_first(df_da_upd)
    if not validate_dataframe_simple(df_hist, text="Updated epexspot df_hist"):
        raise ValueError(f"Failed to validate the updated dataframe for {data_dir}")

    # save
    df_hist.to_parquet(fname)
    if verbose: print(f"Epexspot data is successfully saved to {fname} with shape {df_hist.shape}")

    gc.collect()
