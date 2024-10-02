import pandas as pd
from datetime import datetime, timedelta

from data_modules.collect_data import collect, parse_epexspot

if __name__ == '__main__':

    update:bool = True
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    crop_original = pd.Timestamp(today - timedelta(days=365))
    # crop_original = pd.Timestamp(datetime(year=2023, month=1, day=1), tz='UTC')

    horizon_size = 3*24 # hours

    # ------------- Collect updated data ----------------
    parse_epexspot(raw_datadir='./data/DE-LU/DayAhead_MRC/', datadir='./database/')
    # collect data from api
    collect(today=today,update=update,crop_original=crop_original,horizon_size=horizon_size,
            data_dir='./database/')