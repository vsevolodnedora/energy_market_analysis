import pandas as pd
import os

from datetime import datetime, timedelta
from data_collection_modules import (
    create_openmeteo_from_api,
    update_openmeteo_from_api,
    update_smard_from_api,
    create_smard_from_api,
    update_epexspot_from_files
)



if __name__ == '__main__':

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    db_path = './database/'

    # create_smard_from_api(pd.Timestamp(datetime(year=2015, month=1, day=14),tz='UTC'),
    #                       today=today, data_dir=db_path+'smard/',verbose=True)
    update_smard_from_api(today, db_path + 'smard/', verbose=True)
    update_epexspot_from_files(today, db_path + 'epexspot/', verbose=True)
    # create_openmeteo_from_api(pd.Timestamp(datetime(year=2015, month=1, day=1),tz='UTC'),
    #                           today, db_path + 'openmeteo/', verbose=True)
    update_openmeteo_from_api(today, db_path + 'openmeteo/', verbose=True)


