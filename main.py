import pandas as pd
import os
from dotenv import load_dotenv

from datetime import datetime, timedelta
from data_collection_modules import (
    create_openmeteo_from_api,
    update_openmeteo_from_api,
    update_smard_from_api,
    create_smard_from_api,
    update_epexspot_from_files,
    create_entsoe_from_api,
    update_entsoe_from_api
)



if __name__ == '__main__':

    # Load .env file
    load_dotenv()

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")


    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    db_path = './database/'

    # create_smard_from_api(pd.Timestamp(datetime(year=2015, month=1, day=14),tz='UTC'),
    #                       today=today, data_dir=db_path+'smard/',verbose=True)
    update_smard_from_api(today, db_path + 'smard/', verbose=True)
    update_epexspot_from_files(today, db_path + 'epexspot/', verbose=True)
    # create_openmeteo_from_api(pd.Timestamp(datetime(year=2020, month=1, day=1),tz='UTC'),
    #                           db_path + 'openmeteo/', verbose=True)
    update_openmeteo_from_api(db_path + 'openmeteo/', verbose=True)

    # create_entsoe_from_api(pd.Timestamp(datetime(year=2020, month=1, day=1),tz='UTC'),today,
    #                        db_path + 'entsoe/', entsoe_api_key, verbose=True)
    update_entsoe_from_api(today, db_path + 'entsoe/', entsoe_api_key, verbose=True)