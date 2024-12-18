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

    verbose = True
    task = 'update'
    tasks = ['update', 'create_smard', 'create_openmeteo', 'create_entsoe']

    if task not in tasks:
        raise ValueError(f'Invalid task {task}')

    if task == 'create_smard':
        create_smard_from_api(pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC'),
                              today=today, data_dir=db_path+'smard/',verbose=verbose)

    elif task == 'create_openmeteo':
        create_openmeteo_from_api(pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC'),
                                  db_path + 'openmeteo/', verbose=verbose)

    elif task == 'create_entsoe':
        create_entsoe_from_api(pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC'),
                               today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)

    elif task == 'update':
        update_smard_from_api(today=today, data_dir=db_path + 'smard/', verbose=verbose)
        update_openmeteo_from_api(data_dir=db_path + 'openmeteo/', verbose=verbose)
        update_epexspot_from_files(today=today, data_dir=db_path + 'epexspot/', verbose=verbose)
        update_entsoe_from_api(today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)
