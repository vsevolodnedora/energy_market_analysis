import pandas as pd
import os, sys
from dotenv import load_dotenv

from datetime import datetime
from data_collection_modules import (
    create_openmeteo_from_api,
    create_smard_from_api,
    create_entsoe_from_api,
    loc_solarfarms,
    loc_onshore_windfarms,
    loc_offshore_windfarms,
    loc_cities,
    OpenMeteo
)

def main(task:str, verbose : bool = True):

    # Load .env file
    load_dotenv()

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    start_date = pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC') # for initial collection

    db_path = './database/'

    tasks = [
        'create_smard',
        'create_entsoe',
        'create_openmeteo_windfarms_offshore',
        'create_openmeteo_windfarms_onshore',
        'create_openmeteo_solarfarms',
        'create_openmeteo_cities',
    ]

    print(f"Starting task {task} for start date {start_date} up to date {today}")

    if task == 'create_smard':
        create_smard_from_api(start_date=start_date,
                              today=today, data_dir=db_path+'smard/',verbose=verbose)

    elif task == 'create_entsoe':
        create_entsoe_from_api(start_date=start_date,
                               today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)

    # due to file size limitations on GitHub we need to split the openmeteo data into different files
    elif task == 'create_openmeteo_windfarms_offshore':
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/offshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_offshore_windfarms, start_date = start_date, verbose = verbose
        )

    elif task == 'create_openmeteo_windfarms_onshore':
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/onshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_onshore_windfarms, start_date = start_date, verbose = verbose
        )

    elif task == 'create_openmeteo_solarfarms':
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/solar_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation),
            locations = loc_solarfarms, start_date = start_date, verbose = verbose
        )

    elif task == 'create_openmeteo_cities':
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/cities_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation),
            locations = loc_cities, start_date = start_date, verbose = verbose
        )

    else:
        raise ValueError(f'Invalid task {task}. Available tasks: {tasks}')

if __name__ == '__main__':

    print("launching create_database.py")

    if len(sys.argv) != 2:
        print("Usage: python update_database.py <task>")
        sys.exit(1)

    task_argument = str(sys.argv[1])
    main(task_argument)
