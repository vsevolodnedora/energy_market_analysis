import pandas as pd
import os, sys
from dotenv import load_dotenv

from datetime import datetime
from data_collection_modules import (
    update_openmeteo_from_api,
    update_smard_from_api,
    update_epexspot_from_files,
    update_entsoe_from_api,
    loc_solarfarms,
    loc_onshore_windfarms,
    loc_offshore_windfarms,
    loc_cities,
    OpenMeteo
)

def main(task:str, verbose:bool = True):

    print("launching update_database.py")

    # Load .env file
    load_dotenv()

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    db_path = './database/'

    tasks = [
        'all',
        'update_epexspot'
        'update_entsoe',
        'update_smard',
        'update_openmeteo_windfarms_offshore',
        'update_openmeteo_windfarms_onshore',
        'update_openmeteo_solarfarms',
        'update_openmeteo_cities',
    ]

    # split tasks for lighter workflows
    if task == "update_entsoe" or task == "all":
        update_entsoe_from_api(today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)

    elif task == "update_smard" or task == "all":
        update_smard_from_api(today=today, data_dir=db_path + 'smard/', verbose=verbose)

    elif task == "update_epexspot" or task == "all":
        update_epexspot_from_files(today=today, data_dir=db_path + 'epexspot/', verbose=verbose)

    elif task == "update_openmeteo_windfarms_offshore" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/offshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_offshore_windfarms, verbose = verbose
        )

    elif task == "update_openmeteo_windfarms_onshore" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/onshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_onshore_windfarms, verbose = verbose
        )

    elif task == "update_openmeteo_solarfarms" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/solar_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation),
            locations = loc_solarfarms, verbose = verbose
        )

    elif task == "update_openmeteo_cities" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/cities_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation),
            locations = loc_cities, verbose = verbose
        )
    else:
        raise Exception(f"task {task} is not supported. Supported tasks are: {tasks}")

    print(f"Task '{task}' completed successfully today={today}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python update_database.py <task>")
        sys.exit(1)

    task_argument = str( sys.argv[1] )
    print(f"launching update_database.py for task '{task_argument}'")
    main(task_argument)
