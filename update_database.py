import pandas as pd
import os, sys, time
from dotenv import load_dotenv

from logger import get_logger
logger = get_logger(__name__)


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

def str_time(elapsed_time:float) -> str:
    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main(task:str, freq:str, verbose:bool = True):

    print("launching update_database.py")

    # Load .env file
    load_dotenv()

    if freq not in ['hourly','minutely_15']:
        raise ValueError('freq must be either "hourly" or "minutely_15"')


    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    if freq == 'hourly':
        db_path = './database/'
    else:
        db_path = './database_15min/'

    tasks = [
        'all',
        'update_epexspot',
        'update_entsoe',
        'update_smard',
        'update_openmeteo_windfarms_offshore',
        'update_openmeteo_windfarms_onshore',
        'update_openmeteo_solarfarms',
        'update_openmeteo_cities',
    ]
    if task not in tasks:
        raise Exception(f"task {task} is not supported. Supported tasks are: {tasks}")

    logger.info(f"Starting task {task} for frequency {freq}...")

    start_time = time.time()

    # split tasks for lighter workflows
    if task == "update_entsoe" or task == "all":
        update_entsoe_from_api(
            today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, freq=freq, verbose=verbose
        )

    if task == "update_smard" or task == "all":
        update_smard_from_api(
            today=today, data_dir=db_path + 'smard/', freq=freq, verbose=verbose
        )

    if task == "update_epexspot" or task == "all":
        update_epexspot_from_files(today=today, data_dir=db_path + 'epexspot/', verbose=verbose)

    if task == "update_openmeteo_windfarms_offshore" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/offshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_offshore_windfarms, freq=freq, verbose = verbose
        )

    if task == "update_openmeteo_windfarms_onshore" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/onshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_onshore_windfarms, freq=freq, verbose = verbose
        )

    if task == "update_openmeteo_solarfarms" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/solar_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation),
            locations = loc_solarfarms, freq=freq, verbose = verbose
        )

    if task == "update_openmeteo_cities" or task == "all":
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/cities_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation),
            locations = loc_cities, freq=freq, verbose = verbose
        )

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Task '{task}' for freq '{freq}' completed successfully. Execution time: {str_time(elapsed_time)}")



if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise KeyError("Usage: python update_database.py <task> <freq>")

    task_argument = str( sys.argv[1] )
    freq = str( sys.argv[2] )
    main(task_argument, freq)
