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
    de_regions,
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

    db_path = './database/DE/'

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
        update_epexspot_from_files(today=today, data_dir=db_path + 'epexspot/', freq=freq, verbose=verbose)

    if task == "update_openmeteo_windfarms_offshore" or task == "all":
        for tso_dict in de_regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/offshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in loc_offshore_windfarms if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    if task == "update_openmeteo_windfarms_onshore" or task == "all":
        for tso_dict in de_regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/onshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in loc_onshore_windfarms if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    if task == "update_openmeteo_solarfarms" or task == "all":
        for tso_dict in de_regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/solar/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in loc_solarfarms if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    if task == "update_openmeteo_cities" or task == "all":
        for tso_dict in de_regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/cities/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation) if freq == 'hourly'
                            else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in loc_cities if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Task '{task}' for freq '{freq}' completed successfully. Execution time: {str_time(elapsed_time)}")



if __name__ == '__main__':
    if len(sys.argv) != 3:
        # raise KeyError("Usage: python update_database.py <task> <freq>")
        task_argument = str( 'update_openmeteo_windfarms_onshore' )
        freq = str( 'hourly' )
    else:
        task_argument = str( sys.argv[1] )
        freq = str( sys.argv[2] )
    main(task_argument, freq)
