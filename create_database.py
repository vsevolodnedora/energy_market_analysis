import pandas as pd
import os, sys, time
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
    de_regions,
    OpenMeteo
)

from logger import get_logger
logger = get_logger(__name__)


def str_time(elapsed_time:float) -> str:
    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main(task:str, freq:str, verbose : bool = True):

    if not freq in ['hourly', 'minutely_15']:
        raise ValueError('freq must be "hourly" or "minutely_15". Given: {}'.format(freq))

    # Load .env file
    load_dotenv()

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    if freq == 'hourly':
        start_date = pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC') # can go up to 2015 if needed
    else:
        start_date = pd.Timestamp(datetime(year=2022, month=1, day=1), tz='UTC') # no openmeteo data for 15 min before that

    # start_date = pd.Timestamp(datetime(year=2025, month=2, day=1), tz='UTC') # no openmeteo data for 15 min before that


    db_path = './database/DE/'

    tasks = [
        'create_smard',
        'create_entsoe',
        'create_openmeteo_windfarms_offshore',
        'create_openmeteo_windfarms_onshore',
        'create_openmeteo_solarfarms',
        'create_openmeteo_cities',
    ]

    logger.info(f"Starting task {task} with freq={freq} and start date {start_date} up to date {today}")

    start_time = time.time()

    if task == 'create_smard':
        create_smard_from_api(
            start_date=start_date, today=today, data_dir=db_path+'smard/', freq=freq, verbose=verbose
        )

    elif task == 'create_entsoe':
        create_entsoe_from_api(
            start_date=start_date, today=today, data_dir=db_path + 'entsoe/',
            api_key=entsoe_api_key, freq=freq, verbose=verbose
        )

    # due to file size limitations on GitHub we need to split the openmeteo data into different files
    elif task == 'create_openmeteo_windfarms_offshore':
        for tso_dict in de_regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/offshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind)   if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in loc_offshore_windfarms if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date,  freq=freq, verbose = verbose
            )

    elif task == 'create_openmeteo_windfarms_onshore':
        for tso_dict in de_regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/onshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in loc_onshore_windfarms if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date, freq=freq, verbose = verbose
            )

    elif task == 'create_openmeteo_solarfarms':
        for tso_dict in de_regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/solar/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in loc_solarfarms if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date, freq=freq, verbose = verbose
            )

    elif task == 'create_openmeteo_cities':
        for tso_dict in de_regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/cities/{tso_dict['TSO']}",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in loc_cities if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date, freq=freq, verbose = verbose
            )

    else:
        raise ValueError(f'Invalid task {task}. Available tasks: {tasks}')

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Task '{task}' completed successfully for freq {freq}. Execution time: {str_time(elapsed_time)}")



if __name__ == '__main__':

    print("launching create_database.py")

    if len(sys.argv) != 3:
        raise KeyError("Usage: python update_database.py <task> <freq>")

        # task_argument = 'create_openmeteo_solarfarms'
        # freq = 'hourly'
    else:
        task_argument = str(sys.argv[1])
        freq = str(sys.argv[2])

    main(task_argument, freq)
