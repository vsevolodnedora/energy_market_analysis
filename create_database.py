import pandas as pd
import os, sys, time
from dotenv import load_dotenv

from datetime import datetime
from data_collection_modules import (
    create_openmeteo_from_api,
    create_smard_from_api,
    create_entsoe_from_api,
    countries_metadata,
    OpenMeteo
)

from logger import get_logger
logger = get_logger(__name__)


def str_time(elapsed_time:float) -> str:
    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main(country_code:str, task:str, freq:str, verbose : bool = True):

    if not freq in ['hourly', 'minutely_15']:
        raise ValueError('freq must be "hourly" or "minutely_15". Given: {}'.format(freq))
    if not country_code in ['DE', "FR"]:
        raise ValueError('country code must be "DE" or "FR". Given: {}'.format(country_code))

    c_dict:dict = [dict_ for dict_ in countries_metadata if dict_["code"] == country_code][0]
    if len(list(c_dict.keys())) == 0:
        raise KeyError(f"No country dict found for country code {country_code}. Check your country code.")
    regions = c_dict["regions"]

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

    # start_date = pd.Timestamp(datetime(year=2025, month=2, day=10), tz='UTC') # no openmeteo data for 15 min before that


    db_path = f'./database/{country_code}/'
    if not os.path.exists(db_path):raise FileNotFoundError(f"Database path does not exist: {db_path}")

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

    if task == 'create_smard' and country_code == 'DE':
        create_smard_from_api(
            start_date=start_date, today=today, data_dir=db_path+'smard/', freq=freq, verbose=verbose
        )
    elif task == 'create_smard' and country_code != 'DE':
        raise KeyError(f"SMARD data is only available for 'DE' country code. Given: {country_code}")


    elif task == 'create_entsoe':
        create_entsoe_from_api(
            country_dict=c_dict, start_date=start_date, today=today, data_dir=db_path + 'entsoe/',
            api_key=entsoe_api_key, freq=freq, verbose=verbose
        )

    # due to file size limitations on GitHub we need to split the openmeteo data into different files
    elif (task == 'create_openmeteo_windfarms_offshore') and ('offshore' in list(c_dict['locations'].keys())):
        for tso_dict in regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/offshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind)   if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in c_dict['locations']['offshore'] if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date,  freq=freq, verbose = verbose
            )

    elif (task == 'create_openmeteo_windfarms_onshore') and ("onshore" in list(c_dict['locations'].keys())):
        for tso_dict in regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/onshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in c_dict['locations']['onshore'] if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date, freq=freq, verbose = verbose
            )

    elif (task == 'create_openmeteo_solarfarms') and ("solar" in list(c_dict['locations'].keys())):
        for tso_dict in regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/solar/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in c_dict['locations']['solar'] if loc['TSO'] == tso_dict['TSO']],
                start_date = start_date, freq=freq, verbose = verbose
            )

    elif (task == 'create_openmeteo_cities') and ("cities" in list(c_dict['locations'].keys())):
        for tso_dict in regions:
            create_openmeteo_from_api(
                datadir=db_path + f"openmeteo/cities/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation) if freq == 'hourly'
                            else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in c_dict['locations']['cities'] if loc['TSO'] == tso_dict['TSO']],
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

    if len(sys.argv) != 4:
        raise KeyError("Usage: python update_database.py <country code> <task> <freq>")

        # country_code = 'FR'
        # task_argument = 'create_openmeteo_windfarms_offshore'
        # freq = 'hourly'
    else:
        country_code = sys.argv[1]
        task_argument = str(sys.argv[2])
        freq = str(sys.argv[3])

    main(country_code, task_argument, freq)
