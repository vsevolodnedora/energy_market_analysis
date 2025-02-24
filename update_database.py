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
    OpenMeteo,
    countries_metadata
)

def str_time(elapsed_time:float) -> str:
    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def main_country(country_code:str, task:str, freq:str, verbose:bool = True):

    # Load .env file
    load_dotenv()

    if freq not in ['hourly','minutely_15']:
        raise ValueError('freq must be either "hourly" or "minutely_15"')

    c_dict:dict = [dict_ for dict_ in countries_metadata if dict_["code"] == country_code][0]
    if len(list(c_dict.keys())) == 0:
        raise KeyError(f"No country dict found for country code {country_code}. Check your country code.")
    regions = c_dict["regions"]
    if len(regions) == 0:
        logger.warning(f"No regions (TSOs) dicts found for country code {country_code}.")
    locations = list(c_dict['locations'].keys())
    if len(locations) == 0:
        logger.warning(f"No locations (for weather data) found for country code {country_code}.")

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    db_path = f'./database/{country_code}/'
    if not os.path.exists(db_path):raise FileNotFoundError(f"Database path does not exist: {db_path}")

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

    logger.info(f"Starting task {task} for country {country_code} and frequency {freq}...")

    start_time = time.time()

    # split tasks for lighter workflows
    if task == "update_entsoe" or task == "all":
        update_entsoe_from_api(
            country_dict=c_dict, today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key,
            freq=freq, verbose=verbose
        )

    if country_code=='DE' and (task == "update_smard" or task == "all"):
        update_smard_from_api(
            today=today, data_dir=db_path + 'smard/', freq=freq, verbose=verbose
        )

    if task == "update_epexspot" or task == "all":
        update_epexspot_from_files(
            country_dict=c_dict,today=today, data_dir=db_path + 'epexspot/', freq=freq, verbose=verbose,
            raw_data_dir='./data/DE-LU/DayAhead_MRC/'
        )

    elif (task == 'update_openmeteo_windfarms_offshore') and ('offshore' in locations):
        for tso_dict in regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/offshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in c_dict['locations']['offshore'] if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    elif (task == 'update_openmeteo_windfarms_onshore') and ("onshore" in locations):
        for tso_dict in regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/onshore/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min),
                locations = [loc for loc in c_dict['locations']['onshore'] if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    elif (task == 'update_openmeteo_solarfarms') and ("solar" in locations):
        for tso_dict in regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/solar/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation) if freq == 'hourly' else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in c_dict['locations']['solar'] if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    elif (task == 'update_openmeteo_cities') and ("cities" in locations):
        for tso_dict in regions:
            update_openmeteo_from_api(
                datadir=db_path + f"openmeteo/cities/{tso_dict['TSO']}/",
                variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation) if freq == 'hourly'
                            else
                            (OpenMeteo.vars_basic_15min + OpenMeteo.vars_wind_15min + OpenMeteo.vars_radiation_15min),
                locations = [loc for loc in c_dict['locations']['cities'] if loc['TSO'] == tso_dict['TSO']],
                freq=freq, verbose = verbose
            )

    else:
        logger.warning(f"Task {task} is not supported for country {country_code} and freq {freq}. "
                       f"Supported tasks are: {tasks}. Locations: {list(c_dict['locations'].keys())}")

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Task '{task}' for country {country_code} freq '{freq}' completed successfully. "
                f"Execution time: {str_time(elapsed_time)}")

def main(country_code:str,task:str, freq:str, verbose:bool = True):

    if not country_code == 'all' and not country_code in [d['code'] for d in countries_metadata]:
        raise ValueError(f"country code {country_code} is not supported. "
                         f"Supported countries are: {countries_metadata} (use 'all' for all countries)")

    if (country_code == 'all'):
        for country_code in [d['code'] for d in countries_metadata]:
            main_country(country_code=country_code, task=task, freq=freq, verbose=verbose)
    else:
        main_country(country_code=country_code, task=task, freq=freq, verbose=verbose)


if __name__ == '__main__':

    print("launching update_database.py")

    if len(sys.argv) != 4:
        raise KeyError("Usage: python update_database.py <country_code> <task> <freq>")
        # country_code = 'FR'
        # task_argument = str( 'update_entsoe' )
        # freq = str( 'hourly' )
    else:
        country_code = str( sys.argv[1] )
        task_argument = str( sys.argv[2] )
        freq = str( sys.argv[3] )

    main(country_code, task_argument, freq)
