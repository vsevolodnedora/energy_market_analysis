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
    update_entsoe_from_api,
    loc_solarfarms,
    loc_onshore_windfarms,
    loc_offshore_windfarms,
    loc_cities,
    OpenMeteo
)

if __name__ == '__main__':

    # Load .env file
    load_dotenv()

    # Fetch API key from environment
    entsoe_api_key = os.getenv("ENTSOE_API_KEY")

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    start_date = pd.Timestamp(datetime(year=2021, month=1, day=1), tz='UTC') # for initial collection

    db_path = './database/'

    verbose = True
    task = 'update'
    tasks = [
        'update', 'create_smard', 'create_openmeteo', 'create_entsoe',
    ]

    if task not in tasks:
        raise ValueError(f'Invalid task {task}')

    if task == 'create_smard':
        create_smard_from_api(start_date=start_date,
                              today=today, data_dir=db_path+'smard/',verbose=verbose)

    # due to file size limitations on GitHub we need to split the openmeteo data into different files
    elif task == 'create_openmeteo':
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/offshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_offshore_windfarms, start_date = start_date, verbose = verbose
        )
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/onshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_onshore_windfarms, start_date = start_date, verbose = verbose
        )
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/solar_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation),
            locations = loc_solarfarms, start_date = start_date, verbose = verbose
        )
        create_openmeteo_from_api(
            fpath=db_path + 'openmeteo/cities_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation),
            locations = loc_cities, start_date = start_date, verbose = verbose
        )


    elif task == 'create_entsoe':
        create_entsoe_from_api(start_date=start_date,
                               today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)


    elif task == 'update':

        # --- update database ---
        update_smard_from_api(today=today, data_dir=db_path + 'smard/', verbose=verbose)

        # due to file size limitations on GitHub we need to split the openmeteo data into different files
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/offshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_offshore_windfarms, verbose = verbose
        )
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/onshore_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind),
            locations = loc_onshore_windfarms, verbose = verbose
        )
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/solar_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_radiation),
            locations = loc_solarfarms, verbose = verbose
        )
        update_openmeteo_from_api(
            fpath=db_path + 'openmeteo/cities_history.parquet',
            variables = (OpenMeteo.vars_basic + OpenMeteo.vars_wind + OpenMeteo.vars_radiation),
            locations = loc_cities, verbose = verbose
        )

        update_epexspot_from_files(today=today, data_dir=db_path + 'epexspot/', verbose=verbose)

        update_entsoe_from_api(today=today, data_dir=db_path + 'entsoe/', api_key=entsoe_api_key, verbose=verbose)

    else:
        raise ValueError(f'Invalid task {task}')

    print(f"All tasks in update are completed successfully!")