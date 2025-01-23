import os

from forecasting_modules import (
    update_forecast_production
)

from publish_data import (
    publish_generation,
    publish_to_api
)


def update_forecasts(db_path:str, targets:list, verbose:bool):

    for target in targets:
        update_forecast_production(
            database=db_path, variable=target, outdir='./output/forecasts/', verbose=verbose
        )

def publish_forecasts(db_path:str, targets:list, verbose:bool):

    # check if output directory set up or set them up
    data_dir = "./deploy/data/"
    if not os.path.isdir(data_dir):
        if verbose: print(f"Creating {data_dir}")
        os.mkdir(data_dir)

    data_dir_web = data_dir + "forecasts/"
    if not os.path.isdir(data_dir_web):
        if verbose: print(f"Creating {data_dir_web}")
        os.mkdir(data_dir_web)

    data_dir_api = data_dir + "api/"
    if not os.path.isdir(data_dir_api):
        if verbose: print(f"Creating {data_dir_api}")
        os.mkdir(data_dir_api)

    for target in targets:
        if target == 'wind_offshore': regions = ('DE_50HZ', 'DE_TENNET')
        else: regions = ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')

        # publish data to webpage
        publish_generation(
            target=target,
            avail_regions=regions,
            n_folds = 3,
            metric = 'rmse',
            method_type = 'trained', # 'trained'
            results_root_dir = './output/forecasts/',
            database_dir = db_path,
            output_dir = data_dir_web
        )

        # publish to API folder
        publish_to_api(
            target=target,
            avail_regions=regions,
            method_type = 'forecast', # 'trained'
            results_root_dir = './output/forecasts/',
            output_dir = f'{data_dir_api}forecasts/'
        )

if __name__ == '__main__':

    print("launching update_forecasts.py")

    db_path = './database/'

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load']

    update_forecasts(targets=targets, db_path=db_path, verbose=True)

    publish_forecasts(targets=targets, db_path=db_path, verbose=True)




    # --- update forecasts ---
    # update_forecast_production(
    #     database=db_path, variable='wind_offshore', outdir='./output/forecasts/', verbose=verbose
    # )
    # update_forecast_production(
    #     database=db_path, variable='wind_onshore', outdir='./output/forecasts/', verbose=verbose
    # )
    # update_forecast_production(
    #     database=db_path, variable='solar', outdir='./output/forecasts/', verbose=verbose
    # )
    # update_forecast_production(
    #     database=db_path, variable='load', outdir='./output/forecasts/', verbose=verbose
    # )



    # publish_generation(
    #     target='wind_offshore',
    #     avail_regions=('DE_50HZ', 'DE_TENNET'),
    #     n_folds = 3,
    #     metric = 'rmse',
    #     method_type = 'trained', # 'trained'
    #     results_root_dir = './output/forecasts/',
    #     database_dir = db_path,
    #     output_dir = './deploy/data/forecasts/'
    # )
    # publish_to_api(
    #     target='wind_offshore',
    #     avail_regions=('DE_50HZ', 'DE_TENNET'),
    #     method_type = 'forecast', # 'trained'
    #     results_root_dir = './output/forecasts/',
    #     output_dir = './deploy/data/api/forecasts/'
    # )

    # publish_generation(
    #     target='wind_onshore',
    #     avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
    #     n_folds = 3,
    #     metric = 'rmse',
    #     method_type = 'trained', # 'trained'
    #     results_root_dir = './output/forecasts/',
    #     database_dir = db_path,
    #     output_dir = './deploy/data/forecasts/'
    # )
    # publish_generation(
    #     target='solar',
    #     avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
    #     n_folds = 3,
    #     metric = 'rmse',
    #     method_type = 'trained', # 'trained'
    #     results_root_dir = './output/forecasts/',
    #     database_dir = db_path,
    #     output_dir = './deploy/data/forecasts/'
    # )
    # publish_generation(
    #     target='load',
    #     avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
    #     n_folds = 3,
    #     metric = 'rmse',
    #     method_type = 'trained', # 'trained'
    #     results_root_dir = './output/forecasts/',
    #     database_dir = db_path,
    #     output_dir = './deploy/data/forecasts/'
    # )

    print(f"All tasks in update are completed successfully!")