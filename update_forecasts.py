import os

from forecasting_modules import (
    update_forecast_production
)

from publish_data import (
    publish_generation
)

if __name__ == '__main__':

    print("launching update_forecasts.py")

    db_path = './database/'
    verbose = True

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

    # --- serve forecasts ---
    if not os.path.isdir("./deploy/data/"):
        os.mkdir("./deploy/data/")

    publish_generation(
        target='wind_offshore',
        avail_regions=('DE_50HZ', 'DE_TENNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = './output/forecasts/',
        database_dir = db_path,
        output_dir = './deploy/data/forecasts/'
    )
    publish_generation(
        target='wind_onshore',
        avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = './output/forecasts/',
        database_dir = db_path,
        output_dir = './deploy/data/forecasts/'
    )
    publish_generation(
        target='solar',
        avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = './output/forecasts/',
        database_dir = db_path,
        output_dir = './deploy/data/forecasts/'
    )
    publish_generation(
        target='load',
        avail_regions=('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = './output/forecasts/',
        database_dir = db_path,
        output_dir = './deploy/data/forecasts/'
    )

    print(f"All tasks in update are completed successfully!")