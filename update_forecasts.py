import os

from forecasting_modules import (
    update_forecast_production
)

from logger import get_logger
logger = get_logger(__name__)


if __name__ == '__main__':
    db_path = './database/'

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load']

    # update_forecast_production(
    #     database=db_path, variable='load', outdir='./output/forecasts/', verbose=True
    # )

    for target in targets:
        update_forecast_production(
            database=db_path, variable=target, outdir='./output/forecasts/', verbose=True
        )

    logger.info(f"All tasks in update are completed successfully!")