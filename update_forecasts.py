import os, time

from forecasting_modules import (
    update_forecast_production
)

from logger import get_logger
logger = get_logger(__name__)


if __name__ == '__main__':

    db_path = './database/'

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load', 'energy_mix']

    # update_forecast_production(
    #     database=db_path, variable='energy_mix', outdir='./output/forecasts/', verbose=True
    # )

    start_time = time.time()  # Start the timer

    for target in targets:
        update_forecast_production(
            database=db_path, variable=target, outdir='./output/forecasts/', verbose=True
        )

    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    hours, minutes = divmod(elapsed_time // 60, 60)

    logger.info(
        f"All tasks in update are completed successfully! Execution time: "
        f"{int(hours)} hours and {int(minutes)} minutes."
    )
