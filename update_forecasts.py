import os

from forecasting_modules import (
    update_forecast_production
)


if __name__ == '__main__':
    db_path = './database/'

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load']
    for target in targets:
        update_forecast_production(
            database=db_path, variable=target, outdir='./output/forecasts/', verbose=True
        )

    print(f"All tasks in update are completed successfully!")