import os.path

import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import numpy as np
import gc
import time

import openmeteo_requests

from .locations import locations
from .utils import validate_dataframe

class OpenMeteo:
    variables_standard = (
        # basic
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        # additional
        "precipitation",
        "wind_gusts_10m",
        "cloud_cover",
        'shortwave_radiation'
    )

    def __init__(self, lat, lon, variables=variables_standard, extra_api_params:dict=dict(),verbose:bool=False) -> None:
        self.lat=lat
        self.lon=lon
        self.variables=variables
        self.now = datetime.now() #+ timedelta(hours=12)
        self.extra_api_params = extra_api_params
        self.verbose=verbose

    def get_historical(self, start_date:pd.Timestamp, end_date:pd.Timestamp, url:str or None) -> pd.DataFrame:
        """
        https://open-meteo.com/en/docs/historical-weather-api
        :param start_date:
        :param end_date:
        :param lat:
        :param lon:
        :return:
        """

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('openmeteo.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        if url is None: url = "https://archive-api.open-meteo.com/v1/archive"
        # url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date.strftime('%Y-%m-%d'),#"2015-01-01",
            "end_date": end_date.strftime('%Y-%m-%d'),#"2024-09-14",
            "hourly": ''.join([var+',' for var in self.variables])[:-1], # remove last comma...
        }
        if self.verbose: print(f"Requesting historical data for {params['start_date']} - {params['end_date']}")
        responses = openmeteo.weather_api(url, params=params | self.extra_api_params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        if self.verbose:
            print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation {response.Elevation()} m asl")
            print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
            print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}

        for i, var in enumerate(self.variables):
            data = hourly.Variables(i).ValuesAsNumpy()
            hourly_data[var] = data
        hourly_dataframe = pd.DataFrame(data = hourly_data)

        if os.path.isfile('./openmeteo.cache'):
            os.remove('./openmeteo.cache')

        return hourly_dataframe

    def get_forecast(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('openmeteo.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "forecast_days": 14,
            "hourly": ''.join([var+',' for var in self.variables])[:-1], # remove last comma...
        }
        responses = openmeteo.weather_api(url, params=params | self.extra_api_params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        if self.verbose:
            print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
            print(f"Elevation {response.Elevation()} m asl")
            print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
            print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )
        }

        for i, var in enumerate(self.variables):
            data = hourly.Variables(i).ValuesAsNumpy()
            hourly_data[var] = data

        hourly_dataframe = pd.DataFrame(data = hourly_data)

        return hourly_dataframe

def get_weather_data_from_api(start_date,today,locations:list,verbose:bool):
    df_om_hist = pd.DataFrame()
    for il, location in enumerate(locations):
        if verbose:
            print(f"Processing historic: {location['name']} ({location['type']}) {il}/{len(locations)}")
        # if location['type'] == 'windfarm': om_quants = OpenMeteo.variables_highaltitude
        # else: om_quants = OpenMeteo.variables_standard
        om = OpenMeteo(
            lat=location['lat'],
            lon=location['lon'],
            verbose=False
        )
        try:
            # collect hsitoric data
            df_hist = None
            for i in range(5):
                url="https://archive-api.open-meteo.com/v1/archive"
                try:
                    df_hist = om.get_historical(
                        start_date=start_date,
                        end_date=today+timedelta(hours=15),
                        url=url
                    )
                except Exception as e:
                    if verbose:
                        print(f"API call failed attempt {i+1}/{5} for loc={location['name']} url={url} with error\n{e}")
                    time.sleep(3)
                if verbose: print(f"API call successful. Retrieved df={df_hist.shape}")
                break
            if df_hist is None:
                raise AttributeError(f"Couldn't get historical data for {location['name']}")

            # collect the data for today forecasted previously
            df_hist_forecast = None
            for i in range(5):
                url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
                try:
                    df_hist_forecast = om.get_historical(
                        start_date=today-timedelta(days=3),
                        end_date=today,
                        url=url
                    )
                except Exception as e:
                    if verbose:
                        print(f"API call failed attempt {i+1}/{5} for loc={location['name']} url={url} with error\n{e}")
                    time.sleep(3)
                if verbose:
                    print(f"API call successful. Retrieved df={df_hist_forecast.shape}")
                break
            if df_hist_forecast is None:
                raise AttributeError(f"Couldn't get historical forecast data for {location['name']}")

            # combine the data
            df_hist = pd.concat([df_hist, df_hist_forecast],axis=0)
            df_hist.drop_duplicates(subset='date', keep='last', inplace=True)

            mapping = {name : name+location['suffix'] for name in df_hist.columns if name != 'date'}
            df_hist.rename(columns=mapping, inplace=True)
            if df_om_hist.empty:
                df_om_hist = df_hist
            else:
                df_om_hist = pd.merge(left=df_om_hist, right=df_hist, on='date', how='outer')
        except:
            print(f"Error! Failed to get weather data for {location['name']}")

    if os.path.isfile('./openmeteo.cache'):
        os.remove('./openmeteo.cache')

    return df_om_hist

def get_weather_data_from_api_forecast(locations:list):

    df_om_forecast = pd.DataFrame()
    for il, location in enumerate(locations):
        print(f"Processing forecast: {location['name']} ({location['type']}) {il}/{len(locations)}")
        om = OpenMeteo(
            lat=location['lat'],
            lon=location['lon'],
            verbose=False
        )
        df_forecast = None
        for i in range(5):
            try:
                # collect actual forecast
                df_forecast = om.get_forecast()
            except Exception as e:
                print(f"API call failed attempt {i+1}/{5} for loc={location['name']} forecast, with error\n{e}")
                time.sleep(3)
            print(f"API call successful. Retrieved df={df_forecast.shape}")
            break
        if df_forecast is None:
            raise AttributeError(f"Couldn't get forecast data for {location['name']}")

        # merge data
        mapping = {name : name+location['suffix'] for name in df_forecast.columns if name != 'date'}
        df_forecast.rename(columns=mapping, inplace=True)
        if df_om_forecast.empty:
            df_om_forecast = df_forecast
        else:
            df_om_forecast = pd.merge(left=df_om_forecast, right=df_forecast, on='date', how='outer')

    if os.path.isfile('./openmeteo.cache'):
        os.remove('./openmeteo.cache')

    return df_om_forecast

def check_phys_limits_in_data(df: pd.DataFrame) -> pd.DataFrame:
    physical_limits = dict(
        temperature_2m=(-60.,60.), # degrees, Celsius
        visibility=(0., 1e4), # distance, meters
        surface_pressure=(870.,1080.), # pressure hPa,
        realtive_humidity_2m=(0.,100), # humidity, percent
        wind_speed_10m=(0.,113), # velocity, m/s
        wind_direction_10m=(0.,360), # degree, deg.
        clouds_all=(0., 100.) # percent?
    )
    for key, lim in physical_limits.items():
        for loc in locations:
            kkey = key+loc['suffix']
            if kkey in df:
                df[kkey].where(df[kkey] < lim[0], None)
                df[kkey].where(df[kkey] > lim[1], None)
    return df

# def transform_data(df:pd.DataFrame):
#     key = 'wind_direction_10m'
#     for loc in locations:
#         df[key + '_x' + loc['suffix']] = \
#             df[key + loc['suffix']] * np.cos( np.pi / 180 )
#         df[key + '_y' + loc['suffix']] = \
#             df[key + loc['suffix']] * np.sin( np.pi / 180 )
#         df.drop([key + loc['suffix']], axis=1, inplace=True)
#     return df

# def process_weather_quantities(df:pd.DataFrame, locations):
#     key:str='wind_direction_10m'
#     for loc in locations:
#         df[key + '_x' + loc['suffix']] = \
#             df[key + loc['suffix']] * np.cos( np.pi / 180 )
#         df[key + '_y' + loc['suffix']] = \
#             df[key + loc['suffix']] * np.sin( np.pi / 180 )
#         df.drop([key + loc['suffix']], axis=1, inplace=True)
#     return df

def update_openmeteo_from_api(today:pd.Timestamp,data_dir:str,verbose:bool):

    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)

    first_timestamp = pd.Timestamp(df_hist.dropna(how='any', inplace=False).first_valid_index())
    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())

    start_date = last_timestamp - timedelta(hours=24)
    end_date = today - timedelta(hours=12)

    # ------- UPDATE WEATHER DATA -------------
    if verbose: print(f"Updating SMARD data from {last_timestamp} to {today}")

    df_om_hist = get_weather_data_from_api(
        start_date=start_date,
        today=end_date,
        locations=locations,
        verbose=verbose
    )
    df_om_hist.set_index('date',inplace=True)
    df_om_hist = df_om_hist[:today]

    # check columns
    for col in df_hist.columns:
        if not col in df_om_hist.columns:
            raise IOError(f"Error. col={col} is not in the update dataframe. Cannot continue")

    if not validate_dataframe(df_om_hist, text="Updated openmeteo df_hist"):
        raise ValueError(f"Failed to validate the dataframe for {fname}")

    df_om_hist = check_phys_limits_in_data(df_om_hist)
    df_om_hist = df_om_hist.interpolate(method='linear')

    df_om_hist.to_parquet(fname,engine='pyarrow')
    if verbose:
        print(f'Update for openmeteo historical data is successfully created at '
              f'{fname} with shape {df_om_hist.shape}')

    gc.collect()

    # FORECAST

    df_om_forecast = get_weather_data_from_api_forecast(locations=locations)
    df_om_forecast.set_index('date',inplace=True)
    df_om_forecast = df_om_forecast[today+timedelta(hours=1):] # start with next hour
    if not df_om_forecast.columns.equals(df_om_hist.columns):
        unique_to_df1 = set(df_om_hist.columns) - set(df_om_forecast.columns)
        unique_to_df2 = set(df_om_forecast.columns) - set(df_om_hist.columns)
        raise KeyError(f"! Error. Column mismatch between historical and forecasted weather! unique to "
              f"df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")
    fname = fname.replace('history','forecast')
    df_om_forecast.to_parquet(fname,engine='pyarrow')
    if verbose:
        print(f'Update for openmeteo forecast data is successfully created at '
          f'{fname} with shape {df_om_forecast.shape}')

    gc.collect()

if __name__ == '__main__':
    # todo add tests
    pass