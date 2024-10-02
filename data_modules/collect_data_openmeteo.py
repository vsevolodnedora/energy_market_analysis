import os.path

import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import numpy as np

import openmeteo_requests

from .locations import locations

class OpenMeteo:
    variables_standard = (
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m"
    )
    variables_highaltitude = (
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
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
            # "start_date":'2024-09-16',
            # "end_date":'2024-09-24',
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

def get_weather_data_from_api(start_date,today,locations:list):
    df_om_hist = pd.DataFrame()
    for il, location in enumerate(locations):
        print(f"Processing historic: {location['name']} ({location['type']}) {il}/{len(locations)}")
        if location['type'] == 'windfarm': om_quants = OpenMeteo.variables_highaltitude
        else: om_quants = OpenMeteo.variables_standard
        om = OpenMeteo(
            lat=location['lat'],
            lon=location['lon'],
            variables=om_quants,
            #extra_api_params=location['om_api_pars'], # this gives Error in API call
            verbose=False
        )
        try:
            # collect hsitoric data
            df_hist = om.get_historical(
                # start_date=pd.Timestamp(datetime(year=2016,month=1,day=1)).tz_localize('UTC'),#start_date,
                start_date=start_date,
                end_date=today+timedelta(hours=15),
                # url="https://historical-forecast-api.open-meteo.com/v1/forecast"
                url="https://archive-api.open-meteo.com/v1/archive"
            )

            # collect the data for today forecasted previously
            df_hist_forecast = om.get_historical(
                # start_date=pd.Timestamp(datetime(year=2016,month=1,day=1)).tz_localize('UTC'),#start_date,
                start_date=today-timedelta(days=3),
                end_date=today,#+timedelta(days=1),
                url="https://historical-forecast-api.open-meteo.com/v1/forecast"
                # url="https://archive-api.open-meteo.com/v1/archive"
            )

            df_hist = pd.concat([df_hist, df_hist_forecast],axis=0)
            df_hist.drop_duplicates(subset='date', keep='last', inplace=True)

            mapping = {name : name+location['suffix'] for name in df_hist.columns if name != 'date'}
            df_hist.rename(columns=mapping, inplace=True)
            if df_om_hist.empty:
                df_om_hist = df_hist
            else:
                df_om_hist = pd.merge(left=df_om_hist, right=df_hist, on='date', how='outer')
        except:
            print(f"! Failed to get weather data for {location['name']}")

    if os.path.isfile('./openmeteo.cache'):
        os.remove('./openmeteo.cache')

    return df_om_hist

def get_weather_data_from_api_forecast(locations:list):

    df_om_forecast = pd.DataFrame()
    for il, location in enumerate(locations):
        if location['type'] == 'windfarm': om_quants = OpenMeteo.variables_highaltitude
        else: om_quants = OpenMeteo.variables_standard
        print(f"Processing forecast: {location['name']} ({location['type']}) {il}/{len(locations)}")
        try:
            om = OpenMeteo(
                lat=location['lat'],
                lon=location['lon'],
                variables=om_quants,
                # extra_api_params=location['om_api_pars'], # this gives error in API call
                verbose=False
            )
            # collect actual forecast
            df_forecast = om.get_forecast()

            mapping = {name : name+location['suffix'] for name in df_forecast.columns if name != 'date'}
            df_forecast.rename(columns=mapping, inplace=True)
            if df_om_forecast.empty:
                df_om_forecast = df_forecast
            else:
                df_om_forecast = pd.merge(left=df_om_forecast, right=df_forecast, on='date', how='outer')
        except:
            print(f"! Failed to get weather forecast data for {location['name']}")

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

def transform_data(df:pd.DataFrame):
    key = 'wind_direction_10m'
    for loc in locations:
        df[key + '_x' + loc['suffix']] = \
            df[key + loc['suffix']] * np.cos( np.pi / 180 )
        df[key + '_y' + loc['suffix']] = \
            df[key + loc['suffix']] * np.sin( np.pi / 180 )
        df.drop([key + loc['suffix']], axis=1, inplace=True)
    return df

def process_weather_quantities(df:pd.DataFrame, locations):
    key:str='wind_direction_10m'
    for loc in locations:
        df[key + '_x' + loc['suffix']] = \
            df[key + loc['suffix']] * np.cos( np.pi / 180 )
        df[key + '_y' + loc['suffix']] = \
            df[key + loc['suffix']] * np.sin( np.pi / 180 )
        df.drop([key + loc['suffix']], axis=1, inplace=True)
    return df

if __name__ == '__main__':
    # todo add tests
    pass