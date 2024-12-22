import pandas as pd
from retry_requests import retry
import time
import requests
import openmeteo_requests
from datetime import datetime, timedelta
from data_collection_modules.locations import locations

class OpenMeteo:
    variables_standard = (
        # basic
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        # additional
        "precipitation",
        "wind_gusts_10m",
        "cloud_cover",
        "shortwave_radiation"
    )

    def __init__(self, start_date: pd.Timestamp, locations: dict, verbose: bool = False):
        self.start_date = start_date
        self.locations = locations
        self.verbose = verbose

        today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
        self.today = today.normalize() + pd.DateOffset(hours=today.hour)  # leave only hours

        yesterday = today.normalize() - pd.DateOffset(days=1)
        self.yesterday_last_hour = yesterday

        self.day_before_yesterday = self.yesterday_last_hour.normalize() - pd.DateOffset(days=1)

        tomorrow = today.normalize() + pd.DateOffset(days=1)
        self.tomorrow_first_hour = tomorrow

    def make_request(self, url: str, params: dict) -> pd.DataFrame:
        # No caching session, just a regular retry session
        retry_session = retry(requests.Session(), retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        hourly = []
        for i in range(5):
            try:
                responses = openmeteo.weather_api(url, params=params)
                hourly = []
                for response in responses:
                    hourly.append( response.Hourly() )
            except Exception as e:
                if self.verbose: print(f"Failed to fetch weather data from openmeteo {i}/{5} "
                                       f"for {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')} "
                                       f"with Error:\n{e}")
                time.sleep(20*i)  # in case API is overloaded
                continue
            break

        if len(hourly) == 0:
            raise ConnectionError(
                f"Failed to fetch weather data from OpenMeteo 5 times for"
                f" {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')}"
            )
        if not len(hourly) == len(self.locations):
            raise ValueError(
                f"Requesting data for {len(self.locations)} locations, got only {len(hourly)} hourly data."
            )

        df = pd.DataFrame()
        for hourly, loc in zip(hourly,self.locations):
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }

            for i, var in enumerate(self.variables_standard):
                data = hourly.Variables(i).ValuesAsNumpy()
                hourly_data[var] = data
            hourly_dataframe = pd.DataFrame(data=hourly_data)
            hourly_dataframe.set_index('date', inplace=True)

            # add suffix
            mapping = {name : name + loc['suffix'] for name in df.columns if name != 'date'}
            hourly_dataframe.rename(columns=mapping, inplace=True)

            if df.empty:df = hourly_dataframe
            else: df = pd.merge(df, hourly_dataframe, left_index=True, right_index=True, how='left')

        return df

    def collect(self) -> pd.DataFrame:
        ''' Returns time-series dataframe '''

        # collect historic data
        start_date = self.start_date
        end_date = self.day_before_yesterday  # after that it returns nans
        url = "https://archive-api.open-meteo.com/v1/archive"
        lats = [loc['lat'] for loc in locations]
        lons = [loc['lon'] for loc in locations]
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "hourly": ''.join([var + ',' for var in self.variables_standard])[:-1],
        }
        hist_data = self.make_request(url, params)

        # collect historic forecast (to bridge the data till forecasts starts)
        params['start_date'] = self.day_before_yesterday.strftime('%Y-%m-%d')
        params['end_date'] = self.today.strftime('%Y-%m-%d')
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        hist_forecast_data = self.make_request(url, params)

        # collect forecast
        url = "https://api.open-meteo.com/v1/forecast"
        params['forecast_days'] = 14
        del params['start_date']
        del params['end_date']
        forecast_data = self.make_request(url, params)

        df = hist_data.combine_first(hist_forecast_data)
        df = forecast_data.combine_first(df)

        df.sort_index(inplace=True)
        if not pd.infer_freq(df.index) == 'h':
            raise ValueError("Dataframe must have 'h' frequency for openmeteo")

        return df

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

def create_openmeteo_from_api(start_date:pd.Timestamp, data_dir:str, verbose:bool):
    if verbose:print(f"Collecting historical data from OpenMeteo from {start_date} ({len(locations)})"
                     f"{[loc['name'] for loc in locations]} locations")
    fname = data_dir + 'history.parquet'
    om = OpenMeteo(start_date, locations, verbose=verbose)
    df_hist = om.collect()
    df_hist = df_hist[start_date:]
    df_hist.to_parquet(fname,engine='pyarrow')
    if verbose: print(f"OpenMeteo data collected. Collected {df_hist.shape}. Saving to {fname}")

def update_openmeteo_from_api(data_dir:str, verbose:bool):
    fname = data_dir + 'history.parquet'
    df_hist = pd.read_parquet(fname)
    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
    start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
    if verbose: print(f"Updating openmeteo data from {start_date}. Current data has shape {df_hist.shape}")
    om = OpenMeteo(start_date, locations, verbose=verbose)
    df_om = om.collect()
    if not df_hist.columns.equals(df_om.columns):
        unique_to_df1 = set(df_om.columns) - set(df_hist.columns)
        unique_to_df2 = set(df_hist.columns) - set(df_om.columns)
        raise KeyError(f"! Error. Column mismatch between historical and forecasted weather! unique to "
                       f"df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")
    df_om = df_om.combine_first(df_hist) # overwrite previous forecasts with actual data
    df_om.sort_index(inplace=True)
    idx = df_om.index[-1]-timedelta(days=14) # separate historic and forecasted data
    df_om[:idx].to_parquet(fname,engine='pyarrow')
    df_om[idx+timedelta(hours=1):].to_parquet( fname.replace('history','forecast'),engine='pyarrow' )
    if verbose: print(f"OpenMeteo data updated. "
                      f"Collected df_hist={df_om[:idx].shape} and df_forecast={df_om[idx+timedelta(hours=1):].shape}. ")


#
# class OpenMeteo:
#     variables_standard = (
#         # basic
#         "temperature_2m",
#         "relative_humidity_2m",
#         "surface_pressure",
#         "wind_speed_10m",
#         "wind_speed_100m",
#         "wind_direction_10m",
#         "wind_direction_100m",
#         # additional
#         "precipitation",
#         "wind_gusts_10m",
#         "cloud_cover",
#         "shortwave_radiation"
#     )
#
#     def __init__(self, start_date: pd.Timestamp, location: dict, verbose: bool = False):
#         self.start_date = start_date
#         self.location = location
#         self.verbose = verbose
#
#         today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
#         self.today = today.normalize() + pd.DateOffset(hours=today.hour)  # leave only hours
#
#         yesterday = today.normalize() - pd.DateOffset(days=1)
#         self.yesterday_last_hour = yesterday
#
#         self.day_before_yesterday = self.yesterday_last_hour.normalize() - pd.DateOffset(days=1)
#
#         tomorrow = today.normalize() + pd.DateOffset(days=1)
#         self.tomorrow_first_hour = tomorrow
#
#         self.lat = float(location['lat'])
#         self.lon = float(location['lon'])
#
#     def make_request(self, url: str, params: dict) -> pd.DataFrame:
#         # No caching session, just a regular retry session
#         retry_session = retry(requests.Session(), retries=5, backoff_factor=0.2)
#         openmeteo = openmeteo_requests.Client(session=retry_session)
#
#         hourly = None
#         for i in range(5):
#             try:
#                 responses = openmeteo.weather_api(url, params=params)
#                 response = responses[0]
#                 hourly = response.Hourly()
#             except Exception as e:
#                 if self.verbose: print(f"Failed to fetch weather data from openmeteo {i}/{5} "
#                       f"for {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')} "
#                       f"with Error:\n{e}")
#                 time.sleep(20*i)  # in case API is overloaded
#                 continue
#             break
#
#         if hourly is None:
#             raise ConnectionError(
#                 f"Failed to fetch weather data from OpenMeteo 5 times for"
#                 f" {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')}"
#             )
#
#         hourly_data = {
#             "date": pd.date_range(
#                 start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
#                 end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
#                 freq=pd.Timedelta(seconds=hourly.Interval()),
#                 inclusive="left"
#             )
#         }
#
#         for i, var in enumerate(self.variables_standard):
#             data = hourly.Variables(i).ValuesAsNumpy()
#             hourly_data[var] = data
#         hourly_dataframe = pd.DataFrame(data=hourly_data)
#
#         return hourly_dataframe
#
#     def collect(self) -> pd.DataFrame:
#         ''' Returns time-series dataframe '''
#
#         # collect historic data
#         start_date = self.start_date
#         end_date = self.day_before_yesterday  # after that it returns nans
#         url = "https://archive-api.open-meteo.com/v1/archive"
#         params = {
#             "latitude": self.lat,
#             "longitude": self.lon,
#             "start_date": start_date.strftime('%Y-%m-%d'),
#             "end_date": end_date.strftime('%Y-%m-%d'),
#             "hourly": ''.join([var + ',' for var in self.variables_standard])[:-1],
#         }
#         hist_data = self.make_request(url, params)
#         hist_data.set_index('date', inplace=True)
#
#         # collect historic forecast (to bridge the data till forecasts starts)
#         params['start_date'] = self.day_before_yesterday.strftime('%Y-%m-%d')
#         params['end_date'] = self.today.strftime('%Y-%m-%d')
#         url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
#         hist_forecast_data = self.make_request(url, params)
#         hist_forecast_data.set_index('date', inplace=True)
#
#         # collect forecast
#         url = "https://api.open-meteo.com/v1/forecast"
#         params['forecast_days'] = 14
#         del params['start_date']
#         del params['end_date']
#         forecast_data = self.make_request(url, params)
#         forecast_data.set_index('date', inplace=True)
#
#         df = hist_data.combine_first(hist_forecast_data)
#         df = forecast_data.combine_first(df)
#
#         df.sort_index(inplace=True)
#         if not pd.infer_freq(df.index) == 'h':
#             raise ValueError("Dataframe must have 'h' frequency for openmeteo")
#
#         return df
#
# def check_phys_limits_in_data(df: pd.DataFrame) -> pd.DataFrame:
#     physical_limits = dict(
#         temperature_2m=(-60.,60.), # degrees, Celsius
#         visibility=(0., 1e4), # distance, meters
#         surface_pressure=(870.,1080.), # pressure hPa,
#         realtive_humidity_2m=(0.,100), # humidity, percent
#         wind_speed_10m=(0.,113), # velocity, m/s
#         wind_direction_10m=(0.,360), # degree, deg.
#         clouds_all=(0., 100.) # percent?
#     )
#     for key, lim in physical_limits.items():
#         for loc in locations:
#             kkey = key+loc['suffix']
#             if kkey in df:
#                 df[kkey].where(df[kkey] < lim[0], None)
#                 df[kkey].where(df[kkey] > lim[1], None)
#     return df
#
# def fetch_openmeteo_data_for_all_locations(start_date:pd.Timestamp, verbose:bool)->pd.DataFrame:
#     df_om = pd.DataFrame()
#     for location in locations:
#         if verbose: print(f"Collecting openmeteo from {start_date} for location {location['name']}")
#         om = OpenMeteo(start_date=start_date, location=location, verbose=verbose)
#         df = om.collect()
#         df = check_phys_limits_in_data(df)
#         df = df.interpolate(method='linear')
#         mapping = {name : name + location['suffix'] for name in df.columns if name != 'date'}
#         df.rename(columns=mapping, inplace=True)
#         if df_om.empty: df_om = df
#         else: df_om = pd.merge(left=df_om, right=df, left_index=True, right_index=True)
#     return df_om
#
# def create_openmeteo_from_api(start_date:pd.Timestamp, data_dir:str, verbose:bool):
#     if verbose:print(f"Collecting historical data from OpenMeteo from {start_date} ({len(locations)})"
#                      f"{[loc['name'] for loc in locations]} locations")
#     fname = data_dir + 'history.parquet'
#     df_hist = fetch_openmeteo_data_for_all_locations(start_date=start_date, verbose=verbose)
#     df_hist = df_hist[start_date:]
#     df_hist.to_parquet(fname,engine='pyarrow')
#     if verbose: print(f"OpenMeteo data collected. Collected {df_hist.shape}. Saving to {fname}")
#
# def update_openmeteo_from_api(data_dir:str, verbose:bool):
#     fname = data_dir + 'history.parquet'
#     df_hist = pd.read_parquet(fname)
#     last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
#     start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
#     if verbose: print(f"Updating openmeteo data from {start_date}. Current data has shape {df_hist.shape}")
#     df_om = fetch_openmeteo_data_for_all_locations(start_date=start_date, verbose=verbose)
#     if not df_hist.columns.equals(df_om.columns):
#         unique_to_df1 = set(df_om.columns) - set(df_hist.columns)
#         unique_to_df2 = set(df_hist.columns) - set(df_om.columns)
#         raise KeyError(f"! Error. Column mismatch between historical and forecasted weather! unique to "
#                        f"df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")
#     df_om = df_om.combine_first(df_hist) # overwrite previous forecasts with actual data
#     df_om.sort_index(inplace=True)
#     idx = df_om.index[-1]-timedelta(days=14) # separate historic and forecasted data
#     df_om[:idx].to_parquet(fname,engine='pyarrow')
#     df_om[idx+timedelta(hours=1):].to_parquet( fname.replace('history','forecast'),engine='pyarrow' )
#     if verbose: print(f"OpenMeteo data updated. "
#                       f"Collected df_hist={df_om[:idx].shape} and df_forecast={df_om[idx+timedelta(hours=1):].shape}. ")


if __name__ == '__main__':
    # todo add tests
    from locations import locations
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    om = OpenMeteo(start_date=today - timedelta(days=60), location=locations[0], verbose=True)
    df = om.collect()
    df = check_phys_limits_in_data(df)
    df = df.interpolate(method='linear')
