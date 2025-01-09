import pandas as pd
from retry_requests import retry
import time
import requests
import openmeteo_requests
from datetime import datetime, timedelta


class OpenMeteo:

    vars_basic = (
        "temperature_2m", # degrees, Celsius
        "relative_humidity_2m", # percentage
        "surface_pressure", # pressure hPa,
        "precipitation", # percentage
        "cloud_cover", # percentage
    )
    vars_wind = (
        "wind_speed_10m", # velocity, km/h
        "wind_speed_100m", # velocity, km/h
        "wind_direction_10m",# degree, deg.
        "wind_direction_100m",# degree, deg.
        "wind_gusts_10m", # km/h
    )
    vars_radiation = (
        "shortwave_radiation", # W/m^2
        "direct_radiation", # W/m^2
        "diffuse_radiation", # W/m^2
        "direct_normal_irradiance", # W/m^2
        "global_tilted_irradiance", # W/m^2
        "terrestrial_radiation", # W/m^2
    )

    vars_radiation_instant = (
        "shortwave_radiation_instant", # W/m^2
        "direct_radiation_instant", # W/m^2
        "diffuse_radiation_instant", # W/m^2
        "direct_normal_irradiance_instant", # W/m^2
        "global_tilted_irradiance_instant", # W/m^2
        "terrestrial_radiation_instant" # W/m^2
    )

    phys_limits = {
        "temperature_2m": (-45., 50.),  # Extreme global temperature range; robust for outliers.
        "relative_humidity_2m": (0, 100),  # Physical constraint of humidity percentage.
        "surface_pressure": (900., 1080.),  # Typical for Germany; excludes extreme altitudes.
        "wind_speed_10m": (0., 200),  # km/h Conservative; rare globally, but robust for outliers.
        "wind_speed_100m": (0., 200),  # km/h Same as 10m; aligns with rare global extremes.
        "wind_direction_10m": (0., 360),  # Wind direction inherently constrained to this range.
        "wind_direction_100m": (0., 360),  # Same as 10m; inherent constraint.
        "precipitation": (0, 100),  # Accounts for intense rainfall events in Germany.
        "wind_gusts_10m": (0., 300),  # km/h Conservative for error filtering; rare globally.
        "cloud_cover": (0, 100),  # Physical constraint of cloud coverage percentage.
        "shortwave_radiation": (0, 1000),  # Peak solar radiation under clear skies.
    }

    def __init__(self, start_date: pd.Timestamp, location_list: list, variable_list, verbose: bool = False):
        self.start_date = start_date
        self.location_list = location_list
        self.verbose = verbose
        self.variable_list = variable_list

        # check if it is a variable from the list
        for variable in self.variable_list:
            if (not variable in self.vars_basic) and \
                    (not variable in self.vars_wind) and \
                    (not variable in self.vars_radiation):
                raise Exception(f"{variable} is not a valid variable")

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
        if not len(hourly) == len(self.location_list):
            raise ValueError(
                f"Requesting data for {len(self.location_list)} locations, got {len(hourly)} hourly data."
            )

        df = pd.DataFrame()
        for hourly, loc in zip(hourly, self.location_list):
            hourly_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }

            for i, var in enumerate(self.variable_list):
                data = hourly.Variables(i).ValuesAsNumpy()
                hourly_data[var] = data
            hourly_dataframe = pd.DataFrame(data=hourly_data)
            hourly_dataframe.set_index('date', inplace=True)

            # add suffix
            mapping = {name : name + loc['suffix'] for name in hourly_dataframe.columns if name != 'date'}
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
        lats = [loc['lat'] for loc in self.location_list]
        lons = [loc['lon'] for loc in self.location_list]
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "hourly": ''.join([var + ',' for var in self.variable_list])[:-1],
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

# def check_phys_limits_in_data(df: pd.DataFrame) -> pd.DataFrame:
#
#     for key, lim in OpenMeteo.phys_limits.items():
#         for loc in locations:
#             kkey = key+loc['suffix']
#             if kkey in df:
#                 df[kkey].where(df[kkey] < lim[0], None)
#                 df[kkey].where(df[kkey] > lim[1], None)
#     return df

def create_openmeteo_from_api(fpath:str, locations:list, variables:list, start_date:pd.Timestamp, verbose:bool):
    if verbose: print(f"Collecting historical data from OpenMeteo from {start_date} ({len(locations)})"
                     f"{[loc['name'] for loc in locations]} locations")

    om = OpenMeteo(start_date, locations, variables, verbose=verbose)
    df_hist = om.collect()
    df_hist = df_hist[start_date:]
    idx = df_hist.index[-1]-timedelta(days=14) # separate historic and forecasted data
    df_hist[:idx].to_parquet(fpath,engine='pyarrow')
    df_hist[idx+timedelta(hours=1):].to_parquet( fpath.replace('history','forecast'),engine='pyarrow' )
    if verbose: print(f"OpenMeteo data updated. "
                      f"Collected df_hist={df_hist[:idx].shape} and df_forecast={df_hist[idx+timedelta(hours=1):].shape}. ")

def update_openmeteo_from_api(fpath:str, verbose:bool, locations:list, variables:list, ):
    df_hist = pd.read_parquet(fpath)
    if (df_hist.index[-1] >= pd.Timestamp.today(tz='UTC')):
        raise ValueError(f"Cannot update df_hist as its idx[-1]={df_hist.index[-1]} File={fpath}")
    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
    start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
    if verbose: print(f"Updating openmeteo {fpath} with {len(variables)} variables from {start_date}. "
                      f"Current data has shape {df_hist.shape}")
    om = OpenMeteo(start_date, locations, variables, verbose=verbose)
    df_om = om.collect()
    if not df_hist.columns.equals(df_om.columns):
        unique_to_df1 = set(df_om.columns) - set(df_hist.columns)
        unique_to_df2 = set(df_hist.columns) - set(df_om.columns)
        raise KeyError(f"! Error. Column mismatch between historical and forecasted weather for file {fpath}\n!"
                       f" (expected {len(variables)} variables) unique to"
                       f" df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")
    df_om = df_om.combine_first(df_hist) # overwrite previous forecasts with actual data
    df_om.sort_index(inplace=True)
    idx = df_om.index[-1]-timedelta(days=14) # separate historic and forecasted data
    df_om[:idx].to_parquet(fpath,engine='pyarrow')
    df_om[idx+timedelta(hours=1):].to_parquet( fpath.replace('history', 'forecast'), engine='pyarrow' )
    if verbose: print(f"OpenMeteo file {fpath} updated with {len(variables)} variables.\n"
                      f"Collected df_hist={df_om[:idx].shape} and df_forecast={df_om[idx+timedelta(hours=1):].shape}. ")

if __name__ == '__main__':
    # todo add tests

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    from data_collection_modules.german_locations import loc_cities

    om = OpenMeteo(
        start_date=today - timedelta(days=30),
        location_list=loc_cities, variable_list=OpenMeteo.vars_radiation, verbose=True
    )
    # df = om.collect()
    # df = check_phys_limits_in_data(df)
    # df:pd.DataFrame = df.interpolate(method='linear')
    # df.to_parquet('./tmp_om.parquet')



    # import matplotlib.pyplot as plt
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['diffuse_radiation_ham'], color='red',label='Ham, diffuse_radiation')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['direct_radiation_ham'], color='orange',label='Ham, direct_radiation')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['shortwave_radiation_ham'], color='pink',label='Ham, shortwave_radiation')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['direct_normal_irradiance_ham'], color='magenta',label='Ham, direct_normal_irradiance')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['global_tilted_irradiance_ham'], color='cyan',label='Ham, global_tilted_irradiance')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['terrestrial_radiation_ham'], color='lime',label='Ham, terrestrial_radiation')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['shortwave_radiation_instant_ham'], color='green',label='Ham, shortwave_radiation_instant')
    # plt.plot(df.tail(30*24).index, df.tail(30*24)['direct_radiation_instant_ham'], color='blue',label='Ham, direct_radiation_instant')
    # plt.legend(loc='best')
    # plt.show()

