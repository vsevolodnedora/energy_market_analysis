import pandas as pd
from retry_requests import retry
import time
import os
import random
import requests
import openmeteo_requests
from openmeteo_requests.Client import OpenMeteoRequestsError
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse


from datetime import datetime, timedelta
from pysolar.solar import get_altitude, get_azimuth

from logger import get_logger
logger = get_logger(__name__)




class OpenMeteo:

    # ---------- HOURLY DATA (hourly) ---------------

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
    vars = vars_basic + vars_wind + vars_radiation

    # ----------- 15 MIN DATA (minutely_15) ---------------

    vars_basic_15min = (
        "temperature_2m", # degrees, Celsius
        "relative_humidity_2m",  # percentage
        # "dew_point_2m",
        "apparent_temperature",
        "precipitation"
        # No cloud_cover !
    )
    vars_wind_15min = (
        "wind_speed_10m", # velocity, km/h
        # "wind_speed_100m" No data for 100m !
        "wind_speed_80m", # velocity, km/h
        "wind_direction_10m", # velocity, km/h
        "wind_direction_80m", # velocity, km/h
        "wind_gusts_10m"# velocity, km/h
    )

    vars_radiation_15min = (
        "shortwave_radiation",  # W/m^2
        "direct_radiation",  # W/m^2
        "diffuse_radiation",  # W/m^2
        "direct_normal_irradiance",  # W/m^2
        "global_tilted_irradiance",  # W/m^2
        "terrestrial_radiation" # W/m^2
    )

    phys_limits = {
        "temperature_2m": (-45., 50.),  # Extreme global temperature range; robust for outliers.
        "relative_humidity_2m": (0, 100),  # Physical constraint of humidity percentage.
        "surface_pressure": (900., 1080.),  # Typical for Germany; excludes extreme altitudes.
        "precipitation": (0, 100),  # Accounts for intense rainfall events in Germany.
        "cloud_cover": (0, 100),  # Physical constraint of cloud coverage percentage.

        # wind
        "wind_speed_10m": (0., 200),  # km/h Conservative; rare globally, but robust for outliers.
        "wind_speed_80m": (0., 200),  # km/h Conservative; rare globally, but robust for outliers.
        "wind_speed_100m": (0., 200),  # km/h Same as 10m; aligns with rare global extremes.
        "wind_direction_10m": (0., 360),  # Wind direction inherently constrained to this range.
        "wind_direction_80m": (0., 360),  # Wind direction inherently constrained to this range.
        "wind_direction_100m": (0., 360),  # Same as 10m; inherent constraint.
        "wind_gusts_10m": (0., 300),  # km/h Conservative for error filtering; rare globally.

        # radiation
        "shortwave_radiation": (0., 1400.),  # W/m^2 Clear-sky tropical noon maximum.
        "direct_radiation": (0., 1200.),  # W/m^2 Approximate maximum under clear-sky conditions.
        "diffuse_radiation": (0., 450.),  # W/m^2 Typical maximum for diffuse radiation.
        "direct_normal_irradiance": (0., 1200.),  # W/m^2 Clear-sky maximum at solar noon.
        "global_tilted_irradiance": (0., 1400.),  # W/m^2 Includes combined radiation on a tilted plane.
        "terrestrial_radiation": (200., 2000.),  # W/m^2 Typical range for Earth's surface.
    }

    vars_radiation_instant = (
        "shortwave_radiation_instant", # W/m^2
        "direct_radiation_instant", # W/m^2
        "diffuse_radiation_instant", # W/m^2
        "direct_normal_irradiance_instant", # W/m^2
        "global_tilted_irradiance_instant", # W/m^2
        "terrestrial_radiation_instant" # W/m^2
    )

    def __init__(self, start_date: pd.Timestamp, location_list: list, variable_list:tuple, freq:str, verbose: bool = False):
        self.start_date = start_date
        self.location_list = location_list
        self.verbose = verbose
        self.variable_list = variable_list
        self.freq = freq
        if not self.freq in ['hourly', 'minutely_15']:
            raise NotImplementedError(f'Frequency {self.freq} is not implemented. Use "hourly" or "minutely_15"')

        # check if it is a variable from the list
        if self.freq == 'hourly':
            for variable in self.variable_list:
                if (not variable in self.vars_basic) and \
                        (not variable in self.vars_wind) and \
                        (not variable in self.vars_radiation):
                    raise Exception(f"{variable} is not a valid variable for frequency {self.freq}")
        elif self.freq == 'minutely_15':
            for variable in self.variable_list:
                if (not variable in self.vars_basic_15min) and \
                        (not variable in self.vars_wind_15min) and \
                        (not variable in self.vars_radiation_15min):
                    raise Exception(f"{variable} is not a valid variable for frequency {self.freq}")


        today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
        self.today = today.normalize() + pd.DateOffset(hours=today.hour)  # leave only hours

        yesterday = today.normalize() - pd.DateOffset(days=1)
        self.yesterday_last_hour = yesterday

        self.day_before_yesterday = self.yesterday_last_hour.normalize() - pd.DateOffset(days=1)

        tomorrow = today.normalize() + pd.DateOffset(days=1)
        self.tomorrow_first_hour = tomorrow

    def make_request_15min(self, url: str, params: dict) -> pd.DataFrame:
        # No caching session, just a regular retry session
        retry_session = retry(requests.Session(), retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        minutely_15 = []
        for i in range(5):
            try:
                responses = openmeteo.weather_api(url, params=params)
                minutely_15 = []
                for response in responses:
                    minutely_15.append( response.Minutely15() )
            except Exception as e:
                if self.verbose: logger.error(f"Failed to fetch 15 minute weather data from openmeteo {i}/{5} "
                                       f"for {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')} "
                                       f"with Error:\n{e}")
                time.sleep(20*i)  # in case API is overloaded
                continue
            break

        if len(minutely_15) == 0:
            raise ConnectionError(
                f"Failed to fetch 15 min weather data from OpenMeteo 5 times for"
                f" {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')}"
            )
        if not len(minutely_15) == len(self.location_list):
            raise ValueError(
                f"Requesting 15 min data for {len(self.location_list)} locations, got {len(minutely_15)} 15 min data."
            )

        df = pd.DataFrame()
        for minutely_15ly, loc in zip(minutely_15, self.location_list):
            minutely_15_data = {
                "date": pd.date_range(
                    start=pd.to_datetime(minutely_15ly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(minutely_15ly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=minutely_15ly.Interval()),
                    inclusive="left"
                )
            }

            for i, var in enumerate(self.variable_list):
                data = minutely_15ly.Variables(i).ValuesAsNumpy()
                minutely_15_data[var] = data
            minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
            minutely_15_dataframe.set_index('date', inplace=True)

            # add suffix
            mapping = {name : name + loc['suffix'] for name in minutely_15_dataframe.columns if name != 'date'}
            minutely_15_dataframe.rename(columns=mapping, inplace=True)

            if df.empty:df = minutely_15_dataframe
            else: df = pd.merge(df, minutely_15_dataframe, left_index=True, right_index=True, how='left')

        return df

    def make_request_hourly(self, url: str, params: dict) -> pd.DataFrame:
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
                if self.verbose: logger.error(
                    f"Failed to fetch weather data from openmeteo {i}/{5} "
                    f"for {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')} "
                    f"url={url} "
                    f"with Error:\n{e}"
                )
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

    def _collect_past_actual(self,data_dir:str, freq:str)->pd.DataFrame:
        if freq != 'hourly':
            raise NotImplementedError(f"Frequency {freq} is not implemented for hourly data")
        # collect historic data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": [loc['lat'] for loc in self.location_list],
            "longitude": [loc['lon'] for loc in self.location_list],
            "start_date": self.start_date.strftime('%Y-%m-%d'),
            "end_date": self.today.strftime('%Y-%m-%d'),
            'hourly': ''.join([var + ',' for var in self.variable_list])[:-1], # "hourly" or "minutely_15"
        }

        # making heavy-duty API call (it might overload API so we need to be able to fail and load)
        fname = data_dir  + '/' + f"tmp_hist_{freq}.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            hist_data = pd.read_parquet(fname)
        else:
            hist_data = self.make_request_hourly(url, params)
        return hist_data

    def _collect_past_forecast(self,data_dir:str, freq:str)->pd.DataFrame:
        # collect historic data
        params = {
            "latitude": [loc['lat'] for loc in self.location_list],
            "longitude": [loc['lon'] for loc in self.location_list],
            "start_date": self.start_date.strftime('%Y-%m-%d'),
            "end_date": self.today.strftime('%Y-%m-%d'),
        }
        # collect historic forecast (to bridge the data till forecasts starts)
        # params['start_date'] = self.day_before_yesterday.strftime('%Y-%m-%d')
        # params['end_date'] = self.today.strftime('%Y-%m-%d')
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        fname = data_dir  + '/' + f"tmp_hist_forecast_{freq}.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            hist_forecast_data = pd.read_parquet(fname)
        else:
            if freq == 'hourly':
                params['hourly'] = ''.join([var + ',' for var in self.variable_list])[:-1]
                hist_forecast_data = self.make_request_hourly(url, params)
            else:
                params['minutely_15'] = ''.join([var + ',' for var in self.variable_list])[:-1]
                hist_forecast_data = self.make_request_15min(url, params)
        return hist_forecast_data

    def _collect_forecast(self,data_dir:str, freq:str)->pd.DataFrame:

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'forecast_days':14,
            "latitude": [loc['lat'] for loc in self.location_list],
            "longitude": [loc['lon'] for loc in self.location_list],
        }

        fname = data_dir  + '/' + f"tmp_forecast_{freq}.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            forecast_data = pd.read_parquet(fname)
        else:
            if freq == 'hourly':
                params['hourly']= ''.join([var + ',' for var in self.variable_list])[:-1], # "hourly" or "minutely_15"
                forecast_data = self.make_request_hourly(url, params)
            else:
                params['minutely_15']= ''.join([var + ',' for var in self.variable_list])[:-1], # "hourly" or "minutely_15"
                forecast_data = self.make_request_15min(url, params)
        return forecast_data


def add_solar_elevation_and_azimuth(df: pd.DataFrame, locations, verbose=False):

    # add static features (location dependent)
    for i, loc in enumerate(locations):
        lat, lon, suffix = loc['lat'], loc['lon'], loc['suffix']
        if verbose:
            logger.info(f"Adding solar elevation and/or azimuth to {loc['name']} ({i}/{len(locations)})",)
        elevation_list = []
        azimuth_list = []
        for ts in df.index:
            # ts is a pd.Timestamp in UTC
            # Convert to Python datetime (naive or aware).
            # If needed, ensure it's aware in UTC: ts.to_pydatetime()
            dt_utc = ts.to_pydatetime()
            elevation_deg = get_altitude(lat, lon, dt_utc)
            azimuth_deg = get_azimuth(lat, lon, dt_utc)
            elevation_list.append(elevation_deg)
            azimuth_list.append(azimuth_deg)

        df[f"solar_elevation_deg{suffix}"] = elevation_list
        df[f"solar_azimuth_deg{suffix}"] = azimuth_list
    return df


def create_openmeteo_from_api(
        datadir:str,
        locations:list, variables:tuple, start_date:pd.Timestamp, freq:str, verbose:bool
):

    om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)

    # --- collect past actual data and update database

    logger.info(
        f"Collecting historical actual data from OpenMeteo ({freq}) from {start_date} ({len(locations)})"
        f"{[loc['name'] for loc in locations]} locations"
    )
    if freq == 'hourly':
        df_hist_ = om._collect_past_actual(data_dir=datadir, freq=freq)
        if 'solar' in datadir:
            df_hist_ = add_solar_elevation_and_azimuth(df_hist_, locations, verbose=verbose)
        fpath = datadir + f'history_{freq}.parquet'
        logger.info(f"Writing historical data in {fpath} (shape={df_hist_.shape})")
        df_hist_.to_parquet(fpath)
    else:
        logger.info(f"Openmeteo does not have actual historical data for freq={freq}. Skipping...")

    # --- collect past forecast

    logger.info(
        f"Collecting historical forecasts from OpenMeteo from {start_date} ({len(locations)})"
        f"freq={freq} and {[loc['name'] for loc in locations]} locations"
    )
    df_hist_forecast_ = om._collect_past_forecast(data_dir=datadir, freq=freq)
    if 'solar' in datadir:
        df_hist_forecast_ = add_solar_elevation_and_azimuth(df_hist_forecast_, locations, verbose=verbose)
    fpath = datadir + f'hist_forecast_{freq}.parquet'
    logger.info(f"Writing historical forecasts for in {fpath} (shape={df_hist_forecast_.shape})")
    df_hist_forecast_.to_parquet(fpath)

    # --- collect current forecast

    logger.info(
        f"Collecting forecasts from OpenMeteo ({len(locations)}) for "
        f"freq={freq} and {[loc['name'] for loc in locations]} locations"
    )
    df_forecast_ = om._collect_forecast(data_dir=datadir, freq=freq)
    if 'solar' in datadir:
        df_forecast_ = add_solar_elevation_and_azimuth(df_forecast_, locations, verbose=verbose)
    fpath = datadir + f'forecast_{freq}.parquet'
    logger.info(f"Writing forecasts for {datadir} to {fpath} (shape={df_forecast_.shape})")
    df_forecast_.to_parquet(fpath)

def update_openmeteo_from_api(
        datadir:str, verbose:bool, locations:list, variables:tuple, freq:str
):

    if len(locations) == 0:
        logger.info(f"No locations provided for {datadir} and freq={freq}. Exiting...")
        return

    def _update_historic(
            datadir:str, dtype_label:str, verbose:bool, locations:list, variables:tuple, freq:str
    ):

        fpath = datadir + dtype_label + '_' + freq + '.parquet'
        if not os.path.isfile(fpath): raise FileNotFoundError(f"Historical actual data file is not found {fpath}")
        df_hist = pd.read_parquet(fpath)
        if (df_hist.index[-1] >= pd.Timestamp.today(tz='UTC')):
            logger.warning(f"Cannot fpath={fpath} as its "
                           f"idx[-1]={df_hist.index[-1]} >= today={pd.Timestamp.today(tz='UTC')} ")
        last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
        start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
        logger.info(
            f"Collecting historical actual data from OpenMeteo from {start_date} ({len(locations)})"
            f"{[loc['name'] for loc in locations]} locations"
        )

        # collect data
        om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)
        if dtype_label == 'history': df_hist_upd = om._collect_past_actual(data_dir=datadir, freq=freq)
        elif dtype_label == 'hist_forecast': df_hist_upd = om._collect_past_forecast(data_dir=datadir, freq=freq)
        else: raise ValueError(f"Unknown suffix {dtype_label}")
        if 'solar' in datadir: df_hist_upd = add_solar_elevation_and_azimuth(df_hist_upd, locations, verbose=verbose)

        # check if columns match
        if not df_hist.columns.equals(df_hist_upd.columns):
            unique_to_df1 = set(df_hist_upd.columns) - set(df_hist.columns)
            unique_to_df2 = set(df_hist.columns) - set(df_hist_upd.columns)
            raise KeyError(f"! Error. Column mismatch between historical and forecasted weather for file {fpath}\n!"
                           f" (expected {len(variables)} variables) unique to"
                           f" df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")

        df_hist_upd = df_hist_upd.combine_first(df_hist[:df_hist_upd.index[0]]) # overwrite previous forecasts with actual data
        df_hist_upd = df_hist_upd.loc[~df_hist_upd.index.duplicated(keep='first')]
        df_hist_upd.sort_index(inplace=True)
        logger.info(f"Overriding openmeteo past actual dataframe at {fpath} (shape={df_hist_upd.shape})")
        df_hist_upd.to_parquet(fpath)


    # update past actual data
    if freq == 'hourly':
        _update_historic(
            datadir=datadir, dtype_label='history', verbose=verbose,
            locations=locations, variables=variables, freq=freq
        )

    # update past forecasts data
    _update_historic(
        datadir=datadir, dtype_label='hist_forecast', verbose=verbose,
        locations=locations, variables=variables, freq=freq
    )

    # collect current forecasts
    fpath = datadir + 'forecast' + f"_{freq}" + '.parquet'
    om = OpenMeteo(pd.Timestamp.today(), locations, variables, freq=freq, verbose=verbose)
    df_forecast = om._collect_forecast(data_dir=datadir, freq=freq)
    if 'solar' in datadir: df_forecast = add_solar_elevation_and_azimuth(df_forecast, locations, verbose=verbose)
    logger.info(f"Saving openmeteo latest forecast to {fpath} (shape={df_forecast.shape})")
    df_forecast.to_parquet(fpath)



if __name__ == '__main__':
    # todo add tests

    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours

    from data_collection_modules.german_locations import loc_cities

    om = OpenMeteo(
        start_date=today - timedelta(days=1),
        location_list=loc_cities,
        variable_list=OpenMeteo.vars_basic_15min,
        freq='minutely_15',
        verbose=True
    )

    # df = om.collect('../datababase_15min/openmeteo/')
    # df.to_parquet('./tmp_om.parquet')

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

