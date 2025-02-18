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

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:98.0)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    # Add more to your liking
]

TOR_PROXIES = {
    'http':  'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}

def get_retry_session(use_tor=False) -> requests.Session:
    """Return a requests.Session with random User-Agent and optional Tor proxies."""
    session = requests.Session()

    # Rotate User-Agent
    random_user_agent = random.choice(USER_AGENTS)
    session.headers.update({"User-Agent": random_user_agent})

    # Optionally set Tor proxies
    if use_tor:
        session.proxies.update(TOR_PROXIES)

    # If you want to add retries, do so here:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=5,  # total retry attempts
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],  # retry on these statuses
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

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

    def NEW_make_request_hourly(self, url: str, params: dict) -> pd.DataFrame:

        # 1. Start with a normal session (rotating user agent)
        retry_session = get_retry_session(use_tor=False)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        responses_data = []
        max_tries = 5
        for i in range(max_tries):
            try:
                responses = openmeteo.weather_api(url, params=params)
                responses_data = responses
                break

            except OpenMeteoRequestsError as e:
                # If error body says "429", try fallback to Tor
                if self.verbose:
                    logger.error(
                        f"OpenMeteoRequestsError {i}/{max_tries}: {e}. "
                        f"Trying fallback (Tor) if 429 limit encountered."
                    )

                # If you see that the error JSON or code is 429, do the fallback:
                if "429" in str(e):
                    # fallback with Tor session
                    time.sleep(3)
                    retry_session = get_retry_session(use_tor=True)
                    openmeteo = openmeteo_requests.Client(session=retry_session)
                    # Attempt again
                    try:
                        responses = openmeteo.weather_api(url, params=params)
                        responses_data = responses
                        break
                    except Exception as e_tor:
                        logger.error("Tor fallback also failed: %s", e_tor)
                        # Optionally keep trying; or break early.
                        time.sleep(5 * (i+1))
                        continue
                else:
                    time.sleep(5 * (i+1))
                    continue

            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"General error {i}/{max_tries} "
                        f"for {params.get('start_date', 'N/A')} -> "
                        f"{params.get('end_date', 'N/A')} (url={url}):\n{e}"
                    )
                time.sleep(5 * (i+1))
                continue

        if not responses_data:
            raise ConnectionError(
                f"Failed to fetch weather data from OpenMeteo {max_tries} times "
                f"for {params.get('start_date', 'N/A')} -> {params.get('end_date', 'N/A')}"
            )

        if len(responses_data) != len(self.location_list):
            raise ValueError(
                f"Requested data for {len(self.location_list)} locations, "
                f"but got {len(responses_data)} responses."
            )

        # Construct dataframe
        df = pd.DataFrame()
        for hourly_resp, loc in zip(responses_data, self.location_list):
            # Convert to times and variables
            date_range = pd.date_range(
                start=pd.to_datetime(hourly_resp.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly_resp.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly_resp.Interval()),
                inclusive="left"
            )
            hourly_data = {"date": date_range}
            for i, var in enumerate(self.variable_list):
                data = hourly_resp.Variables(i).ValuesAsNumpy()
                hourly_data[var] = data

            hourly_dataframe = pd.DataFrame(data=hourly_data).set_index('date')
            # add suffix to column names
            rename_map = {
                col: col + loc['suffix']
                for col in hourly_dataframe.columns
            }
            hourly_dataframe.rename(columns=rename_map, inplace=True)

            if df.empty:
                df = hourly_dataframe
            else:
                df = pd.merge(df, hourly_dataframe, left_index=True, right_index=True, how='left')

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
        fname = data_dir  + '/' + "tmp_hist.parquet"
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
        fname = data_dir  + '/' + "tmp_hist_forecast.parquet"
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

        fname = data_dir  + '/' + "tmp_forecast.parquet"
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


    def _collect_hourly(self, data_dir:str) -> pd.DataFrame:
        '''
            Openmeteo data is available as follows:
            - Historic data is up to the end of the previous day.
            - Forecast data is available from the start of the next day
            - Historic forecast is available from the past till future dates.

            We collect them all to create continuous and accurate time-series data as follows:
            [start_date, ... end_of_yersteday][start_of_today ... end_of_today][start_of_tomorrow ... end_date]
            [      historic actual data      ][   historic forecasted data    ][  actual forecasted data      ]

            We acknoledge that using historic forecast is not optimal.
            Thus, when we update the data next day we replace the previously saved historic forecast with
            actual histroic data as it becomes available. This assures that we always have actual observed and
            actual forecasted data. Historic forecast is only used to bridge this gap today.
        '''

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
            'hourly': ''.join([var + ',' for var in self.variable_list])[:-1], # "hourly" or "minutely_15"
        }
        # making heavy-duty API call (it might overload API so we need to be able to fail and load)
        fname = data_dir  + '/' + "tmp_hist.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            hist_data = pd.read_parquet(fname)
        else:
            hist_data = self.make_request_hourly(url, params)
            time.sleep(30)


        # collect historic forecast (to bridge the data till forecasts starts)
        params['start_date'] = self.day_before_yesterday.strftime('%Y-%m-%d')
        params['end_date'] = self.today.strftime('%Y-%m-%d')
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        fname = data_dir  + '/' + "tmp_hist_forecast.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            hist_forecast_data = pd.read_parquet(fname)
        else:
            hist_forecast_data = self.make_request_hourly(url, params)
            time.sleep(30)


        # collect forecast
        url = "https://api.open-meteo.com/v1/forecast"
        params['forecast_days'] = 14
        del params['start_date']
        del params['end_date']
        fname = data_dir  + '/' + "tmp_forecast.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            forecast_data = pd.read_parquet(fname)
        else:
            forecast_data = self.make_request_hourly(url, params)
            time.sleep(30)

        # combine all data to form a continuous dataset
        df = hist_data.combine_first(hist_forecast_data)
        df = forecast_data.combine_first(df)

        df.sort_index(inplace=True)
        if not pd.infer_freq(df.index) == 'h':
            raise ValueError("Dataframe must have 'h' frequency for hourly openmeteo")

        # delete temporary files if present
        if self.verbose:
            logger.info(
                f"Openmeteo data from {start_date} to {end_date} hourly is collected successfully (df={df.shape})."
                  f"Removing temporary files...")
            tmp_files = [
                data_dir  + '/' + "tmp_hist.parquet",
                data_dir  + '/' + "tmp_hist_forecast.parquet",
                data_dir  + '/' + "tmp_forecast.parquet"
            ]
            for tmp_file in tmp_files:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        return df

    def _collect_15_min_forecasts(self, data_dir:str) -> pd.DataFrame:
        '''
            Openmeteo data is available as follows:
            - Forecast data is available from the start of the next day
            - Historic forecast is available from the past till future dates.

            We collect them all to create continuous and accurate time-series data as follows:
            [start_date ... end_of_today][start_of_tomorrow ... end_date]
            [   historic forecasted data    ][  actual forecasted data      ]

            We acknoledge that using historic forecast is not optimal. However, openmeteo does not provide
            15 min actual historic data
        '''

        # collect historic data
        start_date = self.start_date
        end_date = self.day_before_yesterday  # after that it returns nans
        lats = [loc['lat'] for loc in self.location_list]
        lons = [loc['lon'] for loc in self.location_list]


        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            'minutely_15': ''.join([var + ',' for var in self.variable_list])[:-1], # "hourly" or "minutely_15"
        }

        # collect historic forecast (to bridge the data till forecasts starts)
        params['start_date'] = start_date.strftime('%Y-%m-%d')
        params['end_date'] = self.today.strftime('%Y-%m-%d')
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        fname = data_dir  + '/' + "tmp_hist_forecast.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            hist_forecast_data = pd.read_parquet(fname)
        else:
            hist_forecast_data = self.make_request_15min(url, params)
            time.sleep(30)


        # collect forecast
        url = "https://api.open-meteo.com/v1/forecast"
        params['forecast_days'] = 14
        del params['start_date']
        del params['end_date']
        fname = data_dir  + '/' + "tmp_forecast.parquet"
        if os.path.exists(fname):
            if self.verbose: logger.info(f"Loading temporary file: {fname}")
            forecast_data = pd.read_parquet(fname)
        else:
            forecast_data = self.make_request_15min(url, params)
            time.sleep(30)

        # combine all data to form a continuous dataset
        df = forecast_data.combine_first(hist_forecast_data)

        df.sort_index(inplace=True)
        time_diffs = df.index.to_series().diff().dropna()
        if not (time_diffs == pd.Timedelta(minutes=15)).all():
            raise ValueError(f"Dataframe contains irregular time intervals:\n{time_diffs.value_counts()}")

        # delete temporary files if present
        if self.verbose:
            logger.info(
                f"Openmeteo 15min data from {start_date} to {end_date} is collected successfully (df={df.shape})."
                f"Removing temporary files...")
            tmp_files = [
                data_dir  + '/' + "tmp_hist_forecast.parquet",
                data_dir  + '/' + "tmp_forecast.parquet"
            ]
            for tmp_file in tmp_files:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        return df

    def collect(self, data_dir:str):
        if self.freq == 'hourly': return self._collect_hourly(data_dir)
        elif self.freq == 'minutely_15': return self._collect_15_min_forecasts(data_dir)
        else: raise NotImplementedError(f"Data collection for freq={self.freq} not implemented")

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
        datadir:str, suffix:str,
        locations:list, variables:tuple, start_date:pd.Timestamp, freq:str, verbose:bool
):

    om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)

    # --- collect past actual data and update database

    logger.info(
        f"Collecting historical actual data from OpenMeteo from {start_date} ({len(locations)})"
        f"{[loc['name'] for loc in locations]} locations"
    )
    if freq == 'hourly':
        df_hist_ = om._collect_past_actual(data_dir=datadir, freq=freq)
        if 'solar' in suffix:
            df_hist_ = add_solar_elevation_and_azimuth(df_hist_, locations, verbose=verbose)
        fpath = datadir + suffix + '_history.parquet'
        logger.info(f"Writing historical data for {suffix} to {fpath} (shape={df_hist_.shape})")
        df_hist_.to_parquet(fpath)
    else:
        logger.info(f"Openmeteo does not have actual historical data for freq={freq}. Skipping...")

    # --- collect past forecast

    logger.info(
        f"Collecting historical forecasts from OpenMeteo from {start_date} ({len(locations)})"
        f"freq={freq} and {[loc['name'] for loc in locations]} locations"
    )
    df_hist_forecast_ = om._collect_past_forecast(data_dir=datadir, freq=freq)
    if 'solar' in suffix:
        df_hist_forecast_ = add_solar_elevation_and_azimuth(df_hist_forecast_, locations, verbose=verbose)
    fpath = datadir + suffix + '_hist_forecast.parquet'
    logger.info(f"Writing historical forecasts for {suffix} to {fpath} (shape={df_hist_forecast_.shape})")
    df_hist_forecast_.to_parquet(fpath)

    # --- collect current forecast

    logger.info(
        f"Collecting forecasts from OpenMeteo ({len(locations)}) for "
        f"freq={freq} and {[loc['name'] for loc in locations]} locations"
    )
    df_forecast_ = om._collect_forecast(data_dir=datadir, freq=freq)
    if 'solar' in suffix:
        df_forecast_ = add_solar_elevation_and_azimuth(df_forecast_, locations, verbose=verbose)
    fpath = datadir + suffix + '_forecast.parquet'
    logger.info(f"Writing forecasts for {suffix} to {fpath} (shape={df_forecast_.shape})")
    df_forecast_.to_parquet(fpath)

def update_openmeteo_from_api(
        datadir:str, suffix:str, verbose:bool, locations:list, variables:tuple, freq:str
):

    def _update_historic(
            datadir:str, suffix:str, suffix2:str, verbose:bool, locations:list, variables:tuple, freq:str
    ):

        if (freq != 'hourly') and (suffix2 == 'history'):
            logger.info(f"Actual historical data is not available for freq={freq} Skipping...")
            return

        fpath = datadir + suffix + '_' + suffix2 + '.parquet'
        if not os.path.isfile(fpath): raise FileNotFoundError(f"Historical actual data file is not found {fpath}")
        df_hist = pd.read_parquet(fpath)
        if (df_hist.index[-1] >= pd.Timestamp.today(tz='UTC')):
            logger.warning(f"Cannot update {suffix} with {suffix2} as its "
                           f"idx[-1]={df_hist.index[-1]} >= today={pd.Timestamp.today(tz='UTC')} "
                           f"File={fpath}")
        last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
        start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
        logger.info(
            f"Collecting historical actual data from OpenMeteo from {start_date} ({len(locations)})"
            f"{[loc['name'] for loc in locations]} locations"
        )

        # collect data
        om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)
        if suffix2 == 'history': df_hist_upd = om._collect_past_actual(data_dir=datadir, freq=freq)
        elif suffix2 == 'hist_forecast': df_hist_upd = om._collect_past_forecast(data_dir=datadir, freq=freq)
        else: raise ValueError(f"Unknown suffix {suffix2}")
        if 'solar' in suffix: df_hist_upd = add_solar_elevation_and_azimuth(df_hist_upd, locations, verbose=verbose)

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
    _update_historic(
        datadir=datadir, suffix=suffix, suffix2='history', verbose=verbose,
        locations=locations, variables=variables, freq=freq
    )
    # update past forecasts data
    _update_historic(
        datadir=datadir, suffix=suffix, suffix2='hist_forecast', verbose=verbose,
        locations=locations, variables=variables, freq=freq
    )

    # collect current forecasts
    fpath = datadir + suffix + '_' + 'forecast' + '.parquet'
    om = OpenMeteo(pd.Timestamp.today(), locations, variables, freq=freq, verbose=verbose)
    df_forecast = om._collect_forecast(data_dir=datadir, freq=freq)
    if 'solar' in suffix: df_forecast = add_solar_elevation_and_azimuth(df_forecast, locations, verbose=verbose)
    logger.info(f"Saving openmeteo latest forecast to {fpath} (shape={df_forecast.shape})")
    df_forecast.to_parquet(fpath)


def OLD_create_openmeteo_from_api(fpath:str, locations:list, variables:tuple, start_date:pd.Timestamp, freq:str, verbose:bool):

    if verbose: logger.info(f"Collecting historical data from OpenMeteo from {start_date} ({len(locations)})"
                     f"{[loc['name'] for loc in locations]} locations")

    om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)
    df_hist = om.collect(data_dir=os.path.dirname(fpath))
    if 'solar' in fpath:
        df_hist = add_solar_elevation_and_azimuth(df_hist, locations, verbose=verbose)

    df_hist = df_hist[start_date:]
    idx = df_hist.index[-1]-timedelta(days=14) # separate historic and forecasted data
    df_hist[:idx].to_parquet(fpath,engine='pyarrow')
    if freq=='hourly':
        df_hist[idx+timedelta(hours=1):].to_parquet( fpath.replace('history','forecast'),engine='pyarrow' )
    elif freq=='minutely_15':
        df_hist[idx+timedelta(minutes=15):].to_parquet( fpath.replace('history','forecast'),engine='pyarrow' )
    else: raise NotImplementedError(f"Frequency {freq} not implemented. Use 'hourly' or 'minutely_15' instead.")
    if verbose: logger.info(
        f"OpenMeteo data updated. Freq: {freq} "
        f"Collected df_hist={df_hist[:idx].shape} and df_forecast={df_hist[idx+timedelta(hours=1):].shape}. ")

def OLD_update_openmeteo_from_api(fpath:str, verbose:bool, locations:list, variables:tuple, freq:str):
    df_hist = pd.read_parquet(fpath)
    if (df_hist.index[-1] >= pd.Timestamp.today(tz='UTC')):
        raise ValueError(f"Cannot update df_hist as its idx[-1]={df_hist.index[-1]} File={fpath}")

    last_timestamp = pd.Timestamp(df_hist.dropna(how='all', inplace=False).last_valid_index())
    start_date = last_timestamp - timedelta(days=3) # overwrite previous historic forecast with actual data
    if verbose: logger.info(f"Updating openmeteo {fpath} with {len(variables)} variables from {start_date}. "
                      f"Current data has shape {df_hist.shape}")

    om = OpenMeteo(start_date, locations, variables, freq=freq, verbose=verbose)
    df_om = om.collect(data_dir=os.path.dirname(fpath))

    if 'solar' in fpath:
        df_om = add_solar_elevation_and_azimuth(df_om, locations, verbose=verbose)

    # check if columns match
    if not df_hist.columns.equals(df_om.columns):
        unique_to_df1 = set(df_om.columns) - set(df_hist.columns)
        unique_to_df2 = set(df_hist.columns) - set(df_om.columns)
        raise KeyError(f"! Error. Column mismatch between historical and forecasted weather for file {fpath}\n!"
                       f" (expected {len(variables)} variables) unique to"
                       f" df_om_hist={unique_to_df1} Unique to df_om_forecast={unique_to_df2}")

    # combine, sort, split, save
    df_om = df_om.combine_first(df_hist) # overwrite previous forecasts with actual data
    df_om.sort_index(inplace=True)
    idx = df_om.index[-1]-timedelta(days=14) # separate historic and forecasted data
    df_om[:idx].to_parquet(fpath,engine='pyarrow')

    # save forecast that strates at the next timestep
    if freq == 'hourly': idx_1 = idx + timedelta(hours=1)
    elif freq == 'minutely_15': idx_1 = idx + timedelta(minutes=15)
    else: raise NotImplementedError(f"Frequency {freq} not implemented. Use 'hourly' or 'minutely_15' instead.")
    df_om[idx_1:].to_parquet( fpath.replace('history', 'forecast'), engine='pyarrow' )

    if verbose: logger.info(f"OpenMeteo file {fpath} updated with {len(variables)} variables (freq: {freq}).\n"
                      f"Collected df_hist={df_om[:idx].shape} and df_forecast={df_om[idx_1:].shape}. ")


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

    df = om.collect('../datababase_15min/openmeteo/')
    df.to_parquet('./tmp_om.parquet')

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

