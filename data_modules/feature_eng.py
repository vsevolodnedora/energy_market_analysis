import copy

import optuna
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from math import radians, sin, cos
import joblib
import gc
from pysolar.solar import get_altitude, get_azimuth

from data_collection_modules.collect_data_openmeteo import OpenMeteo
from data_collection_modules.german_locations import all_locations
from data_modules.utils import validate_dataframe


from logger import get_logger
logger = get_logger(__name__)

def create_holiday_weekend_series(df_index):
    # Create a Germany holiday calendar
    de_holidays = holidays.Germany()

    # Generate a Series with the index from the DataFrame
    date_series = pd.Series(index=df_index, dtype=int)

    # Loop over each date in the index
    for date in date_series.index:
        # Check if the date is a holiday or a weekend (Saturday=5, Sunday=6)
        if date in de_holidays:
            date_series[date] = 1
        elif date.weekday() >= 5:
            date_series[date] = 1
        else:
            date_series[date] = 0

    return date_series

def create_time_features(index)->pd.DataFrame:
    df_time_featues = pd.DataFrame(index=index)
    df_time_featues['hour'] = df_time_featues.index.hour
    df_time_featues['dayofweek'] = df_time_featues.index.dayofweek
    # df_time_featues['month'] = df_time_featues.index.month
    # df_time_featues['dayofyear'] = df_time_featues.index.dayofyear
    # df_time_featues['weekofyear'] = df_time_featues.index.isocalendar().week.astype(int)
    # df_time_featues['quarter'] = df_time_featues.index.quarter

    # add, encode holidays: Remove timezone from the index; Convert holiday_set to datetime and normalize
    # holiday_set = set(holidays.CountryHoliday('DE', years=range(
    #     df_time_featues.index.year.min(), df_time_featues.index.year.max()+1)))
    # holiday_dates = pd.to_datetime(list(holiday_set)).normalize()
    df_time_featues['is_holiday'] = create_holiday_weekend_series(index)#df_time_featues.index.tz_localize(None).normalize().isin(holiday_dates).astype(int)

    # Create cyclical features for hour and day of week
    df_time_featues['hour_sin'] = np.sin(2. * np.pi * df_time_featues['hour'] / 24.)
    df_time_featues['hour_cos'] = np.cos(2. * np.pi * df_time_featues['hour'] / 24.)
    df_time_featues['day_sin'] = np.sin(2. * np.pi * df_time_featues['dayofweek'] / 7.)
    df_time_featues['day_cos'] = np.cos(2. * np.pi * df_time_featues['dayofweek'] / 7.)

    cyc_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
    df_time_featues[cyc_cols] = df_time_featues[cyc_cols].round(8)

    return df_time_featues


# --- weather-related helper function ---

def compute_air_density(pressure:pd.Series, temperature:pd.Series):
    R_d = 287.05  # J/(kg·K)
    # pressure -- convert hPa -> Pa
    air_density = pressure * 100.0 / (R_d * (temperature + 273.15))
    return air_density

def compute_air_density_moist(temperature:pd.Series,pressure:pd.Series,humidity:pd.Series):
    # T_K = temperature + 273.15
    # p_total_pa = pressure * 100.0
    # # vapor pressure (Magnus formula)
    # T_C = temperature
    # RH = humidity / 100.0
    # e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH
    # e_pa = e_hpa * 100.0
    # p_d_pa = p_total_pa - e_pa
    #
    # R_d = 287.05  # J/(kg·K)
    # R_v = 461.50  # J/(kg·K)
    # air_density_moist = ( p_d_pa / (R_d * T_K) + e_pa / (R_v * T_K) )
    # return air_density_moist
    temperature = np.asarray(temperature).flatten()
    pressure = np.asarray(pressure).flatten()
    humidity = np.asarray(humidity).flatten()

    # Convert temperature °C -> K
    T_K = temperature + 273.15
    # Convert total pressure from hPa -> Pa
    p_total_pa = pressure * 100.0

    # --- Compute vapor pressure in hPa using the Magnus formula ---
    # e (hPa) = 6.112 * exp(17.67 * T / (T + 243.5)) * (RH/100)
    # T is in °C here, so let's call it T_C to be explicit
    T_C = temperature
    RH = humidity / 100.0  # from % to fraction

    e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH
    # convert e from hPa -> Pa
    e_pa = e_hpa * 100.0

    # partial pressure of dry air
    p_d_pa = p_total_pa - e_pa

    # constants
    R_d = 287.05   # J/(kg·K) for dry air
    R_v = 461.50   # J/(kg·K) for water vapor

    # moist air density:
    # ρ_moist = (p_d / (R_d * T)) + (e / (R_v * T))
    return ( p_d_pa / (R_d * T_K) + e_pa / (R_v * T_K) )

def compute_dew_point_temperature(temperature:pd.Series, humidity:pd.Series):
    # Using the Magnus formula variant
    # T_dew in °C
    T_C = temperature
    RH_frac = humidity / 100.0
    # Prevent log(0) by ensuring RH_frac>0
    RH_safe = RH_frac.where(RH_frac > 0, other=0.0001)
    # Commonly used constants for Magnus formula
    a = 17.62
    b = 243.12
    # e_s (saturation vapor pressure in hPa) = 6.112 * exp(a * T / (b + T))
    # Then actual vapor pressure e = e_s * RH
    # T_dew = (b * gamma) / (a - gamma)
    # gamma = ln(RH) + (a * T / (b + T))
    gamma = np.log(RH_safe) + (a * T_C / (b + T_C))
    T_dew = (b * gamma) / (a - gamma)
    dew_point_temperature = T_dew
    return dew_point_temperature

def compute_vapor_pressure(temperature:pd.Series, humidity:pd.Series):
    temperature = np.asarray(temperature).flatten()
    humidity = np.asarray(humidity).flatten()
    T_C = temperature # in Celsius
    RH_frac = humidity / 100.0
    # e (hPa) = 6.112 * exp(17.67 * T / (T + 243.5)) * RH
    e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH_frac
    lvapor_pressure = e_hpa
    return lvapor_pressure # in hPa

def compute_wind_sheer(wind_speed_high: pd.Series, wind_speed_low: pd.Series):
    # Convert inputs to numpy arrays and flatten
    wind_speed_high = np.asarray(wind_speed_high).flatten()
    wind_speed_low = np.asarray(wind_speed_low).flatten()

    # Replace zeros in wind_speed_low with the smallest positive value in the series
    min_positive = np.min(wind_speed_low[wind_speed_low > 0]) if np.any(wind_speed_low > 0) else 1e-6
    wind_speed_low = np.where(wind_speed_low <= 0, min_positive, wind_speed_low)

    # Compute wind shear
    res = wind_speed_high / wind_speed_low
    sheer = np.log(np.maximum(res, 1e-10)) / np.log(10.0)  # Ensure no log(0) issues

    # Replace infinite values with NaN
    sheer = np.where(np.isinf(sheer), np.nan, sheer)

    return sheer

def compute_turbulence_intensity(wind_speed:pd.Series, window:int):
    # TI = rolling std / rolling mean of wind speed
    rolling_std = wind_speed.rolling(window=window).std()
    rolling_mean = wind_speed.rolling(window=window).mean()
    ti = rolling_std / rolling_mean
    return ti

def compute_wind_ramp(wind_speed:pd.Series):
    ramp = wind_speed.diff()
    return ramp

def compute_gust_factor(wind_speed:pd.Series, wind_gust:pd.Series):
    # Convert wind_gust_10m and wind_speed_10m from km/h to m/s to keep consistent units
    gust_m_s = wind_gust * (1000.0 / 3600.0)
    ws_m_s = wind_speed * (1000.0 / 3600.0)

    # gust_factor = gust / wind_speed (handle zero or near-zero speeds)
    gf = gust_m_s / ws_m_s.replace({0: np.nan})  # or clip small speeds
    return gf.replace([np.inf, -np.inf], np.nan)

def compute_wind_chill(temperature:pd.Series, wind_speed:pd.Series):
    wind_speed_ms = wind_speed / 3.6  # convert km/h -> m/s
    T_C = temperature
    wind_chill = (
            13.12 + 0.6215 * T_C
            - 11.37 * (wind_speed_ms ** 0.16)
            + 0.3965 * T_C * (wind_speed_ms ** 0.16)
    )
    return wind_chill

def compute_humidex(temperature:pd.Series, humidity:pd.Series):
    T_C = temperature
    RH = humidity / 100.0
    e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH
    e_pa = e_hpa * 100.0
    dewpoint_term = 0.5555 * (e_pa / 100.0 - 10.0)
    humidex = T_C + dewpoint_term
    return humidex

def compute_wind_power_density(wind_speed:pd.Series, air_density:pd.Series):
    wind_speed_ms_100m = wind_speed / 3.6
    wind_power_density = ( 0.5 * air_density * (wind_speed_ms_100m ** 3) )
    return wind_power_density


# --- spatial aggregation helper functions ---

def _haversine_distance(lat1: np.floating, lon1: np.floating, lat2: float, lon2: float) -> float:
    """
    Compute the Haversine distance (in km) between two lat/lon points.
    """
    rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in km
    return r * c

def _strip_suffix(col_name: str, suffixes: list[str]) -> str:
    """
    Remove the location suffix (e.g. '_F1') from the end of a column name.
    If no suffix is matched, return the original column name.
    """
    for sfx in suffixes:
        if col_name.endswith(sfx):
            return col_name.replace(sfx, "")
    return col_name

def _weighted_average(sub_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Compute a weighted average across columns in sub_df using `weights` (suffix -> weight).
    Falls back to simple mean if the sum of weights is 0.
    """
    sum_w = sum(weights.values())
    if sum_w == 0:
        # Fallback to mean across columns
        return sub_df.mean(axis=1)
    # Weighted sum
    weighted_sum = pd.Series(0.0, index=sub_df.index)
    for sfx in sub_df.columns:
        weighted_sum += sub_df[sfx] * weights[sfx]
    return weighted_sum / sum_w

def _build_base_feature_map(
        df: pd.DataFrame, suffixes: list[str]
) -> dict[str, list[tuple[str, str]]]:
    """
    Build a mapping from base-feature-name -> list of (suffix, col_name).
    E.g. "TMP" -> [("_F1", "TMP_F1"), ("_F2", "TMP_F2"), ... ]
    """
    all_cols = df.columns.tolist()
    base_feature_map = {}
    for sfx in suffixes:
        # Columns that end with this suffix
        cols_with_suffix = [c for c in all_cols if c.endswith(sfx)]
        for col_name in cols_with_suffix:
            base_feat = _strip_suffix(col_name, [sfx])
            if base_feat not in base_feature_map:
                base_feature_map[base_feat] = []
            base_feature_map[base_feat].append((sfx, col_name))
    return base_feature_map


# --- weather feature engineering ---

class WeatherBasedFE:

    def __init__(self, config:dict, verbose:bool):
        self.config = copy.deepcopy(config)
        self.verbose = verbose

        # get list of locations names (windfarms, solarfarms, citites etc)
        self.loc_names = self.config.get("locations", [])
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")
        # get list of dict() for locations. Each dict has latitude, longitude, suffix, etc.
        self.locations:list[dict] = [loc for loc in all_locations if loc['name'] in self.loc_names]
        if len(self.locations) == 0:
            raise ValueError("No locations configured.")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
            Given dataframe with raw features 'df', preprocess the dataframe based on
            config() (engineer features, drop features, add lags, aggregate based on suffix (location)
            and return resulting dataframe
        '''

        # Process each location separately
        processed_dfs = []
        for loc in self.locations:
            loc_df = self._preprocess_location(df.copy(), loc['suffix'])
            processed_dfs.append(loc_df)

        # Combine horizontally: each location df with engineered features
        combined_df = pd.concat(processed_dfs, axis=1)

        # Apply spatial aggregation if configured
        if (len(self.locations) > 1) and (self.config["spatial_agg_method"] != "None"):
            combined_df = self._apply_spatial_aggregation(combined_df)
        else:
            if self.verbose:
                logger.info("No spatial aggregation method specified.")

        if self.verbose:
            logger.info(f"Preprocessing result Shapes {df.shape} -> {combined_df.shape}"
                  f" Start {df.index[0]} -> {combined_df.index[0]}"
                  f" End {df.index[-3]} -> {combined_df.index[-1]}")

        expected_range = pd.date_range(
            start=combined_df.index.min(), end=combined_df.index.max(), freq='h'
        )

        if not combined_df.index.equals(expected_range):
            raise ValueError("combined_df must be continuous with hourly frequency.")

        return combined_df

    def _preprocess_location(self, df:pd.DataFrame, location_suffix:str) -> pd.DataFrame:
        raise NotImplementedError("Preprocessing not implemented in base class.")

    def _apply_spatial_aggregation(self, df:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Aggregation is not implemented in base class.")

    @staticmethod
    def selector_for_optuna(trial: optuna.Trial, fixed:dict)->dict:
        raise NotImplementedError("Selector for optuna is not implemented in base class.")


class WeatherWindPowerFE(WeatherBasedFE):

    name = "Wind Power"

    options = [
        "compute_air_density",
        "compute_air_density_moiust_air_correction",
        "encode_wind_direction",
        "compute_wind_shear",
        "turbulence_window",
        "compute_wind_ramp",
        "gust_factor",
        "dew_point_temperature",
        "vapor_pressure",
        "lags_choice",
        "precip_lags_choice",
        "drop_raw_main_features",
        "drop_raw_wind_features"
    ]

    def __init__(self, config: dict, verbose:bool):
        """
        Initialize with a configuration dictionary.
        """
        super().__init__(config, verbose)

    def _preprocess_location(self, df: pd.DataFrame, location_suffix: str) -> pd.DataFrame:
        """
        Preprocess meteorological features for a single location.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with raw meteorological data.
        location_suffix : str
            The suffix identifying which location columns to extract.

        Returns
        -------
        pd.DataFrame
            A DataFrame with engineered features for this location.
        """

        for key in self.options:
            if not key in self.config.keys():
                raise ValueError(f"Key {key} not found in config.")

        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()

        # -- Extract key meteorological columns
        temp_col = f"temperature_2m{location_suffix}"  # Celsius
        press_col = f"surface_pressure{location_suffix}"  # hPa
        humid_col = f"relative_humidity_2m{location_suffix}"  # %
        precip_col = f"precipitation{location_suffix}"  # mm
        cloud_col = f"cloud_cover{location_suffix}"  # %
        main_featurs = [temp_col, press_col, precip_col, cloud_col]

        wind_speed_10m_col = f"wind_speed_10m{location_suffix}"  # in km/h
        wind_speed_100m_col = f"wind_speed_100m{location_suffix}"  # in km/h
        wind_dir_100m_col = f"wind_direction_100m{location_suffix}"  # degrees (0 to 360)
        wind_gust_10m_col = f"wind_gusts_10m{location_suffix}"  # in km/h
        wind_features = [temp_col, press_col, precip_col, cloud_col]

        # 2. Compute Air Density (dry) & Wind Power Density
        if self.config["compute_air_density"]:
            loc_df["air_density" + location_suffix] = compute_air_density(loc_df[press_col],loc_df[temp_col])
            loc_df["wind_power_density" + location_suffix] = compute_wind_power_density(
                loc_df[wind_speed_100m_col],loc_df["air_density" + location_suffix]
            )
        # 2a. Compute Air Density With Moist Air Correction
        if self.config["compute_air_density_moiust_air_correction"]:
            loc_df["air_density_moist" + location_suffix] = compute_air_density_moist(
                loc_df[temp_col], loc_df[press_col], loc_df[humid_col]
            )
        # 3. Encode Wind Direction (Cyclic)
        if self.config["encode_wind_direction"]:
            loc_df["wind_dir_sin" + location_suffix] = np.sin(np.deg2rad(np.asarray(loc_df[wind_dir_100m_col])))
            loc_df["wind_dir_cos" + location_suffix] = np.cos(np.deg2rad(np.asarray(loc_df[wind_dir_100m_col])))
            loc_df.drop(columns=[wind_dir_100m_col], inplace=True)
        # 4. Compute Wind Shear
        if self.config["compute_wind_shear"]:
            loc_df["wind_shear" + location_suffix] = compute_wind_sheer(
                loc_df[wind_speed_100m_col], loc_df[wind_speed_10m_col]
            )
        # 5. Turbulence Intensity
        if self.config["turbulence_window"] > 0:
            loc_df["turbulence_intensity" + location_suffix] = compute_turbulence_intensity(
                loc_df[wind_speed_100m_col],
                self.config["turbulence_window"]
            )
        # 6. Wind Ramp
        if self.config["compute_wind_ramp"]:
            # Simple difference in wind speed from one timestep to the next
            loc_df["wind_ramp" + location_suffix] = loc_df[wind_speed_100m_col].diff()
        # 8. Gust Factor
        if self.config["gust_factor"]:
            loc_df["gust_factor" + location_suffix] = compute_gust_factor(
                loc_df[wind_speed_10m_col], loc_df[wind_gust_10m_col]
            )
        # 9. Dew Point Temperature
        if self.config["dew_point_temperature"]:
            loc_df["dew_point_temperature" + location_suffix] = compute_dew_point_temperature(
                loc_df[temp_col], loc_df[humid_col]
            )
        # 10. Vapor Pressure (Magnus formula in hPa)
        if self.config["vapor_pressure"]:
            loc_df["vapor_pressure" + location_suffix] = compute_vapor_pressure(
                loc_df[temp_col], loc_df[humid_col]
            )
        # 7. Lags
        if self.config["lags_choice"] != "none":
            if not self.config["lags_choice"] in ["small", "large"]:
                raise ValueError("lags_choice must be 'small' or 'large'")
            if self.config["lags_choice"] == "low": lags = [1, 6]
            else: lags = [1, 6, 12]
            for lag in lags:
                loc_df[f"wind_speed_lag_{lag}{location_suffix}"] = loc_df[wind_speed_100m_col].shift(lag)

        # 7a. Precipitation Lags
        if self.config["precip_lags_choice"] != "none":
            if not self.config["precip_lags_choice"] in ["small", "large"]:
                raise ValueError("precip_lags_choice must be 'small' or 'large'")
            if self.config["precip_lags_choice"] == "small": precip_lags = [1, 6]
            else: precip_lags = [1, 6, 12, 24]
            # Similar logic to wind_speed lags
            for lag in precip_lags:
                loc_df[f"precip_lag_{lag}{location_suffix}"] = loc_df[precip_col].shift(lag)
        # 8. Drop raw features if requested
        if self.config["drop_raw_main_features"]:
            loc_df.drop(columns=main_featurs, inplace=True, errors="ignore")
        if self.config["drop_raw_wind_features"]:
            loc_df.drop(
                columns=[feat for feat in wind_features if feat != wind_speed_100m_col],
                inplace=True, errors="ignore"
            )

        # if self.config.get("drop_raw_features", False):
        #     features_to_drop = self.config.get("features_to_drop", [])
        #     drop_cols = [f"{feat}{location_suffix}" for feat in features_to_drop if f"{feat}{location_suffix}" in loc_df.columns]
        #     loc_df.drop(columns=drop_cols, inplace=True, errors="ignore")



        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features across multiple wind farms using a specified spatial method.
        (Refactored version preserving original logic.)
        """
        if len(self.locations) <= 1:
            raise ValueError("Cannot apply spatial aggregation on one location")

        # Extract config
        method: str = self.config["spatial_agg_method"]

        # Collect suffixes from self.locations
        suffixes = [loc["suffix"] for loc in self.locations]

        # Build the base feature map
        base_feature_map = _build_base_feature_map(df, suffixes)

        # Build location metadata for wind
        loc_meta = {}
        for loc in self.locations:
            sfx = loc['suffix']
            loc_meta[sfx] = {
                "lat"   : loc['lat'],
                "lon"   : loc['lon'],
                "capacity": loc['capacity'],
                "n_turbines": loc['n_turbines'],
                # Additional wind-specific fields:
                "elevation": loc.get('elevation', 0.0),
                "z0": loc.get('z0', 0.0),
                "terrain_category": loc.get('terrain_category', 'I'),
            }

        # Compute centroid lat/lon for distance-based weighting
        mean_lat = np.mean([loc_meta[sfx]["lat"] for sfx in suffixes])
        mean_lon = np.mean([loc_meta[sfx]["lon"] for sfx in suffixes])

        # Create the aggregated DataFrame
        aggregated_df = pd.DataFrame(index=df.index)

        # Iterate over each base feature
        for base_feat, columns_with_suffixes in base_feature_map.items():
            # Build a sub-DataFrame with the time series for each suffix
            sub_data = {}
            for (sfx, c_name) in columns_with_suffixes:
                sub_data[sfx] = df[c_name]
            sub_df = pd.DataFrame(sub_data, index=df.index)

            # Apply the method
            if method == "mean":
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

            elif method == "max":
                aggregated_df[base_feat + "_agg"] = sub_df.max(axis=1)

            elif method == "idw":
                # 1 / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = 1.0 / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "capacity":
                # Weighted by capacity
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["capacity"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "n_turbines":
                # Weighted by number of turbines
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["n_turbines"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_capacity":
                # capacity / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    cap_i = loc_meta[sfx]["capacity"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = cap_i / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_n_turbines":
                # n_turbines / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    tur_i = loc_meta[sfx]["n_turbines"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = tur_i / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            else:
                raise KeyError(f"Spatial aggregation method {method} not recognized")

        return aggregated_df

    @staticmethod
    def selector_for_optuna(trial: optuna.Trial, fixed:dict) -> dict:
        # Boolean features
        compute_air_density = trial.suggest_categorical("compute_air_density", [True, False])
        compute_wind_power_density = trial.suggest_categorical("compute_wind_power_density", [True, False])
        encode_wind_direction = trial.suggest_categorical("encode_wind_direction", [True, False])
        compute_wind_shear = trial.suggest_categorical("compute_wind_shear", [True, False])
        compute_wind_ramp = trial.suggest_categorical("compute_wind_ramp", [True, False])

        # Additional booleans for new features
        compute_air_density_moiust_air_correction = trial.suggest_categorical(
            "compute_air_density_moiust_air_correction", [True, False]
        )
        dew_point_temperature = trial.suggest_categorical("dew_point_temperature", [True, False])
        vapor_pressure = trial.suggest_categorical("vapor_pressure", [True, False])
        gust_factor = trial.suggest_categorical("gust_factor", [True, False])

        # Turbulence intensity window size if enabled
        turbulence_window = trial.suggest_categorical("turbulence_window", [0, 2, 6])

        # Lags can be a list, decide length and values
        lags_choice = trial.suggest_categorical("lags_choice", ["none", "small", "large"])

        # Precipitation lags
        precip_lags_choice = trial.suggest_categorical("precip_lags_choice", ["none", "small", "large"])

        # Raw feature dropping
        drop_raw_main_features = trial.suggest_categorical("drop_raw_main_features", [True, False])
        drop_raw_wind_features = trial.suggest_categorical("drop_raw_wind_features", [True, False])


        if not "spatial_agg_method" in fixed:
            spatial_agg_method = trial.suggest_categorical(
                "spatial_agg_method", [
                    "None",
                    "mean",
                    "max",
                    "idw",
                    "capacity",
                    "distance_capacity",
                    "distance_n_turbines"
                ]
            )
        else:
            spatial_agg_method = fixed["spatial_agg_method"]


        # Construct and return the config dictionary
        config = {
            "compute_air_density": compute_air_density,
            "compute_wind_power_density": compute_wind_power_density,
            "encode_wind_direction": encode_wind_direction,
            "compute_wind_shear": compute_wind_shear,
            "compute_wind_ramp": compute_wind_ramp,
            "compute_air_density_moiust_air_correction": compute_air_density_moiust_air_correction,
            "dew_point_temperature": dew_point_temperature,
            "vapor_pressure": vapor_pressure,
            "gust_factor": gust_factor,
            "turbulence_window": turbulence_window,  # either False or dict
            "lags_choice": lags_choice,  # wind speed lags
            "precip_lags_choice": precip_lags_choice,  # precipitation lags
            "drop_raw_main_features": drop_raw_main_features,
            "drop_raw_wind_features": drop_raw_wind_features,
            "spatial_agg_method": spatial_agg_method,
        }


        return config


class WeatherSolarPowerFE(WeatherBasedFE):

    options = [
        "compute_cloud_cover_fraction",
        "compute_clear_sky_fraction",
        "compute_air_density",
        "compute_air_density_moist",
        "compute_direct_ratio",
        "compute_diffuse_ratio",
        "compute_dni_ratio",
        "compute_global_tilted_ratio",
        "use_solar_geometry",
        "dew_point_temperature",
        "vapor_pressure",
        "precip_lags_option",
        "cloud_lags_option",
        "shortwave_lags_option",
        "drop_raw_solar_features",
        "drop_raw_features"
    ]

    def __init__(self, config: dict, verbose: bool):
        """
        Initialize with a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration for the feature engineering steps.
        verbose : bool
            Whether to print intermediate steps and debugging info.
        """
        super().__init__(config, verbose)

    def _preprocess_location(self, df: pd.DataFrame, location_suffix: str) -> pd.DataFrame:
        """
        Preprocess meteorological features for a single location.
        Incorporates:
          - Solar-specific features (ratios of direct/diffuse, etc.)
          - Cloud cover features
          - Air density calculations (dry or moist)
          - Solar geometry (elevation/azimuth)
          - Precipitation/Cloud lags
          - Dew point temperature, vapor pressure
          - and more, based on config toggles.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with raw meteorological data.
            Expected to have a DatetimeIndex in UTC.
        location_suffix : str
            The suffix identifying which location columns to extract.

        Returns
        -------
        pd.DataFrame
            A DataFrame with engineered solar-specific features for this location.
        """

        for key in self.options:
            if not key in self.config.keys():
                raise ValueError(f"Key {key} not found in config.")


        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()

        # 2. Identify key meteorological columns for solar
        temp_col = f"temperature_2m{location_suffix}"        # °C
        press_col = f"surface_pressure{location_suffix}"     # hPa
        humid_col = f"relative_humidity_2m{location_suffix}" # %
        precip_col = f"precipitation{location_suffix}"       # mm
        cloud_col = f"cloud_cover{location_suffix}"          # %
        main_feat = [temp_col, press_col, humid_col, precip_col, cloud_col]

        shortwave_col = f"shortwave_radiation{location_suffix}"        # W/m^2
        direct_col = f"direct_radiation{location_suffix}"              # W/m^2
        diffuse_col = f"diffuse_radiation{location_suffix}"            # W/m^2
        dni_col = f"direct_normal_irradiance{location_suffix}"         # W/m^2
        global_tilted_col = f"global_tilted_irradiance{location_suffix}"   # W/m^2
        terrestrial_col = f"terrestrial_radiation{location_suffix}"         # W/m^2
        rad_features = [shortwave_col, direct_col, diffuse_col, dni_col, global_tilted_col, terrestrial_col]

        # 3. Cloud Cover Features
        if self.config.get("compute_cloud_cover_fraction", False):
            loc_df[f"cloud_cover_fraction{location_suffix}"] = loc_df[cloud_col] / 100.0
        # Add a "clear_sky_fraction"
        if self.config.get("compute_clear_sky_fraction", False):
            loc_df[f"clear_sky_fraction{location_suffix}"] = 1.0 - (loc_df[cloud_col] / 100.0)

        # 4. Air Density (Dry and/or Moist)
        if self.config["compute_air_density"]:
            loc_df[f"air_density{location_suffix}"] = compute_air_density(
                loc_df[press_col], loc_df[temp_col]
            )

        # Moist air density correction (optional)
        if self.config["compute_air_density_moist"]:
            loc_df[f"air_density_moist{location_suffix}"] = compute_air_density_moist(
                loc_df[temp_col], loc_df[press_col], loc_df[humid_col]
            )

        # 5. Solar-Specific Ratios
        sw = loc_df[shortwave_col].replace(0, np.inf)
        if self.config["compute_direct_ratio"]:
            loc_df[f"direct_ratio{location_suffix}"] = loc_df[direct_col] / sw
        if self.config["compute_diffuse_ratio"]:
            loc_df[f"diffuse_ratio{location_suffix}"] = loc_df[diffuse_col] / sw
        if self.config["compute_dni_ratio"]:
            loc_df[f"dni_ratio{location_suffix}"] = loc_df[dni_col] / sw
        if self.config["compute_global_tilted_ratio"]:
            loc_df[f"global_tilted_ratio{location_suffix}"] = loc_df[global_tilted_col] / sw

        # 6. Solar Geometry (Elevation, Azimuth, etc.)
        if not self.config["use_solar_geometry"]:
            loc_df.drop(columns=[
                f"solar_elevation_deg{location_suffix}",
                f"solar_azimuth_deg{location_suffix}"],
                inplace=True, errors="ignore"
            )

        # 7. Dew Point Temperature (Magnus formula)
        if self.config["dew_point_temperature"]:
            loc_df[f"dew_point_temperature{location_suffix}"] = compute_dew_point_temperature(
                loc_df[temp_col], loc_df[humid_col]
            )


        # 8. Vapor Pressure (Magnus formula in hPa)
        if self.config["vapor_pressure"]:
            loc_df[f"vapor_pressure{location_suffix}"] = compute_vapor_pressure(
                loc_df[temp_col], loc_df[humid_col]
            )

        # 9. Lags (Example: precipitation, cloud, shortwave, etc.)
        if self.config["precip_lags_option"] != "none":
            if self.config["precip_lags_option"] == "small": precip_lags = [1, 6]
            elif self.config["precip_lags_option"] == "large": precip_lags = [1, 6, 12, 24]
            else: raise KeyError(f"{self.config['precip_lags_option']} not recognized")
            for lag in precip_lags:
                loc_df[f"precip_lag_{lag}{location_suffix}"] = loc_df[precip_col].shift(lag)

        if self.config["cloud_lags_option"] != "none":
            if self.config["cloud_lags_option"] == "small": cloud_lags = [1, 3]
            elif self.config["cloud_lags_option"] == "medium": cloud_lags = [1, 3, 6]
            elif self.config["cloud_lags_option"] == "large": cloud_lags = [1, 3, 6, 12]
            else: raise KeyError(f"{self.config['cloud_lags_option']} not recognized")
            for lag in cloud_lags:
                loc_df[f"cloud_lag_{lag}{location_suffix}"] = loc_df[cloud_col].shift(lag)

        if self.config["shortwave_lags_option"] != "none":
            if self.config["shortwave_lags_option"] == "small": shortwave_lags = [1, 3]
            elif self.config["shortwave_lags_option"] == "medium": shortwave_lags = [1, 3, 6]
            elif self.config["shortwave_lags_option"] == "large": shortwave_lags = [1, 3, 6, 12]
            else: raise KeyError(f"{self.config['shortwave_lags_option']} not recognized")
            for lag in shortwave_lags:
                loc_df[f"shortwave_lag_{lag}{location_suffix}"] = loc_df[shortwave_col].shift(lag)

        # 10. Drop raw features if requested
        if self.config.get("drop_raw_solar_features", False):
            loc_df.drop(
                columns=rad_features,
                inplace=True,
                errors="ignore"
            )
        if self.config.get("drop_raw_features", False):
            loc_df.drop(
                columns=main_feat,
                inplace=True,
                errors="ignore"
            )

        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features across multiple solar farms using a specified spatial method.
        (Refactored version preserving original logic.)
        """
        if len(self.locations) <= 1:
            raise ValueError("Cannot apply spatial aggregation on a single location")

        method: str = self.config["spatial_agg_method"]
        suffixes = [loc["suffix"] for loc in self.locations]

        # Build the base feature map
        base_feature_map = _build_base_feature_map(df, suffixes)

        # Build location metadata
        loc_meta = {}
        for loc in self.locations:
            sfx = loc["suffix"]
            loc_meta[sfx] = {
                "lat": loc["lat"],
                "lon": loc["lon"],
                "capacity": loc.get("capacity", 1.0),
                "n_panels": loc.get("n_panels", 1),
                "elevation": loc.get("elevation", 0.0),
            }

        # Centroid for distance-based
        mean_lat = np.mean([loc_meta[sfx]["lat"] for sfx in suffixes])
        mean_lon = np.mean([loc_meta[sfx]["lon"] for sfx in suffixes])

        aggregated_df = pd.DataFrame(index=df.index)

        for base_feat, columns_with_suffixes in base_feature_map.items():
            sub_data = {}
            for (sfx, c_name) in columns_with_suffixes:
                sub_data[sfx] = df[c_name]
            sub_df = pd.DataFrame(sub_data, index=df.index)

            if method == "mean":
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

            elif method == "max":
                aggregated_df[base_feat + "_agg"] = sub_df.max(axis=1)

            elif method == "idw":
                # 1 / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = 1.0 / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "capacity":
                # Weighted by capacity
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["capacity"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "n_panels":
                # Weighted by number of panels
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["n_panels"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_capacity":
                # capacity / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    cap_i = loc_meta[sfx]["capacity"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = cap_i / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_n_panels":
                # n_panels / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    n_pan = loc_meta[sfx]["n_panels"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = n_pan / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            else:
                raise KeyError(f"Aggregation method '{method}' not supported")

        return aggregated_df

    @staticmethod
    def selector_for_optuna(trial: optuna.Trial, fixed: dict) -> dict:
        """
        Example method for hyperparameter selection with Optuna.

        Parameters
        ----------
        trial : optuna.Trial
            The trial object for hyperparameter search.
        fixed : dict
            A dictionary of fixed parameters (if any).

        Returns
        -------
        dict
            A configuration dictionary to be used by WeatherSolarPowerFE.
        """

        # ----- New booleans for cloud cover features -----
        compute_cloud_cover_fraction = trial.suggest_categorical(
            "compute_cloud_cover_fraction", [True, False]
        )
        compute_clear_sky_fraction = trial.suggest_categorical(
            "compute_clear_sky_fraction", [True, False]
        )

        # ----- New booleans for solar geometry -----
        use_solar_geometry = trial.suggest_categorical(
            "use_solar_geometry", [True, False]
        )

        # ----- New booleans for air density (dry vs. moist) -----
        compute_air_density = trial.suggest_categorical(
            "compute_air_density", [True, False]
        )
        compute_air_density_moist = trial.suggest_categorical(
            "compute_air_density_moist", [True, False]
        )

        # ----- Existing solar-specific booleans -----
        compute_direct_ratio = trial.suggest_categorical(
            "compute_direct_ratio", [True, False]
        )
        compute_diffuse_ratio = trial.suggest_categorical(
            "compute_diffuse_ratio", [True, False]
        )
        compute_dni_ratio = trial.suggest_categorical(
            "compute_dni_ratio", [True, False]
        )
        compute_global_tilted_ratio = trial.suggest_categorical(
            "compute_global_tilted_ratio", [True, False]
        )
        # compute_terrestrial_ratio = trial.suggest_categorical(
        #     "compute_terrestrial_ratio", [False, True]  # e.g., maybe not as common
        # )

        # ----- Additional booleans for dew point and vapor pressure -----
        dew_point_temperature = trial.suggest_categorical(
            "dew_point_temperature", [True, False]
        )
        vapor_pressure = trial.suggest_categorical(
            "vapor_pressure", [True, False]
        )

        # ----- Precipitation lags (existing logic) -----
        precip_lags_option = trial.suggest_categorical(
            "precip_lags_option", ["none", "small", "large"]
        )


        # ----- New: Cloud cover lags -----
        cloud_lags_option = trial.suggest_categorical(
            "cloud_lags_option", ["none", "small", "medium", "large"]
        )

        # ----- New: Shortwave lags -----
        shortwave_lags_option = trial.suggest_categorical(
            "shortwave_lags_option", ["none", "small", "medium", "large"]
        )

        # ----- Raw feature dropping -----
        drop_raw_solar_features = trial.suggest_categorical(
            "drop_raw_solar_features", [True, False]
        )
        drop_raw_features = trial.suggest_categorical(
            "drop_raw_features", [True, False]
        )

        # ----- Spatial aggregation (unchanged from prior logic) -----
        if "spatial_agg_method" not in fixed:
            spatial_agg_method = trial.suggest_categorical(
                "spatial_agg_method",
                [
                    "None",
                    "mean",
                    "max",
                    "idw",
                    "capacity",
                    "n_panels",
                    "distance_capacity",
                    "distance_n_panels",
                ],
            )
        else:
            spatial_agg_method = fixed["spatial_agg_method"]

        # ----- Construct and return the config dictionary -----
        config = {
            "locations": fixed.get("locations", []),

            # New features
            "compute_cloud_cover_fraction": compute_cloud_cover_fraction,
            "compute_clear_sky_fraction": compute_clear_sky_fraction,
            "use_solar_geometry": use_solar_geometry,
            "compute_air_density": compute_air_density,
            "compute_air_density_moist": compute_air_density_moist,

            # Existing solar features
            "compute_direct_ratio": compute_direct_ratio,
            "compute_diffuse_ratio": compute_diffuse_ratio,
            "compute_dni_ratio": compute_dni_ratio,
            "compute_global_tilted_ratio": compute_global_tilted_ratio,
            # "compute_terrestrial_ratio": compute_terrestrial_ratio,

            # Dew point & vapor pressure
            "dew_point_temperature": dew_point_temperature,
            "vapor_pressure": vapor_pressure,

            # Lag configurations
            "precip_lags_option": precip_lags_option,
            "cloud_lags_option": cloud_lags_option,
            "shortwave_lags_option": shortwave_lags_option,

            # Dropping raw features
            "drop_raw_solar_features": drop_raw_solar_features,
            "drop_raw_features": drop_raw_features,

            # Spatial aggregation
            "spatial_agg_method": spatial_agg_method
        }

        return config


class WeatherLoadFE(WeatherBasedFE):

    options = [
        "compute_heating_degree_hours",
        "compute_cooling_degree_hours",
        "compute_dew_point_spread",
        "compute_temp_gradient",
        "compute_wind_chill",
        "compute_humidex",
        "compute_pressure_trend",
        "compute_air_density",
        "compute_rain_indicator",
        "compute_wind_speed_gradient",
        "compute_wind_components",
        "compute_wind_power_density",
        "compute_cloud_cover_fraction",
        "compute_effective_solar",
        "temp_lags_option",
        "precip_lags_option",
        "cloud_lags_option",
        "rolling_temp_option",
        "drop_basic_meteo_features",
        "drop_wind_meteo_features",
        "drop_rad_meteo_features"
    ]

    def __init__(self, config: dict, verbose: bool):
        """
        Initialize with a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration for the feature engineering steps.
        verbose : bool
            Whether to print intermediate steps and debugging info.
        """
        super().__init__(config, verbose)


    def _preprocess_location( self, df: pd.DataFrame, location_suffix: str ) -> pd.DataFrame:
        """
        Preprocess meteorological features for a single location.
        Incorporates:
          - Temperature-based features (degree days, dew point spread, gradient, etc.)
          - Wind chill, humidex
          - Wind-based features (wind components, wind speed gradient)
          - Solar-related features (effective solar, if desired)
          - Lags, rolling/aggregated features
          - Optional dropping of raw features
          - Excludes time features (hour-of-day, etc.), which will be added elsewhere.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with raw meteorological data (hourly).
            About 1 year of hourly data, used to forecast up to 187 hours ahead iteratively.
            Expected to have a DatetimeIndex in UTC.
        location_suffix : str
            The suffix identifying which location columns to extract.

        Returns
        -------
        pd.DataFrame
            A DataFrame with engineered load-specific features for this location.
        """

        for key in self.options:
            if not key in self.config.keys():
                raise ValueError(f"Key {key} not found in config.")

        # 1. Select relevant columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()

        # 2. Identify key meteorological columns (keeping track of units)
        temp_col       = f"temperature_2m{location_suffix}"         # [°C]
        press_col      = f"surface_pressure{location_suffix}"        # [hPa]
        humid_col      = f"relative_humidity_2m{location_suffix}"    # [%]
        precip_col     = f"precipitation{location_suffix}"           # [mm]
        cloud_col      = f"cloud_cover{location_suffix}"             # [%]
        base_feat = [temp_col, press_col, precip_col, cloud_col]

        wind_speed_10m_col  = f"wind_speed_10m{location_suffix}"     # [km/h]
        wind_speed_100m_col = f"wind_speed_100m{location_suffix}"    # [km/h]
        wind_dir_10m_col    = f"wind_direction_10m{location_suffix}" # [degrees]
        wind_dir_100m_col   = f"wind_direction_100m{location_suffix}"# [degrees]
        wind_gust_10m_col   = f"wind_gusts_10m{location_suffix}"     # [km/h]
        wind_feat = [wind_speed_10m_col, wind_dir_10m_col, wind_gust_10m_col, wind_speed_100m_col, wind_dir_100m_col]

        shortwave_col     = f"shortwave_radiation{location_suffix}"       # [W/m^2]
        direct_col        = f"direct_radiation{location_suffix}"          # [W/m^2]
        diffuse_col       = f"diffuse_radiation{location_suffix}"         # [W/m^2]
        dni_col           = f"direct_normal_irradiance{location_suffix}"  # [W/m^2]
        global_tilted_col = f"global_tilted_irradiance{location_suffix}"  # [W/m^2]
        terrestrial_col   = f"terrestrial_radiation{location_suffix}"     # [W/m^2]
        rad_feat = [shortwave_col, direct_col, diffuse_col, dni_col, global_tilted_col, terrestrial_col]

        # 3. Temperature-based features (HDH, CDH, dew point spread, gradient)
        if self.config["compute_heating_degree_hours"]:
            hdh_threshold = self.config.get("hdh_threshold", 18)
            loc_df[f"HDH{location_suffix}"] = np.clip(hdh_threshold - loc_df[temp_col], 0, None)

        if self.config["compute_cooling_degree_hours"]:
            cdh_threshold = self.config.get("cdh_threshold", 22)
            loc_df[f"CDH{location_suffix}"] = np.clip(loc_df[temp_col] - cdh_threshold, 0, None)

        # 3.2 Dew Point Spread
        if self.config["compute_dew_point_spread"]:
            loc_df["dew_point_temperature" + location_suffix] = compute_dew_point_temperature(
                loc_df[temp_col], loc_df[humid_col]
            )
            loc_df[f"dew_point_spread{location_suffix}"] = \
                loc_df[temp_col] - loc_df["dew_point_temperature" + location_suffix]

        # 3.3 Temperature Gradient (one-hour difference)
        if self.config["compute_temp_gradient"]:
            loc_df[f"temp_gradient{location_suffix}"] = loc_df[temp_col] - loc_df[temp_col].shift(1)

        # 4. Wind chill, Humidex (already in your original code, but shown here)
        if self.config["compute_wind_chill"]:
            loc_df[f"wind_chill{location_suffix}"] = compute_wind_chill(
                loc_df[temp_col], loc_df[wind_speed_10m_col]
            )

        if self.config["compute_humidex"]:
            loc_df[f"humidex{location_suffix}"] = compute_humidex(
                loc_df[temp_col], loc_df[humid_col]
            )

        # 5. Pressure-based features (pressure trend, air density if needed): 5.1 Pressure Trend
        if self.config["compute_pressure_trend"]:
            loc_df[f"pressure_trend{location_suffix}"] = loc_df[press_col] - loc_df[press_col].shift(1)

        # 5.2 Air density (from original code)
        if self.config["compute_air_density"]:
            loc_df[f"air_density{location_suffix}"] = compute_air_density(
                loc_df[press_col], loc_df[temp_col]
            )

        # 6. Precipitation features (binary indicator, existing rolling sums, etc.)
        if self.config["compute_rain_indicator"]:
            loc_df[f"rain_indicator{location_suffix}"] = (loc_df[precip_col] > 0).astype(int)

        # 7. Wind-based features (speed gradient, wind components, wind power density)
        if self.config["compute_wind_speed_gradient"]:
            loc_df[f"wind_speed_gradient{location_suffix}"] = (
                    loc_df[wind_speed_10m_col] - loc_df[wind_speed_10m_col].shift(1)
            )

        # 7.2 Wind Components (convert from speed/direction to U/V)
        if self.config["compute_wind_components"]:
            wind_speed_ms_10m = loc_df[wind_speed_10m_col] / 3.6
            loc_df[f"wind_u{location_suffix}"] = (
                    wind_speed_ms_10m * np.cos(np.deg2rad(loc_df[wind_dir_10m_col]))
            )
            loc_df[f"wind_v{location_suffix}"] = (
                    wind_speed_ms_10m * np.sin(np.deg2rad(loc_df[wind_dir_10m_col]))
            )

        # 7.3 Wind Power Density (re-using air_density if computed)
        if self.config["compute_wind_power_density"]:
            air_density = compute_air_density(loc_df[press_col], loc_df[temp_col])
            loc_df[f"wind_power_density_100m{location_suffix}"] = compute_wind_power_density(
                loc_df[wind_speed_100m_col], air_density
            )

        # 8. Radiation (solar) features
        # 8.1 Convert cloud cover % to fraction if desired
        if self.config["compute_cloud_cover_fraction"]:
            loc_df[f"cloud_cover_fraction{location_suffix}"] = loc_df[cloud_col] / 100.0

        # 8.2 Effective Solar Radiation: (1 - cloud_cover) * shortwave_radiation
        if self.config["compute_effective_solar"]:
            # If fraction not yet computed, do it here
            cloud_fraction = loc_df[cloud_col] / 100.0
            loc_df[f"effective_solar{location_suffix}"] = (
                    (1 - cloud_fraction) * loc_df[shortwave_col]
            )

        # 9. Lags (temp, cloud, etc.) and rolling features (kept from original)
        if self.config["temp_lags_option"] != "none":
            if self.config["temp_lags_option"] == "small": temp_lags = [1, 3]
            elif self.config["temp_lags_option"] == "medium": temp_lags = [1, 3, 6]
            else: raise KeyError(f"temp_lags_option = {self.config['temp_lags_option']} is not supported")
            for lag in temp_lags:
                loc_df[f"temp_lag_{lag}{location_suffix}"] = loc_df[temp_col].shift(lag)

        if self.config["precip_lags_option"] != "none":
            if self.config["precip_lags_option"] == "small": precip_lags = [1, 3]
            elif self.config["precip_lags_option"] == "medium": precip_lags = [1, 3, 6]
            else: raise KeyError(f"precip_lags_option = {self.config['precip_lags_option']} is not supported")
            for lag in precip_lags:
                loc_df[f"precip_lags_lag_{lag}{location_suffix}"] = loc_df[precip_col].shift(lag)

        if self.config["cloud_lags_option"] != "none":
            if self.config["cloud_lags_option"] == "small": cloud_lags = [1, 3]
            elif self.config["cloud_lags_option"] == "medium": cloud_lags = [1, 3, 6]
            else: raise KeyError(f"cloud_lags_option = {self.config['cloud_lags_option']} is not supported")
            for lag in cloud_lags:
                loc_df[f"cloud_lags_lag_{lag}{location_suffix}"] = loc_df[cloud_col].shift(lag)

        if self.config["rolling_temp_option"] != "none":
            if self.config["rolling_temp_option"] == "short": rolling_temp = [3, 6]
            elif self.config["rolling_temp_option"] == "long": rolling_temp = [6, 12, 24]
            else: raise KeyError(f"rolling_temp_option = {self.config['rolling_temp_option']} is not supported")
            for window in rolling_temp:
                loc_df[f"temp_roll_{window}{location_suffix}"] = (
                    loc_df[temp_col].rolling(window=window, min_periods=1).mean()
                )

        # 10. Drop basic meteo features if requested (avoid duplicating columns)
        if self.config["drop_basic_meteo_features"]:
            # Keep temperature, unless you explicitly want to drop it too
            basic_cols = [c for c in base_feat if c != temp_col]
            loc_df.drop(columns=[c for c in basic_cols if c in loc_df.columns],
                        inplace=True, errors="ignore")

        if self.config["drop_wind_meteo_features"]:
            loc_df.drop(columns=[c for c in wind_feat if c in loc_df.columns],
                        inplace=True, errors="ignore")

        if self.config["drop_rad_meteo_features"]:
            loc_df.drop(columns=[c for c in rad_feat if c in loc_df.columns],
                        inplace=True, errors="ignore")

        # Return DataFrame with added features
        return loc_df


    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features across multiple locations (cities) using a specified spatial method.
        (Refactored version preserving original logic.)
        """
        if len(self.locations) <= 1:
            raise ValueError("Cannot apply spatial aggregation on a single location")

        method: str = self.config["spatial_agg_method"]
        suffixes = [loc["suffix"] for loc in self.locations]

        # Build the base feature map
        base_feature_map = _build_base_feature_map(df, suffixes)

        # Prepare location metadata
        loc_meta = {}
        for loc in self.locations:
            sfx = loc["suffix"]
            loc_meta[sfx] = {
                "lat": loc["lat"],
                "lon": loc["lon"],
                # Distinct load-related fields:
                "population": loc.get("population", 1),
                "energy": loc.get("total_energy_consumption", 1.0),
            }

        mean_lat = np.mean([loc_meta[sfx]["lat"] for sfx in suffixes])
        mean_lon = np.mean([loc_meta[sfx]["lon"] for sfx in suffixes])

        aggregated_df = pd.DataFrame(index=df.index)

        for base_feat, columns_with_suffixes in base_feature_map.items():
            sub_data = {}
            for (sfx, c_name) in columns_with_suffixes:
                sub_data[sfx] = df[c_name]
            sub_df = pd.DataFrame(sub_data, index=df.index)

            if method == "mean":
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

            elif method == "max":
                aggregated_df[base_feat + "_agg"] = sub_df.max(axis=1)

            elif method == "idw":
                # 1 / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = 1.0 / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "population":
                # Weighted by population
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["population"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "energy":
                # Weighted by total energy consumption
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["energy"]
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_population":
                # population / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    pop_i = loc_meta[sfx]["population"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = pop_i / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            elif method == "distance_energy":
                # total_energy_consumption / distance^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    en_i = loc_meta[sfx]["energy"]
                    d_km = _haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = en_i / (d_km**2)
                aggregated_df[base_feat + "_agg"] = _weighted_average(sub_df, weights)

            else:
                raise KeyError(f"Aggregation method {method} not recognized")

        return aggregated_df

    @staticmethod
    def selector_for_optuna(trial: optuna.Trial, fixed: dict) -> dict:
        """
        Example method for hyperparameter selection with Optuna for load forecasting.

        Parameters
        ----------
        trial : optuna.Trial
            The trial object for hyperparameter search.
        fixed : dict
            A dictionary of fixed parameters (if any).

        Returns
        -------
        dict
            A configuration dictionary to be used by your load-forecasting Feature Engineering pipeline.
        """

        # 1. Temperature-based features
        compute_heating_degree_hours = trial.suggest_categorical(
            "compute_heating_degree_hours", [True, False]
        )
        compute_cooling_degree_hours = trial.suggest_categorical(
            "compute_cooling_degree_hours", [True, False]
        )
        compute_dew_point_spread = trial.suggest_categorical(
            "compute_dew_point_spread", [True, False]
        )
        compute_temp_gradient = trial.suggest_categorical(
            "compute_temp_gradient", [True, False]
        )

        # 2. Wind & pressure features
        compute_wind_chill = trial.suggest_categorical(
            "compute_wind_chill", [True, False]
        )
        compute_humidex = trial.suggest_categorical(
            "compute_humidex", [True, False]
        )
        compute_wind_components = trial.suggest_categorical(
            "compute_wind_components", [True, False]
        )
        compute_wind_power_density = trial.suggest_categorical(
            "compute_wind_power_density", [True, False]
        )
        compute_pressure_trend = trial.suggest_categorical(
            "compute_pressure_trend", [True, False]
        )
        compute_wind_speed_gradient = trial.suggest_categorical(
            "compute_wind_speed_gradient", [True, False]
        )

        # 3. Other meteorological features
        compute_air_density = trial.suggest_categorical(
            "compute_air_density", [True, False]
        )
        compute_rain_indicator = trial.suggest_categorical(
            "compute_rain_indicator", [True, False]
        )

        compute_cloud_cover_fraction = trial.suggest_categorical(
            "compute_cloud_cover_fraction", [True, False]
        )
        compute_effective_solar = trial.suggest_categorical(
            "compute_effective_solar", [True, False]
        )

        # 4. Lags for temperature, precipitation, cloud, etc.
        temp_lags_option = trial.suggest_categorical(
            "temp_lags_option", ["none", "small", "medium"]
        )

        precip_lags_option = trial.suggest_categorical(
            "precip_lags_option", ["none", "small", "medium"]
        )

        cloud_lags_option = trial.suggest_categorical(
            "cloud_lags_option", ["none", "small", "medium"]
        )

        # Optional rolling temperature windows
        rolling_temp_option = trial.suggest_categorical(
            "rolling_temp_option", ["none", "short", "long"]
        )

        # 5. Dropping raw or radiation features
        drop_basic_meteo_features = trial.suggest_categorical(
            "drop_basic_meteo_features", [True, False]
        )
        drop_wind_meteo_features = trial.suggest_categorical(
            "drop_wind_meteo_features", [True, False]
        )
        drop_rad_meteo_features = trial.suggest_categorical(
            "drop_rad_meteo_features", [True, False]
        )

        # 6. Spatial aggregation method (for multiple locations)
        if "spatial_agg_method" not in fixed:
            spatial_agg_method = trial.suggest_categorical(
                "spatial_agg_method",
                [
                    "None",   # If "None", skip or handle outside
                    "mean",
                    "max",
                    "idw",
                    "population",
                    "energy",
                    "distance_population",
                    "distance_energy",
                ],
            )
        else:
            spatial_agg_method = fixed["spatial_agg_method"]

        # 7. Construct and return the config dictionary
        config = {
            "locations": fixed.get("locations", []),

            # Temperature-based toggles
            "compute_heating_degree_hours": compute_heating_degree_hours,
            "compute_cooling_degree_hours": compute_cooling_degree_hours,
            "compute_dew_point_spread": compute_dew_point_spread,
            "compute_temp_gradient": compute_temp_gradient,

            # Wind & pressure toggles
            "compute_wind_chill": compute_wind_chill,
            "compute_humidex": compute_humidex,
            "compute_wind_components": compute_wind_components,
            "compute_wind_power_density": compute_wind_power_density,
            "compute_pressure_trend": compute_pressure_trend,
            "compute_wind_speed_gradient":compute_wind_speed_gradient,

            # Other meteo toggles
            "compute_air_density": compute_air_density,
            "compute_rain_indicator": compute_rain_indicator,
            "compute_cloud_cover_fraction": compute_cloud_cover_fraction,
            "compute_effective_solar": compute_effective_solar,

            # Lag settings
            "temp_lags_option": temp_lags_option,
            "precip_lags_option": precip_lags_option,
            "cloud_lags_option": cloud_lags_option,
            "rolling_temp_option": rolling_temp_option,

            # Dropping raw features
            "drop_basic_meteo_features": drop_basic_meteo_features,
            "drop_wind_meteo_features": drop_wind_meteo_features,
            "drop_rad_meteo_features":drop_rad_meteo_features,

            # Spatial aggregation
            "spatial_agg_method": spatial_agg_method,
        }

        return config


def physics_informed_feature_engineering(df_hist_:pd.DataFrame, df_forecast_:pd.DataFrame, config:dict, verbose:bool):

    if config['feature_engineer'] in ['WeatherWindPowerFE','WeatherSolarPowerFE','WeatherLoadFE']:
        if verbose:logger.info(f"Performing feature engineering with {config['feature_engineer']}")
        n_cols = len(df_hist_.columns)
        if config['feature_engineer'] == 'WeatherWindPowerFE':  o_feat = WeatherWindPowerFE( config, verbose=verbose )
        elif config['feature_engineer'] == 'WeatherSolarPowerFE': o_feat = WeatherSolarPowerFE( config, verbose=verbose )
        elif config['feature_engineer'] == 'WeatherLoadFE': o_feat = WeatherLoadFE( config, verbose=verbose )
        else: raise NotImplementedError(f"Feature engineering method {config['feature_engineer']} is not implemented")

        # extract non-weather features that will not participate in feature engineering
        non_weather_feat = [
            key for key in df_hist_.columns.tolist() if not any(key.startswith(var) for var in OpenMeteo.vars)
        ]

        # combine dataframes
        n_h, n_f = len(df_hist_), len(df_forecast_)
        df_tmp = pd.concat([df_hist_,df_forecast_],axis=0)
        df_non_weather_tmp = df_tmp[non_weather_feat].copy()
        assert len(df_tmp) == n_h + n_f

        # apply feature engineering
        df_tmp:pd.DataFrame = o_feat(df_tmp)

        # split back into hist and forecast
        df_tmp = pd.merge(df_tmp, df_non_weather_tmp, left_index=True, right_index=True, how="left")
        df_tmp = validate_dataframe(df_tmp,name='df_tmp_feature_engineering',log_func=logger.info)
        df_hist_, df_forecast_ = df_tmp[:df_forecast_.index[0]-pd.Timedelta(hours=1)], df_tmp[df_forecast_.index[0]:]

        assert len(df_forecast_) == n_f

        df_hist_ = df_hist_.dropna(inplace=False) # in case there are lagged features -- nans are introguded

        if verbose: logger.info(f"Feature engineering ({config['feature_engineer']}) "
                          f"From {n_cols} now using {len(df_hist_)} features (excl. target)")
    else:
        if verbose:
            logger.warning(f"Feature engineering is not implemented for {config['feature_engineer']}. "
                  f"Using {len(df_hist_.columns)} raw features.")

    return df_hist_, df_forecast_
