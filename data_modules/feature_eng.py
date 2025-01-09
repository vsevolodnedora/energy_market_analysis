import copy

import optuna
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from math import radians, sin, cos
import joblib
import gc

from data_collection_modules.german_locations import all_locations


def create_holiday_weekend_series(df_index):
    # Create a Germany holiday calendar
    de_holidays = holidays.Germany()

    # Generate a Series with the index from the DataFrame
    date_series = pd.Series(index=df_index, dtype=int)

    # Loop over each date in the index
    for date in date_series.index:
        # Check if the date is a holiday or a weekend (Saturday=5, Sunday=6)
        if date in de_holidays or date.weekday() >= 5:
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
    df_time_featues['hour_sin'] = np.sin(2 * np.pi * df_time_featues['hour'] / 24)
    df_time_featues['hour_cos'] = np.cos(2 * np.pi * df_time_featues['hour'] / 24)
    df_time_featues['day_sin'] = np.sin(2 * np.pi * df_time_featues['dayofweek'] / 7)
    df_time_featues['day_cos'] = np.cos(2 * np.pi * df_time_featues['dayofweek'] / 7)

    return df_time_featues

class WeatherFeatureEngineer_OLD:
    def __init__(self, config: dict, verbose:bool):
        """
        Initialize with a configuration dictionary.
        """
        self.config = config
        self.verbose = verbose
        self.loc_names = self.config.get("locations", [])
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")
        self.locations:list[dict] = [loc for loc in all_locations if loc['name'] in self.loc_names]
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")
        self.sp_agg_config = self.config.get("spatial_aggregation", {})

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Process each location separately
        processed_dfs = []
        for loc in self.locations:
            loc_df = self._preprocess_location(df.copy(), loc['suffix'])
            processed_dfs.append(loc_df)

        # Combine horizontally: each location df with engineered features
        combined_df = pd.concat(processed_dfs, axis=1)

        # Apply spatial aggregation if configured
        if self.sp_agg_config.get("method"):
            combined_df = self._apply_spatial_aggregation(combined_df)

        if self.verbose:
            print(f"Preprocessing result Shapes {df.shape} -> {combined_df.shape}"
                  f" Start {df.index[0]} -> {combined_df.index[0]}"
                  f" End {df.index[-3]} -> {combined_df.index[-1]}")

        expected_range = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='h')
        if not combined_df.index.equals(expected_range):
            raise ValueError("combined_df must be continuous with hourly frequency.")

        return combined_df

    def _preprocess_location(self, df: pd.DataFrame, location_suffix: str) -> pd.DataFrame:
        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()

        # Store original feature names for potential dropping later
        original_cols = loc_df.columns.tolist()

        # Extract key column names
        wind_speed_10m_col = f"wind_speed_10m{location_suffix}"
        wind_speed_100m_col = f"wind_speed_100m{location_suffix}"
        wind_dir_100m_col = f"wind_direction_100m{location_suffix}"
        temp_col = f"temperature_2m{location_suffix}"
        press_col = f"surface_pressure{location_suffix}"


        # Compute Air Density
        if self.config.get("compute_air_density", False) and temp_col in loc_df.columns and press_col in loc_df.columns:
            temp_K = loc_df[temp_col] + 273.15
            R_specific = 287.05  # J/(kg*K)
            loc_df["air_density" + location_suffix] = (loc_df[press_col]*100.0) / (R_specific * temp_K)

        # Compute Wind Power Density
        if self.config.get("compute_wind_power_density", False) and wind_speed_100m_col in loc_df.columns and ("air_density" + location_suffix) in loc_df.columns:
            loc_df["wind_power_density" + location_suffix] = 0.5 * loc_df["air_density" + location_suffix] * (loc_df[wind_speed_100m_col]**3)

        # Encode Wind Direction (Cyclic)
        if self.config.get("encode_wind_direction", False) and wind_dir_100m_col in loc_df.columns:
            loc_df["wind_dir_sin" + location_suffix] = np.sin(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df["wind_dir_cos" + location_suffix] = np.cos(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df.drop(columns=[wind_dir_100m_col], inplace=True)

        # Wind Shear
        if self.config.get("compute_wind_shear", False) and wind_speed_10m_col in loc_df.columns and wind_speed_100m_col in loc_df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                loc_df["wind_shear" + location_suffix] = (np.log(loc_df[wind_speed_100m_col]/loc_df[wind_speed_10m_col])) / np.log(100/10)
            loc_df["wind_shear" + location_suffix] = loc_df["wind_shear" + location_suffix].replace([np.inf, -np.inf], np.nan)

        # Turbulence Intensity
        if isinstance(self.config.get("compute_turbulence_intensity", False), dict) and wind_speed_100m_col in loc_df.columns:
            window = self.config["compute_turbulence_intensity"].get("window", 3)
            rolling_std = loc_df[wind_speed_100m_col].rolling(window=window, min_periods=1).std()
            rolling_mean = loc_df[wind_speed_100m_col].rolling(window=window, min_periods=1).mean()
            loc_df["turbulence_intensity" + location_suffix] = rolling_std / rolling_mean

        # Wind Ramp
        if self.config.get("compute_wind_ramp", False) and wind_speed_100m_col in loc_df.columns:
            loc_df["wind_ramp" + location_suffix] = loc_df[wind_speed_100m_col].diff(1)

        # Lags
        if "lags" in self.config and wind_speed_100m_col in loc_df.columns:
            for lag in self.config["lags"]:
                loc_df[f"wind_speed_lag_{lag}{location_suffix}"] = loc_df[wind_speed_100m_col].shift(lag)

        # Drop raw features if requested
        if self.config.get("drop_raw_features", False):
            features_to_drop = self.config.get("features_to_drop", [])
            drop_cols = [f"{feat}{location_suffix}" for feat in features_to_drop if f"{feat}{location_suffix}" in loc_df.columns]
            loc_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Spatial aggregation based on config
        method = self.sp_agg_config.get("method", "mean")
        loc_suffixes = [loc['suffix'] for loc in self.locations]


        # Identify engineered features (non-location-specific suffix stripped)
        # For simplicity, just find unique suffix patterns and group by them
        # We'll assume all columns contain a suffix and that suffix matches one of the locations
        all_features = df.columns.tolist()

        # Group columns by feature name ignoring suffix
        # e.g., "wind_shear_hsee" => feature base "wind_shear"
        feature_groups = {}
        for feat in all_features:
            for loc in loc_suffixes:
                if feat.endswith(loc):
                    base_name = feat.replace(loc, '')
                    if base_name not in feature_groups:
                        feature_groups[base_name] = []
                    feature_groups[base_name].append(feat)
                    break

        if method == "mean":
            # Compute mean across farms for each base feature
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "mean"] = df[cols].mean(axis=1)
            return agg_df

        elif method == "max":
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "max"] = df[cols].max(axis=1)
            return agg_df

        elif method in ["idw-c", "idw-wc-c", "idw-wc-n"]:
            # compute reference point location
            reference_point = self._compute_reference_point_for_spatial_averaging(method)
            # Inverse Distance Weighting
            agg_df = pd.DataFrame(index=df.index)
            # Compute distances of each farm to the reference point
            distances = []
            for loc in self.locations:
                lat, lon = loc['lat'], loc['lon']
                d = self._haversine_distance(reference_point, (lat, lon))
                distances.append(d if d != 0 else 1e-6)  # avoid zero-division

            weights = 1.0 / np.array(distances)
            weights = weights / weights.sum()

            for base_name, cols in feature_groups.items():
                # IDW combination of features
                mat = df[cols].values
                agg_col = np.sum(mat * weights, axis=1)
                agg_df[base_name + "idw"] = agg_col
            return agg_df

        elif method == "cluster":
            # Placeholder for clustering-based approach
            # For demonstration, just return mean until cluster logic is implemented
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "clustered"] = df[cols].mean(axis=1)
            return agg_df

        else:
            # Default fallback, return original df if method unknown
            return df

    def _compute_reference_point_for_spatial_averaging(self, method:str)->tuple:
        # compute refernce point
        reference_point = (0, 0)
        if method == "idw-c":
            # Calculate centroid
            centroid_lat = np.mean([loc['lat'] for loc in self.locations])
            centroid_lon = np.mean([loc['lon'] for loc in self.locations])
            reference_point = (centroid_lat, centroid_lon)
        elif method == "idw-wc-c":
            # Include weights, for example, based on capacity
            weights = [loc['capacity'] for loc in self.locations]
            weighted_lat = np.average([loc['lat'] for loc in self.locations], weights=weights)
            weighted_lon = np.average([loc['lon'] for loc in self.locations], weights=weights)
            reference_point = (weighted_lat, weighted_lon)
        elif method == "idw-wc-n":
            # Include weights, for example, based on number of turbines
            weights = [loc['n_turbines'] for loc in self.locations]
            weighted_lat = np.average([loc['lat'] for loc in self.locations], weights=weights)
            weighted_lon = np.average([loc['lon'] for loc in self.locations], weights=weights)
            reference_point = (weighted_lat, weighted_lon)
        return reference_point

    @staticmethod
    def _haversine_distance(p1, p2):
        # p1, p2 = (lat, lon) in degrees
        R = 6371.0  # Earth radius in km
        lat1, lon1 = map(radians, p1)
        lat2, lon2 = map(radians, p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2*np.arcsin(np.sqrt(a))
        return R * c

    @staticmethod
    def selector_for_optuna(trial:optuna.Trial) -> dict:
        # Boolean features
        compute_air_density = trial.suggest_categorical("compute_air_density", [True, False])
        compute_wind_power_density = trial.suggest_categorical("compute_wind_power_density", [True, False])
        encode_wind_direction = trial.suggest_categorical("encode_wind_direction", [True, False])
        compute_wind_shear = trial.suggest_categorical("compute_wind_shear", [True, False])
        compute_wind_ramp = trial.suggest_categorical("compute_wind_ramp", [True, False])

        # Turbulence intensity window size if enabled
        use_turbulence = trial.suggest_categorical("compute_turbulence_intensity", [False, True])
        turbulence_config = {"window": trial.suggest_int("turbulence_window", 2, 6)} if use_turbulence else False

        # Lags can be a list, decide length and values
        lag_choice = trial.suggest_categorical("lags_option", ["none", "small", "large"])
        if lag_choice == "none":
            lags = []
        elif lag_choice == "small":
            lags = [1, 6]
        else:
            lags = [1, 6, 12, 24]


        # Raw feature dropping
        drop_raw_features = trial.suggest_categorical("drop_raw_features", [True, False])
        features_to_drop = [
            "precipitation", "cloud_cover", "shortwave_radiation", "relative_humidity_2m", "wind_direction_10m", "wind_gusts_10m"
        ]

        # Spatial aggregation method
        spatial_method = trial.suggest_categorical("spatial_method", [
            "mean", "max", "idw-c",'idw-wc-c','idw-wc-n', "cluster"])
        sp_agg_config = {
            "method": spatial_method
        }


        # Build the config dictionary
        config = {
            "compute_air_density": compute_air_density,
            "compute_wind_power_density": compute_wind_power_density,
            "encode_wind_direction": encode_wind_direction,
            "compute_wind_shear": compute_wind_shear,
            "compute_turbulence_intensity": turbulence_config,
            "compute_wind_ramp": compute_wind_ramp,
            "lags": lags,
            "drop_raw_features": drop_raw_features,
            "features_to_drop": features_to_drop,
            "spatial_aggregation": sp_agg_config,
        }
        return config

class WeatherFeatureEngineer:
    def __init__(self, config: dict, verbose:bool):
        """
        Initialize with a configuration dictionary.
        """
        self.config = config
        self.verbose = verbose
        self.loc_names = self.config.get("locations", [])
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")
        self.locations:list[dict] = [loc for loc in all_locations if loc['name'] in self.loc_names]
        if len(self.locations) == 0:
            raise ValueError("No locations configured.")
        self.sp_agg_config = self.config.get("spatial_aggregation", {})


    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Process each location separately
        processed_dfs = []
        for loc in self.locations:
            loc_df = self._preprocess_location(df.copy(), loc['suffix'])
            processed_dfs.append(loc_df)

        # Combine horizontally: each location df with engineered features
        combined_df = pd.concat(processed_dfs, axis=1)

        # Apply spatial aggregation if configured
        if self.sp_agg_config.get("method"):
            combined_df = self._apply_spatial_aggregation(combined_df)

        if self.verbose:
            print(f"Preprocessing result Shapes {df.shape} -> {combined_df.shape}"
                  f" Start {df.index[0]} -> {combined_df.index[0]}"
                  f" End {df.index[-3]} -> {combined_df.index[-1]}")

        expected_range = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='h')
        if not combined_df.index.equals(expected_range):
            raise ValueError("combined_df must be continuous with hourly frequency.")

        return combined_df

    def _check_cols(self, df: pd.DataFrame, location_suffix:str):
        ''' expected columns from openmeteo dataframe '''
        expected_cols = [
            f"wind_speed_10m{location_suffix}", # velocity, km/h
            f"wind_speed_100m{location_suffix}", # velocity, km/h
            f"wind_direction_100m{location_suffix}", # degrees (0 to 180)
            f"temperature_2m{location_suffix}", # degrees C
            f"surface_pressure{location_suffix}" # hPa
            f"relative_humidity_2m{location_suffix}", # percent
            f"precipitation{location_suffix}", # mm
            f"wind_gusts_10m{location_suffix}", # velocity, km/h
            f"cloud_cover{location_suffix}", # percentage (0 to 100)
            f"relative_humidity_2m{location_suffix}", # percentage (0 to 100)
        ]
        for col in expected_cols:
            if not col in df.columns:
                raise ValueError(f"Expected column {col} to be present.")

    def _preprocess_location(self, df: pd.DataFrame, location_suffix: str) -> pd.DataFrame:
        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()
        elevation:float = loc_df['elevation'] # meters
        z0:float = loc_df['z0'] # meters (roughness length)
        terrain_catergoty:str = loc_df['terrain_category'] # "I", "II" or "III", # as defined in the Eurocode standards,

        self._check_cols(loc_df, location_suffix)

        # Extract key column names
        wind_speed_10m_col = f"wind_speed_10m{location_suffix}"
        wind_speed_100m_col = f"wind_speed_100m{location_suffix}"
        wind_dir_100m_col = f"wind_direction_100m{location_suffix}"
        temp_col = f"temperature_2m{location_suffix}"
        press_col = f"surface_pressure{location_suffix}"


        # Compute Air Density
        if self.config["compute_air_density"]:
            temp_K = loc_df[temp_col] + 273.15
            R_specific = 287.05  # J/(kg*K)
            loc_df["air_density" + location_suffix] = (loc_df[press_col]*100.0) / (R_specific * temp_K)
            # Compute Wind Power Density
            if self.config["compute_wind_power_density"]:
                loc_df["wind_power_density" + location_suffix] = \
                    0.5 * loc_df["air_density" + location_suffix] * (loc_df[wind_speed_100m_col]**3)


        # Encode Wind Direction (Cyclic)
        if self.config.get("encode_wind_direction", False):
            loc_df["wind_dir_sin" + location_suffix] = np.sin(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df["wind_dir_cos" + location_suffix] = np.cos(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df.drop(columns=[wind_dir_100m_col], inplace=True)

        # Wind Shear
        if self.config.get("compute_wind_shear", False):
            with np.errstate(divide='ignore', invalid='ignore'):
                loc_df["wind_shear" + location_suffix] = \
                    (np.log(loc_df[wind_speed_100m_col]/loc_df[wind_speed_10m_col])) / np.log(100./10.)

            loc_df["wind_shear" + location_suffix] = loc_df["wind_shear" + location_suffix].replace([np.inf, -np.inf], np.nan)

        # Turbulence Intensity
        if isinstance(self.config.get("compute_turbulence_intensity", False), dict):
            window = self.config["compute_turbulence_intensity"].get("window", 3)
            rolling_std = loc_df[wind_speed_100m_col].rolling(window=window, min_periods=1).std()
            rolling_mean = loc_df[wind_speed_100m_col].rolling(window=window, min_periods=1).mean()
            loc_df["turbulence_intensity" + location_suffix] = rolling_std / rolling_mean

        # Wind Ramp
        if self.config.get("compute_wind_ramp", False) :
            loc_df["wind_ramp" + location_suffix] = loc_df[wind_speed_100m_col].diff(1)

        # Lags
        if "lags" in self.config :
            for lag in self.config["lags"]:
                loc_df[f"wind_speed_lag_{lag}{location_suffix}"] = loc_df[wind_speed_100m_col].shift(lag)

        # Drop raw features if requested
        if self.config.get("drop_raw_features", False):
            features_to_drop = self.config.get("features_to_drop", [])
            drop_cols = [f"{feat}{location_suffix}" for feat in features_to_drop if f"{feat}{location_suffix}" in loc_df.columns]
            loc_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Examle of locations:
         self.locations = [
             {
                "name": "Hüselitz Wind Farm",
                "capacity": 151.8,
                "n_turbines": 46,
                "lat": 52.5347,
                "lon": 11.7321,
                "location": "Lower Saxony",
                "TSO": "50Hertz",
                "suffix":"_won_hueselitz",
                "type": "onshore wind farm",
            },
            {
                "name": "Werder/Kessin Wind Farm",
                "capacity": 148.05,
                "n_turbines": 32,
                "lat": 53.7270,
                "lon": 13.3362,
                "TSO": "50Hertz",
                "suffix":"_won_werder",
                "type": "onshore wind farm",
            }
        ]
        :param df:
        :return:
        '''
        # Spatial aggregation based on config
        method = self.sp_agg_config.get("method", "mean")
        loc_suffixes = [loc['suffix'] for loc in self.locations]

        # Identify engineered features (non-location-specific suffix stripped)
        # For simplicity, just find unique suffix patterns and group by them
        all_features = df.columns.tolist()

        # Group columns by feature name ignoring suffix e.g., "wind_shear_hsee" => feature base "wind_shear"
        feature_groups = {}
        for feat in all_features:
            for loc in loc_suffixes:
                if feat.endswith(loc):
                    base_name = feat.replace(loc, '')
                    if base_name not in feature_groups:
                        feature_groups[base_name] = []
                    feature_groups[base_name].append(feat)
                    break

        if method == "mean":
            # Compute mean across farms for each base feature
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "mean"] = df[cols].mean(axis=1)
            return agg_df

        elif method == "max":
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "max"] = df[cols].max(axis=1)
            return agg_df

        elif method in ["idw-c", "idw-wc-c", "idw-wc-n"]:
            # compute reference point location
            reference_point = self._compute_reference_point_for_spatial_averaging(method)
            # Inverse Distance Weighting
            agg_df = pd.DataFrame(index=df.index)
            # Compute distances of each farm to the reference point
            distances = []
            for loc in self.locations:
                lat, lon = loc['lat'], loc['lon']
                d = self._haversine_distance(reference_point, (lat, lon))
                distances.append(d if d != 0 else 1e-6)  # avoid zero-division

            weights = 1.0 / np.array(distances)
            weights = weights / weights.sum()

            for base_name, cols in feature_groups.items():
                # IDW combination of features
                mat = df[cols].values
                agg_col = np.sum(mat * weights, axis=1)
                agg_df[base_name + "idw"] = agg_col
            return agg_df

        elif method == "cluster":
            # Placeholder for clustering-based approach
            # For demonstration, just return mean until cluster logic is implemented
            agg_df = pd.DataFrame(index=df.index)
            for base_name, cols in feature_groups.items():
                agg_df[base_name + "clustered"] = df[cols].mean(axis=1)
            return agg_df

        else:
            # Default fallback, return original df if method unknown
            return df

    def _compute_reference_point_for_spatial_averaging(self, method:str)->tuple:
        # compute refernce point
        reference_point = (0, 0)
        if method == "idw-c":
            # Calculate centroid
            centroid_lat = np.mean([loc['lat'] for loc in self.locations])
            centroid_lon = np.mean([loc['lon'] for loc in self.locations])
            reference_point = (centroid_lat, centroid_lon)
        elif method == "idw-wc-c":
            # Include weights, for example, based on capacity
            weights = [loc['capacity'] for loc in self.locations]
            weighted_lat = np.average([loc['lat'] for loc in self.locations], weights=weights)
            weighted_lon = np.average([loc['lon'] for loc in self.locations], weights=weights)
            reference_point = (weighted_lat, weighted_lon)
        elif method == "idw-wc-n":
            # Include weights, for example, based on number of turbines
            weights = [loc['n_turbines'] for loc in self.locations]
            weighted_lat = np.average([loc['lat'] for loc in self.locations], weights=weights)
            weighted_lon = np.average([loc['lon'] for loc in self.locations], weights=weights)
            reference_point = (weighted_lat, weighted_lon)
        return reference_point

    @staticmethod
    def _haversine_distance(p1, p2):
        # p1, p2 = (lat, lon) in degrees
        R = 6371.0  # Earth radius in km
        lat1, lon1 = map(radians, p1)
        lat2, lon2 = map(radians, p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2*np.arcsin(np.sqrt(a))
        return R * c

    @staticmethod
    def selector_for_optuna(trial:optuna.Trial) -> dict:
        # Boolean features
        compute_air_density = trial.suggest_categorical("compute_air_density", [True, False])
        compute_wind_power_density = trial.suggest_categorical("compute_wind_power_density", [True, False])
        encode_wind_direction = trial.suggest_categorical("encode_wind_direction", [True, False])
        compute_wind_shear = trial.suggest_categorical("compute_wind_shear", [True, False])
        compute_wind_ramp = trial.suggest_categorical("compute_wind_ramp", [True, False])

        # Turbulence intensity window size if enabled
        use_turbulence = trial.suggest_categorical("compute_turbulence_intensity", [False, True])
        turbulence_config = {"window": trial.suggest_int("turbulence_window", 2, 6)} if use_turbulence else False

        # Lags can be a list, decide length and values
        lag_choice = trial.suggest_categorical("lags_option", ["none", "small", "large"])
        if lag_choice == "none":
            lags = []
        elif lag_choice == "small":
            lags = [1, 6]
        else:
            lags = [1, 6, 12, 24]


        # Raw feature dropping
        drop_raw_features = trial.suggest_categorical("drop_raw_features", [True, False])
        features_to_drop = [
            "precipitation", "cloud_cover", "shortwave_radiation", "relative_humidity_2m", "wind_direction_10m", "wind_gusts_10m"
        ]

        # Spatial aggregation method
        spatial_method = trial.suggest_categorical("spatial_method", [
            "mean", "max", "idw-c",'idw-wc-c','idw-wc-n', "cluster"])
        sp_agg_config = {
            "method": spatial_method
        }


        # Build the config dictionary
        config = {
            "compute_air_density": compute_air_density,
            "compute_wind_power_density": compute_wind_power_density,
            "encode_wind_direction": encode_wind_direction,
            "compute_wind_shear": compute_wind_shear,
            "compute_turbulence_intensity": turbulence_config,
            "compute_wind_ramp": compute_wind_ramp,
            "lags": lags,
            "drop_raw_features": drop_raw_features,
            "features_to_drop": features_to_drop,
            "spatial_aggregation": sp_agg_config,
        }
        return config

class WeatherFeatureEngineer_2:
    def __init__(self, config: dict, verbose:bool):
        """
        Initialize with a configuration dictionary.
        """
        self.config = config
        self.verbose = verbose
        self.loc_names = self.config.get("locations", [])
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")
        self.locations:list[dict] = [loc for loc in all_locations if loc['name'] in self.loc_names]
        if len(self.locations) == 0:
            raise ValueError("No locations configured.")


    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Process each location separately
        processed_dfs = []
        for loc in self.locations:
            loc_df = self._preprocess_location(df.copy(), loc['suffix'])
            processed_dfs.append(loc_df)

        # Combine horizontally: each location df with engineered features
        combined_df = pd.concat(processed_dfs, axis=1)

        # Apply spatial aggregation if configured
        if (not self.config.get("spatial_agg_method", None) is "None") and (len(self.locations) > 1):
            combined_df = self._apply_spatial_aggregation(combined_df)

        if self.verbose:
            print(f"Preprocessing result Shapes {df.shape} -> {combined_df.shape}"
                  f" Start {df.index[0]} -> {combined_df.index[0]}"
                  f" End {df.index[-3]} -> {combined_df.index[-1]}")

        expected_range = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='h')
        if not combined_df.index.equals(expected_range):
            raise ValueError("combined_df must be continuous with hourly frequency.")

        return combined_df

    def _check_cols(self, df: pd.DataFrame, location_suffix:str):
        ''' expected columns from openmeteo dataframe '''
        expected_cols = [
            f"wind_speed_10m{location_suffix}", # velocity, km/h
            f"wind_speed_100m{location_suffix}", # velocity, km/h
            f"wind_direction_100m{location_suffix}", # degrees (0 to 180)
            f"temperature_2m{location_suffix}", # degrees C
            f"surface_pressure{location_suffix}" # hPa
            f"relative_humidity_2m{location_suffix}", # percent
            f"precipitation{location_suffix}", # mm
            f"wind_gusts_10m{location_suffix}", # velocity, km/h
            f"cloud_cover{location_suffix}", # percentage (0 to 100)
            f"relative_humidity_2m{location_suffix}", # percentage (0 to 100)
        ]
        for col in expected_cols:
            if not col in df.columns:
                raise ValueError(f"Expected column {col} to be present.")
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

        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()


        # -- Extract key meteorological columns
        wind_speed_10m_col = f"wind_speed_10m{location_suffix}"  # in km/h
        wind_speed_100m_col = f"wind_speed_100m{location_suffix}"  # in km/h
        wind_dir_100m_col = f"wind_direction_100m{location_suffix}"  # degrees (0 to 360)
        temp_col = f"temperature_2m{location_suffix}"  # Celsius
        press_col = f"surface_pressure{location_suffix}"  # hPa
        humid_col = f"relative_humidity_2m{location_suffix}"  # %
        precip_col = f"precipitation{location_suffix}"  # mm
        wind_gust_10m_col = f"wind_gusts_10m{location_suffix}"  # in km/h
        cloud_col = f"cloud_cover{location_suffix}"  # %

        # ------------------------------
        # 2. Compute Air Density (dry) & Wind Power Density
        # ------------------------------
        if self.config["compute_air_density"]:
            # Convert pressure hPa -> Pa
            P_Pa = loc_df[press_col] * 100.0
            # Convert temperature °C -> K
            T_K = loc_df[temp_col] + 273.15
            R_d = 287.05  # J/(kg·K) (dry air gas constant)

            # ρ (dry) = P / (R_d * T)
            loc_df["air_density" + location_suffix] = P_Pa / (R_d * T_K)

            # Convert wind speed from km/h to m/s
            wind_speed_m_s = loc_df[wind_speed_100m_col] * (1000.0 / 3600.0)

            # wind_power_density = 0.5 * rho * v^3
            loc_df["wind_power_density" + location_suffix] = (
                    0.5 * loc_df["air_density" + location_suffix] * (wind_speed_m_s ** 3)
            )

        # ------------------------------
        # 2a. Compute Air Density With Moist Air Correction
        # ------------------------------
        if self.config.get("compute_air_density_moiust_air_correction", False):
            # Convert temperature °C -> K
            T_K = loc_df[temp_col] + 273.15
            # Convert total pressure from hPa -> Pa
            p_total_pa = loc_df[press_col] * 100.0

            # --- Compute vapor pressure in hPa using the Magnus formula ---
            # e (hPa) = 6.112 * exp(17.67 * T / (T + 243.5)) * (RH/100)
            # T is in °C here, so let's call it T_C to be explicit
            T_C = loc_df[temp_col]
            RH = loc_df[humid_col] / 100.0  # from % to fraction

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
            loc_df["air_density_moist" + location_suffix] = (
                    p_d_pa / (R_d * T_K) + e_pa / (R_v * T_K)
            )

        # ------------------------------
        # 3. Encode Wind Direction (Cyclic)
        # ------------------------------
        if self.config.get("encode_wind_direction", False):
            loc_df["wind_dir_sin" + location_suffix] = np.sin(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df["wind_dir_cos" + location_suffix] = np.cos(np.deg2rad(loc_df[wind_dir_100m_col]))
            loc_df.drop(columns=[wind_dir_100m_col], inplace=True)

        # ------------------------------
        # 4. Compute Wind Shear
        # ------------------------------
        if self.config.get("compute_wind_shear", False):
            # alpha = ln(V_100 / V_10) / ln(100/10)
            numerator = np.log(loc_df[wind_speed_100m_col] / loc_df[wind_speed_10m_col])
            denominator = np.log(100.0 / 10.0)

            loc_df["wind_shear" + location_suffix] = numerator / denominator
            # handle infinities
            loc_df["wind_shear" + location_suffix] = loc_df["wind_shear" + location_suffix].replace([np.inf, -np.inf], np.nan)

        # ------------------------------
        # 5. Turbulence Intensity
        # ------------------------------
        if isinstance(self.config.get("compute_turbulence_intensity", False), dict):
            window = self.config["compute_turbulence_intensity"].get("window", 3)
            # TI = rolling std / rolling mean of wind speed
            rolling_std = loc_df[wind_speed_100m_col].rolling(window=window).std()
            rolling_mean = loc_df[wind_speed_100m_col].rolling(window=window).mean()
            loc_df["turbulence_intensity" + location_suffix] = rolling_std / rolling_mean

        # ------------------------------
        # 6. Wind Ramp
        # ------------------------------
        if self.config.get("compute_wind_ramp", False):
            # Simple difference in wind speed from one timestep to the next
            loc_df["wind_ramp" + location_suffix] = loc_df[wind_speed_100m_col].diff()

        # ------------------------------
        # 7. Lags
        # ------------------------------
        if "lags" in self.config:
            for lag in self.config["lags"]:
                loc_df[f"wind_speed_lag_{lag}{location_suffix}"] = loc_df[wind_speed_100m_col].shift(lag)

        # ------------------------------
        # 7a. Precipitation Lags
        # ------------------------------
        if self.config.get("precip_lags", False):
            # Similar logic to wind_speed lags
            for lag in self.config["precip_lags"]:
                loc_df[f"precip_lag_{lag}{location_suffix}"] = loc_df[precip_col].shift(lag)

        # ------------------------------
        # 8. Gust Factor
        # ------------------------------
        if self.config.get("gust_factor", False):
            # Convert wind_gust_10m and wind_speed_10m from km/h to m/s to keep consistent units
            gust_m_s = loc_df[wind_gust_10m_col] * (1000.0 / 3600.0)
            ws_m_s = loc_df[wind_speed_10m_col] * (1000.0 / 3600.0)

            # gust_factor = gust / wind_speed (handle zero or near-zero speeds)
            gf = gust_m_s / ws_m_s.replace({0: np.nan})  # or clip small speeds
            loc_df["gust_factor" + location_suffix] = gf.replace([np.inf, -np.inf], np.nan)

        # ------------------------------
        # 9. Dew Point Temperature
        # ------------------------------
        if self.config.get("dew_point_temperature", False):
            # Using the Magnus formula variant
            # T_dew in °C
            T_C = loc_df[temp_col]  # in Celsius
            RH_frac = loc_df[humid_col] / 100.0

            # Prevent log(0) by ensuring RH_frac>0
            RH_safe = RH_frac.where(RH_frac > 0, other=0.0001)

            # Commonly used constants for Magnus formula
            a = 17.62
            b = 243.12  # °C

            # e_s (saturation vapor pressure in hPa) = 6.112 * exp(a * T / (b + T))
            # Then actual vapor pressure e = e_s * RH
            # T_dew = (b * gamma) / (a - gamma)
            # gamma = ln(RH) + (a * T / (b + T))
            gamma = np.log(RH_safe) + (a * T_C / (b + T_C))

            T_dew = (b * gamma) / (a - gamma)
            loc_df["dew_point_temperature" + location_suffix] = T_dew

        # ------------------------------
        # 10. Vapor Pressure (Magnus formula in hPa)
        # ------------------------------
        if self.config.get("vapor_pressure", False):
            T_C = loc_df[temp_col]  # in Celsius
            RH_frac = loc_df[humid_col] / 100.0

            # e (hPa) = 6.112 * exp(17.67 * T / (T + 243.5)) * RH
            e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH_frac
            loc_df["vapor_pressure" + location_suffix] = e_hpa  # in hPa

        # ------------------------------
        # 8. Drop raw features if requested
        # ------------------------------
        if self.config.get("drop_raw_features", False):
            features_to_drop = self.config.get("features_to_drop", [])
            drop_cols = [f"{feat}{location_suffix}" for feat in features_to_drop if f"{feat}{location_suffix}" in loc_df.columns]
            loc_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features across multiple wind farms using a specified spatial method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame. It contains columns for all wind farms,
            each column having a unique suffix (e.g. '_F1', '_F2', etc.).

        Returns
        -------
        pd.DataFrame
            A DataFrame with aggregated features (one column per base feature).
        """

        if len(self.locations) <= 1:
            raise ValueError("Cannot apply spatial aggregation on one location")

        # Extract the aggregation method from config
        method: str = self.config.get("spatial_agg_method")  # e.g. "mean", "max", "idw", "capacity", etc.
        all_features = df.columns.tolist()

        # 1. Group columns by the wind-farm suffix
        # ------------------------------------------------
        features_per_wind_farm = {}
        suffixes = []
        for loc in self.locations:
            suffix = loc['suffix']  # e.g. '_F1'
            suffixes.append(suffix)
            cols_with_suffix = [c for c in all_features if c.endswith(suffix)]
            features_per_wind_farm[suffix] = cols_with_suffix

        # 2. Build a mapping from "base feature name" -> columns for each suffix
        # ------------------------------------------------
        def strip_suffix(col_name: str, available_suffixes) -> str:
            for sfx in available_suffixes:
                if col_name.endswith(sfx):
                    return col_name.replace(sfx, "")
            return col_name  # if no match, return as-is

        base_feature_map = {}  # { base_feature: [ (suffix, col_name), ... ] }
        for loc in self.locations:
            suffix = loc['suffix']
            for col_name in features_per_wind_farm[suffix]:
                base_feat = strip_suffix(col_name, [suffix])
                if base_feat not in base_feature_map:
                    base_feature_map[base_feat] = []
                base_feature_map[base_feat].append((suffix, col_name))

        # 3. Prepare for more complex methods: compute distances, weights, etc.
        # ------------------------------------------------
        loc_meta = {}
        for loc in self.locations:
            sfx = loc['suffix']
            loc_meta[sfx] = {
                "lat": loc['lat'],
                "lon": loc['lon'],
                "capacity": loc['capacity'],
                "n_turbines": loc['n_turbines'],
                "elevation": loc['elevation'] if 'elevation' in loc else 0.,
                "z0": loc['z0'] if 'z0' in loc else 0.,
                "terrain_category": loc['terrain_category'] if 'terrain_category' in loc else 'I',
            }

        def haversine_distance(lat1, lon1, lat2, lon2):
            # lat/lon in degrees -> convert to radians
            rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = rlat2 - rlat1
            dlon = rlon2 - rlon1
            a = sin(dlat/2)**2 + cos(rlat1)*cos(rlat2)*sin(dlon/2)**2
            c = 2 * np.asin(np.sqrt(a))
            # Earth radius ~6371 km
            r = 6371
            return r * c

        # For IDW or distance-based methods, define a reference point (e.g. centroid)
        mean_lat = np.mean([loc_meta[sfx]["lat"] for sfx in suffixes])
        mean_lon = np.mean([loc_meta[sfx]["lon"] for sfx in suffixes])

        # 4. Create a new DataFrame of aggregated features
        aggregated_df = pd.DataFrame(index=df.index)

        # 5. Apply the chosen method for each base feature
        # ------------------------------------------------
        for base_feat, columns_with_suffixes in base_feature_map.items():
            # Gather each suffix's time series into a small DataFrame
            sub_data = {}
            for (sfx, c_name) in columns_with_suffixes:
                sub_data[sfx] = df[c_name]
            sub_df = pd.DataFrame(sub_data, index=df.index)  # columns = suffixes

            if method == "mean":
                # Simple average across wind farms
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

            elif method == "max":
                # Max across wind farms
                aggregated_df[base_feat + "_agg"] = sub_df.max(axis=1)

            elif method == "idw":
                # Inverse Distance Weighting around (mean_lat, mean_lon)
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)  # avoid division by zero
                    weights[sfx] = 1.0 / (d_km ** 2)

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "capacity":
                # Weight by farm's capacity (no distance factor)
                # aggregated_value(t) = sum( capacity_i * x_i(t) ) / sum(capacity_i)
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["capacity"]

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "n_turbines":
                # Weight by farm's number of turbines (no distance factor)
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["n_turbines"]

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "distance_capacity":
                # Combine distance-based weighting with capacity
                # weight_i = capacity_i / distance_i^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    cap_i = loc_meta[sfx]["capacity"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = cap_i / (d_km ** 2)

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "distance_n_turbines":
                # Combine distance-based weighting with number of turbines
                # weight_i = n_turbines_i / distance_i^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    n_turb = loc_meta[sfx]["n_turbines"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = n_turb / (d_km ** 2)

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            else:
                # Fallback: if unknown method, just do mean
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

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
        use_turbulence = trial.suggest_categorical("compute_turbulence_intensity", [False, True])
        turbulence_config = (
            {"window": trial.suggest_int("turbulence_window", 2, 6)}
            if use_turbulence
            else False
        )

        # Lags can be a list, decide length and values
        lag_choice = trial.suggest_categorical("lags_option", ["none", "small", "large"])
        if lag_choice == "none":
            lags = []
        elif lag_choice == "small":
            lags = [1, 6]
        else:
            lags = [1, 6, 12, 24]

        # Precipitation lags
        precip_lags_choice = trial.suggest_categorical("precip_lags_option", ["none", "small", "large"])
        if precip_lags_choice == "none":
            precip_lags = []
        elif precip_lags_choice == "small":
            precip_lags = [1, 6]
        else:
            precip_lags = [1, 6, 12, 24]

        # Raw feature dropping
        drop_raw_features = trial.suggest_categorical("drop_raw_features", [True, False])
        features_to_drop = [
            "precipitation",
            "cloud_cover",
            "shortwave_radiation",
            "relative_humidity_2m",
            "wind_direction_10m",
            "wind_gusts_10m"
        ]

        # Spatial aggregation method
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
        sp_agg_config = {
            "method": spatial_agg_method
        }

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
            "compute_turbulence_intensity": turbulence_config,  # either False or dict
            "lags": lags,  # wind speed lags
            "precip_lags": precip_lags,  # precipitation lags
            "drop_raw_features": drop_raw_features,
            "features_to_drop": features_to_drop,
            "spatial_agg_config": sp_agg_config,
        }


        return config