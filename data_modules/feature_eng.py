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




class WeatherWindPowerFE:
    def __init__(self, config: dict, verbose:bool):
        """
        Initialize with a configuration dictionary.
        """
        self.config = config
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


'''
    
    Please Rewrite and Adapt Feature Engineering Class for Solar Power Generation
    
    Adapt an existing Python class (WeatherWindPowerFE) designed for feature engineering in wind power forecasting 
    to work for solar power generation forecasting. The updated class should incorporate solar-specific features, 
    preprocessing logic. Engineer features that are most relevant for solar power generation with gradient 
    boosting and random forest family of models. Do not include temporal features (those are added elsewhere). 
    
    Use the following meteorological columns inside _preprocess_location(): 
    
    temp_col = f"temperature_2m{location_suffix}"  # Celsius
    press_col = f"surface_pressure{location_suffix}"  # hPa
    humid_col = f"relative_humidity_2m{location_suffix}"  # %
    precip_col = f"precipitation{location_suffix}"  # mm
    cloud_col = f"cloud_cover{location_suffix}"  # %
    shortwave_col = f"shortwave_radiation{location_suffix}"  # W/m^2
    direct_col = f"direct_radiation{location_suffix}"  # W/m^2
    diffuse_col = f"diffuse_radiation{location_suffix}"  # W/m^2
    dni_col = f"direct_normal_irradiance{location_suffix}"  # W/m^2
    global_tilted_col = f"global_tilted_irradiance{location_suffix}"  # W/m^2
    terrestrial_col = f"terrestrial_radiation{location_suffix}"  # W/m^2
    
'''


class WeatherSolarPowerFE:

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
        self.config = config
        self.verbose = verbose

        # Get list of location names (solar farms, cities, etc.)
        self.loc_names = self.config.get("locations", [])
        if len(self.loc_names) == 0:
            raise ValueError("No locations configured.")

        # Get list of dict() for locations. Each dict has latitude, longitude, suffix, etc.
        self.locations: list[dict] = [loc for loc in all_locations if loc["name"] in self.loc_names]
        if len(self.locations) == 0:
            raise ValueError("No locations configured.")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a DataFrame with raw features `df`, preprocess per the config,
        engineer features, drop features, apply aggregation, and return the result.

        Returns
        -------
        pd.DataFrame
            A DataFrame with solar-specific engineered features for each location.
        """

        # Process each location separately
        processed_dfs = []
        for loc in self.locations:
            loc_df = self._preprocess_location(df.copy(), loc["suffix"])
            processed_dfs.append(loc_df)

        # Combine horizontally: each location's engineered features side by side
        combined_df = pd.concat(processed_dfs, axis=1)

        # Apply spatial aggregation if configured
        if (self.config.get("spatial_agg_method", None) != "None") and (len(self.locations) > 1):
            combined_df = self._apply_spatial_aggregation(combined_df)

        if self.verbose:
            print(
                f"Preprocessing result: {df.shape} -> {combined_df.shape} | "
                f"Start {df.index[0]} -> {combined_df.index[0]} | "
                f"End {df.index[-1]} -> {combined_df.index[-1]}"
            )

        # Ensure continuous hourly frequency
        expected_range = pd.date_range(
            start=combined_df.index.min(), end=combined_df.index.max(), freq="h"
        )
        if not combined_df.index.equals(expected_range):
            raise ValueError("combined_df must be continuous with hourly frequency.")

        return combined_df

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

        # 1. Select columns for this location
        cols = [c for c in df.columns if c.endswith(location_suffix)]
        loc_df = df[cols].copy()

        # 2. Identify key meteorological columns for solar
        temp_col = f"temperature_2m{location_suffix}"        # °C
        press_col = f"surface_pressure{location_suffix}"     # hPa
        humid_col = f"relative_humidity_2m{location_suffix}" # %
        precip_col = f"precipitation{location_suffix}"       # mm
        cloud_col = f"cloud_cover{location_suffix}"          # %
        shortwave_col = f"shortwave_radiation{location_suffix}"        # W/m^2
        direct_col = f"direct_radiation{location_suffix}"              # W/m^2
        diffuse_col = f"diffuse_radiation{location_suffix}"            # W/m^2
        dni_col = f"direct_normal_irradiance{location_suffix}"         # W/m^2
        global_tilted_col = f"global_tilted_irradiance{location_suffix}"   # W/m^2
        terrestrial_col = f"terrestrial_radiation{location_suffix}"         # W/m^2

        # 3. Cloud Cover Features
        # ----------------------------------------------------------
        # Example: convert cloud cover from % to fraction
        if self.config.get("compute_cloud_cover_fraction", False):
            loc_df[f"cloud_cover_fraction{location_suffix}"] = loc_df[cloud_col] / 100.0

        # Example: add a "clear_sky_fraction"
        if self.config.get("compute_clear_sky_fraction", False):
            loc_df[f"clear_sky_fraction{location_suffix}"] = 1.0 - (loc_df[cloud_col] / 100.0)

        # 4. Air Density (Dry and/or Moist)
        # ----------------------------------------------------------
        # ρ (dry) = P / (R_d * T)
        if self.config.get("compute_air_density", False):
            R_d = 287.05  # J/(kg·K)
            loc_df[f"air_density{location_suffix}"] = (
                    loc_df[press_col] * 100.0  # convert hPa -> Pa
                    / (R_d * (loc_df[temp_col] + 273.15))
            )

        # Moist air density correction (optional)
        if self.config.get("compute_air_density_moist", False):
            T_K = loc_df[temp_col] + 273.15
            p_total_pa = loc_df[press_col] * 100.0
            # vapor pressure (Magnus formula)
            T_C = loc_df[temp_col]
            RH = loc_df[humid_col] / 100.0
            e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH
            e_pa = e_hpa * 100.0
            p_d_pa = p_total_pa - e_pa

            R_d = 287.05  # J/(kg·K)
            R_v = 461.50  # J/(kg·K)
            loc_df[f"air_density_moist{location_suffix}"] = (
                    p_d_pa / (R_d * T_K) + e_pa / (R_v * T_K)
            )

        # 5. Solar-Specific Ratios
        # ----------------------------------------------------------
        # Ratios: direct/shortwave, diffuse/shortwave, etc.
        # Replace zero shortwave with np.inf to avoid div-by-zero (rations than will be 0)
        sw = loc_df[shortwave_col].replace(0, np.inf)

        if self.config.get("compute_direct_ratio", True):
            loc_df[f"direct_ratio{location_suffix}"] = loc_df[direct_col] / sw

        if self.config.get("compute_diffuse_ratio", True):
            loc_df[f"diffuse_ratio{location_suffix}"] = loc_df[diffuse_col] / sw

        if self.config.get("compute_dni_ratio", True):
            loc_df[f"dni_ratio{location_suffix}"] = loc_df[dni_col] / sw

        if self.config.get("compute_global_tilted_ratio", False):
            loc_df[f"global_tilted_ratio{location_suffix}"] = loc_df[global_tilted_col] / sw
        #
        # if self.config.get("compute_terrestrial_ratio", False):
        #     loc_df[f"terrestrial_ratio{location_suffix}"] = loc_df[terrestrial_col] / sw

        # 6. Solar Geometry (Elevation, Azimuth, etc.)
        # ----------------------------------------------------------
        if not self.config.get("use_solar_geometry", False):
            loc_df.drop(columns=[
                f"solar_elevation_deg{location_suffix}",
                f"solar_azimuth_deg{location_suffix}"],
                inplace=True, errors="ignore"
            )

        # 7. Dew Point Temperature (Magnus formula)
        # ----------------------------------------------------------
        if self.config.get("dew_point_temperature", False):
            T_C = loc_df[temp_col]
            RH_frac = loc_df[humid_col] / 100.0
            RH_safe = RH_frac.where(RH_frac > 0, other=0.0001)
            a = 17.62
            b = 243.12
            gamma = np.log(RH_safe) + (a * T_C / (b + T_C))
            T_dew = (b * gamma) / (a - gamma)
            loc_df[f"dew_point_temperature{location_suffix}"] = T_dew

        # 8. Vapor Pressure (Magnus formula in hPa)
        # ----------------------------------------------------------
        if self.config.get("vapor_pressure", False):
            T_C = loc_df[temp_col]
            RH_frac = loc_df[humid_col] / 100.0
            e_hpa = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5)) * RH_frac
            loc_df[f"vapor_pressure{location_suffix}"] = e_hpa

        # 9. Lags (Example: precipitation, cloud, shortwave, etc.)
        # ----------------------------------------------------------
        # Precipitation lags
        if "precip_lags" in self.config and self.config["precip_lags"]:
            for lag in self.config["precip_lags"]:
                loc_df[f"precip_lag_{lag}{location_suffix}"] = loc_df[precip_col].shift(lag)

        # Cloud cover lags
        if "cloud_lags" in self.config and self.config["cloud_lags"]:
            for lag in self.config["cloud_lags"]:
                loc_df[f"cloud_lag_{lag}{location_suffix}"] = loc_df[cloud_col].shift(lag)

        # Example of shortwave lag, if desired
        if "shortwave_lags" in self.config and self.config["shortwave_lags"]:
            for lag in self.config["shortwave_lags"]:
                loc_df[f"shortwave_lag_{lag}{location_suffix}"] = loc_df[shortwave_col].shift(lag)

        # 10. Drop raw features if requested
        # ----------------------------------------------------------
        if self.config.get("drop_raw_solar_features", False):
            loc_df.drop(
                columns=[f"{var}{location_suffix}" for var in OpenMeteo.vars_radiation
                         if f"{var}{location_suffix}" in loc_df.columns],
                inplace=True, errors="ignore"
            )
        if self.config.get("drop_raw_features", False):
            loc_df.drop(
                columns=[f"{var}{location_suffix}" for var in OpenMeteo.vars_basic
                         if f"{var}{location_suffix}" in loc_df.columns],
                inplace=True, errors="ignore"
            )

        return loc_df

    def _apply_spatial_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features across multiple solar farms using a specified spatial method.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame. It contains columns for each solar farm,
            each column having a unique suffix (e.g. '_S1', '_S2', etc.).

        Returns
        -------
        pd.DataFrame
            A DataFrame with aggregated features (one column per base feature).
        """

        if len(self.locations) <= 1:
            raise ValueError("Cannot apply spatial aggregation on a single location")

        method: str = self.config.get("spatial_agg_method")  # e.g. "mean", "max", "idw", "capacity", etc.
        all_features = df.columns.tolist()

        # 1. Group columns by the location suffix
        features_per_site = {}
        suffixes = []
        for loc in self.locations:
            suffix = loc["suffix"]
            suffixes.append(suffix)
            cols_with_suffix = [c for c in all_features if c.endswith(suffix)]
            features_per_site[suffix] = cols_with_suffix

        # 2. Build a mapping from "base feature name" -> columns for each suffix
        def strip_suffix(col_name: str, available_suffixes) -> str:
            for sfx in available_suffixes:
                if col_name.endswith(sfx):
                    return col_name.replace(sfx, "")
            return col_name  # if no match, return as-is

        base_feature_map = {}  # { base_feature: [ (suffix, col_name), ... ] }
        for loc in self.locations:
            suffix = loc["suffix"]
            for col_name in features_per_site[suffix]:
                base_feat = strip_suffix(col_name, [suffix])
                if base_feat not in base_feature_map:
                    base_feature_map[base_feat] = []
                base_feature_map[base_feat].append((suffix, col_name))

        # 3. Prepare location metadata
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

        # Haversine distance
        def haversine_distance(lat1, lon1, lat2, lon2):
            rlat1, rlon1, rlat2, rlon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = rlat2 - rlat1
            dlon = rlon2 - rlon1
            a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Earth radius in km
            return r * c

        # Centroid coordinates for IDW or distance-based
        mean_lat = np.mean([loc_meta[sfx]["lat"] for sfx in suffixes])
        mean_lon = np.mean([loc_meta[sfx]["lon"] for sfx in suffixes])

        # 4. Create a new DataFrame for aggregated features
        aggregated_df = pd.DataFrame(index=df.index)

        # 5. Apply the chosen aggregation method for each base feature
        for base_feat, columns_with_suffixes in base_feature_map.items():
            # Gather each suffix's time series
            sub_data = {}
            for (sfx, c_name) in columns_with_suffixes:
                sub_data[sfx] = df[c_name]
            sub_df = pd.DataFrame(sub_data, index=df.index)

            if method == "mean":
                aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)

            elif method == "max":
                aggregated_df[base_feat + "_agg"] = sub_df.max(axis=1)

            elif method == "idw":
                # Inverse Distance Weighting around (mean_lat, mean_lon)
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)  # avoid division by zero
                    weights[sfx] = 1.0 / (d_km**2)

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "capacity":
                # Weight by farm's capacity
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

            elif method == "n_panels":
                # Weight by the number of solar panels
                weights = {}
                for sfx in sub_df.columns:
                    weights[sfx] = loc_meta[sfx]["n_panels"]
                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "distance_capacity":
                # weight_i = capacity_i / distance_i^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    cap_i = loc_meta[sfx]["capacity"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = cap_i / (d_km**2)

                sum_weights = sum(weights.values())
                if sum_weights == 0:
                    aggregated_df[base_feat + "_agg"] = sub_df.mean(axis=1)
                else:
                    weighted_sum = pd.Series(0.0, index=df.index)
                    for sfx in sub_df.columns:
                        weighted_sum += sub_df[sfx] * weights[sfx]
                    aggregated_df[base_feat + "_agg"] = weighted_sum / sum_weights

            elif method == "distance_n_panels":
                # weight_i = n_panels_i / distance_i^2
                weights = {}
                for sfx in sub_df.columns:
                    lat_i = loc_meta[sfx]["lat"]
                    lon_i = loc_meta[sfx]["lon"]
                    n_pan = loc_meta[sfx]["n_panels"]
                    d_km = haversine_distance(mean_lat, mean_lon, lat_i, lon_i)
                    d_km = max(d_km, 0.001)
                    weights[sfx] = n_pan / (d_km**2)

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
        precip_lags_choice = trial.suggest_categorical(
            "precip_lags_option", ["none", "small", "large"]
        )
        if precip_lags_choice == "none":
            precip_lags = []
        elif precip_lags_choice == "small":
            precip_lags = [1, 6]
        else:
            precip_lags = [1, 6, 12, 24]

        # ----- New: Cloud cover lags -----
        cloud_lags_choice = trial.suggest_categorical(
            "cloud_lags_option", ["none", "small", "medium", "large"]
        )
        if cloud_lags_choice == "none":
            cloud_lags = []
        elif cloud_lags_choice == "small":
            cloud_lags = [1, 3]
        elif cloud_lags_choice == "medium":
            cloud_lags = [1, 3, 6]
        else:  # "large"
            cloud_lags = [1, 3, 6, 12]

        # ----- New: Shortwave lags -----
        shortwave_lags_choice = trial.suggest_categorical(
            "shortwave_lags_option", ["none", "small", "medium", "large"]
        )
        if shortwave_lags_choice == "none":
            shortwave_lags = []
        elif shortwave_lags_choice == "small":
            shortwave_lags = [1, 3]
        elif shortwave_lags_choice == "medium":
            shortwave_lags = [1, 3, 6]
        else:  # "large"
            shortwave_lags = [1, 3, 6, 12]

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

        sp_agg_config = {"method": spatial_agg_method}

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
            "precip_lags": precip_lags,
            "cloud_lags": cloud_lags,
            "shortwave_lags": shortwave_lags,

            # Dropping raw features
            "drop_raw_solar_features": drop_raw_solar_features,
            "drop_raw_features": drop_raw_features,

            # Spatial aggregation
            "spatial_agg_method": spatial_agg_method,
            "spatial_agg_config": sp_agg_config,
        }

        return config


def physics_informed_feature_engineering(df_hist_:pd.DataFrame, df_forecast_:pd.DataFrame, config:dict, verbose:bool):

    if config['feature_engineer'] == 'WeatherWindPowerFE':

        if verbose:print(f"Performing feature engineering with {config['feature_engineer']}")
        o_feat = WeatherWindPowerFE( config, verbose=verbose )
        n_h, n_f = len(df_hist_), len(df_forecast_)
        df_tmp = pd.concat([df_hist_,df_forecast_],axis=0)
        assert len(df_tmp) == n_h + n_f
        df_tmp = o_feat(df_tmp)
        df_tmp = validate_dataframe(df_tmp)
        df_hist_, df_forecast_ = df_tmp[:df_forecast_.index[0]-pd.Timedelta(hours=1)], df_tmp[df_forecast_.index[0]:]

        assert len(df_forecast_) == n_f
        # assert len(df_hist_) == n_h
        df_hist_ = df_hist_.dropna(inplace=False) # in case there are lagged features -- nans are introguded

    elif config['feature_engineer'] == 'WeatherSolarPowerFE':

        if verbose:print(f"Performing feature engineering with {config['feature_engineer']}")
        o_feat = WeatherSolarPowerFE( config, verbose=verbose )
        n_h, n_f = len(df_hist_), len(df_forecast_)
        df_tmp = pd.concat([df_hist_,df_forecast_],axis=0)
        assert len(df_tmp) == n_h + n_f
        df_tmp = o_feat(df_tmp)
        df_tmp = validate_dataframe(df_tmp)
        df_hist_, df_forecast_ = df_tmp[:df_forecast_.index[0]-pd.Timedelta(hours=1)], df_tmp[df_forecast_.index[0]:]

        assert len(df_forecast_) == n_f
        # assert len(df_hist_) == n_h
        df_hist_ = df_hist_.dropna(inplace=False) # in case there are lagged features -- nans are introguded

    elif config['feature_engineer'] is None:

        df_hist_ = pd.DataFrame(index=df_hist_.index)
        df_forecast_ = pd.DataFrame(index=df_forecast_.index)

    return df_hist_, df_forecast_
