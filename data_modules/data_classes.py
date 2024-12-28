import copy

import optuna
import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta
from math import radians, sin, cos
import joblib
import gc

from statsmodels.tsa.deterministic import DeterministicProcess, Fourier

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from data_collection_modules.utils import compare_columns, validate_dataframe_simple
from data_collection_modules.locations import locations
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

def _seasonal_imputation_with_historical_values(column:pd.Series, period:str='weekly')->pd.Series:
    """
    Replaces zero values in the specified column of a DataFrame with imputed values
    based on the historical average for that same time period in past cycles.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index and the column to be imputed.
    column (str): The name of the column in which to replace zero values.
    period (str): Imputation period, options are 'daily', 'weekly', or 'monthly'.

    Returns:
    pd.DataFrame: DataFrame with zero values replaced in the specified column.
    """
    column = column.copy()  # To avoid modifying the original DataFrame

    # Identify zero values
    zero_indices = column[column == 0].index
    if len(zero_indices) > 0:
        print(f'WARNING: There are {len(zero_indices)} zero values in the {column.name} column. Imputing using {period} average.')

    for idx in zero_indices:
        # Select similar historical periods based on the period setting
        if period == 'daily':
            mask = (column.index.hour == idx.hour) & (column.index < idx)
        elif period == 'weekly':
            mask = (column.index.dayofweek == idx.dayofweek) & (column.index.hour == idx.hour) & (column.index < idx)
        elif period == 'monthly':
            mask = (column.index.day == idx.day) & (column.index.hour == idx.hour) & (column.index < idx)
        else:
            raise ValueError("Period must be 'daily', 'weekly', or 'monthly'")

        # Get historical values for the same time period
        similar_period_values = column.loc[mask]

        # Calculate the mean of the historical values, ignoring zeros if they exist
        imputed_value = similar_period_values[similar_period_values != 0].mean()

        # Replace zero with the imputed value if it's not NaN, otherwise leave it as zero
        if not np.isnan(imputed_value):
            column.at[idx] = imputed_value

    return column

def _adjust_dataframe_to_divisible_by_N(df, N, verbose:bool):
    """
    Adjusts a DataFrame so that its row count is divisible by N by removing rows from the oldest (top) if necessary.

    Args:
    df (pd.DataFrame): Input DataFrame indexed by pd.Timestamp.
    N (int): The divisor for the row count.

    Returns:
    pd.DataFrame: Adjusted DataFrame with row count divisible by N.
    """
    # Check the number of rows in the DataFrame
    num_rows = len(df)

    # Calculate the remainder when the row count is divided by N
    remainder = num_rows % N

    # If remainder is not zero, remove rows starting from the oldest (top)
    if remainder != 0:
        rows_to_remove = remainder
        df = df.iloc[rows_to_remove:]
        if verbose:print(f"Cropping dataframe to be devisibile by {N}. "
                         f"N rows: {num_rows} -> {len(df)}. Removing {rows_to_remove}")

    return df

def _create_time_features(index)->pd.DataFrame:
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
        self.locations:list[dict] = [loc for loc in locations if loc['name'] in self.loc_names]
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

def suggest_values_for_ds_pars_optuna(feature_engineering_pipeline, trial, fixed:dict):
    config = {}
    if feature_engineering_pipeline == 'WeatherFeatureEngineer':
        config.update( WeatherFeatureEngineer.selector_for_optuna(trial)  )

    if'log_target' in fixed:
        config['log_target'] = fixed['log_target']
    else:
        config['log_target'] = trial.suggest_categorical("log_target", [True, False])

    config['lags_target'] = trial.suggest_categorical("lags_target", [None, 1, 6, 12]) if not 'lags_target' in fixed else fixed['lags_target']

    return config


class HistForecastDataset:
    '''
    Scale, impute, engineer and normalize the time-series
    '''

    def __init__(
            self,
            df_historic:pd.DataFrame, df_forecast:pd.DataFrame or None, pars:dict
    ):

        expected_range = pd.date_range(start=df_historic.index.min(), end=df_historic.index.max(), freq='h')
        if not df_historic.index.equals(expected_range):
            raise ValueError("full_index must be continuous with hourly frequency.")

        self.target_key = pars['target']
        self.verbose = pars['verbose']

        if pars['copy_input']:
            if pars['verbose']:print("Copyting df_historic and df_forecast for dataclass")
            self.df_historic_ = copy.deepcopy( df_historic[[col for col in df_historic.columns if col != self.target_key]] )
            self.df_target_ = copy.deepcopy( df_historic[self.target_key] )
            self.df_forecast_ = copy.deepcopy( df_forecast )
        else:
            self.df_historic_ = df_historic[[col for col in df_historic.columns if col != self.target_key]]
            self.df_target_ = df_historic[self.target_key]
            self.df_forecast_ = df_forecast

        self.original_features = copy.deepcopy( self.df_historic_.columns.tolist() )
        self.forecast_horizon = pars['forecast_horizon']

        if self.forecast_horizon is None and self.df_forecast_ is None:
            raise ValueError("Either forecast_horizon or df_forecast must be set")
        if not len(self.df_forecast_) % 24 == 0:
            raise ValueError(f"Horizon must be divisible by 24 (at least one day). Given {self.df_forecast_.shape}")

        # if no forecast data is given, create an empty dataframe to be filled later
        if self.df_forecast_ is None:
            self.df_forecast_ = pd.DataFrame(
                index = pd.date_range(start=self.df_target_.index[-1] + pd.Timedelta(hours=1),
                                      periods=self.forecast_horizon, freq=self.df_target_.index.freq)
            )

        # check if columns are the same
        if not compare_columns(df_historic[[col for col in df_historic.columns if col != self.target_key]], df_forecast):
            raise ValueError("df_historic and df_forecast must have same columns")

        # check if there are no nans or missing values or non-monotonicities
        is_df_valid = validate_dataframe_simple(self.df_historic_)
        if not is_df_valid:
            # todo implement imputing mechanism
            raise ValueError("Nans in the history dataframe")

        self.set_pars = copy.deepcopy(pars)
        self.curr_config = None

        # containers for the processed features and target variables
        self.df_exog_hist = None
        self.df_exog_forecast = None
        self.df_target_hist = None

    # ------------- PIPELINE ----------------

    def process_data(self, config:dict, df_hist_:pd.DataFrame, df_forecast_:pd.DataFrame, df_target_:pd.Series) \
            ->tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

        # CHECKS
        if not df_hist_.index.intersection(df_forecast_.index).empty:
            raise ValueError("Historic and forecast dataframes should not overlap")
        last_index_df1 = df_hist_.index[-1]
        first_index_df2 = df_forecast_.index[0]
        expected_next_index = last_index_df1 + pd.Timedelta(hours=1)
        if first_index_df2 != expected_next_index:
            raise ValueError("Forecast dataframe should start exactly one timestep after the historic index")
        if not len(df_forecast_) % 24 == 0:
            raise ValueError(f"Horizon must be divisible by 24 (at least one day). Given {len(df_forecast_)}")
        if not validate_dataframe_simple(df_hist_):
            raise ValueError(f"Historic dataframe failed to validate")
        if not validate_dataframe_simple(df_forecast_):
            raise ValueError(f"Forecast dataframe failed to validate")

        ''' ------- HISTORIC TARGET ------- '''

        # 1. TARGET

        self.do_log_target = config['log_target']
        if self.do_log_target:
            # Add small constant to prevent log(0)
            df_target = np.log10(df_target_ + 1e-8)
        else:
            df_target = df_target_

        target_scaler_name = config['target_scaler']
        do_load_scaler = False
        if target_scaler_name == 'StandardScaler': self.target_scaler = StandardScaler()
        elif target_scaler_name == 'MinMaxScaler': self.target_scaler = MinMaxScaler()
        elif target_scaler_name == 'MaxAbsScaler': self.target_scaler = MaxAbsScaler()
        elif target_scaler_name == 'RobustScaler': self.target_scaler = RobustScaler()
        else:
            if self.verbose: print(f"Loading target scaler: {target_scaler_name}")
            self.target_scaler = joblib.load(target_scaler_name) # dir + 'target_scaler.pkl'
            do_load_scaler = True
        if do_load_scaler:
            if self.verbose:
                print(f"Using pre-fitted scaler for target={self.target_key}")
            column = self.target_scaler.transform(df_target.values.reshape(-1, 1)).flatten()
            df_target = pd.Series( column, index=df_target.index )
        else:
            print(f"Fitting scaler for target={self.target_key}")
            column = self.target_scaler.fit_transform(df_target.values.reshape(-1, 1)).flatten()
            df_target = pd.Series( column, index=df_target.index )

        self.transform_target_series = lambda data: pd.Series(
            (
                self.target_scaler.transform(
                    (np.log10(data + 1e-8) if self.do_log_target else data).values.reshape(-1, 1)
                ).flatten()
                if self.target_scaler is not None
                else (np.log10(data + 1e-8) if self.do_log_target else data).values.reshape(-1, 1)
            ),
            index=data.index,
            name=data.name
        )

        self.inv_transform_target_series = lambda data: pd.Series(
            (
                (10 ** self.target_scaler.inverse_transform(data.values.reshape(-1, 1)).flatten() - 1e-8)
                if self.do_log_target
                else self.target_scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
            ),
            index=data.index,
            name=data.name
        )


        ''' ---- TARGET-AWARE FEATURE ENGINEERING ----- '''

        if config['feature_engineer'] == 'WeatherFeatureEngineer':
            if self.verbose:print(f"Performing feature engineering with {config['feature_engineer']}")
            o_feat = WeatherFeatureEngineer(config, verbose=self.verbose)
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

        expected_range = pd.date_range(start=df_hist_.index.min(), end=df_hist_.index.max(), freq='h')
        if not df_hist_.index.equals(expected_range):
            raise ValueError("df_hist_ must be continuous with hourly frequency.")

        ''' ---- BASIC FEATURE ENGINEERING AND HISTORIC FEATURES ----- '''

        exog = pd.DataFrame(index=df_hist_.index) # container for engineered / processed features

        do_exog_feature_engineering = len(df_hist_.columns) > 0
        if do_exog_feature_engineering:
            feature_scaler_name = config['feature_scaler']
            do_load_feature_scaler = False
            if feature_scaler_name == 'StandardScaler': self.feature_scaler = StandardScaler()
            elif feature_scaler_name == 'MinMaxScaler': self.feature_scaler = MinMaxScaler()
            elif feature_scaler_name == 'MaxAbsScaler': self.feature_scaler = MaxAbsScaler()
            elif feature_scaler_name == 'RobustScaler': self.feature_scaler = RobustScaler()
            else:
                if self.verbose: print(f"Loading feature scaler: {feature_scaler_name}")
                self.feature_scaler = joblib.load(feature_scaler_name) # dir + 'target_scaler.pkl'
                do_load_feature_scaler = True
            if do_load_feature_scaler:
                X_scaled = self.feature_scaler.transform(df_hist_)
                print(f"Using pre-fitted scaler for features")
            else:
                X_scaled = self.feature_scaler.fit_transform(df_hist_)
                print(f"Fitting scaler for {len(df_hist_.columns)} columns")
            df_hist_scaled = pd.DataFrame(X_scaled, index=df_hist_.index, columns=df_hist_.columns)
            exog = exog.merge(df_hist_scaled, left_index=True, right_index=True, how='left')
        else:
            self.feature_scaler = None

        if config['add_cyclical_time_features']:
            df_time_features = _create_time_features(index=df_target_.index)
            exog = exog.merge(df_time_features, left_index=True, right_index=True, how='left')

        if 'fourier_features' in config.keys() and not config['fourier_features'] is None:
            fourier_features = Fourier(**config['fourier_features'])#(period=24, order=3)
            dp = DeterministicProcess(
                index=df_target_.index,
                constant=True,  # True - Include constant term for better stability
                order=1, # 1 Include linear trend
                seasonal=False,
                additional_terms=[fourier_features],
                drop=True,
            )
            df_fourier_features = dp.in_sample()
            df_fourier_features.drop(labels=['const','trend'],inplace=True,axis=1)
            exog = exog.merge(df_fourier_features, left_index=True, right_index=True, how='left')

        if 'lags_target' in config.keys() and not config['lags_target'] is None:
            self.lags_target = config['lags_target']
            for lag in range(1, self.lags_target + 1):
                exog[f'{self.target_key}_lag_{lag}'] = df_target.shift(lag)

            # Drop rows with NaN values due to lagging and save the final dataframe for forecasting later
            exog.dropna(inplace=True)

            # adjust the target column if lags were added (remove rows with nans)
            df_target = df_target[exog.index]

        else:
            self.lags_target = None

        exog = _adjust_dataframe_to_divisible_by_N( exog, len(df_forecast_), self.verbose )
        df_target = df_target[exog.index]

        if not validate_dataframe_simple(exog):
            raise ValueError("Error in validating dataframe with engineered features")

        ''' ---- BASIC FEATURE ENGINEERING AND FUTURE FEATURES ----- '''

        exog_forecast = pd.DataFrame(index=df_forecast_.index)
        if do_exog_feature_engineering:
            X_scaled = self.feature_scaler.transform(df_forecast_)
            df_forecast_scaled = pd.DataFrame(X_scaled, index=df_forecast_.index, columns=df_forecast_.columns)
            exog_forecast = exog_forecast.merge(df_forecast_scaled, left_index=True, right_index=True, how='left')

        if config['add_cyclical_time_features']:
            df_time_features = _create_time_features(index=exog_forecast.index)
            exog_forecast = exog_forecast.merge(df_time_features, left_index=True, right_index=True, how='left')

        if 'fourier_features' in config.keys() and not config['fourier_features'] is None:
            df_fourier_features = dp.out_of_sample(steps=len(df_forecast_))
            df_fourier_features.drop(labels=['const','trend'],inplace=True,axis=1)
            exog_forecast = exog_forecast.merge(df_fourier_features, left_index=True, right_index=True, how='left')

        if 'lags_target' in config.keys() and not config['lags_target'] is None:
            for lag in range(1, self.lags_target+1):
                exog_forecast[f'{self.target_key}_lag_{lag}'] = -1

        if not validate_dataframe_simple(exog_forecast):
            raise ValueError("Error in validating dataframe with engineered forecasted features")

        if not compare_columns(exog, exog_forecast):
            raise ValueError("Error in validating dataframe with engineered forecasted features")

        expected_range = pd.date_range(start=exog.index.min(), end=exog.index.max(), freq='h')
        if not exog.index.equals(expected_range):
            raise ValueError("exog must be continuous with hourly frequency.")


        return exog, exog_forecast, df_target

    def run_preprocess_pipeline(self, config:dict):
        self.df_exog_hist, self.df_exog_forecast, self.df_target_hist = self.process_data(
            config, self.df_historic_, self.df_forecast_, self.df_target_
        )

    # --------------- access to the dataset (class interface) ---------------
    @property
    def init_pars(self)->dict:
        return self.set_pars
    @property
    def hist_idx(self)->pd.DatetimeIndex:
        '''
:return: pd.DatetimeIndex -- index column for the dataframe with historical data
'''
        return self.df_target_hist.index
    @property
    def target_hist(self)->pd.Series:
        '''
        :return: pd.Series with transformed (scaled, normalized) target values from df_hist, scaled
        '''
        return self.df_target_hist
    @property
    def exog_hist(self) ->pd.DataFrame:
        '''
        :return: pd.Dataframe[time, features] dataframe with exogenous features for df_hist, scaled and extended
        '''
        return self.df_exog_hist
    @property
    def forecast_idx(self)->pd.DatetimeIndex:
        '''
:return: pd.DatetimeIndex -- index column for the dataframe with forecasted data
'''
        return self.df_exog_forecast.index
    @property
    def exog_forecast(self)->pd.DataFrame:
        '''
        :return: [time, features] dataframe with exogenous features for df_forecast, scaled and extended
        '''
        return self.df_exog_forecast
    def reset_engineered(self):
        del self.df_exog_hist; self.df_exog_hist = None
        del self.df_target_hist; self.df_target_hist = None
        del self.df_exog_forecast; self.df_exog_forecast = None
        del self.curr_config; self.curr_config = None
        self.lags_target = None
        gc.collect()


if __name__ == '__main__':
    # todo add test
    pass