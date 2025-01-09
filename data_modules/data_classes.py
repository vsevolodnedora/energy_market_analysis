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
from data_modules.utils import validate_dataframe
from data_modules.feature_eng import (
    create_holiday_weekend_series,
    create_time_features,
    WeatherFeatureEngineer,
    WeatherFeatureEngineer_2
)



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

def suggest_values_for_ds_pars_optuna(feature_engineering_pipeline, trial, fixed:dict):
    config = {}

    if feature_engineering_pipeline == 'WeatherFeatureEngineer':
        config.update( WeatherFeatureEngineer_2.selector_for_optuna(trial, fixed)  )

    if'log_target' in fixed:
        config['log_target'] = fixed['log_target']
    else:
        config['log_target'] = trial.suggest_categorical("log_target", [True, False])

    config['lags_target'] = trial.suggest_categorical(
        "lags_target", [None, 1, 6, 12]
    ) if not 'lags_target' in fixed else fixed['lags_target']

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
            o_feat = WeatherFeatureEngineer_2( config, verbose=self.verbose )
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
            df_time_features = create_time_features(index=df_target_.index)
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
            df_time_features = create_time_features(index=exog_forecast.index)
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