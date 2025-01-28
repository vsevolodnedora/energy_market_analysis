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
    physics_informed_feature_engineering,
    WeatherWindPowerFE,
    WeatherSolarPowerFE,
    WeatherLoadFE
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

def suggest_values_for_ds_pars_optuna(feature_engineering_pipeline:str, trial:optuna.Trial, fixed:dict):
    config = {}

    if feature_engineering_pipeline == 'WeatherWindPowerFE':
        config.update( WeatherWindPowerFE.selector_for_optuna( trial, fixed )  )
    elif feature_engineering_pipeline == 'WeatherSolarPowerFE':
        config.update( WeatherSolarPowerFE.selector_for_optuna( trial, fixed )  )
    elif feature_engineering_pipeline == 'WeatherLoadFE':
        config.update( WeatherLoadFE.selector_for_optuna( trial, fixed )  )
    else:
        raise ValueError(f"Unknown feature engineering pipeline: {feature_engineering_pipeline}")

    if'log_target' in fixed: config['log_target'] = fixed['log_target']
    else: config['log_target'] = trial.suggest_categorical("log_target", [True, False])

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
            df_historic:pd.DataFrame,
            df_forecast:pd.DataFrame or None,
            pars:dict
    ):

        expected_range = pd.date_range(start=df_historic.index.min(), end=df_historic.index.max(), freq='h')
        if not df_historic.index.equals(expected_range):
            raise ValueError("full_index must be continuous with hourly frequency.")


        self.targets_list : list = pars['targets']

        self.verbose = pars['verbose']
        if len(self._targets_list) == 1 and len(df_historic.columns) - len(df_forecast.columns) != 1:
            raise ValueError(
                f"For one target {self._targets_list[0]} the "
                f"df_histric should have exactly 1 extra column with respect to df_forecast."
                f"Found that df_historic has {len(df_historic.columns)} columns, "
                f"while df_forecast has {len(df_forecast.columns)} columns."
            )


        if pars['copy_input']:
            if pars['verbose']:print("Copyting df_historic and df_forecast for dataclass")
            self.df_historic_ = copy.deepcopy(
                df_historic[[col for col in df_historic.columns if not col in list(self._targets_list)]]
            )
            self.df_target_ : pd.DataFrame = copy.deepcopy( df_historic[self._targets_list] )
            self.df_forecast_ : pd.DataFrame = copy.deepcopy( df_forecast )
        else:
            self.df_historic_ : pd.DataFrame = df_historic[[
                col for col in df_historic.columns if not col in list(self._targets_list)
            ]]
            self.df_target_ : pd.DataFrame = df_historic[self._targets_list]
            self.df_forecast_ : pd.DataFrame = df_forecast


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

        if not compare_columns(
                df_historic[[col for col in df_historic.columns if not col in list(self._targets_list)]],
                df_forecast
        ):
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

    def process_data(self, config:dict, df_hist_:pd.DataFrame, df_forecast_:pd.DataFrame, df_target_:pd.DataFrame) \
            ->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if not isinstance(df_target_, pd.DataFrame):
            raise ValueError("df_target_ must be a pandas DataFrame")

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

        self.do_log_target = config['log_target']
        if self.do_log_target:
            # Add small constant to prevent log(0)
            df_target_ = np.log10(df_target_ + 1e-8)
        else:
            df_target_ = df_target_

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
                print(f"Using pre-fitted scaler for targets_list={self._targets_list}")
            y_scaled = self.target_scaler.transform( df_target_ )
            df_target = pd.DataFrame(y_scaled, index=df_target_.index, columns=df_target_.columns)
        else:
            if self.verbose:
                print(f"Fitting scaler for targets_list={self._targets_list}")
            y_scaled = self.target_scaler.fit_transform(df_target_)
            df_target = pd.DataFrame(y_scaled, index=df_target_.index, columns=df_target_.columns)

        ''' --- Target Transformers --- '''

        self.transform_target = lambda data : (
            pd.DataFrame(
                self.target_scaler.transform(
                    (np.log10(data + 1e-8) if self.do_log_target else data)
                ) if self.target_scaler is not None else (
                    np.log10(data + 1e-8) if self.do_log_target else data
                ),
                index=data.index,
                columns=data.columns
            )
        )
        self.inv_transform_target = lambda data: (
            pd.DataFrame(
                self.target_scaler.inverse_transform(data),
                index=data.index,
                columns=data.columns
            ).pipe(lambda df: 10 ** df - 1e-8 if self.do_log_target else df)
        )

        ''' ---- add physics-informed features ----- '''
        df_hist_, df_forecast_ = physics_informed_feature_engineering(df_hist_, df_forecast_, config, self.verbose)

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
                for target_ in df_target_.columns:
                    exog[f'{target_}_lag_{lag}'] = df_target[target_].shift(lag)

            # Drop rows with NaN values due to lagging and save the final dataframe for forecasting later
            exog.dropna(inplace=True)

            # adjust the target column if lags were added (remove rows with nans)
            df_target = df_target.loc[exog.index]

        else:
            self.lags_target = None

        exog = _adjust_dataframe_to_divisible_by_N( exog, len(df_forecast_), self.verbose )
        df_target = df_target.loc[exog.index]

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
                for target_ in df_target_.columns:
                    exog_forecast[f'{target_}_lag_{lag}'] = -1

        if not validate_dataframe_simple(exog_forecast):
            raise ValueError("Error in validating dataframe with engineered forecasted features")

        if not compare_columns(exog, exog_forecast):
            raise ValueError("Error in validating dataframe with engineered forecasted features")

        expected_range = pd.date_range(start=exog.index.min(), end=exog.index.max(), freq='h')
        if not exog.index.equals(expected_range):
            raise ValueError("exog must be continuous with hourly frequency.")

        for col in exog.columns:
            for target_ in self._targets_list:
                if str(col).startswith(target_) and not str(col).__contains__('lag'):
                    raise ValueError("Exogenous has target without lags.")

        return exog, exog_forecast, df_target

    def run_preprocess_pipeline(self, config:dict):
        self.df_exog_hist, self.df_exog_forecast, self.df_target_hist = self.process_data(
            config, self.df_historic_, self.df_forecast_, self.df_target_
        )

    def inverse_transform_targets(self, result:pd.DataFrame) -> pd.DataFrame:
        ''' column-wise inverse transformation separately for _actual, _fitted, _lower, _upper'''
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Inverse transformation result must be a pandas DataFrame")

        df_res = pd.DataFrame()

        for type in ["_actual", "_fitted", "_lower", "_upper"]:
            df_i = result[[f"{col}{type}" for col in self.targets_list]]

            if len(df_i) > 0:
                df_i_unscaled = self.inv_transform_target(df_i)
                if df_res.empty : df_res = df_i_unscaled.copy()
                else: df_res = pd.merge(df_res, df_i_unscaled, left_index=True, right_index=True, how='left')
        # df_res = df_res.reindex(columns=result.columns, fill_value=np.nan)
        return df_res

    def transform_targets(self, df:pd.DataFrame) -> pd.DataFrame:
        ''' column-wise inverse transformation separately for _actual, _fitted, _lower, _upper'''
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Inverse transformation result must be a pandas DataFrame")

        df_res = pd.DataFrame()

        for type in ["_actual", "_fitted", "_lower", "_upper"]:
            df_i = df[[f"{col}{type}" for col in self.targets_list]]
            df_i.rename(columns={f"{col}{type}" : f"{col}" for col in self.targets_list }, inplace=True)

            if len(df_i) > 0:
                df_i_unscaled = self.transform_target(df_i)
                df_i_unscaled.rename(columns={f"{col}" : f"{col}{type}" for col in self.targets_list }, inplace=True)
                if df_res.empty : df_res = df_i_unscaled.copy()
                else: df_res = pd.merge(df_res, df_i_unscaled, left_index=True, right_index=True, how='left')
        # df_res = df_res.reindex(columns=result.columns, fill_value=np.nan)
        return df_res

        # for target_ in self.targets_list:
        #     if f"{target_}_lower" in result.columns and f"{target_}_upper" in result.columns:
        #         df_i = result[[f"{target_}_actual", f"{target_}_fitted", f"{target_}_lower", f"{target_}_upper"]]
        #     else:
        #         df_i = result[[f"{target_}_actual", f"{target_}_fitted"]]
        #     df_i_inv = self.inv_transform_target()
        #
        #
        #
        # actual_detransformed_ = self.inv_transform_target(result[[f"{col}_actual" for col in self._targets_list]])
        # fitted_detransformed_ = self.inv_transform_target(result[[f"{col}_fitted" for col in  self._targets_list]])
        # result_detransformed = pd.merge(actual_detransformed_, fitted_detransformed_, left_index=True, right_index=True)
        # for target_ in self._targets_list:
        #
        #
        #
        # if f"_lower" in result.columns and f"_upper" in result.columns:
        #     lower_detransformed_=self.inv_transform_target(result[[f"{col}_lower" for col in  self._targets_list]])
        #     upper_detransformed_=self.inv_transform_target(result[[f"{col}_upper" for col in  self._targets_list]])
        #     result_detransformed = pd.merge(result_detransformed, lower_detransformed_, left_index=True, right_index=True)
        #     result_detransformed = pd.merge(result_detransformed, upper_detransformed_, left_index=True, right_index=True)
        # return result_detransformed

    # --------------- access to the dataset (class interface) ---------------
    @property
    def targets_list(self)->list:
        return self._targets_list
    @targets_list.setter
    def targets_list(self, value):
        """Setter for targets_list"""
        if not isinstance(value, list):  # Add validation if necessary
            raise ValueError("targets_list must be a list.")
        self._targets_list = value

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