import copy
import joblib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import ElasticNetCV, ElasticNet, Ridge
from mapie.regression import MapieRegressor
import xgboost as xgb
import shap
import logging
import holidays
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from logger import get_logger
logger = get_logger(__name__)

class BaseMultiTargetForecaster:

    def __init__(self, targets:list, alpha:float=0.05, verbose:bool=False):
        self.targets = targets
        self.alpha = alpha
        self.name = 'BaseClass'
        self.model : CatBoostRegressor or MultiOutputRegressor
        self.verbose = verbose

    def predict(self, X_scaled:pd.DataFrame, y_scaled:pd.DataFrame or None) -> pd.DataFrame:
        '''
        Compute trained model predictions for each timestep in the given train data set.
        :param X_scaled: time-series dataframe with exogenous variables. (same as used for fitting)
        :param y_scaled: time-series dataframe with target. (same as used for fitting)
        :return: pd.DataFrame with columns:
            [f'{target}_actual',f'{target}_fitted',f'{target}_lower',f'{target}_upper'] where the last two
        correspond to 95% confidence intervals.
        '''
        # predict with model
        res = self.model.predict(X_scaled)

        # lower = np.array( pis[:, 0, 0] )
        # upper = np.array( pis[:, 1, 0] )

        # form results

        res_df = {}
        for i, target_ in enumerate(self.targets):
            if y_scaled is None:
                _y = pd.Series([np.nan] * len(X_scaled), index=X_scaled.index)
            else:
                _y = y_scaled[target_]
            res_df[f'{target_}_actual'] = _y.values
            res_df[f'{target_}_fitted'] = res[:, i]
            res_df[f'{target_}_lower'] = np.zeros_like(res[:, i])
            res_df[f'{target_}_upper'] = np.zeros_like(res[:, i])

        results = pd.DataFrame(res_df, index=X_scaled.index)
        return results

    def fit(self, X_scaled:pd.DataFrame, y_scaled:pd.DataFrame)->None:
        '''

        :param X_scaled: dataframe with scaled, normalized features
        :param y_scaled: dataframe with scaled, normalized targets (multiple targets)
        :return: None
        '''
        # Check if base model is pre-fitted
        if not hasattr(self.model, "booster_") or (hasattr(self.model, "estimators_") and self.model.estimators_):
            if self.verbose: logger.info(f"Base model {self.name} is not fitted. Fitting using X={X_scaled.shape}")
            self.model.fit(X_scaled, y_scaled)
        if len(X_scaled) == 0 or len(y_scaled) == 0:
            raise ValueError(
                f"Empty dataframe is passed for training: "
                f"X_scaled={X_scaled.shape} and y_scaled={y_scaled.shape}"
            )

        # fit the CatBoostRegressor model (using multiRMSE for multi-target forecasting)
        self.model.fit(X_scaled, y_scaled)

    def forecast_window(self, X_test: pd.DataFrame, y_train_scaled: pd.DataFrame or None, lags_target: int or None ) \
            -> pd.DataFrame:
        """
        Compute trained model forecast for each timestep in the given test data set,
        *iteratively* updating any lagged target features if needed.

        :param X_test: time-series dataframe (forecast horizon) with exogenous + (possibly) lagged target columns.
        :param y_train_scaled: historical target data used to fill initial lags. Must have columns = self.targets.
        :param lags_target: number of target lags used in the model features. If None, no iterative procedure is required.
        :return: pd.DataFrame with columns:
            [
                f'{target}_actual',
                f'{target}_fitted',
                f'{target}_lower',
                f'{target}_upper'
            ]
            for each target in self.targets.
        """
        # If no lagged features are needed, we can simply call predict on X_test
        if lags_target is None:
            return self.predict(X_test, None)

        if y_train_scaled is None:
            raise ValueError("Requires y_train_scaled for lagged target(s).")

        # Make sure y_train_scaled has all the required target columns
        missing_targets = set(self.targets) - set(y_train_scaled.columns)
        if missing_targets:
            raise ValueError(
                f"y_train_scaled is missing required target columns: {missing_targets}"
            )

        # --------------------------------------------------------------------------
        # Optional: Check time continuity to prevent data leakage
        # --------------------------------------------------------------------------
        # For example, ensure the last timestamp in y_train_scaled plus one step
        # matches the first timestamp in X_test. This is optional and depends on
        # how your data/time indices are arranged.

        # time_delta = y_train_scaled.index[-1] - y_train_scaled.index[-2]
        # if y_train_scaled.index[-1] + time_delta != X_test.index[0]:
        #     raise ValueError(
        #         f"Time index mismatch: y_train_scaled ends at {y_train_scaled.index[-1]}, "
        #         f"but X_test starts at {X_test.index[0]}."
        #     )

        # --------------------------------------------------------------------------
        # Iterative Forecasting
        # --------------------------------------------------------------------------
        # We will store the predictions for each step and each target.
        forecast_values = []   # Will be a list of shape (len(X_test), n_targets)
        X_futures = []

        # Loop over each future time step
        for i in range(len(X_test)):
            # We copy the row of features for the i-th forecast step
            X_future = X_test.iloc[[i]].copy(deep=True)

            # Update lag features for each target, if needed
            for t_idx, target_ in enumerate(self.targets):
                for lag in range(1, lags_target + 1):
                    lag_col = f"{target_}_lag_{lag}"
                    if lag_col not in X_future.columns:
                        raise ValueError(
                            f"Column '{lag_col}' not found in X_test. "
                            "All lag features must match model training features."
                        )

                    # If we have already forecasted beyond the "training window" for this lag,
                    # then use the predicted values from previous steps
                    if i - lag >= 0:
                        # Use the forecast from the (i - lag)-th time step for this target
                        X_future.at[X_future.index[0], lag_col] = forecast_values[i - lag][t_idx]
                    else:
                        # Otherwise, use the historical y_train_scaled for initial lags
                        # Make sure to index the correct location:
                        X_future.at[X_future.index[0], lag_col] =\
                            y_train_scaled[target_].iloc[y_train_scaled.shape[0] - lag + i]


            # Now predict for this single time step (which includes the updated lags)
            fcst = self.model.predict(X_future)  # shape (1, n_targets)
            # Store the row of predictions (fcst[0] is shape (n_targets,))
            forecast_values.append(fcst[0])
            X_futures.append(X_future)

        # Convert our list of forecasts to a NumPy array of shape (len(X_test), n_targets)
        forecast_array = np.array(forecast_values)

        # --------------------------------------------------------------------------
        # Build the Result DataFrame
        #   - For multi-target: create columns for each target
        #   - Fill actual with NaN (since future actual is often unknown),
        #     or fill with any known values if you do have them
        #   - Lower/Upper set to zero in this example (no intervals)
        # --------------------------------------------------------------------------
        res_df = {}
        for t_idx, target_ in enumerate(self.targets):
            # In many forecasting scenarios, you won't have actual future values,
            # so we set them to NaN. If you do know the actuals, you could pass them here.
            res_df[f"{target_}_actual"] = [np.nan] * len(X_test)

            res_df[f"{target_}_fitted"] = forecast_array[:, t_idx]

            # Dummy placeholders; replace with real intervals if you have them
            res_df[f"{target_}_lower"] = np.zeros_like(forecast_array[:, t_idx])
            res_df[f"{target_}_upper"] = np.zeros_like(forecast_array[:, t_idx])

        forecast_df = pd.DataFrame(res_df, index=X_test.index)

        # Optionally store the final "X_futures" used for each step
        # so you can inspect them or compute feature importance if desired
        self.X_futures_df = pd.concat(X_futures, axis=0)
        self.X_futures_df.index = X_test.index

        return forecast_df

    def save_model(self, file_path: str):
        """
        Save the trained model to a file using joblib.
        """
        # if hasattr(self.model.estimator, "fit"):
        #     raise ValueError("The model does not appear to be trained.")

        joblib.dump(self.model, file_path)
        # print(f"Model saved to {file_path}")

    def load_model(self, file_path: str):
        """
        Load a trained model from a file using joblib.
        """
        self.model = joblib.load(file_path)
        # print(f"Model loaded from {file_path}")

    def reset_model(self):
        del self.model; self.model = None
        # del self.X_futures_df;  self.X_futures_df = pd.DataFrame()



class MultiTargetCatBoostRegressor(BaseMultiTargetForecaster):

    def __init__(self, targets: list, model: CatBoostRegressor, alpha: float, verbose: bool): # lags_target:int or None,

        super().__init__(targets, alpha, verbose)

        self.name='MultiTargetCatBoostRegressor'
        self.model = model


class MultiTargetLGBMMultiTargetForecaster(BaseMultiTargetForecaster):

    def __init__(self, targets: list, model: MultiOutputRegressor, alpha: float, verbose: bool): # lags_target:int or None,

        super().__init__(targets, alpha, verbose)

        self.name='MultiTargetLGBMMultiTargetForecaster'
        self.model = model



def instantiate_base_multitarget_forecaster(model_name:str, targets:list, model_pars:dict, verbose:bool) \
        -> BaseMultiTargetForecaster:
    # if 'l1_ratio' in model_pars: del model_pars['l1_ratio']
    # train the forecasting model several times to evaluate its performance, get all results
    if model_name == 'MultiTargetElasticNet':
        extra_pars = {}
        return MultiTargetLGBMMultiTargetForecaster(
            model=MultiOutputRegressor(ElasticNet(**(model_pars | extra_pars))),
            targets=targets, alpha=0.05, verbose=verbose
        )

    elif model_name == 'MultiTargetCatBoost':
        if len(targets) > 1: extra_pars =  {
            'loss_function': 'MultiRMSE', 'eval_metric': 'MultiRMSE',
            "allow_writing_files":False,"silent":True
        }
        else: extra_pars = {
            'loss_function': 'RMSE', 'eval_metric': 'RMSE',
            "allow_writing_files":False,"silent":True
        }
        return MultiTargetCatBoostRegressor(
            model=CatBoostRegressor(**(model_pars | extra_pars)),  # Multivariate regression objective}),
            targets=targets, alpha=0.05, verbose=verbose
        )

    elif model_name == 'MultiTargetLGBM':
        extra_pars = {'importance_type': 'gain', "verbose":-1} # Use 'gain' importance for feature selection
        return MultiTargetLGBMMultiTargetForecaster(
            model=MultiOutputRegressor(LGBMRegressor(**(model_pars | extra_pars))),
            targets=targets, alpha=0.05, verbose=verbose
        )

    else:
        raise NotImplementedError(f"Fine-tuning parameter set for {model_name} not implemented")