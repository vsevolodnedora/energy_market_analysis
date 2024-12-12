
import copy, re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import os.path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNetCV, ElasticNet, Ridge
from mapie.regression import MapieRegressor
import xgboost as xgb
import shap
from datetime import datetime, timedelta
import holidays
from sklearn.decomposition import PCA
import matplotlib.dates as mdates
from mapie.regression import MapieRegressor
import json
import csv
import gc
import inspect
import optuna
import joblib
import pickle
from prophet import Prophet
from sklearn.preprocessing import StandardScaler

from sklearn.utils.validation import check_is_fitted

from scipy.stats import friedmanchisquare

from forecasting_modules.utils import (
    compute_timeseries_split_cutoffs,
    compute_error_metrics,
    compute_error_metrics_aggregate_over_horizon,
    compute_error_metrics_aggregate_over_cv_runs
)
from data_modules.data_classes import (
    validate_dataframe,
    HistForecastDatasetBase,
    HistForecastDataset
)
from data_modules.data_vis import plot_time_series_with_residuals
from forecasting_modules.ensemble_model import (
    EnsembleForecaster
)
from forecasting_modules.base_models import (
    BaseForecaster,
    XGBoostMapieRegressor,
    ElasticNetMapieRegressor,
    ProphetForecaster
)

def load_data(target:str,limit_train_to:timedelta=timedelta(days=365)) \
        ->tuple[pd.DataFrame, pd.DataFrame]:
    datapath = '../tmp_database/'
    df_history = pd.read_parquet(f"{datapath}history.parquet")
    df_forecast = pd.read_parquet(f"{datapath}forecast.parquet")

    # for brevity and due to evolving market conditions we focus here only on 1 year of data
    # df_history = df_history[pd.Timestamp(df_history.dropna(how='any', inplace=False).last_valid_index()) - limit_train_to:]

    # assure that the columns in both dataframes match
    df_features = df_history[[col for col in list(df_history.columns) if col != target]]
    if not df_features.columns.equals(df_forecast.columns):
        raise IOError("The DataFrames have different columns.")

    print(f"History: {df_history.shape} from {df_history.index[0]} to {df_history.index[-1]} ({len(df_history.index)/7/24} weeks)")
    print(f"Forecast: {df_forecast.shape} from {df_forecast.index[0]} to {df_forecast.index[-1]} ({len(df_forecast.index)/24} days)")

    return df_history, df_forecast

def _get_weather_features(df_history:pd.DataFrame):
    # list of openmeteo feature names (excluding suffix added for different locations)
    patterns = [
        'cloud_cover', 'precipitation', 'relative_humidity_2m', 'shortwave_radiation',
        'surface_pressure', 'temperature_2m', 'wind_direction_10m', 'wind_gusts_10m', 'wind_speed_10m'
    ]
    # Use the filter method to get columns that match the regex pattern
    weather_columns = df_history.filter(
        regex='|'.join([f"{pattern}_(fran|hsee|mun|solw|stut)" for pattern in patterns])
    ).columns.tolist()
    # Display or use the filtered columns as needed
    print(f"Weather features found {len(weather_columns)}")
    return weather_columns

def save_optuna_results(study:optuna.Study, extra_pars:dict, outdir:str):
    """
    Save all results from an Optuna study to multiple formats.

    Args:
        study (optuna.study.Study): The completed Optuna study to save.
        outdir (str): Outdir for the files to create.

    Files created:
        - {outdir}_best_parameters.json: Best parameters in JSON format.
        - {outdir}_best_trial_details.json: Details of the best trial in JSON format.
        - {outdir}_best_parameters.csv: Best parameters in CSV format.
        - {outdir}_complete_study_results.csv: Full study results in CSV format.
    """

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Save best parameters to JSON
    best_params = study.best_params
    if extra_pars: best_params = best_params | extra_pars
    with open(f'{outdir}best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    # Save details of the best trial to JSON
    best_trial = study.best_trial
    best_trial_details = {
        'trial_id': best_trial.number,
        'best_value': study.best_value,
        'params': best_params
    }
    with open(f'{outdir}best_trial_details.json', 'w') as f:
        json.dump(best_trial_details, f, indent=4)

    # Save best parameters to CSV
    with open(f'{outdir}_best_parameters.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        for key, value in best_params.items():
            writer.writerow([key, value])

    # Convert the complete study results to a DataFrame and save to CSV
    results_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    results_df.to_csv(f'{outdir}_complete_study_results.csv', index=False)

def get_parameters_for_optuna_trial(model_name, trial:optuna.trial):

    if model_name == 'XGBoost':
        param = {
            # 'objective': self.optim_pars['objective'],#'reg:squarederror',
            # 'eval_metric': self.optim_pars['eval_metric'],#'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }

    elif model_name == 'ElasticNet':
        param = {
            'l1_ratio': trial.suggest_float('l1_ratio', 1e-5, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        }

    elif model_name == 'Prophet':
        param = {
            # 'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
            # 'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
            # 'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
            'n_changepoints': trial.suggest_int('n_changepoints', 5, 100),
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
            'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False]),
        }

    else:
        raise NotImplementedError(f"Fine-tuning parameter set for {model_name} not implemented")

    return param

def instantiate_base_forecaster(model_name:str, target:str, lags_target:int or None, model_pars:dict, verbose:bool)->BaseForecaster:
    # train the forecasting model several times to evaluate its performance, get all results
    if model_name == 'XGBoost':
        return XGBoostMapieRegressor(
            model=MapieRegressor(
                xgb.XGBRegressor(**model_pars),
                method='naive', cv='prefit'#TimeSeriesSplit(n_splits=5)
            ), target=target, alpha=0.05, lags_target=lags_target, verbose=verbose)

    elif model_name == 'ElasticNet':
        return ElasticNetMapieRegressor(
            model=MapieRegressor(
                ElasticNet(**(model_pars | {'max_iter':10000, 'tol':1e-2})),
                method='naive', cv='prefit'#TimeSeriesSplit(n_splits=5)
            ), target=target, alpha=0.05, lags_target=lags_target, verbose=verbose)

    elif model_name == 'Prophet':
        return ProphetForecaster( params = model_pars, target=target, alpha=0.05, verbose=verbose)

    else:
        raise NotImplementedError(f"Fine-tuning parameter set for {model_name} not implemented")

def get_ts_cutoffs(ds:HistForecastDataset,folds:int):
    if ds is None:
        raise ReferenceError("Dataset class is not initialized")
    horizon = len(ds.get_forecast_index()) # number of timesteps to forecast
    if not horizon % 24 == 0:
        raise ValueError(f"Horizon must be divisible by 24 (at least one day). Given {horizon}")
    cutoffs = compute_timeseries_split_cutoffs(
        ds.get_index(),
        horizon=horizon,
        delta=horizon,
        folds=folds,
        min_train_size=3*30*24
    )
    return cutoffs

def get_ensemble_name_and_model_names(model_name:str):
    match = re.search(r'\[(.*?)\]', model_name)
    if match: meta_model = match.group(1)
    else: raise NameError(f"Model name {model_name} does not contain '[meta_model_name]' string")

    # extract base-models names
    match = re.search(r'\((.*?)\)', model_name)
    if match: model_names = match.group(1).split(',')  # Split by comma
    else: raise NameError(f"Model name {model_name} does not contain '(model_name_1,model_name_2)' string")

    return meta_model, model_names

class TaskPaths:

    def __init__(self, target:str, model_label:str, working_dir:str,verbose:bool):
        self.working_dir = working_dir
        self.model_label = model_label
        self.target = target
        self.verbose = verbose
        self.finetuned_name = 'finetuning'
        self.trained_name = 'trained'
        self.forecast_name = 'forecast'

        if not os.path.isdir(working_dir):
            raise FileNotFoundError(f"Working directory does not exists {self.working_dir}")
        self.to_target = self.working_dir + self.target + '/'
        self._get_create(self.to_target)

        if self.model_label.__contains__('ensemble'):
            self.name_base_model, self.names_base_models = get_ensemble_name_and_model_names(self.model_label)
            self.model_label = 'meta_' + self.name_base_model + ''.join(['_' + m for m in self.names_base_models]) + '/'
        else:
            self.name_base_model = self.model_label
            self.model_label = self.model_label

        self.to_model = self.working_dir + self.target + '/' + self.model_label + '/'

        self._get_create(self.to_model)

    def _get_create(self, dir:str)->None:
        if not os.path.isdir(dir):
            if self.verbose: print(f"Creating {dir}")
            os.mkdir(dir)

    def to_target(self):
        return self.to_target

    def to_model(self):
        return self.to_model

    def to_finetuned(self):
        dir = self.to_model + self.finetuned_name + '/'
        self._get_create(dir)
        return dir

    def to_trained(self):
        dir = self.to_model + self.trained_name + '/'
        self._get_create(dir)
        return dir

    def to_forecast(self):
        dir = self.to_model + self.forecast_name + '/'
        self._get_create(dir)
        return dir

    def to_dir(self,dir:str)->str:
        if dir == 'finetuning': return self.to_finetuned()
        if dir == 'trained': return self.to_trained()
        elif dir == 'forecast': return self.to_forecast()
        else: raise ValueError(f"Directory {dir} is not supported. Expected 'trained' or 'forecast'")

class BaseModelTasks(TaskPaths):

    def __init__(self, target: str, model_label: str, working_dir: str, verbose: bool = False):

        super().__init__(target, model_label, working_dir, verbose)

        self.X_for_model = None
        self.y_for_model = None
        self.verbose = verbose
        self.ds = None
        self.forecaster = None

        self.results = {}
        self.metrics = {}
        self.contributions = {}
        self.cutoffs = []

        self.model_dataset_pars = None
        self.model_pars = None

    def clear(self):
        del self.X_for_model
        del self.y_for_model
        del self.results
        del self.metrics
        del self.contributions
        gc.collect()

    def set_dataset_from_ds(self, ds:HistForecastDatasetBase):
        self.ds = ds

    def set_dataset_from_df(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame, pars:dict):
        if len(df_hist.columns)-1 != len(df_forecast.columns):
            raise ValueError(f'df_hist and df_forecast must have same number of columns')
        if len(df_hist.columns) == 0 or len(df_forecast.columns) == 0:
            raise ValueError(f'df_hist and df_forecast must have at least one column')
        pars = pars | {'verbose':self.verbose}
        self.ds = HistForecastDataset( df_hist=df_hist, df_forecast=df_forecast, **pars )

    def set_dataset_from_finetuned(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame)->tuple[dict,dict]:

        dir = self.to_finetuned()
        if self.verbose:
            print(f"Setting dataset for base model {self.model_label} from finetuning directory {dir}")

        with open(dir+'dataset.json', 'r') as f:
            self.model_dataset_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.model_dataset_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.model_dataset_pars.items() if key in accepted_params}
        accepted_params = accepted_params | {'verbose':self.verbose}
        self.ds = HistForecastDataset( df_hist=df_hist, df_forecast=df_forecast, **accepted_params )

        return accepted_params, unexpected_params

    def set_load_dataset_from_dir(self, dir:str, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Setting dataset for base model {self.model_label} from directory {dir}")

        with open(dir+'dataset.json', 'r') as f:
            self.model_dataset_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.model_dataset_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.model_dataset_pars.items() if key in accepted_params}

        target_scaler = joblib.load(dir + 'target_scaler.pkl')

        feature_scaler = joblib.load(dir + 'feature_scaler.pkl')

        ds_pars = accepted_params | {
            'target_scaler':target_scaler, 'feature_scaler':feature_scaler, 'verbose':self.verbose
        }
        self.ds = HistForecastDataset( df_hist=df_hist, df_forecast=df_forecast, **ds_pars )

        return accepted_params , unexpected_params

    def set_forecaster_from_dir(self, dir:str) -> tuple[dict,dict]:

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Setting base forecaster {self.model_label} from directory {dir}")

        with open(dir+'best_parameters.json', 'r') as f:
            self.model_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.model_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.model_pars.items() if key in accepted_params}

        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.ds.target, lags_target=self.ds.lags_target,
            model_pars=accepted_params, verbose=self.verbose
        )

        return accepted_params, unexpected_params

    def load_forecaster_from_dir(self, dir:str) -> tuple[dict,dict]:

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Loading base forecaster {self.model_label} from directory {dir}")

        if self.ds is None:
            raise ReferenceError(f"Dataset class for base forecaster {self.model_label} is not initialized")

        with open(dir+'best_parameters.json', 'r') as f:
            self.model_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.model_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.model_pars.items() if key in accepted_params}

        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.ds.target, lags_target=self.ds.lags_target,
            model_pars=accepted_params, verbose=self.verbose
        )
        self.forecaster.load_model(dir + 'model.joblib')

        self.forecaster.lag_y_past = self.ds.get_target_transformed()

        return accepted_params, unexpected_params

    # def set_ts_cutoffs(self, folds:int):
    #     self.cutoffs = get_ts_cutoffs(self.ds, folds=folds)

    # def get_create_finetuning_dir(self)->str:
    #     # create folder for the hyperparameter search
    #     if not os.path.isdir(self.model_dir):
    #         os.makedirs(self.model_dir)
    #
    #     outdir__ = self.model_dir + 'finetuning/'
    #     if not os.path.isdir(outdir__):
    #         os.makedirs(outdir__)
    #
    #     return outdir__

    # def get_create_trained_dir(self)->str:
    #     # create folder for the hyperparameter search
    #     if not os.path.isdir(self.model_dir):
    #         os.makedirs(self.model_dir)
    #
    #     outdir__ = self.model_dir + 'trained/'
    #     if not os.path.isdir(outdir__):
    #         os.makedirs(outdir__)
    #
    #     return outdir__

    # def get_create_forecast_dir(self)->str:
    #     # create folder for the hyperparameter search
    #     if not os.path.isdir(self.model_dir):
    #         os.makedirs(self.model_dir)
    #
    #     outdir__ = self.model_dir + 'forecast/'
    #     if not os.path.isdir(outdir__):
    #         os.makedirs(outdir__)
    #
    #     return outdir__

    def get_average_metrics(self)->dict:
        keys = self.metrics[list(self.metrics)[0]].keys()
        res = {
            key: np.mean(
                [self.metrics[timestamp][key] for timestamp in list(self.metrics.keys())]
            ) for key in keys
        }
        return res

    def print_average_metrics(self, prefix:str, metrics:dict):
        print(prefix + f"RMSE={metrics['rmse']:.1f} "
              f"CI_width={metrics['prediction_interval_width']:.1f} "
              f"sMAPE={metrics['smape']:.2f}")

    def cv_train_test(self, folds:int, X_train:pd.DataFrame or None, y_train:pd.DataFrame or None, do_fit:bool):

        if self.ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if self.forecaster is None:
            raise ReferenceError("Forecaster class is not initialized")
        if folds < 1:
            raise ValueError(f"CV is not possible with folds < 1 Given={folds}")
        if not self.cutoffs:
            self.cutoffs = get_ts_cutoffs(self.ds, folds=folds)
        if len(self.cutoffs) != folds:
            raise ValueError(f"There are already {self.cutoffs} cutoffs while passed folds={folds}")
        if X_train is None and y_train is None:
            self.X_for_model: pd.DataFrame = self.ds.get_exogenous_trasnformed()
            self.y_for_model: pd.Series = self.ds.get_target_transformed()
            if self.verbose:
                print(f"Using full model dataset from {self.y_for_model.index[0]} to {self.y_for_model.index[-1]} "
                      f"with {len(self.y_for_model)/24/7} weeks "
                      f"({len(self.y_for_model)/len(self.ds.get_forecast_index())} horizons)")
            if self.verbose:
                print(f"Running CV {folds} folds for {self.model_label}. "
                      f"Setting {len(self.X_for_model.columns)} "
                      f"features from dataset (lags_target={self.ds.lags_target}))")
        else:
            self.X_for_model = X_train
            self.y_for_model = y_train
            if self.verbose:
                print(f"Running CV {folds} folds for {self.model_label}. "
                      f"Given {len(self.X_for_model.columns)} features (lags_target={self.ds.lags_target}))")

        cutoffs = self.cutoffs # (self.ds, folds=folds)

        target = self.ds.target

        # sanity check that CV cutoffs are correctly set assuming multi-day forecasting horizon
        delta = timedelta(hours=len(self.ds.get_forecast_index()))
        for idx, cutoff in enumerate(cutoffs):
            # Train matrix should have negth devisible for the length of the forecasting horizon,
            # ane be composed of N segments each of which start at 00 hour and ends at 23 hour
            train_mask = self.y_for_model.index < cutoff
            # test mask should start at 00 hour and end on 23 hour (several full days)
            test_mask = (self.y_for_model.index >= cutoff) \
                        & (self.y_for_model.index < cutoff + delta)

            train_idx = self.y_for_model[train_mask].index
            test_idx = self.y_for_model[test_mask].index

            if len(train_idx) == 0 or len(test_idx) == 0:
                if self.verbose:
                    print(f"Warning! Empty train data batch idx={idx}/{len(cutoffs)}. Skipping.")
                    continue

            if not len(train_idx) % len(test_idx) == 0:
                print(f"Train: {train_idx[0]} to {train_idx[-1]} ({len(train_idx)/7/24} weeks, "
                      f"{len(train_idx)/len(test_idx)} horizons) Horizon={len(test_idx)/7/24} weeks")
                print(f"Test: {test_idx[0]} to {test_idx[-1]}")
                raise ValueError("Train set size should be divisible by the test size")

            if not len(test_idx) & 24:
                print(f"Test: {test_idx[0]} to {test_idx[-1]} N={len(test_idx)}")
                raise ValueError("Test set size should be divisible by 24")

        # perform CV for each fold (cutoff)
        for idx, cutoff in enumerate(cutoffs):
            train_mask = self.y_for_model.index < cutoff
            test_mask = (self.y_for_model.index >= cutoff) \
                        & (self.y_for_model.index < cutoff + timedelta(hours=len(self.ds.get_forecast_index())))

            if len(self.y_for_model[train_mask]) == 0 or len(self.y_for_model[test_mask]) == 0:
                if self.verbose:
                    print(f"Warning! Empty train data batch idx={idx}/{len(cutoffs)}. Skipping.")
                    continue

            # fit MapieRegressor(estimator=xbg.XBGRegressor(), method='naive', cv='prefit') model on the past data for current fold
            if do_fit: self.forecaster.fit(self.X_for_model[train_mask], self.y_for_model[train_mask])
            # get forecast for the next 'horizon' (data unseen during train time)
            result:pd.DataFrame = self.forecaster.forecast_window( self.X_for_model[test_mask], self.y_for_model[train_mask] )
            # 'result' has f'{target}_actual' and f'{target}_fitted' columns with N=horizon number of timesteps
            result[f'{target}_actual'] = self.y_for_model[test_mask] # add actual target values to result for error estimation
            # undo transformations
            if len(result[f'{target}_fitted'][~np.isfinite(result[f'{target}_fitted'])]) > 0:
                raise ValueError(f"Forecasting result contains nans. Fitted={result[f'{target}_fitted']}")
            # collect results (Dataframe with actual, fitted, lower and upper) for each fold (still transformed!)
            self.results[cutoff] = result
            # compute error metrics (RMSE, sMAPE...) over the entire forecasted window
            self.metrics[cutoff] = compute_error_metrics( target, result.apply(self.ds.inv_transform_target_series) )
            # print error metrics
            if self.verbose:
                self.print_average_metrics(
                    f"Fold {idx}/{len(cutoffs)} cutoff={cutoff} | "
                    f"{'FIT' if do_fit else 'INF'} | {self.model_label} | " + \
                    f"Train={self.X_for_model[train_mask].shape} ",
                    self.metrics[cutoff]
                )
        # print averaged over all CV folds error metrics
        if self.verbose:
            self.print_average_metrics(
                f"Average over {len(self.metrics)} folds | {self.model_label} | ",
                self.get_average_metrics())


    # def full_train(self):
    #     if self.forecaster is None:
    #         raise ReferenceError("Forecaster class is not initialized")
    #     if self.verbose:
    #         print(f"Training {self.model_label} on the entire dataset (X_train={self.X_for_model.shape})")
    #     self.forecaster.fit(self.X_for_model, self.y_for_model)

    def finetune(self, trial:optuna.Trial, cv_metrics_folds:int):
        if self.ds is None:
            raise ReferenceError("Dataset class is not initialized")

        self.forecaster = None

        params = get_parameters_for_optuna_trial(self.name_base_model, trial=trial)
        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.ds.target, lags_target=self.ds.lags_target,
            model_pars=params, verbose=self.verbose
        )

        self.cv_train_test(folds = cv_metrics_folds, X_train=None, y_train=None, do_fit=True)

        average_metrics = self.get_average_metrics()
        res = float( average_metrics['rmse'] ) # Average over all CV folds

        del self.results; self.results = {}
        del self.metrics; self.metrics = {}
        del self.contributions; self.contributions = {}

        gc.collect()

        return res


    def save_results(self, dir:str):

        dir = self.to_dir(dir=dir)
        if len(self.results.keys()) == 0:
            raise ReferenceError(f"No results found for base model {self.model_label}")

        metadata = {
            'start_date':self.ds.get_index()[0].isoformat(),
            'end_date':self.ds.get_index()[-1].isoformat(),
            'horizon':len(self.ds.get_forecast_index()),
            'features':self.ds.get_exogenous_trasnformed().columns.tolist(),
            'error_metrics':{key.isoformat() : val for key, val in self.metrics.items()}
        }
        with open(dir+'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        df_result = pd.concat([
            self.results[key].apply(self.ds.inv_transform_target_series) for key in list(self.results.keys())
        ], axis=0)
        df_result.to_csv(dir+'result.csv')

        if self.verbose:
            print(f"Results of {self.model_label} fits are saved into {dir}")

    def save_full_model(self, dir:str):

        dir = self.to_dir(dir=dir)

        if self.ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if self.forecaster is None:
            raise ReferenceError("Forecaster is not initialized")

        check_is_fitted(self.forecaster.model)
        self.forecaster.save_model(dir + 'model.joblib')

        if not self.ds.target_scaler is None:
            joblib.dump(self.ds.target_scaler, dir+'target_scaler.pkl')

        if not self.ds.feature_scaler is None:
            joblib.dump(self.ds.feature_scaler, dir+'feature_scaler.pkl')

        with open(dir+'dataset.json', 'w') as f:
            json.dump(self.model_dataset_pars, f, indent=4)

        with open(dir+'best_parameters.json', 'w') as f:
            json.dump(self.model_pars, f, indent=4)

        if self.verbose:
            print(f"Metadata for the trained base model {self.model_label} and dataset are saved into {dir}")

    def run_save_forecast(self, folds:int):
        if self.ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if self.forecaster is None:
            raise ReferenceError("Forecaster is not initialized")
        #
        #
        # self.cv_train_test(folds=folds, X_train=None, y_train=None, do_fit=False)
        # self.save_full_model(dir='forecast')

        X_scaled = self.ds.get_forecast_exogenous()
        y_scaled = self.ds.get_target_transformed()
        forecast = self.forecaster.forecast_window(X_scaled, y_scaled)
        forecast = forecast.apply(self.ds.inv_transform_target_series)
        if self.verbose: print(f"Saving {self.to_forecast() + 'forecast.csv'}")
        forecast.to_csv(self.to_forecast() + 'forecast.csv')

class EnsembleModelTasks(BaseModelTasks):

    def __init__(self, target:str, model_label:str, working_dir: str, verbose: bool = False):

        super().__init__(target, model_label, working_dir, verbose)

        # self.meta_model_name = model_name
        # self.base_model_names = base_model_names
        # self.workingdir = workingdir
        # self.model_dir = workingdir + 'meta_'+model_name+''.join(['_'+m for m in base_model_names])+'/'
        self.verbose = verbose
        self.ds = None
        self.base_models = {
            model_name : BaseModelTasks(self.target, model_name, self.working_dir, verbose)
            for model_name in self.names_base_models
        }

        self.X_for_model = None
        self.y_for_model = None

        self.X_meta = None
        self.y_meta = None
        self.cutoffs = None

    def set_meta_X_y(self, X_meta:pd.DataFrame, y_meta:pd.Series):
        self.y_meta = y_meta
        self.X_meta = X_meta


    def set_datasets_for_base_models_from_ds(self, ds:HistForecastDataset):
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.set_dataset_from_ds(ds)

    def set_datasets_for_base_models_from_finetuned(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.set_dataset_from_finetuned(df_hist=df_hist, df_forecast=df_forecast)

    def set_load_datasets_for_base_models_from_dir(self, dir:str, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.set_load_dataset_from_dir(dir=dir, df_hist=df_hist, df_forecast=df_forecast)

    def set_base_models_from_dir(self, dir:str):
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.set_forecaster_from_dir(dir=dir)


    def load_base_models_from_dir(self,dir:str):
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.load_forecaster_from_dir(dir=dir)

    def update_pretrain_base_models(self, cv_folds_base:int, do_fit:bool):
        if self.verbose:
            print(f"Running CV {'train-test' if do_fit else 'test only'} {list(self.base_models.keys())} "
                  f"base model using their respective datasets")
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.cv_train_test(cv_folds_base, None, None, do_fit=do_fit)

    def create_X_y_for_model_from_base_models_cv_folds(
            self,
            cv_folds_base_to_use:int,
            use_base_models_pred_intervals:bool
    ):

        target = self.ds.target

        if use_base_models_pred_intervals: features_from_base_models = ['fitted','lower','upper']
        else: features_from_base_models = ['fitted']

        cv_folds_base = len(self.base_models[list(self.base_models.keys())[-1]].results)
        if self.verbose:
            print(f"Combining {cv_folds_base} "
                  f"base-model CV folds to create a training set for meta-model {self.model_label}")

        if cv_folds_base < 3:
            raise ValueError(f"Number of CV folds for base models is too small. Should be >3 Given: {cv_folds_base}")

        cutoffs = get_ts_cutoffs(self.ds, folds=cv_folds_base)
        cutoffs = cutoffs[-cv_folds_base_to_use:]

        X_meta_train_list, y_meta_train_list = [], []
        for i_cutoff, cutoff in enumerate(cutoffs):

            # get indexes for train-test split
            resutls:dict = self.base_models[list(self.base_models.keys())[0]].results
            if not cutoff in list(resutls.keys()):
                raise ValueError(f"Expected cutoff {cutoff} is not in results.keys()={list(resutls.keys())}")

            target_actual = resutls[cutoff][f'{target}_actual']
            curr_tst_index = target_actual.index
            forecasts_df = pd.DataFrame(index=curr_tst_index)

            # extract results from base model forecasts
            for base_model_name, base_model_class in self.base_models.items():
                for key in features_from_base_models:
                    forecasts_df[f'base_model_{base_model_name}_{key}'] = \
                        base_model_class.results[cutoff][f'{target}_{key}']
            # add meta-features
            if not self.X_meta is None:
                forecasts_df = forecasts_df.merge(self.X_meta.loc[curr_tst_index], left_index=True, right_index=True)

            X_meta_train_list.append(forecasts_df)
            y_meta_train_list.append(target_actual)

        # Concatenate all folds and check shapes
        X_meta_model_train = pd.concat(X_meta_train_list)
        y_meta_model_train = pd.Series(pd.concat(y_meta_train_list))

        # check if shapes are correct
        model_names = list(self.base_models.keys())
        if not self.X_meta is None:
            if not (len(X_meta_model_train.columns) ==
                    len(self.X_meta.columns) + len(model_names)*len(features_from_base_models)):
                raise ValueError(f"Expected {len(self.X_meta.columns)+len(model_names)} columns in "
                                 f"X_meta_model_train. Got={len(X_meta_model_train.columns)}")
        else:
            if not (len(X_meta_model_train.columns) == len(model_names)):
                raise ValueError(f"Expected {len(model_names)} columns in "
                                 f"X_meta_model_train. Got={len(X_meta_model_train.columns)}")

        self.X_for_model = X_meta_model_train
        self.y_for_model = y_meta_model_train

        if self.verbose:
            print(f"Trining data for meta-model {self.model_label} is collected. "
                  f"Shape: X_train={self.X_for_model.shape}")

    def cv_train_test_ensemble(self, cv_folds_base:int, do_fit:bool):
        self.cv_train_test(folds=cv_folds_base, X_train=self.X_for_model, y_train=self.y_for_model, do_fit=do_fit)

    def finetune(self, trial:optuna.Trial, cv_metrics_folds:int):

        cv_folds_base = len(self.base_models[list(self.base_models.keys())[-1]].results)
        # if not (cv_folds_base > cv_metrics_folds+3):
        #     raise ValueError("There should be more CV folds for base models then for ensemble model. "
        #                      f"Given CV ensemble {cv_metrics_folds}+3 folds and base models {cv_folds_base} folds")

        if self.ds is None:
            raise ReferenceError("Dataset class is not initialized")
        self.forecaster = None

        params = get_parameters_for_optuna_trial(self.name_base_model, trial=trial)
        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.ds.target, lags_target=self.ds.lags_target,
            model_pars=params, verbose=self.verbose
        )

        self.create_X_y_for_model_from_base_models_cv_folds(
            cv_folds_base_to_use=cv_folds_base,
            use_base_models_pred_intervals = False
        )
        # trial.suggest_int(
        #         'cv_folds_base_to_use', cv_metrics_folds+3, cv_folds_base
        #     ),
        #     use_base_models_pred_intervals=trial.suggest_categorical(
        #         'use_base_models_pred_intervals', [True, False])
        # )

        self.cv_train_test_ensemble(cv_folds_base=cv_metrics_folds, do_fit=True)

        average_metrics = self.get_average_metrics()
        res = float( average_metrics['rmse'] ) # Average over all CV folds

        del self.X_for_model; self.X_for_model = None
        del self.y_for_model; self.y_for_model = None
        del self.results; self.results = {}
        del self.metrics; self.metrics = {}
        del self.contributions; self.contributions = {}

        gc.collect()

        return res


    def run_save_forecast(self, folds:int):

        use_base_models_pred_intervals = self.model_pars['use_base_models_pred_intervals']
        if use_base_models_pred_intervals: features_from_base_models = ['fitted','lower','upper']
        else: features_from_base_models = ['fitted']

        X_test = pd.DataFrame()
        for base_model_name, base_model_class in self.base_models.items():
            dir = base_model_class.to_forecast()
            df_i = pd.read_csv(dir+'forecast.csv', index_col='date')
            df_i.index = pd.to_datetime(df_i.index)
            df_i = df_i.apply(self.ds.transform_target_series)
            for key in features_from_base_models:
                X_test[f'base_model_{base_model_name}_{key}'] = df_i[f'{self.ds.target}_{key}']

        if not self.X_meta is None:
            X_test = X_test.merge(self.X_meta, left_index=True, right_index=True)

        if not validate_dataframe(X_test) or len(X_test) != len(self.ds.get_forecast_index()):
            raise ValueError("Validation check of forecasts_df failed.")

        forecast = self.forecaster.forecast_window( X_test, self.y_meta )
        forecast = forecast.apply(self.ds.inv_transform_target_series)

        dir = self.to_forecast()
        if self.verbose: print(f"Saving {dir+'forecast.csv'}")
        forecast.to_csv(dir+'forecast.csv')

def select_best_model(metrics):
    """
    Selects the best-performing model using different evaluation approaches.

    Args:
        metrics (dict): A nested dictionary where metrics[model_name][window_start_timestamp][metric_name] = value.

    Returns:
        dict: A dictionary with approach names as keys and the best model names as values.
    """

    # Collect models and windows
    models = list(metrics.keys())
    windows = set()
    for model in metrics:
        windows.update(metrics[model].keys())
    windows = sorted(windows)  # Oldest to newest

    # Create weights for windows, giving higher weights to more recent windows
    num_windows = len(windows)
    window_weights = np.arange(1, num_windows + 1)  # Linear weights
    window_weights = window_weights / window_weights.sum()  # Normalize weights to sum to 1
    window_weights_series = pd.Series(window_weights, index=windows)

    # Define the metrics to consider
    metric_names = ['mse', 'rmse', 'mae', 'mape', 'smape', 'bias', 'variance', 'std',
                    'r2', 'prediction_interval_coverage', 'prediction_interval_width']

    # Initialize a dictionary to store DataFrames for each metric
    metric_dfs = {metric_name: pd.DataFrame(index=windows, columns=models) for metric_name in metric_names}

    # Populate the DataFrames with metric values
    for model in models:
        for window in metrics[model]:
            for metric_name in metric_names:
                value = metrics[model][window].get(metric_name, np.nan)
                metric_dfs[metric_name].loc[window, model] = value

    # Initialize the result dictionary
    best_models = {}

    # Approach 1: Weighted Aggregate Error Metrics
    # Weighted Mean and Median RMSE
    for agg_func in ['mean', 'median']:
        for metric in ['rmse', 'smape']:
            metric_df = metric_dfs[metric].astype(float)
            # Apply weights
            weighted_metric = metric_df.mul(window_weights_series, axis=0)
            if agg_func == 'mean':
                agg_metric = weighted_metric.sum()
            elif agg_func == 'median':
                # Weighted median is more complex; we'll approximate by sorting
                # and selecting the value where the cumulative weight reaches 50%
                agg_metric = {}
                for model in models:
                    sorted_metrics = metric_df[model].dropna().sort_values()
                    sorted_weights = window_weights_series.loc[sorted_metrics.index]
                    cum_weights = sorted_weights.cumsum()
                    median_idx = cum_weights >= 0.5
                    if not median_idx.any():
                        median_value = np.nan
                    else:
                        median_value = sorted_metrics[median_idx.idxmax()]
                    agg_metric[model] = median_value
                agg_metric = pd.Series(agg_metric)
            best_model = agg_metric.idxmin()
            best_models[f'Weighted {agg_func.capitalize()} {metric.upper()}'] = best_model

    # Weighted Worst-Case Performance
    # Since worst-case is a single value, weighting doesn't directly apply,
    # but we can focus on recent worst cases.
    recent_windows = windows[-max(1, num_windows // 3):]  # Use the most recent third of windows
    for metric in ['rmse', 'smape']:
        metric_df = metric_dfs[metric].loc[recent_windows]
        max_metric = metric_df.max()
        best_model_max = max_metric.idxmin()
        best_models[f'Recent Worst-Case {metric.upper()}'] = best_model_max

    # Approach 2: Weighted Stability of Performance
    for metric in ['rmse', 'smape']:
        metric_df = metric_dfs[metric].astype(float)
        # Calculate weighted standard deviation
        mean_metric = metric_df.mul(window_weights_series, axis=0).sum()
        deviations = metric_df.subtract(mean_metric, axis=1)
        weighted_var = (deviations ** 2).mul(window_weights_series, axis=0).sum()
        weighted_std = np.sqrt(weighted_var)
        cv_metric = weighted_std / mean_metric.abs()

        best_model_std = weighted_std.idxmin()
        best_model_cv = cv_metric.idxmin()

        best_models[f'Weighted Std of {metric.upper()}'] = best_model_std
        best_models[f'Weighted CV of {metric.upper()}'] = best_model_cv

    # Approach 3: Weighted Pairwise Comparisons
    for metric in ['rmse', 'smape']:
        metric_df = metric_dfs[metric].astype(float)
        # For each window, find the best model
        best_per_window = metric_df.idxmin(axis=1)
        # Weight the counts based on window weights
        weighted_counts = best_per_window.map(window_weights_series).groupby(best_per_window).sum()
        best_model_pairwise = weighted_counts.idxmax()
        best_models[f'Weighted Pairwise Comparison {metric.upper()}'] = best_model_pairwise

    # Approach 5: Statistical Tests on Recent Windows
    for metric in ['rmse', 'smape']:
        metric_df = metric_dfs[metric].loc[recent_windows].dropna()
        if len(models) >= 2 and metric_df.shape[0] >= 2:
            # Perform Friedman test on recent windows
            friedman_stat, p_value = friedmanchisquare(
                *[metric_df[model].values for model in models]
            )
            if p_value < 0.05:
                best_model_friedman = metric_df.mean().idxmin()
                best_models[f'Recent Friedman Test {metric.upper()}'] = best_model_friedman
            else:
                best_models[f'Recent Friedman Test {metric.upper()}'] = 'No significant difference'
        else:
            best_models[f'Recent Friedman Test {metric.upper()}'] = 'Not enough data'

    return best_models

class ForecastingTaskSingleTarget:

    def __init__(self, target:str, task:dict, outdir:str, verbose:bool):
        # main output directory
        self.verbose = verbose
        if not os.path.isdir(outdir):
            if self.verbose: print(f"Creating {outdir}")
            os.mkdir(outdir)

        # init dataclass
        self.target = task['target']
        self.outdir_ = outdir


        # load dataset
        df_history, df_forecast = load_data(target)
        print(f"Loaded dataset has features: {df_history.columns.tolist()}")
        # df_forecast = df_forecast[1:] # remove df_history[-1] hour

        # restrict to required features
        features = task['features']
        features_to_restrict : list = []
        for feature in features:
            # preprocess some features
            if feature  == 'weather':
                # TODO IMPROVE (USE OPENMETEO CLASS HERE)
                weather_features:list = _get_weather_features(df_history)
                features_to_restrict += weather_features
        if not features:
            if verbose: print(f"No features selected for {target}. Using all features: \n{df_forecast.columns.tolist()}")
            features_to_restrict = df_forecast.columns.tolist()

        # remove unnecessary features from the dataset
        print(f"Restricting dataframe from {len(df_history.columns)} features to {len(features_to_restrict)}")
        self.df_history = df_history[features_to_restrict + [target]]
        self.df_forecast = df_forecast[features_to_restrict]
        if not validate_dataframe(self.df_forecast):
            raise ValueError("Nans in the df_forecast after restricting. Cannot continue.")

    # ------- FINETUNING ------------
    def process_finetuning_task_ensemble(self, ft_task):

        model_label = ft_task['model']
        dataset_pars = ft_task['dataset_pars']
        finetuning_pars = ft_task['finetuning_pars']

        # common for all tasks for a given quantity
        dataset_pars['target'] = self.target

        # use actual weather features
        if 'limit_pca_to_features' in dataset_pars.keys() and dataset_pars['limit_pca_to_features'] == 'weather':
            dataset_pars['limit_pca_to_features'] = _get_weather_features(self.df_history)

        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_, verbose=self.verbose
        )

        ft_outdir = wrapper.to_finetuned()
        with open(ft_outdir+'dataset.json', 'w') as f:
            json.dump(dataset_pars, f, indent=4)

        ensemble_features = dataset_pars['ensemble_features']; del dataset_pars['ensemble_features']
        wrapper.set_dataset_from_df(self.df_history, self.df_forecast, pars=dataset_pars)
        wrapper.set_datasets_for_base_models_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_base_models_from_dir(dir='finetuning')
        if ensemble_features == 'cyclic_time':
            wrapper.set_meta_X_y(
                wrapper.ds._create_time_features( wrapper.ds.get_index() ),
                wrapper.ds.get_target_transformed()
            )

        wrapper.update_pretrain_base_models(
            cv_folds_base=finetuning_pars['cv_folds_base'], do_fit=True
        )

        if self.verbose:
            print(f"Performing optimization study for meta-{wrapper.name_base_model} as {wrapper.model_label}")
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: wrapper.finetune(trial, cv_metrics_folds=finetuning_pars['cv_folds']),
            n_trials=finetuning_pars['n_trials']
        )
        save_optuna_results(study, {
            'cv_folds_base':finetuning_pars['cv_folds_base'],
            'use_base_models_pred_intervals':finetuning_pars['use_base_models_pred_intervals']
        }, ft_outdir)

        wrapper.clear()

    def process_finetuning_task_base(self, ft_task):

        model_label = ft_task['model']
        dataset_pars = ft_task['dataset_pars']
        finetuning_pars = ft_task['finetuning_pars']

        # common for all tasks for a given quantity
        dataset_pars['target'] = self.target

        # use actual weather features
        if 'limit_pca_to_features' in dataset_pars.keys() and dataset_pars['limit_pca_to_features'] == 'weather':
            dataset_pars['limit_pca_to_features'] = _get_weather_features(self.df_history)

        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_dataset_from_df(self.df_history, self.df_forecast, dataset_pars)
        # wrapper.set_ts_cutoffs(folds=finetuning_pars['cv_folds'])

        if self.verbose:
            print(f"Performing optimization study for {self.target} with base model {model_label}")
        study = optuna.create_study(direction='minimize')
        # study.optimize(wrapper.finetune, n_trials=finetuning_pars['n_trials']) # todo move this into finetuning_pars
        study.optimize(
            lambda trial: wrapper.finetune(trial, cv_metrics_folds=finetuning_pars['cv_folds']),
            n_trials=finetuning_pars['n_trials']
        )

        ft_outdir = wrapper.to_finetuned()
        save_optuna_results(study, {}, ft_outdir)
        with open(ft_outdir+'dataset.json', 'w') as f:
            json.dump(dataset_pars, f, indent=4)

        wrapper.clear()

    # ------- TRAINING ---------
    def process_training_task_ensemble(self, t_task):
        model_label = t_task['model']
        pars = t_task['pars']

        # train_ensemble_model(
        #     outdir_, model_name, target, df_history, df_forecast, pars,False
        # )
        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        ds_pars, extra_pars = wrapper.set_dataset_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_datasets_for_base_models_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_base_models_from_dir(dir='finetuning')
        if extra_pars['ensemble_features'] == 'cyclic_time':
            wrapper.set_meta_X_y(
                wrapper.ds._create_time_features( wrapper.ds.get_index() ),
                wrapper.ds.get_target_transformed()
            )
        model_pars, extra_model_pars = wrapper.set_forecaster_from_dir(dir='finetuning')
        wrapper.update_pretrain_base_models(
            cv_folds_base=extra_model_pars['cv_folds_base'],#['cv_folds_base_to_use'],
            do_fit=True
        )
        wrapper.create_X_y_for_model_from_base_models_cv_folds(
            cv_folds_base_to_use=extra_model_pars['cv_folds_base'],#['cv_folds_base_to_use'],
            use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals'],#['use_base_models_pred_intervals']
        )
        wrapper.cv_train_test(
            folds=pars['cv_folds'],
            X_train=wrapper.X_for_model,y_train=wrapper.y_for_model,
            do_fit=True
        )

        t_outdir = wrapper.to_dir('trained')
        wrapper.save_full_model(dir='trained') # trained
        wrapper.save_results(dir='trained')
        with open(t_outdir+'dataset.json', 'w') as f:
            json.dump(ds_pars | extra_pars, f, indent=4)

    def process_training_task_base(self, t_task):
        model_label = t_task['model']
        pars = t_task['pars']

        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        ds_pars, extra_pars = wrapper.set_dataset_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_forecaster_from_dir(dir='finetuning')

        wrapper.cv_train_test(folds=pars['cv_folds'], X_train=None, y_train=None, do_fit=True)

        t_outdir = wrapper.to_dir('trained')
        wrapper.save_full_model(dir='trained') # trained
        wrapper.save_results(dir='trained')
        with open(t_outdir+'dataset.json', 'w') as f:
            json.dump(ds_pars | extra_pars, f, indent=4)

    # ------ FORECASTING -------

    def process_forecasting_task_ensemble(self, f_task):
        model_label = f_task['model']
        folds = f_task['past_folds']
        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        ds_pars, extra_pars = wrapper.set_load_dataset_from_dir(
            dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast
        )
        wrapper.set_load_datasets_for_base_models_from_dir(
            dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast
        )
        wrapper.load_base_models_from_dir(dir='trained')
        if extra_pars['ensemble_features'] == 'cyclic_time':
            wrapper.set_meta_X_y(
                wrapper.ds._create_time_features( wrapper.ds.get_index() ),
                wrapper.ds.get_target_transformed()
            )

        model_pars, extra_model_pars = wrapper.load_forecaster_from_dir(dir='trained')
        wrapper.update_pretrain_base_models(
            cv_folds_base=folds, do_fit=False
        )
        wrapper.create_X_y_for_model_from_base_models_cv_folds(
            cv_folds_base_to_use=extra_model_pars['cv_folds_base'],
            use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals']
        )
        wrapper.cv_train_test_ensemble(cv_folds_base=folds, do_fit=False)
        wrapper.save_results(dir='forecast')


        # forecast (different feature set)
        if extra_pars['ensemble_features'] == 'cyclic_time':
            wrapper.set_meta_X_y(
                wrapper.ds._create_time_features( wrapper.ds.get_forecast_index() ),
                wrapper.ds.get_target_transformed()
            )

        # wrapper.update_pretrain_base_models(
        #     cv_folds_base=folds, do_fit=False
        # )
        # wrapper.create_X_y_for_model_from_base_models_cv_folds(
        #     cv_folds_base_to_use=extra_model_pars['cv_folds_base_to_use'],
        #     use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals']
        # )
        # wrapper.cv_train_test_ensemble(cv_folds_base=extra_model_pars['cv_folds_base_to_use'], do_fit=False)
        wrapper.run_save_forecast(folds=folds)

    def process_forecasting_task_base(self, f_task):
        model_label = f_task['model']
        folds = f_task['past_folds']
        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_load_dataset_from_dir(dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast)
        wrapper.load_forecaster_from_dir(dir='trained')
        wrapper.cv_train_test(folds=folds, X_train=None, y_train=None, do_fit=False)
        wrapper.save_results(dir='forecast')
        wrapper.run_save_forecast(folds=folds)

    # ------ OTHERS --------

    @staticmethod
    def _load_trained_model(target:str,model_label:str, working_dir:str, train_forecast:str, verbose:bool)->dict:
        paths = TaskPaths(
            target=target, model_label=model_label, working_dir=working_dir, verbose=verbose
        )

        # choose which results to load -- from training run or from inference run
        if train_forecast == 'forecast': dir = paths.to_forecast()
        elif train_forecast == 'train': dir = paths.to_trained()
        else: raise ValueError('train_forecast must be either forecast or train')

        with open(dir + "metadata.json", 'r') as file:
            train_metadata = json.load(file)

        horizon = int(train_metadata['horizon'])
        df_results = pd.read_csv(dir + 'result.csv',index_col=0,parse_dates=True)
        if len(df_results) % horizon != 0:
            raise ValueError(f"Expected number of rows in results to be divisible by horizon for {model_label}. "
                             f"Given: df_results={len(df_results)} horizon={horizon} ({len(df_results) % horizon})")

        df_results = [df_results.iloc[i:i + horizon] for i in range(0, len(df_results), horizon)]
        metrics = train_metadata['error_metrics']
        metrics = [val for (key, val) in metrics.items()]
        if not len(metrics) == len(df_results):
            raise ValueError(f"Expected same number of results and metrics. "
                             f"Given: n_results={len(df_results)} n_metrics={len(metrics)}")

        ave_metrics = {
            key: np.mean( [metrics[i][key] for i in range(len((metrics)))] ) for key in list(metrics[0].keys())
        }

        task_i = {}
        task_i['results'] = df_results
        task_i['metrics'] = metrics

        df_forecast = pd.read_csv(paths.to_forecast() + 'forecast.csv',index_col=0,parse_dates=True)

        task_i['forecast'] = df_forecast
        task_i['metrics'].append(ave_metrics)
        return task_i

    def process_task_plot_predict_forecast(self, task):
        plotting_tasks = []
        for t_task in task['task_plot']:
            task_i = self._load_trained_model(
                self.target,
                model_label=t_task['model'],
                working_dir=self.outdir_,
                train_forecast=t_task['train_forecast'],
                verbose=self.verbose
            )
            n = t_task['n']
            if n > len(task_i['metrics'])-1:
                raise ValueError(f"Requested to plot n={n} "
                                 f"past forecasts while only {len(task_i['metrics'])-1} are avaialble")
            task_i['results'] = task_i['results'][-n:]
            task_i['metrics'] = task_i['metrics'][-n-1:]
            plotting_tasks.append(task_i | t_task)

            # model_label = t_task['model']
            # paths = TaskPaths(
            #     target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
            # )
            #
            # n = t_task['n']
            # task_i = copy.deepcopy(t_task)
            #
            # with open(paths.to_trained() + "metadata.json", 'r') as file:
            #     train_metadata = json.load(file)
            #
            # horizon = int(train_metadata['horizon'])
            # df_results = pd.read_csv(paths.to_trained() + 'result.csv',index_col=0,parse_dates=True)
            # if len(df_results) % horizon != 0:
            #     raise ValueError(f"Expected number of rows in results to be divisible by horizon for {model_label}. "
            #                      f"Given: df_results={len(df_results)} horizon={horizon} ({len(df_results) % horizon})")
            #
            # df_results = [df_results.iloc[i:i + horizon] for i in range(0, len(df_results), horizon)]
            # metrics = train_metadata['error_metrics']
            # metrics = [val for (key, val) in metrics.items()]
            # if not len(metrics) == len(df_results):
            #     raise ValueError(f"Expected same number of results and metrics. "
            #                      f"Given: n_results={len(df_results)} n_metrics={len(metrics)}")
            #
            # ave_metrics = {
            #     key: np.mean( [metrics[i][key] for i in range(len((metrics)))] ) for key in list(metrics[0].keys())
            # }
            #
            # if n > len(metrics):
            #     raise ValueError(f"Requested to plot n={n} past forecasts while only {len(metrics)} are avaialble")
            # task_i['results'] = df_results[-n:]
            # task_i['metrics'] = metrics[-n:]
            #
            # df_forecast = pd.read_csv(paths.to_forecast() + 'forecast.csv',index_col=0,parse_dates=True)
            #
            # task_i['forecast'] = df_forecast
            # task_i['metrics'].append(ave_metrics)
            #
            # plotting_tasks.append(task_i)
        plot_time_series_with_residuals(plotting_tasks, target=self.target, ylabel=task["label"])
        return plotting_tasks

    def process_task_determine_the_best_model(self, task):
        metrics = {}
        for t_task in task['task_select_best']:
            model_label = t_task['model']
            paths = TaskPaths(
                target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
            )
            with open(paths.to_trained() + "metadata.json", 'r') as file:
                train_metadata = json.load(file)
            metrics[model_label] = train_metadata['error_metrics']
        result = select_best_model(metrics)
        with open(paths.to_target + 'forecasting_selection_result.json', 'w') as f:
            json.dump(result, f, indent=4)


def main():
    tasks = [
        {
            "target": "wind_offshore",
            "label": "Wind off-shore [MW]",
            "features":[],
            "task_fine_tuning":[
                # {'model':'Prophet',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'feature_pca_pars' : None,
                #      'limit_pca_to_features':None,#'weather',
                #      'fourier_features' : {'period':24, 'order':3},
                #      'add_cyclical_time_features':True,
                #      'lags_target':None,'log_target':True,
                #      'copy_input':True
                #  },
                # 'finetuning_pars':{'n_trials':120,'optim_metric':'rmse','cv_folds':3}},

                # {'model':'XGBoost',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'limit_pca_to_features':None,#'weather',
                #      'feature_pca_pars':None,#{'n_components':0.95},
                #      'fourier_features': {},
                #      'add_cyclical_time_features':True,
                #      'lags_target':24,'log_target':True,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':5,'optim_metric':'rmse','cv_folds':3}},
                #
                # {'model':'ElasticNet',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'limit_pca_to_features':None,#'weather',
                #      'feature_pca_pars':None,#{'n_components':0.95},
                #      'fourier_features': {},
                #      'add_cyclical_time_features':True,
                #      'lags_target':24,'log_target':True,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':5,'optim_metric':'rmse','cv_folds':3}},

                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
                 'dataset_pars': {
                     'forecast_horizon':None,
                     'target_scaler':'StandardScaler',
                     'feature_scaler':'StandardScaler',
                     'limit_pca_to_features':None,#'weather',
                     'feature_pca_pars':None,#{'n_components':0.95},
                     'add_cyclical_time_features':False,
                     'fourier_features': {},
                     'ensemble_features': 'cyclic_time',
                     'log_target':True,
                     'lags_target': None,
                     'copy_input':True

                 },
                 'finetuning_pars':{'n_trials':5,
                                    'optim_metric':'rmse',
                                    'cv_folds':3,
                                    'cv_folds_base':35,
                                    'use_base_models_pred_intervals':False}}
            ],
            "task_training":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                # {'model':'XGBoost', 'pars':{'cv_folds':5}},
                # {'model':'ElasticNet', 'pars':{'cv_folds':5}},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':5}}
            ],
            "task_forecasting":[
                # {'model':'Prophet'},
                {'model':'XGBoost', 'past_folds':5},
                {'model':'ElasticNet', 'past_folds':5},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','past_folds':5}
            ],
            "task_plot":[
                # {'model':'Prophet', 'n':2,
                #  'name':"Prophet",'lw':0.7,'color':"red",'ci_alpha':0.0},
                {'model':'XGBoost','n':2,
                 'name':'XGBoost','lw':0.7,'color':"green",'ci_alpha':0.0,
                 'train_forecast':'train'},
                {'model':'ElasticNet','n':2,
                 'name':'ElasticNet','lw':0.7,'color':"blue",'ci_alpha':0.0,
                 'train_forecast':'train'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','n':2,
                 'name':'Ensemble','lw':1.0,'color':"purple",'ci_alpha':0.2,
                 'train_forecast':'train'},
            ],
            "task_select_best":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                {'model':'XGBoost'},
                {'model':'ElasticNet'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)'},
            ]
        }
    ]

    outdir = './output/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for task in tasks:
        target = task['target']
        processor = ForecastingTaskSingleTarget(target=target,task=task,outdir=outdir,verbose=True)

        # process task to fine-tune the forecasting model
        if task['task_fine_tuning']:
            for ft_task in task['task_fine_tuning']:
                if ft_task['model'].__contains__('ensemble'):
                    processor.process_finetuning_task_ensemble(ft_task)
                else:
                    processor.process_finetuning_task_base(ft_task)

        # train forecasting model on full dataset assuming hyperparameters are in finetuning dir
        if task['task_training']:
            for t_task in task['task_training']:
                if t_task['model'].__contains__('ensemble'):
                    processor.process_training_task_ensemble(t_task)
                else:
                    processor.process_training_task_base(t_task)

        # forecast with trained model
        if task['task_forecasting']:
            for t_task in task['task_forecasting']:
                if t_task['model'].__contains__('ensemble'):
                    processor.process_forecasting_task_ensemble(t_task)
                else:
                    processor.process_forecasting_task_base(t_task)

        if task['task_plot']:
            processor.process_task_plot_predict_forecast(task)

        if task['task_select_best']:
            processor.process_task_determine_the_best_model(task)


if __name__ == '__main__':
    main()