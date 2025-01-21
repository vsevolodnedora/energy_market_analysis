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

from data_collection_modules.utils import validate_dataframe_simple
from forecasting_modules.utils import (
    compute_timeseries_split_cutoffs,
    compute_error_metrics,
    compute_error_metrics_aggregate_over_horizon,
    compute_error_metrics_aggregate_over_cv_runs,
    save_datetime_now
)
from data_modules.data_classes import (
    validate_dataframe,
    HistForecastDataset,
    create_time_features,
    suggest_values_for_ds_pars_optuna
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

def load_data(target:str,datapath:str, verbose:bool) \
        ->tuple[pd.DataFrame, pd.DataFrame]:

    df_history = pd.read_parquet(f"{datapath}history.parquet")
    df_forecast = pd.read_parquet(f"{datapath}forecast.parquet")

    # for brevity and due to evolving market conditions we focus here only on 1 year of data
    # df_history = df_history[pd.Timestamp(df_history.dropna(how='any', inplace=False).last_valid_index()) - limit_train_to:]

    # assure that the columns in both dataframes match
    df_features = df_history[[col for col in list(df_history.columns) if col != target]]
    if not df_features.columns.equals(df_forecast.columns):
        raise IOError("The DataFrames have different columns.")

    if verbose:
        print(f"History: {df_history.shape} from {df_history.index[0]} to {df_history.index[-1]} ({len(df_history.index)/7/24} weeks)")
        print(f"Forecast: {df_forecast.shape} from {df_forecast.index[0]} to {df_forecast.index[-1]} ({len(df_forecast.index)/24} days)")

    return df_history, df_forecast

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
    with open(f'{outdir}best_parameters.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        for key, value in best_params.items():
            writer.writerow([key, value])

    # Convert the complete study results to a DataFrame and save to CSV
    results_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    results_df.to_csv(f'{outdir}complete_study_results.csv', index=False)

    save_datetime_now(outdir) # save when the training was done


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
            'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 1.0, log=False),
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=False)
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

def instantiate_base_forecaster(model_name:str, target:str, model_pars:dict, verbose:bool)->BaseForecaster:
    # if 'l1_ratio' in model_pars: del model_pars['l1_ratio']
    # train the forecasting model several times to evaluate its performance, get all results
    if model_name == 'XGBoost':
        return XGBoostMapieRegressor(
            model=MapieRegressor(
                xgb.XGBRegressor(**model_pars),
                method='naive', cv='prefit'#TimeSeriesSplit(n_splits=5)
            ), target=target, alpha=0.05, verbose=verbose)

    elif model_name == 'ElasticNet':
        return ElasticNetMapieRegressor(
            model=MapieRegressor(
                ElasticNet(**(model_pars | { 'max_iter':1e6, 'tol':1e-8 })),
                method='naive', cv='prefit'#TimeSeriesSplit(n_splits=5)
            ), target=target, alpha=0.05, verbose=verbose)

    elif model_name == 'Prophet':
        return ProphetForecaster( params = model_pars, target=target, alpha=0.05, verbose=verbose)

    else:
        raise NotImplementedError(f"Fine-tuning parameter set for {model_name} not implemented")

# def get_ts_cutoffs(ds:HistForecastDataset,folds:int)->tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
#     if ds is None:
#         raise ReferenceError("Dataset class is not initialized")
#     horizon = len(ds.forecast_idx) # number of timesteps to forecast
#     if not horizon % 24 == 0:
#         raise ValueError(f"Horizon must be divisible by 24 (at least one day). Given {horizon}")
#     return compute_timeseries_split_cutoffs(
#         ds.hist_idx,
#         horizon=horizon,
#         folds=folds,
#         min_train_size=5*horizon
#     )

def get_ensemble_name_and_model_names(model_name:str):
    match = re.search(r'\[(.*?)\]', model_name)
    if match: meta_model = match.group(1)
    else: raise NameError(f"Model name {model_name} does not contain '[meta_model_name]' string")

    # extract base-models names
    match = re.search(r'\((.*?)\)', model_name)
    if match: model_names = match.group(1).split(',')  # Split by comma
    else: raise NameError(f"Model name {model_name} does not contain '(model_name_1,model_name_2)' string")

    return meta_model, model_names

def analyze_model_performance(data: pd.DataFrame, n_folds: int, metric: str)->tuple[pd.DataFrame, pd.DataFrame]:
    # Validate inputs
    if metric not in ['mse', 'rmse', 'mae', 'mape']:
        raise ValueError(f"Invalid metric '{metric}'. Choose from 'mse', 'rmse', 'mae', 'mape'.")

    # Filter data by the number of folds
    data['horizon'] = pd.to_datetime(data['horizon'])
    data = data.sort_values(by=['method', 'model_label', 'horizon'], ascending=[True, True, False])
    recent_data = data.groupby(['method', 'model_label']).head(n_folds)

    # Compute the average error for each model and method
    avg_errors = (recent_data
                  .groupby(['method', 'model_label'])[metric]
                  .mean()
                  .reset_index()
                  .rename(columns={metric: f'avg_{metric}'}))

    # Determine the best model for each method (trained and forecast)
    best_models = avg_errors.loc[avg_errors.groupby('method')[f'avg_{metric}'].idxmin()]

    # Check for model drift (forecast errors systematically larger than trained errors)
    trained_errors = avg_errors[avg_errors['method'] == 'trained']
    forecast_errors = avg_errors[avg_errors['method'] == 'forecast']

    drift_data = pd.merge(trained_errors, forecast_errors, on='model_label', suffixes=('_trained', '_forecast'))
    drift_data['error_difference'] = drift_data[f'avg_{metric}_forecast'] - drift_data[f'avg_{metric}_trained']
    drift_data['drift_detected'] = drift_data['error_difference'] > 0

    # Summarize drift detection
    drift_summary = drift_data[['model_label', 'error_difference', 'drift_detected']]

    return best_models, drift_summary




def write_summary(summary: dict):
    # Collect datagrams
    datagram = []

    # Iterate through the methods (train, forecast)
    for method, models in summary.items():
        for model_label, horizons in models.items():
            for horizon, metrics in horizons.items():
                # Create a single entry for each horizon
                datagram.append({
                    "method": method,
                    "model_label": model_label,
                    "horizon": horizon,
                    **metrics
                })

    return datagram
def plot_metric_evolution(file_path: str, metric: str):
    # Load the dataframe
    df = pd.read_csv(file_path)

    # Filter relevant columns
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in the dataframe.")

    # Sort the dataframe by horizon and convert horizon to datetime
    df['horizon'] = pd.to_datetime(df['horizon'])
    df = df.sort_values(by='horizon')

    markers = ['s', 'o', 'v', '^', 'P']

    # Plot the metric evolution
    fig, ax = plt.subplots(figsize=(8, 4))
    for marker, model_label in zip(markers, df['model_label'].unique()):
        if model_label.__contains__('ensemble'): markerfacecolor = 'black'
        else: markerfacecolor = 'None'
        method_df_trained = df[(df['model_label'] == model_label) & (df['method'] == 'trained')]
        ax.plot(method_df_trained['horizon'], method_df_trained[metric], linestyle='None',
                marker=marker, label=model_label, color='black',
                markerfacecolor=markerfacecolor, markeredgecolor='black', markersize=8
                )
        # method_df_forecast = df[(df['model_label'] == model_label) & (df['method'] == 'trained')]
        # ax.plot(method_df_forecast['horizon'], method_df_forecast[metric], linestyle='None', marker=marker)

    # Set x-ticks at unique horizon values
    unique_horizons = df['horizon'].sort_values().unique()
    ax.set_xticks(unique_horizons)
    ax.set_xticklabels([
        h.strftime('%Y-%m-%d') for h in unique_horizons], rotation=0, ha='center'#'right'
    )

    ax.grid(True, linestyle='-', alpha=0.4)
    ax.tick_params(axis='x', direction='in', bottom=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)
    # Set border lines transparent by setting the edge color and alpha
    ax.spines['top'].set_edgecolor((1, 1, 1, 0))  # Transparent top border
    ax.spines['right'].set_edgecolor((1, 1, 1, 0))  # Transparent right border
    ax.spines['left'].set_edgecolor((1, 1, 1, 0))  # Transparent left border
    ax.spines['bottom'].set_edgecolor((1, 1, 1, 0))  # Transparent bottom border

    # Make x and y ticks transparent
    ax.tick_params(axis='x', color=(1, 1, 1, 0))  # Transparent x ticks
    ax.tick_params(axis='y', color=(1, 1, 1, 0))  # Transparent y ticks

    # ax.xaxis.set_major_locator(mdates.DayLocator())
    # ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format as "Dec

    ax.set_title(f"{metric.upper()} for Several Out-of-Sample Forecasts")
    ax.set_xlabel("Starting Date of the Forecasting Horizon")
    ax.set_ylabel(f"Horizon Averaged {metric.upper()}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(file_path.replace(".csv",".png"), dpi=600)
    plt.show()

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


def get_average_metrics(metrics:dict)->dict:
    keys = metrics[list(metrics)[0]].keys()
    res = {
        key: np.mean(
            [metrics[timestamp][key] for timestamp in list(metrics.keys())]
        ) for key in keys
    }
    return res


class TaskPaths:

    train_forecast = ['trained','forecast']

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

        # self.X_for_model = None
        # self.y_for_model = None
        self.verbose = verbose
        self.base_ds = None
        self.forecaster = None

        self.results = {}
        self.metrics = {}
        self.contributions = {}
        self.cutoffs = []

        self.model_dataset_pars = None
        self.optuna_pars = None


    def train_evaluate_out_of_sample(
            self, folds:int, X_train: pd.DataFrame or None, y_train: pd.DataFrame or None,
            ds: HistForecastDataset or None, do_fit:bool)->tuple[pd.DataFrame,pd.Series]:

        print('\n')
        print(f'--- ENTERING train_evaluate_out_of_sample() model={self.model_label} --- ')

        if ds is None:
            ds = self.base_ds
        else:
            if self.verbose:
                print(f"Using external dataset for {self.model_label}")

        if ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if ds.exog_hist is None or ds.target_hist is None:
            raise ValueError("Preprocessing of the features in the dataset class is not done")
        if self.forecaster is None:
            raise ReferenceError("Forecaster class is not initialized")
        if folds < 1:
            raise ValueError(f"CV is not possible with folds < 1 Given={folds}")

        # if not self.cutoffs:
        #     self.cutoffs = get_ts_cutoffs(ds, folds=folds) # last one is the latest one
        #     if self.verbose:
        #         print(f"Setting cutoffs N={len(self.cutoffs)} with cutoff[0]={self.cutoffs[0]}")
        # if len(self.cutoffs) != folds:
        #     raise ValueError(f"There are already {self.cutoffs} cutoffs while passed folds={folds}")

        X_for_model = X_train
        y_for_model = y_train

        if X_train is None and y_train is None:
            X_for_model: pd.DataFrame = ds.exog_hist
            y_for_model: pd.Series = ds.target_hist
            if self.verbose:
                print(f"Using full model dataset from {y_for_model.index[0]} to {y_for_model.index[-1]} "
                      f"with {len(y_for_model)/24/7} weeks "
                      f"({len(y_for_model)/len(ds.forecast_idx)} horizons)")
            if self.verbose:
                print(f"Running CV {folds} folds for {self.model_label}. "
                      f"Setting {len(X_for_model.columns)} "
                      f"features from dataset (lags_target={ds.lags_target}))")
        else:
            if self.verbose:
                print(f"Using external dataset X_train={X_train.shape} y_train={y_train.shape} "
                      f"Running CV {folds} folds for {self.model_label}. "
                      f"Given {len(X_for_model.columns)} features (lags_target={ds.lags_target}))")

        target = ds.target_key

        if not len(y_for_model) % len(ds.forecast_idx) == 0:
            print(f"Train: {y_for_model.index[0]} to {y_for_model.index[-1]} ({len(y_for_model.index)/7/24} weeks, "
                  f"{len(y_for_model.index)/len(ds.forecast_idx)} horizons) Horizon={len(ds.forecast_idx)/7/24} weeks")
            print(f"Test: {ds.forecast_idx[0]} to {ds.forecast_idx[-1]}")
            raise ValueError("Train set size should be divisible by the test size")

        # clean up (TODO: refactor the class to avoid this...)
        del self.results; self.results = {}
        del self.metrics; self.metrics = {}
        del self.contributions; self.contributions = {}

        # cutoffs, splits = get_ts_cutoffs(ds, folds=folds) # last one is the latest one
        cutoffs, splits = compute_timeseries_split_cutoffs(
            # ds.hist_idx,
            X_for_model.index,
            horizon=len(ds.forecast_idx),
            folds=folds,
            # min_train_size=len(X_for_model.index) - folds * len(ds.forecast_idx),
        )

        if self.verbose:
            print(f"Setting cutoffs N={len(cutoffs)} with cutoff[0]={cutoffs[0]}")

        # sanity check that CV cutoffs are correctly set assuming multi-day forecasting horizon
        for idx, (c, (train_idx, test_idx)) in enumerate(zip(cutoffs, splits), start=1):
            # train_idx = y_for_model[train_i].index
            # test_idx = y_for_model[test_i].index

            # in case there is no data due to externally set cutoffs
            if idx > 0 and len(train_idx) == 0 or len(test_idx) == 0:
                if self.verbose:
                    print(f"Warning! Empty train data batch idx={idx}/{len(cutoffs)}. Skipping.")
                continue

            if not len(train_idx) % len(test_idx) == 0:
                print(f"Train: {train_idx[0]} to {train_idx[-1]} ({len(train_idx)/7/24} weeks, "
                      f"{len(train_idx)/len(test_idx)} horizons) Horizon={len(test_idx)/7/24} weeks")
                print(f"Test: {test_idx[0]} to {test_idx[-1]}")
                raise ValueError("Train set size should be divisible by the test size")

            # if not len(test_idx) & 24:
            #     print(f"Test: {test_idx[0]} to {test_idx[-1]} N={len(test_idx)}")
            #     raise ValueError("Test set size should be divisible by 24")

            if train_idx[-1] > X_for_model.index[-1]:
                raise ValueError(f"train_idx[-1]={train_idx[-1]} > X_for_model.index[-1]={X_for_model.index[-1]}")
            if train_idx[0] < X_for_model.index[0]:
                raise ValueError(f"train_idx[0]={train_idx[0]} < X_for_model.index[0]={X_for_model.index[0]}")

        for idx, (c, (train_idx, test_idx)) in enumerate(zip(cutoffs, splits), start=1):

            if len(train_idx) == 0 or len(test_idx) == 0:
                if self.verbose:
                    print(f"Warning! Empty train data batch idx={idx}/{len(cutoffs)}. Skipping.")
                continue

            # fit MapieRegressor(estimator=xbg.XBGRegressor(), method='naive', cv='prefit') model on the past data for current fold
            if do_fit: self.forecaster.fit(X_for_model.loc[train_idx], y_for_model.loc[train_idx])
            # get forecast for the next 'horizon' (data unseen during train time)
            result:pd.DataFrame = self.forecaster.forecast_window(
                X_for_model.loc[test_idx], y_for_model.loc[train_idx], lags_target=ds.lags_target,
            )
            # 'result' has f'{target}_actual' and f'{target}_fitted' columns with N=horizon number of timesteps
            result[f'{target}_actual'] = y_for_model.loc[test_idx] # add actual target values to result for error estimation
            result_detransformed = result.apply(ds.inv_transform_target_series)
            if not validate_dataframe_simple(result_detransformed):
                if self.verbose:
                    print(f"Error! Nans in the forecasted dataframe for "
                          f"model={self.model_label} target={target} features={len(X_for_model.columns)} "
                          f"idx={idx} lags={ds.lags_target} Number of nans={len(result_detransformed.isna().sum())} ")
                raise ValueError(f"Forecasting result contains nans. Fitted={result[f'{target}_fitted']}")
                # result_detransformed.interpolate(method='time', inplace=True)
                # result = result_detransformed.apply(ds.transform_target_series)
            # collect results (Dataframe with actual, fitted, lower and upper) for each fold (still transformed!)
            self.results[c] = result
            # compute error metrics (RMSE, sMAPE...) over the entire forecasted window
            self.metrics[c] = compute_error_metrics(target, result_detransformed)
            # print error metrics
            if self.verbose:
                self.print_average_metrics(
                    f"Fold {idx}/{len(cutoffs)} cutoff={c} | "
                    f"{'FIT' if do_fit else 'INF'} | {self.model_label} | " + \
                    f"Train={X_for_model.loc[train_idx].shape} ",
                    self.metrics[c]
                )

        # print averaged over all CV folds error metrics
        if self.verbose:
            self.print_average_metrics(
                f"Average over {len(self.metrics)} folds | {self.model_label} | ",
                get_average_metrics(self.metrics))

        # self.X_for_model = X_for_model
        # self.y_for_model = y_for_model

        print(f'--- ENTERING train_evaluate_out_of_sample() model={self.model_label} --- ')
        return X_for_model, y_for_model

    def clear(self):
        # del self.X_for_model
        # del self.y_for_model
        del self.results
        del self.metrics
        del self.contributions
        gc.collect()

    # ---------- SET DATASET FOR THE BASE MODEL ------------

    def set_dataset_from_ds(self, ds:HistForecastDataset):
        self.base_ds = ds

    def _set_dataset_from_df(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame, pars:dict)->HistForecastDataset:
        if len(df_hist.columns)-1 != len(df_forecast.columns):
            raise ValueError(f'df_hist and df_forecast must have same number of columns')
        if len(df_hist.columns) == 0 or len(df_forecast.columns) == 0:
            raise ValueError(f'df_hist and df_forecast must have at least one column')
        pars = pars | {'verbose':self.verbose, 'target':self.target}
        base_ds = HistForecastDataset(
            df_historic=df_hist, df_forecast=df_forecast, pars=pars
        )
        return base_ds

    def set_dataset_from_df(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame, pars:dict):
        self.base_ds = self._set_dataset_from_df(df_hist, df_forecast, pars)

    def _set_dataset_from_finetuned(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame)->HistForecastDataset:

        dir = self.to_finetuned()
        if self.verbose:
            print(f"Setting dataset for base model {self.model_label} from finetuning directory {dir}")

        with open(dir+'dataset.json', 'r') as f:
            self.model_dataset_pars = json.load(f)

        with open(dir+'best_parameters.json', 'r') as f:
            self.optuna_pars = json.load(f)

        # accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        # unexpected_params = {key: value for key, value in self.model_dataset_pars.items() if key not in accepted_params}
        # accepted_params = {key: value for key, value in self.model_dataset_pars.items() if key in accepted_params}
        self.model_dataset_pars = self.model_dataset_pars | {'verbose':self.verbose}
        ds = HistForecastDataset(df_historic=df_hist, df_forecast=df_forecast, pars=self.model_dataset_pars)
        return ds

    def set_dataset_from_finetuned(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        self.base_ds = self._set_dataset_from_finetuned(df_hist, df_forecast)
        self.base_ds.run_preprocess_pipeline(self.optuna_pars | self.model_dataset_pars)

    def _set_load_dataset_from_dir(self, dir:str, df_hist:pd.DataFrame, df_forecast:pd.DataFrame)->HistForecastDataset:

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Setting dataset for base model {self.model_label} from directory {dir}")

        with open(dir+'dataset.json', 'r') as f:
            self.model_dataset_pars = json.load(f)

        with open(dir+'best_parameters.json', 'r') as f:
            self.optuna_pars = json.load(f)

        # accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        # unexpected_params = {key: value for key, value in self.model_dataset_pars.items() if key not in accepted_params}
        # accepted_params = {key: value for key, value in self.model_dataset_pars.items() if key in accepted_params}

        # target_scaler = joblib.load(dir + 'target_scaler.pkl')
        #
        # feature_scaler = joblib.load(dir + 'feature_scaler.pkl')

        self.model_dataset_pars['target_scaler'] = dir + 'target_scaler.pkl'
        self.model_dataset_pars['feature_scaler'] = dir + 'feature_scaler.pkl'
        self.model_dataset_pars['verbose'] = self.verbose
        #
        # ds_pars = accepted_params | {
        #     'target_scaler':target_scaler, 'feature_scaler':feature_scaler, 'verbose':self.verbose
        # }
        ds = HistForecastDataset(df_historic=df_hist, df_forecast=df_forecast, pars = self.model_dataset_pars)
        return ds

    def set_load_dataset_from_dir(self, dir:str, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        self.base_ds = self._set_load_dataset_from_dir(dir, df_hist, df_forecast)
        self.base_ds.run_preprocess_pipeline(self.optuna_pars | self.model_dataset_pars)

    # ---------- SET MODEL --------------

    def set_forecaster_from_dir(self, dir:str) -> tuple[dict,dict]:

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Setting base forecaster {self.model_label} from directory {dir}")

        with open(dir+'best_parameters.json', 'r') as f:
            self.optuna_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.optuna_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.optuna_pars.items() if key in accepted_params}

        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.target,
            model_pars=accepted_params, verbose=self.verbose
        )

        return accepted_params, unexpected_params

    def load_forecaster_from_dir(self, dir:str) -> tuple[dict,dict]:

        dir = self.to_dir(dir=dir)

        if self.verbose:
            print(f"Loading base forecaster {self.model_label} from directory {dir}")

        # if self.base_ds is None:
        #     raise ReferenceError(f"Dataset class for base forecaster {self.model_label} is not initialized")

        with open(dir+'best_parameters.json', 'r') as f:
            self.optuna_pars = json.load(f)

        accepted_params = list(inspect.signature(HistForecastDataset.__init__).parameters.keys())[1:]  # Exclude 'self'
        unexpected_params = {key: value for key, value in self.optuna_pars.items() if key not in accepted_params}
        accepted_params = {key: value for key, value in self.optuna_pars.items() if key in accepted_params}

        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.target,
            model_pars=accepted_params, verbose=self.verbose
        )
        self.forecaster.load_model(dir + 'model.joblib')

        # self.forecaster.lag_y_past = self.base_ds.target_hist

        return accepted_params, unexpected_params

    # ----------



    def print_average_metrics(self, prefix:str, metrics:dict):
        print(prefix + f"RMSE={metrics['rmse']:.1f} "
                       f"CI_width={metrics['prediction_interval_width']:.1f} "
                       f"sMAPE={metrics['smape']:.2f}")


    def finetune(self, trial:optuna.Trial, cv_metrics_folds:int):
        if self.base_ds is None:
            raise ReferenceError("Dataset class is not initialized")

        self.base_ds.reset_engineered()
        ds_config = suggest_values_for_ds_pars_optuna(
            self.base_ds.init_pars['feature_engineer'], trial=trial, fixed = self.base_ds.init_pars
        )
        ds_config.update(self.base_ds.init_pars)
        self.base_ds.run_preprocess_pipeline(config=ds_config)

        self.forecaster = None
        params = get_parameters_for_optuna_trial(self.name_base_model, trial=trial)
        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.base_ds.target_key,
            model_pars=params, verbose=self.verbose
        )

        try:
            self.train_evaluate_out_of_sample(folds = cv_metrics_folds, ds=None, X_train=None, y_train=None, do_fit=True)
        except ValueError as e:
            if self.verbose: print(f"ERROR! Optimization iteration run failed! with \n{e}")
            self.forecaster.reset_model()
            gc.collect()
            return np.inf

        average_metrics = get_average_metrics(self.metrics)
        res = float( average_metrics['rmse'] ) # Average over all CV folds

        self.forecaster.reset_model()

        gc.collect()

        return res


    def save_results(self, dir:str, ds:HistForecastDataset or None):

        if ds is None:
            ds = self.base_ds

        dir = self.to_dir(dir=dir)
        if len(self.results.keys()) == 0:
            raise ReferenceError(f"No results found for base model {self.model_label}")

        metadata = {
            'start_date':ds.hist_idx[0].isoformat(),
            'end_date':ds.hist_idx[-1].isoformat(),
            'horizon':len(ds.forecast_idx),
            'features':ds.exog_hist.columns.tolist(),
            'error_metrics':{key.isoformat() : val for key, val in self.metrics.items()}
        }
        with open(dir+'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        df_result = pd.concat([
            self.results[key].apply(ds.inv_transform_target_series) for key in list(self.results.keys())
        ], axis=0)
        df_result.to_csv(dir+'result.csv')

        save_datetime_now(dir) # save when the training was done

        if self.verbose:
            print(f"Results of {self.model_label} fits are saved into {dir}")

    def save_full_model(self, dir:str, ds:HistForecastDataset or None):

        if ds is None:
           ds = self.base_ds

        dir = self.to_dir(dir=dir)

        if ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if self.forecaster is None:
            raise ReferenceError("Forecaster is not initialized")

        check_is_fitted(self.forecaster.model)
        self.forecaster.save_model(dir + 'model.joblib')

        if not ds.target_scaler is None:
            joblib.dump(ds.target_scaler, dir + 'target_scaler.pkl')

        if not ds.feature_scaler is None:
            joblib.dump(ds.feature_scaler, dir + 'feature_scaler.pkl')

        with open(dir+'dataset.json', 'w') as f:
            json.dump(self.model_dataset_pars, f, indent=4)

        with open(dir+'best_parameters.json', 'w') as f:
            json.dump(self.optuna_pars, f, indent=4)

        if self.verbose:
            print(f"Metadata for the trained base model {self.model_label} and dataset are saved into {dir}")

    def run_save_forecast(self, X_test:pd.DataFrame or None, y_train:pd.DataFrame or None, folds:int):
        if self.base_ds is None:
            raise ReferenceError("Dataset class is not initialized")
        if self.forecaster is None:
            raise ReferenceError("Forecaster is not initialized")
        #
        #
        # self.cv_train_test(folds=folds, X_train=None, y_train=None, do_fit=False)
        # self.save_full_model(dir='forecast')

        if X_test is None: X_test = self.base_ds.exog_forecast
        if y_train is None: y_train = self.base_ds.target_hist # needed for lagged target as features
        forecast = self.forecaster.forecast_window(X_test, y_train, lags_target=self.base_ds.lags_target)
        forecast = forecast.apply(self.base_ds.inv_transform_target_series)
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
        self.meta_ds = None
        self.base_models = {
            model_name : BaseModelTasks(self.target, model_name, self.working_dir, verbose)
            for model_name in self.names_base_models
        }

        # self.X_ensemble = None
        # self.y_ensemble = None

        # self.X_meta = None
        # self.y_meta = None
        self.cutoffs = None

    def set_dataset_from_df(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame, pars:dict):
        self.meta_ds = self._set_dataset_from_df(df_hist, df_forecast, pars)

    def set_dataset_from_finetuned(self, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        self.meta_ds = self._set_dataset_from_finetuned(df_hist, df_forecast)
        self.meta_ds.run_preprocess_pipeline(self.optuna_pars | self.model_dataset_pars)

    def set_load_dataset_from_dir(self, dir:str, df_hist:pd.DataFrame, df_forecast:pd.DataFrame):
        self.meta_ds = self._set_load_dataset_from_dir(dir, df_hist, df_forecast)
        self.meta_ds.run_preprocess_pipeline(self.optuna_pars | self.model_dataset_pars)

    # def set_meta_X_y(self, X_meta:pd.DataFrame or None, y_meta:pd.Series or None):
    #     if ((not X_meta is None) and (not y_meta is None)):
    #         if self.verbose:
    #             print(f"Manually setting X_meta={len(X_meta)} and y_meta={len(y_meta)} "
    #                   f"(While current meta dataset has "
    #                   f"X_meta={len(self.meta_ds.exog_hist)} and y_meta={len(self.meta_ds.target_hist)}).")
    #         self.X_meta = X_meta
    #         self.y_meta = y_meta
    #     else:
    #         self.X_meta = self.meta_ds.exog_hist
    #         self.y_meta = self.meta_ds.target_hist
    #
    #     if not len(self.X_meta) == len(self.y_meta):
    #         raise ValueError(f"Size of X_meta={self.X_meta.shape} and y_meta={self.y_meta.shape} do not match")


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

    def train_evaluate_out_of_sample_base_models(self, cv_folds_base:int, do_fit:bool):
        if self.verbose:
            print(f"Running CV {'train-test' if do_fit else 'test only'} {list(self.base_models.keys())} "
                  f"base model using their respective datasets")
        for base_model_name, base_model_class in self.base_models.items():
            base_model_class.train_evaluate_out_of_sample(
                cv_folds_base+1, None, None, ds=None, do_fit=do_fit
            )

    def create_X_y_for_model_from_base_models_cv_folds(
            self, X_meta:pd.DataFrame or None, y_meta : pd.Series or None,
            cv_folds_base_to_use:int,
            use_base_models_pred_intervals:bool ) -> tuple[pd.DataFrame, pd.Series]:

        if ((not X_meta is None) and (not y_meta is None)):
            if self.verbose:
                print(f"Manually setting X_meta={len(X_meta)} and y_meta={len(y_meta)} "
                      f"(While current meta dataset has "
                      f"X_meta={len(self.meta_ds.exog_hist)} and y_meta={len(self.meta_ds.target_hist)}).")
            X_meta = X_meta
            y_meta = y_meta
        else:
            X_meta = self.meta_ds.exog_hist
            y_meta = self.meta_ds.target_hist

        if not len(X_meta) == len(self.meta_ds.exog_hist):
            raise ValueError('X_meta and y_meta have different length')

        if not len(X_meta) == len(y_meta):
            raise ValueError(f"Size of X_meta={X_meta.shape} and y_meta={y_meta.shape} do not match")

        target = self.meta_ds.target_key


        if use_base_models_pred_intervals: features_from_base_models = ['fitted','lower','upper']
        else: features_from_base_models = ['fitted']

        cv_folds_base = len(self.base_models[list(self.base_models.keys())[-1]].results)
        if self.verbose:
            print(f"Combining {cv_folds_base} "
                  f"base-model CV folds to create a training set for meta-model {self.model_label}")

        if cv_folds_base < 3:
            raise ValueError(f"Number of CV folds for base models is too small. Should be >3 Given: {cv_folds_base}")

        X_meta_train_list, y_meta_train_list = [], []


        # cutoffs, splits = get_ts_cutoffs(self.meta_ds, folds=cv_folds_base) # last one is the latest one
        cutoffs, splits = compute_timeseries_split_cutoffs(
            X_meta.index,
            horizon=len(self.meta_ds.forecast_idx),
            folds=cv_folds_base,
            # min_train_size=5*len(self.meta_ds.forecast_idx)
        )

        if self.verbose:
            print(f"Setting cutoffs N={len(cutoffs)} with cutoff[0]={cutoffs[0]}")

        cutoffs = cutoffs[-cv_folds_base_to_use:]
        splits = splits[-cv_folds_base_to_use:]

        # sanity check that CV cutoffs are correctly set assuming multi-day forecasting horizon
        for idx, (c, (train_i, test_i)) in enumerate(zip(cutoffs, splits), start=1):

        # cutoffs = get_ts_cutoffs(self.meta_ds, folds=cv_folds_base)
        # cutoffs = cutoffs[-cv_folds_base_to_use:]
        #
        # X_meta_train_list, y_meta_train_list = [], []
        # for i_cutoff, cutoff in enumerate(cutoffs):

            # get indexes for train-test split
            resutls:dict = self.base_models[list(self.base_models.keys())[0]].results
            # if not cutoff in list(resutls.keys()):
            #     raise ValueError(f"Expected cutoff {cutoff} is not in results.keys()={list(resutls.keys())}")

            target_actual = resutls[c][f'{target}_actual']
            curr_tst_index = target_actual.index
            forecasts_df = pd.DataFrame(index=curr_tst_index)

            # extract results from base model forecasts
            for base_model_name, base_model_class in self.base_models.items():
                for key in features_from_base_models:
                    forecasts_df[f'base_model_{base_model_name}_{key}'] = \
                        base_model_class.results[c][f'{target}_{key}']

            # add meta-features
            if not X_meta is None:
                # plt.plot(curr_tst_index, [1] * len(curr_tst_index), '|', color='blue')
                # Plot testing indices
                # plt.plot(X_meta.index, [2] * len(X_meta.index), '|', color='orange')
                # plt.show()
                forecasts_df = forecasts_df.merge(X_meta.loc[curr_tst_index], left_index=True, right_index=True)

            X_meta_train_list.append(forecasts_df)
            y_meta_train_list.append(target_actual)

        # Concatenate all folds and check shapes
        X_meta_model_train = pd.concat(X_meta_train_list)
        y_meta_model_train = pd.Series(pd.concat(y_meta_train_list))

        # check if shapes are correct
        model_names = list(self.base_models.keys())
        if not X_meta is None:
            if not (len(X_meta_model_train.columns) ==
                    len(X_meta.columns) + len(model_names)*len(features_from_base_models)):
                raise ValueError(f"Expected {len(X_meta.columns)+len(model_names)} columns in "
                                 f"X_meta_model_train. Got={len(X_meta_model_train.columns)}")
        else:
            if not (len(X_meta_model_train.columns) == len(model_names)):
                raise ValueError(f"Expected {len(model_names)} columns in "
                                 f"X_meta_model_train. Got={len(X_meta_model_train.columns)}")

        # self.X_ensemble = X_meta_model_train
        # self.y_ensemble = y_meta_model_train

        if self.verbose:
            print(f"Trining data for meta-model {self.model_label} is collected")

        return X_meta_model_train, y_meta_model_train

    # def cv_train_test_ensemble(self, cv_folds_base:int, do_fit:bool):
    #     self.train_evaluate_out_of_sample(
    #         folds=cv_folds_base, X_train=self.X_for_model, y_train=self.y_for_model, ds=self.meta_ds, do_fit=do_fit
    #     )

    def finetune(self, trial:optuna.Trial, cv_metrics_folds:int):

        cv_folds_base = len(self.base_models[list(self.base_models.keys())[-1]].results)
        # if not (cv_folds_base > cv_metrics_folds+3):
        #     raise ValueError("There should be more CV folds for base models then for ensemble model. "
        #                      f"Given CV ensemble {cv_metrics_folds}+3 folds and base models {cv_folds_base} folds")

        if self.meta_ds is None:
            raise ReferenceError("Dataset class is not initialized")

        self.meta_ds.reset_engineered()
        ds_config = suggest_values_for_ds_pars_optuna(
            self.meta_ds.init_pars['feature_engineer'], trial=trial, fixed = self.meta_ds.init_pars
        )
        ds_config.update(self.meta_ds.init_pars)
        self.meta_ds.run_preprocess_pipeline(config=ds_config)


        self.forecaster = None
        params = get_parameters_for_optuna_trial(self.name_base_model, trial=trial)
        self.forecaster = instantiate_base_forecaster(
            model_name=self.name_base_model, target=self.meta_ds.target_key,
            model_pars=params, verbose=self.verbose
        )

        use_base_models_pred_intervals = trial.suggest_categorical(
                'use_base_models_pred_intervals', [True, False]
        )

        # self.set_meta_X_y(X_meta=self.meta_ds.exog_hist, y_meta=self.meta_ds.target_hist)
        X_ensemble, y_ensemble = self.create_X_y_for_model_from_base_models_cv_folds(
            X_meta=self.meta_ds.exog_hist, y_meta=self.meta_ds.target_hist,
            cv_folds_base_to_use=cv_folds_base,
            use_base_models_pred_intervals = use_base_models_pred_intervals
        )

        self.train_evaluate_out_of_sample(
            folds=cv_metrics_folds, X_train=X_ensemble, y_train=y_ensemble, ds=self.meta_ds, do_fit=True
        )


        average_metrics = get_average_metrics(self.metrics)
        res = float( average_metrics['rmse'] ) # Average over all CV folds

        del X_ensemble; X_ensemble = None
        del y_ensemble; y_ensemble = None
        del self.results; self.results = {}
        del self.metrics; self.metrics = {}
        del self.contributions; self.contributions = {}

        gc.collect()

        return res


    def run_save_forecast(self, X_test:pd.DataFrame or None, y_train:pd.DataFrame or None, folds:int):

        use_base_models_pred_intervals = self.optuna_pars['use_base_models_pred_intervals']
        if use_base_models_pred_intervals: features_from_base_models = ['fitted','lower','upper']
        else: features_from_base_models = ['fitted']

        X_test_ = pd.DataFrame(index=self.meta_ds.forecast_idx)
        for base_model_name, base_model_class in self.base_models.items():
            dir = base_model_class.to_forecast()
            df_i = pd.read_csv(dir+'forecast.csv', index_col='date')
            df_i.index = pd.to_datetime(df_i.index)
            df_i = df_i.apply(base_model_class.base_ds.transform_target_series)
            for key in features_from_base_models:
                vals = df_i[f'{self.meta_ds.target_key}_{key}']
                if vals.isnull().values.any():
                    raise ValueError(f"Found NaNs in loaded {base_model_name} forecast. ")
                X_test_[f'base_model_{base_model_name}_{key}'] = vals
                if len(X_test_) != len(self.meta_ds.forecast_idx):
                    raise ValueError(f"Expected {len(self.meta_ds.forecast_idx)} columns in X_test. Index mismatch.")

        if not X_test is None:
            X_test_ = X_test_.merge(X_test, left_index=True, right_index=True)

        if not validate_dataframe_simple(X_test_) or len(X_test_) != len(self.meta_ds.forecast_idx):
            raise ValueError("Validation check of forecasts_df failed.")

        pars = self.optuna_pars | self.model_dataset_pars
        if y_train is None: y_train = self.meta_ds.target_hist # needed for lagged target as features
        forecast = self.forecaster.forecast_window( X_test_, y_train, lags_target=pars['lags_target'] )
        forecast = forecast.apply(self.meta_ds.inv_transform_target_series)

        dir = self.to_forecast()
        if self.verbose: print(f"Saving {dir+'forecast.csv'}")
        forecast.to_csv(dir+'forecast.csv')


class ForecastingTaskSingleTarget:

    # def __init__(self, target:str, task:dict, outdir:str, verbose:bool):
    #     # main output directory
    #     self.verbose = verbose
    #     if not os.path.isdir(outdir):
    #         if self.verbose: print(f"Creating {outdir}")
    #         os.mkdir(outdir)
    #
    #     # init dataclass
    #     self.target = task['target']
    #     self.outdir_ = outdir
    #
    #
    #     # load dataset
    #     df_history, df_forecast = load_data(target, verbose=verbose)
    #     print(f"Loaded dataset has features: {df_history.columns.tolist()}")
    #     # df_forecast = df_forecast[1:] # remove df_history[-1] hour
    #
    #     # restrict to required features
    #     features = task['features']
    #     features_to_restrict : list = []
    #     for feature in features:
    #         # preprocess some features
    #         if feature  == 'weather':
    #             # TODO IMPROVE (USE OPENMETEO CLASS HERE)
    #             weather_features:list = _get_weather_features(df_history)
    #             features_to_restrict += weather_features
    #     if not features:
    #         if verbose: print(f"No features selected for {target}. Using all features: \n{df_forecast.columns.tolist()}")
    #         features_to_restrict = df_forecast.columns.tolist()
    #
    #     # remove unnecessary features from the dataset
    #     print(f"Restricting dataframe from {len(df_history.columns)} features to {len(features_to_restrict)}")
    #     self.df_history = df_history[features_to_restrict + [target]]
    #     self.df_forecast = df_forecast[features_to_restrict]
    #     if not validate_dataframe(self.df_forecast):
    #         raise ValueError("Nans in the df_forecast after restricting. Cannot continue.")

    def __init__(self, df_history:pd.DataFrame, df_forecast:pd.DataFrame, task:dict, outdir:str, verbose:bool):
        self.df_history = df_history
        self.df_forecast = df_forecast
        self.task = task
        self.target = task['target']
        self.verbose = verbose
        self.outdir_ = outdir
        if not os.path.isdir(self.outdir_):
            if self.verbose:
                print(f"Creating {self.outdir_}")
            os.makedirs(self.outdir_)
        # # main output directory
        # self.verbose = verbose
        # if not os.path.isdir(outdir):
        #     if self.verbose: print(f"Creating {outdir}")
        #     os.mkdir(outdir)
        #
        # # init dataclass
        # self.target = task['target']
        # self.outdir_ = outdir
        #
        # if self.verbose: print(f"Given dataset has features: {df_history.columns.tolist()}")
        # # df_forecast = df_forecast[1:] # remove df_history[-1] hour
        #
        # # restrict to required features
        # features = task['features']
        # features_to_restrict : list = []
        # for feature in features:
        #     # preprocess some features
        #     if feature  == 'weather':
        #         # TODO IMPROVE (USE OPENMETEO CLASS HERE)
        #         weather_features:list = _get_weather_features(df_history)
        #         features_to_restrict += weather_features
        # if not features:
        #     if verbose: print(f"No features selected for {self.target}. Using all features: \n{df_forecast.columns.tolist()}")
        #     features_to_restrict = df_forecast.columns.tolist()
        #
        # # remove unnecessary features from the dataset
        # print(f"Restricting dataframe from {len(df_history.columns)} features to {len(features_to_restrict)}")
        # self.df_history = df_history[features_to_restrict + [self.target]]
        # self.df_forecast = df_forecast[features_to_restrict]
        # if not validate_dataframe(self.df_forecast):
        #     raise ValueError("Nans in the df_forecast after restricting. Cannot continue.")


    # def prepare_dataset_for_task(self, df_history:pd.DataFrame, df_forecast:pd.DataFrame, task:dict):
    #
    #     # load dataset
    #     # df_history, df_forecast = load_data(target,datapath=datapath, verbose=verbose)
    #     print(f"Loaded dataset has features: {df_history.columns.tolist()}")
    #     # df_forecast = df_forecast[1:] # remove df_history[-1] hour
    #
    #     # restrict to required features
    #     target = task["target"]
    #     features = task['features']
    #     features_to_restrict : list = []
    #     for feature in features:
    #         # preprocess some features
    #         if feature  == 'weather':
    #             # TODO IMPROVE (USE OPENMETEO CLASS HERE)
    #             weather_features:list = _get_weather_features(df_history)
    #             features_to_restrict += weather_features
    #     if not features:
    #         if self.verbose: print(f"No features selected for {target}. Using all features: \n{df_forecast.columns.tolist()}")
    #         features_to_restrict = df_forecast.columns.tolist()
    #
    #     # remove unnecessary features from the dataset
    #     print(f"Restricting dataframe from {len(df_history.columns)} features to {len(features_to_restrict)}")
    #     df_history = df_history[features_to_restrict + [target]]
    #     df_forecast = df_forecast[features_to_restrict]
    #     if not validate_dataframe(df_forecast):
    #         raise ValueError("Nans in the df_forecast after restricting. Cannot continue.")
    #
    #     return df_history, df_forecast

    # ------- FINETUNING ------------
    def process_finetuning_task_ensemble(self, ft_task):

        model_label = ft_task['model']
        dataset_pars = ft_task['dataset_pars']
        finetuning_pars = ft_task['finetuning_pars']

        # common for all tasks for a given quantity
        dataset_pars['target'] = self.target

        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_, verbose=self.verbose
        )

        ft_outdir = wrapper.to_finetuned()
        with open(ft_outdir+'dataset.json', 'w') as f:
            json.dump(dataset_pars, f, indent=4)

        # ensemble_features = dataset_pars['ensemble_features']; del dataset_pars['ensemble_features']
        wrapper.set_dataset_from_df(self.df_history, self.df_forecast, pars=dataset_pars)
        wrapper.set_datasets_for_base_models_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_base_models_from_dir(dir='finetuning')
        # if ensemble_features == 'cyclic_time':
        #     wrapper.set_meta_X_y(
        #         # _create_time_features(wrapper.ds.hist_idx()),
        #         _create_time_features(self.df_history.index),
        #         # wrapper.ds.target_hist()
        #         self.df_history[self.target]
        #     )


        wrapper.train_evaluate_out_of_sample_base_models(
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

        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_dataset_from_df(self.df_history, self.df_forecast, dataset_pars)

        if self.verbose:
            print(f"Performing optimization study for {self.target} with base model {model_label}")
        study = optuna.create_study(direction='minimize')
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

        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_dataset_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_datasets_for_base_models_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_base_models_from_dir(dir='finetuning')

        model_pars, extra_model_pars = wrapper.set_forecaster_from_dir(dir='finetuning')
        wrapper.train_evaluate_out_of_sample_base_models(
            cv_folds_base=extra_model_pars['cv_folds_base'],#['cv_folds_base_to_use'],
            do_fit=True
        )

        # wrapper.set_meta_X_y(X_meta=wrapper.meta_ds.exog_hist, y_meta=wrapper.meta_ds.target_hist)
        X_ensemble, y_ensemble = wrapper.create_X_y_for_model_from_base_models_cv_folds(
            X_meta=wrapper.meta_ds.exog_hist, y_meta=wrapper.meta_ds.target_hist,
            cv_folds_base_to_use=extra_model_pars['cv_folds_base'],
            use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals'],
        )
        wrapper.train_evaluate_out_of_sample(
            folds=pars['cv_folds'], X_train=X_ensemble,y_train=y_ensemble, ds=wrapper.meta_ds,
            do_fit=True
        )

        t_outdir = wrapper.to_dir('trained')
        wrapper.save_full_model(dir='trained', ds=wrapper.meta_ds) # trained
        wrapper.save_results(dir='trained', ds=wrapper.meta_ds)
        with open(t_outdir+'dataset.json', 'w') as f:
            json.dump(wrapper.model_dataset_pars | wrapper.optuna_pars, f, indent=4)


    def process_training_task_base(self, t_task):
        model_label = t_task['model']
        pars = t_task['pars']

        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_dataset_from_finetuned(self.df_history, self.df_forecast)
        wrapper.set_forecaster_from_dir(dir='finetuning')

        wrapper.train_evaluate_out_of_sample(folds=pars['cv_folds'], X_train=None, y_train=None, ds=None, do_fit=True)

        t_outdir = wrapper.to_dir('trained')
        wrapper.save_full_model(dir='trained', ds=None) # trained
        wrapper.save_results(dir='trained', ds=None)
        with open(t_outdir+'dataset.json', 'w') as f:
            json.dump(wrapper.model_dataset_pars | wrapper.optuna_pars, f, indent=4)

    # ------ FORECASTING -------

    def process_forecasting_task_ensemble(self, f_task):
        model_label = f_task['model']
        folds = f_task['past_folds']
        wrapper = EnsembleModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_load_dataset_from_dir(
            dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast
        )
        wrapper.set_load_datasets_for_base_models_from_dir(
            dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast
        )
        wrapper.load_base_models_from_dir(dir='trained')
        model_pars, extra_model_pars = wrapper.load_forecaster_from_dir(dir='trained')
        wrapper.train_evaluate_out_of_sample_base_models(cv_folds_base=folds, do_fit=False)

        # wrapper.set_meta_X_y( X_meta=wrapper.meta_ds.exog_hist, y_meta=wrapper.meta_ds.target_hist)
        X_ensemble, y_ensemble = wrapper.create_X_y_for_model_from_base_models_cv_folds(
            X_meta=wrapper.meta_ds.exog_hist, y_meta=wrapper.meta_ds.target_hist,
            cv_folds_base_to_use=extra_model_pars['cv_folds_base'],
            use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals']
        )
        wrapper.train_evaluate_out_of_sample(
            folds=folds, X_train=X_ensemble,y_train=y_ensemble, ds=wrapper.meta_ds,
            do_fit=False
        )
        wrapper.save_results(dir='forecast', ds=wrapper.meta_ds)


        # forecast (different feature set)
        # if extra_pars['ensemble_features'] == 'cyclic_time':
        #     wrapper.set_meta_X_y(
        #         _create_time_features(wrapper.meta_ds.forecast_idx),
        #         wrapper.meta_ds.target_hist
        #     )

        # wrapper.update_pretrain_base_models(
        #     cv_folds_base=folds, do_fit=False
        # )
        # wrapper.create_X_y_for_model_from_base_models_cv_folds(
        #     cv_folds_base_to_use=extra_model_pars['cv_folds_base_to_use'],
        #     use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals']
        # )
        # wrapper.cv_train_test_ensemble(cv_folds_base=extra_model_pars['cv_folds_base_to_use'], do_fit=False)
        # wrapper.set_meta_X_y(X_meta=wrapper.meta_ds.exog_forecast, y_meta=np.zeros_like(wrapper.meta_ds.exog_forecast))
        # wrapper.create_X_y_for_model_from_base_models_cv_folds(
        #     cv_folds_base_to_use=extra_model_pars['cv_folds_base'],
        #     use_base_models_pred_intervals=extra_model_pars['use_base_models_pred_intervals']
        # )
        wrapper.run_save_forecast(X_test=wrapper.meta_ds.exog_forecast, y_train=None, folds=folds)

    def process_forecasting_task_base(self, f_task):
        model_label = f_task['model']
        folds = f_task['past_folds']
        wrapper = BaseModelTasks(
            target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
        )
        wrapper.set_load_dataset_from_dir(dir='trained', df_hist=self.df_history, df_forecast=self.df_forecast)
        wrapper.load_forecaster_from_dir(dir='trained')
        wrapper.train_evaluate_out_of_sample(folds=folds, X_train=None, y_train=None, ds=None, do_fit=False)
        wrapper.save_results(dir='forecast', ds=None)
        wrapper.run_save_forecast(X_test=None, y_train=None, folds=folds)

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

    def process_task_determine_the_best_model(self, task:dict, outdir:str):
        train_forecast:list[str] = TaskPaths.train_forecast # ['train', 'forecast']
        metrics = {}

        # save the metrics for each forecasted horizon during (i) initial train run (ii) latest forecast run
        for method in train_forecast:
            metrics[method] = {}
            for t_task in task['task_summarize']:
                model_label:str = t_task['model']
                paths = TaskPaths(
                    target=self.target, model_label=model_label, working_dir=self.outdir_,verbose=self.verbose
                )
                with open(paths.to_dir(method) + "metadata.json", 'r') as file:
                    train_metadata = json.load(file)
                metrics[method][model_label] = train_metadata['error_metrics']

        # Generate the datagram
        datagram = write_summary(metrics)
        df = pd.DataFrame(datagram)
        df.to_csv(outdir + "summary_metrics.csv", index=False)


        # determine the best model in train run (over the last n_folds_best forecasts)
        task_ = task['task_summarize'][0] # same for all
        df = pd.read_csv( outdir + "summary_metrics.csv" )
        res_models, res_drifts = analyze_model_performance(
            data=df, n_folds=task_['n_folds_best'], metric=task_['summary_metric']
        )
        res_models = res_models[res_models['method'] == task_['method_for_best']] # train or forecast
        best_model:str = str(res_models['model_label'].values[0]) # the best performing model
        with open(outdir + "best_model.txt", 'w') as file:
            file.write(best_model)

        # plot_metric_evolution(file_path=outdir+"summary_metrics.csv",metric='rmse')

        #
        # summary_metric = t_task['summary_metric']
        #
        # # Iterate through models and their respective error metrics
        # for model_label, error_metrics in metrics.items():
        #     for forecast_time, metric_values in error_metrics.items():
        #         if summary_metric in metric_values:
        #             summary_data.append({
        #                 "model": model_label,
        #                 "forecast_time": forecast_time,
        #                 "metric_value": metric_values[summary_metric]
        #             })
        #         else:
        #             raise KeyError(f"Metric '{summary_metric}' not found in error metrics for model '{model_label}' at time '{forecast_time}'")
        #
        # # Create a DataFrame
        # df = pd.DataFrame(summary_data)
        #
        # # Calculate average value for each model
        # average_metrics = (
        #     df.groupby("model")["metric_value"].mean().reset_index().rename(columns={"metric_value": "average_metric"})
        # )
        #
        # # Append average row per model to the DataFrame
        # df = df.merge(average_metrics, on="model")
        # df = pd.concat([
        #     df,
        #     pd.DataFrame({
        #         "model": ["Overall Average"],
        #         "forecast_time": ["Average"],
        #         "metric_value": [df["metric_value"].mean()],
        #         "average_metric": [df["metric_value"].mean()]
        #     })
        # ], ignore_index=True)
        # df.to_csv(outdir+t_task['train_forecast']+'summary.csv', index=False)
        #


if __name__ == '__main__':
    # add tests
    pass