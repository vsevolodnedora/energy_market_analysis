import pandas as pd
import numpy as np
import copy, json

from scipy.stats import friedmanchisquare
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Callable


def compute_error_metrics(target:list,result:pd.DataFrame)->dict:

    def smape(actual, predicted):
        """
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

        Parameters:
        actual (array-like): Array of actual values.
        predicted (array-like): Array of predicted values.

        Returns:
        float: sMAPE value as a percentage.
        """
        actual = np.array(actual)
        predicted = np.array(predicted)

        # Avoid division by zero using (|actual| + |predicted|) in the denominator
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        smape_value = np.mean(2 * np.abs(predicted - actual) / denominator) * 100

        return smape_value

    res_dict ={}
    for target_ in target:
        # extract arrays
        y_true = result[f'{target_}_actual'].values
        y_pred = result[f'{target_}_fitted'].values
        y_lower = result[f'{target_}_lower'].values if f'{target_}_lower' in result.columns else np.zeros_like(y_true)
        y_upper = result[f'{target_}_upper'].values if f'{target_}_lower' in result.columns else np.zeros_like(y_true)
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

        # if not np.all(np.isfinite(y_true)):
        #     print ("WARNIGN! y_true contains NaN, infinity, or values too large for dtype('float64').")
        #     y_true = np.nan_to_num(y_true, nan=0.0, posinf=1e10, neginf=-1e10)
        # if not np.all(np.isfinite(y_pred)):
        #     print ("WARNING! y_pred contains NaN, infinity, or values too large for dtype('float64').")
        #     y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)

        # compute metrics
        res_dict[target_] = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true+1e-10, y_pred) * 100,
            'smape': smape(y_true, y_pred),
            'bias': np.mean(y_pred - y_true),
            'variance': np.var(y_pred - y_true),
            'std': np.std(y_pred - y_true),
            'r2':r2_score(y_true, y_pred),
            'prediction_interval_coverage':coverage,
            'prediction_interval_width':np.mean(y_upper - y_lower)
        }

    return res_dict

def compute_error_metrics_aggregate_over_horizon(
        target:str, cv_result:list[pd.DataFrame], unscaler:Callable[[pd.Series], pd.Series] = None)->dict:
    ''' compute error metrics for each forecasted hour using forecasted horizon to aggregate over
        and compute mean and std of the result (aggregating over cv_runs) '''
    cv_metrics = []
    for i in range(len(cv_result)):
        if unscaler is None:
            cv_metrics.append(compute_error_metrics(target, cv_result[i]))
        else:
            # apply function that takes pd.Seris to invert scale the target column
            cv_metrics.append(compute_error_metrics(target, cv_result[i].apply(unscaler)))

    res = {'mean':{}, 'std':{}}
    for metric in cv_metrics[0].keys():
        res['mean'][metric] = np.mean([cv_metrics[i][metric] for i in range(len(cv_metrics))])
        res['std'][metric] = np.std([cv_metrics[i][metric] for i in range(len(cv_metrics))])
    return res

def compute_error_metrics_aggregate_over_cv_runs(
        target:str, cv_result: list[pd.DataFrame], unscaler:Callable[[pd.Series], pd.Series] or None) -> list[dict]:
    ''' Compute error metrics for each forecasted hour using cross-validation runs to aggregate over '''
    n_hours_forecasted = len(cv_result[0].iloc[:, 0])  # Access first column with .iloc
    folds = len(cv_result)
    entries = list(cv_result[0].columns)

    cv_results = copy.deepcopy(cv_result)
    if not unscaler is None:
        for i in range(len(cv_results)):
            cv_results[i] = cv_results[i].apply(unscaler)

    # Reshape the data so that each DataFrame for each hour contains values for all CV runs
    tmp_list = [pd.DataFrame() for _ in range(n_hours_forecasted)]
    for i_hour in range(n_hours_forecasted):
        tmp_dict = {key: [] for key in entries}
        for key in entries:
            for i_cv in range(folds):
                tmp_dict[key].append(float(cv_results[i_cv][key].iloc[i_hour]))  # Use .iloc[i_hour] here
        tmp_list[i_hour] = pd.DataFrame(tmp_dict, columns=entries, index=[i_cv for i_cv in range(folds)])

    # Compute error metrics for each hour aggregating over CV runs
    res = [{} for _ in range(n_hours_forecasted)]
    for i_hour in range(n_hours_forecasted):
        res[i_hour] = compute_error_metrics(target, tmp_list[i_hour])

    return res

def analyze_model_performance( data: pd.DataFrame, n_folds: int, metric: str )->tuple[dict, dict]:

    # targets = list(data['target'].unique())

    # Validate inputs
    if metric not in ['mse', 'rmse', 'mae', 'mape']:
        raise ValueError(f"Invalid metric '{metric}'. Choose from 'mse', 'rmse', 'mae', 'mape'.")

    targets = data['target'].unique()
    horizons = []
    for target in targets:
        horizons_ = data.loc[data['target']==target]['horizon'].unique().tolist()
        if horizons == []: horizons = horizons_
        else:
            if not horizons_ == horizons_:
                raise ValueError("All horizons must have same number of horizons.")

    # select last n_folds
    latest_timestamps = pd.Series(
        [pd.Timestamp(d,tz='UTC') for d in horizons],
    ).nlargest(n_folds).tolist()

    data['horizon'] = pd.to_datetime(data["horizon"])
    data = data[data["horizon"].isin(latest_timestamps)]
    ave_metric = data.groupby(by=['target', 'method', 'model_label'])[metric].mean().reset_index()
    trained_metrics = ave_metric[ave_metric['method'] == 'trained'].drop(columns=['method'],inplace=False)
    forecast_metrics = ave_metric[ave_metric['method'] == 'forecast'].drop(columns=['method'],inplace=False)

    # Finding the best model for each target based on the lowest RMSE
    best_models_train = trained_metrics.loc[trained_metrics.groupby("target")[metric].idxmin()]
    # Creating the JSON structure
    json_output_train = {
        row["target"]: {
            "method": "trained",
            "model_label": row["model_label"],
            "avg_rmse": row[metric],
        } for _, row in best_models_train.iterrows()
    }

    # Finding the best model for each target based on the lowest RMSE
    best_models_forecast = forecast_metrics.loc[forecast_metrics.groupby("target")[metric].idxmin()]
    # Creating the JSON structure
    json_output_forecast = {
        row["target"]: {
            "method": "trained",
            "model_label": row["model_label"],
            "avg_rmse": row[metric],
        } for _, row in best_models_forecast.iterrows()
    }

    return json_output_train, json_output_forecast

def write_summary(summary: dict):
    # Collect datagrams
    datagram = []

    # Iterate through the methods (train, forecast)
    for method, models in summary.items():
        for model_label, horizons in models.items():
            for horizon, metrics in horizons.items():
                # Extract the wind_offshore key and its associated metrics
                for target_key, target_metrics in metrics.items():
                    # Create a single entry for each horizon with wind_offshore as the root key
                    datagram.append({
                        "target": target_key,
                        "method": method,
                        "model_label": model_label,
                        "horizon": horizon,
                        **target_metrics
                    })

    return datagram

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
    sample_target = list(metrics.values())[0]
    sample_keys = list(sample_target[list(sample_target.keys())[0]].keys())

    # Compute average values for each metric and target
    averages = {
        target: {
            key: np.mean([metrics[timestamp][target][key] for timestamp in metrics])
            for key in sample_keys
        }
        for target in list(sample_target.keys())
    }
    return averages
