import pandas as pd
import numpy as np
import copy
from datetime import timedelta

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


from typing import Callable

def compute_timeseries_split_cutoffs(
        full_index:pd.DatetimeIndex, horizon:int, delta:int , folds:int, min_train_size:int,
)->list[pd.Timestamp]:

    if folds == 0: return []

    # Number of timesteps to forecast
    horizon_duration = pd.Timedelta(hours=horizon) # ~150 hours
    max_date:pd.Timestamp = full_index.max()
    min_date:pd.Timestamp = full_index.min()
    last_cutoff = max_date - horizon_duration
    # Calculate the time delta between cutoffs
    delta = pd.Timedelta(hours=delta if delta else horizon) # if n_steps_back == horizon : non-overlapping windows
    # Generate cutoffs starting from the end of the time series
    cutoffs = [last_cutoff - i * delta + pd.Timedelta(hours=1) for i in range(folds)]
    if (cutoffs[-1]-min_date < timedelta(hours=min_train_size)):
        raise ValueError(f"Not enough train data for {len(cutoffs)}-cross-validation. "
                         f"(Need {min_train_size} hours at least)"
                         f"Last cutoff = {cutoffs[-1]} min_date={min_date}")
    return cutoffs[::-1] # invert that the last one is the latest one


def compute_error_metrics(target:str,result:pd.DataFrame)->dict:
    res = copy.deepcopy(result)
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
    # undo normalization used everywhere
    # res = res.apply(self.dss[0].inv_transform_target_series)
    # extract arrays
    y_true = res[f'{target}_actual'].values
    y_pred = res[f'{target}_fitted'].values
    y_lower = res[f'{target}_lower'].values
    y_upper = res[f'{target}_upper'].values
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    # compute metrics
    res_dict = {
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

