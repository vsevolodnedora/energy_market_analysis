import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy, json
from datetime import timedelta, datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Callable


def visualize_splits(
        full_index: pd.DatetimeIndex,
        cutoffs: list[pd.Timestamp],
        train_test_splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
):
    """
    Visualizes the train and test splits for time series cross-validation.

    Args:
        full_index: DatetimeIndex containing timestamps for the entire dataset
        cutoffs: List of cutoff timestamps used for splitting
        train_test_splits: List of tuples containing train and test indices
    """
    plt.figure(figsize=(15, 8))

    for i, (train_idx, test_idx) in enumerate(train_test_splits):
        # Plot training indices
        plt.plot(train_idx, [i + 1] * len(train_idx), '|', label=f'Train {i + 1}' if i == 0 else "", color='blue')
        # Plot testing indices
        plt.plot(test_idx, [i + 1] * len(test_idx), '|', label=f'Test {i + 1}' if i == 0 else "", color='orange')

    # Plot cutoffs
    for cutoff in cutoffs:
        plt.axvline(cutoff, color='red', linestyle='--', label='Cutoff' if cutoff == cutoffs[0] else "")

    plt.title("Train-Test Splits Visualization")
    plt.xlabel("Time")
    plt.ylabel("Fold")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_timeseries_split_cutoffs(
        full_index: pd.DatetimeIndex,
        horizon: int,
        folds: int
        # min_train_size: int
) -> tuple[list[pd.Timestamp], list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]:
    """
    Computes cutoffs and corresponding train and test indices for time series cross-validation.

    Args:
        full_index: DatetimeIndex containing timestamps for the entire dataset
        horizon: Forecast horizon in hours
        folds: Number of folds for cross-validation
        min_train_size: Minimum required training size in hours

    Returns:
        Tuple containing a list of cutoffs and a list of tuples with train and test indices
    """

    min_train_size = len(full_index) - folds * horizon

    # print(f"| folds={folds} min_train_size={min_train_size} full_index={len(full_index)} |")

    if horizon % 24 != 0:
        raise ValueError("Horizon must be divisible by 24 (whole days).")
    if min_train_size % 24 != 0 or min_train_size < 1:
        raise ValueError("Minimum train size must be divisible by 24 (whole days).")
    # Check if full_index is continuous (hourly data)
    expected_range = pd.date_range(start=full_index.min(), end=full_index.max(), freq='h')
    if not full_index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")

    cutoffs = []
    train_test_splits = []

    step = horizon
    current_index = len(full_index) - 1



    while len(cutoffs) < folds and current_index >= 0:
        cutoff = full_index[current_index]

        # Check if test period fits within the full index
        test_end = cutoff + pd.Timedelta(hours=horizon - 1)
        if test_end not in full_index:
            current_index -= 1
            continue

        # Determine train and test indices
        train_end = cutoff - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=min_train_size - 1)
        test_start = cutoff

        if train_start not in full_index or train_end not in full_index:
            current_index -= 1
            continue

        train_idx = full_index[(full_index >= train_start) & (full_index <= train_end)]
        test_idx = full_index[(full_index >= test_start) & (full_index <= test_end)]

        if len(train_idx) < 1:
            raise ValueError(f"For cutoff={cutoff}, there are no train indices for "
                             f"train_start = {train_start} train_end = {train_end} test_start = {test_start}; "
                             f"full_index = {len(full_index)} len(cutoffs)= {len(cutoffs)} min_train_size={min_train_size}")
        if len(test_idx) < 1:
            raise ValueError(f"For cutoff={cutoff}, there are no train indices for "
                             f"train_start = {train_start} train_end = {train_end} test_start = {test_start}; "
                             f"full_index = {len(full_index)} len(cutoffs)= {len(cutoffs)} min_train_size={min_train_size}")

        # Ensure train and test indices end at 23:00 and start at 00:00
        if train_idx[-1].hour != 23 or train_idx[0].hour != 0:
            current_index -= 1
            continue
        if test_idx[-1].hour != 23 or test_idx[0].hour != 0:
            current_index -= 1
            continue

        # Check divisibility conditions
        if len(train_idx) % len(test_idx) != 0:
            current_index -= 1
            continue

        assert len(test_idx) == horizon
        assert len(train_idx) % horizon == 0

        # Append valid cutoff and splits
        cutoffs.append(cutoff)
        train_test_splits.append((train_idx, test_idx))

        # Move to the next potential cutoff point
        current_index -= step

        # print(f"\tFor cutoff={cutoff} | "
        #       f"(folds={folds}) len(cutoffs)={len(cutoffs)} min_train_size={min_train_size} "
        #       f"full_index={len(full_index)} | train={len(train_idx)} test={len(test_idx)} | "
        #       f"train_start = {train_start} | train_end = {train_end} | test_start = {test_start}")

    if len(cutoffs) < folds:
        raise ValueError("Unable to generate the required number of folds with the given constraints. ")

    # invert so that the last fold is the latest fold
    cutoffs = cutoffs[::-1]
    train_test_splits = train_test_splits[::-1]

    # visualize_splits(full_index, cutoffs, train_test_splits)

    return cutoffs, train_test_splits



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

def save_datetime_now(outdir:str):
    # save when fine-tuning was done
    today = pd.Timestamp(datetime.today()).tz_localize(tz='UTC')
    today = today.normalize() + pd.DateOffset(hours=today.hour) # leave only hours
    with open(f'{outdir}datetime.json', "w") as file:
        json.dump({"datetime": today.isoformat()}, file)


def convert_ensemble_string(input_string):
    # Extract the ensemble name and the components
    ensemble_name = input_string.split('[')[1].split(']')[0]
    components = input_string.split('(')[1].split(')')[0].split(',')

    # Construct the desired format
    output_string = f"meta_{ensemble_name}_" + "_".join(components)
    return output_string