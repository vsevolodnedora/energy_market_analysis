import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy, json
from datetime import timedelta, datetime
import optuna
import os, csv, re


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


def get_ensemble_name_and_model_names(model_name:str):
    match = re.search(r'\[(.*?)\]', model_name)
    if match: meta_model = match.group(1)
    else: raise NameError(f"Model name {model_name} does not contain '[meta_model_name]' string")

    # extract base-models names
    match = re.search(r'\((.*?)\)', model_name)
    if match: model_names = match.group(1).split(',')  # Split by comma
    else: raise NameError(f"Model name {model_name} does not contain '(model_name_1,model_name_2)' string")

    return meta_model, model_names


