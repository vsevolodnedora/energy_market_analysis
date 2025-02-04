'''
This script takes data from './output/' which is produced by forecasting and other models
and converts it into a format that is expected by JS code in './deploy/'
Created files are then saved into './deploy/data/
'''
import logging

import pandas as pd
import json
import os
import copy
import numpy as np
from datetime import datetime, timedelta

from forecasting_modules import compute_error_metrics, analyze_model_performance, convert_ensemble_string
from data_collection_modules.german_locations import de_regions
from data_collection_modules.collect_data_entsoe import entsoe_generation_type_mapping
from data_collection_modules.collect_data_smard import DataEnergySMARD

from logger import get_logger
logger = get_logger(__name__)

from typing import Optional

def convert_csv_to_json(df, target, output_dir, prefix, cols):
    """
    Converts a CSV file with time-series data into separate JSON files for each column.

    Args:
        file_path (str): Path to the input .csv file.
        target (str): Target column base name.
        output_dir (str): Directory to save the JSON files.
    """
    # Ensure the index is a pd.Timestamp object
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The index of the CSV file must be a pd.DatetimeIndex.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the columns and save as JSON
    for data_col in cols:
        column = f"{target}_{data_col}"
        if column in df.columns:
            # Convert data to the desired format
            json_data = [
                [int(timestamp.timestamp() * 1000), float(f"{value:.2f}")] for timestamp, value in df[column].items()
            ]

            # Save to a JSON file
            output_path = os.path.join(output_dir, f"{prefix}_{data_col}.json")
            with open(output_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
        else:
            logger.info(f"Column '{column}' not found in the CSV file. (columns={df.columns})")

def save_to_json(df:pd.DataFrame, metadata:dict, fname_json:str, verbose:bool):
    # Convert the index (timestamps) to ISO 8601 strings and reset the index
    df_reset = df.reset_index()
    df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')  # Format as ISO 8601 string

    # Convert DataFrame to JSON serializable format
    json_data = {
        "metadata": metadata,
        "data": df_reset.reset_index()
            .rename(columns={'date': 'datetime'})
            .to_dict(orient='records')
    }
    # Save as JSON file

    with open(fname_json, 'w') as f:
        json.dump(json_data, f, indent=4)
    if verbose: logger.info(f"Saved {fname_json}")

def create_output_dirs(verbose:bool)->tuple[str,str,str]:
    # check if output directory set up or set them up
    data_dir = "./deploy/data/"
    if not os.path.isdir(data_dir):
        if verbose: logger.info(f"Creating {data_dir}")
        os.mkdir(data_dir)

    data_dir_web = data_dir + "forecasts/"
    if not os.path.isdir(data_dir_web):
        if verbose: logger.info(f"Creating {data_dir_web}")
        os.mkdir(data_dir_web)

    data_dir_api = data_dir + "api/"
    if not os.path.isdir(data_dir_api):
        if verbose: logger.info(f"Creating {data_dir_api}")
        os.mkdir(data_dir_api)

    return data_dir, data_dir_web, data_dir_api

def publish_to_api(
        run_label:str='wind_offshore',
        target='wind_offshore',
        avail_regions=('DE_50HZ', 'DE_TENNET'),
        method_type='forecast',  # 'trained'
        results_root_dir='forecasting_modules/output/',
        output_dir='deploy/data/api/',
        verbose: bool = True
):
    """
    Load model forecasts (previous and current) and combine them into a single time-series file and
    save as a JSON file that can be requested via an API.
    Each forecast file initially has 4 columns:
        [f"{target}_actual", f"{target}_fitted", f"{target}_lower", f"{target}_upper"]

    For the API, we only provide the last three, renaming them for simplicity to:
        ["forecast", "ci_lower", "ci_upper"]

    Dataframes have index as pd.Timestamp in UTC, and the same format is provided in the JSON.

    :param target: The forecasting target.
    :param avail_regions: Available regions for forecasting.
    :param method_type: Type of method used (e.g., 'forecast', 'trained').
    :param results_root_dir: Directory containing forecast results.
    :param output_dir: Directory to store API-ready JSON files.
    :param verbose: Whether to print progress information.
    :return: None
    """

    if target != run_label:
        raise NotImplementedError(f"Target '{target}' for run_label '{run_label}' not implemented.")

    if not os.path.isdir(output_dir):
        if verbose: logger.info(f"Creating directory '{output_dir}'")
        os.makedirs(output_dir)

    df_results = pd.DataFrame()
    for de_reg in de_regions:
        if de_reg['name'] in avail_regions:
            key = de_reg['TSO']
            suffix = de_reg['suffix']
            var = target + suffix

            # Load the best model name
            best_model = None
            with open(f"{results_root_dir}{var}/best_model.txt") as f:
                best_model = str(f.read())
            if 'ensemble' in best_model:
                best_model = convert_ensemble_string(best_model)

            # Load past forecasts
            df_res = pd.read_csv(
                f"{results_root_dir}{var}/{best_model}/{method_type}/result.csv",
                index_col=0, parse_dates=True)
            df_res.columns = [col.replace(suffix, '') for col in df_res.columns]  # remove TSO suffix

            # Load current forecast
            df_forecast = pd.read_csv(
                f"{results_root_dir}{var}/{best_model}/{method_type}/forecast.csv",
                index_col=0, parse_dates=True)
            df_forecast.columns = [col.replace(suffix, '') for col in df_forecast.columns]  # remove TSO suffix

            # load timestamp when the model was last trained

            with open(f"{results_root_dir}{var}/{best_model}/{'finetuning'}/datetime.json", "r") as file:
                finetune_time = pd.to_datetime(json.load(file)['datetime'])

            with open(f"{results_root_dir}{var}/{best_model}/{'trained'}/datetime.json", "r") as file:
                train_time = pd.to_datetime(json.load(file)['datetime'])

            with open(f"{results_root_dir}{var}/{best_model}/{'forecast'}/datetime.json", "r") as file:
                forecast_time = pd.to_datetime(json.load(file)['datetime'])

            # Combine past and current forecasts
            df = pd.concat([df_res, df_forecast], axis=0)  # stack dataframes along index
            df.sort_index(inplace=True)  # Ensure the index is sorted
            df = df[~df.index.duplicated(keep='first')]
            df.drop(columns=[f"{target}_actual"], inplace=True)  # Drop the actual data column
            df.rename(columns={
                f"{target}_fitted": "forecast",
                f"{target}_lower": "ci_lower",
                f"{target}_upper": "ci_upper"
            }, inplace=True)

            # Convert the index (timestamps) to ISO 8601 strings and reset the index
            fname = f"{target.lower()}_{key.lower()}.json"

            # generate metadata
            metadata = {
                "file": fname,
                "data_keys":list(df.columns.tolist()),
                "target_name": target,
                "tso_region": de_reg['name'],
                "model_label": best_model,
                "finetune_datetime":finetune_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "train_datetime": train_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "forecast_datetime": forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "source": "https://vsevolodnedora.github.io/energy_market_analysis/",
                "forecast_horizon_hours":len(df_forecast),
                "units": "MW",
                "notes": "None"
            }

            save_to_json(df, metadata, f"{output_dir}{fname}", verbose)

            # Aggregate for total results
            if df_results.empty:
                df_results = df.copy()
            else:
                df_results += df.copy()

    # Save total combined data
    fname = f"{target.lower()}_total.json"
    fpath_total_json = f"{output_dir}{fname}"

    # generate metadata
    metadata = {
        "file": fname,
        "data_keys":list(df_results.columns.tolist()),
        "target_name": target,
        "tso_region": 'DE',
        "model_label": "N/A",
        "finetune_datetime":finetune_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "train_datetime": train_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "forecast_datetime": forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "source": "https://vsevolodnedora.github.io/energy_market_analysis/",
        "forecast_horizon_hours":len(df_forecast),
        "units": "MW",
        "notes": "Aggregated over all regions."
    }

    save_to_json(df_results, metadata, fpath_total_json, verbose)

def OLD_publish_generation(
        run_label:str='wind_offshore',
        target='wind_offshore',
        avail_regions=('DE_50HZ', 'DE_TENNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = 'forecasting_modules/output/',
        database_dir = 'database/',
        output_dir = 'deploy/data/forecasts/',
        verbose:bool = True,
):

    def retain_most_recent_entries(data:dict, N:int):
        # Sort keys by timestamp (assuming keys are sortable timestamps)
        sorted_keys = sorted(data.keys(), reverse=True)
        # Select the most recent N keys
        most_recent_keys = sorted_keys[:N]
        # Create a new dictionary with only the most recent N entries
        recent_entries = {key: data[key] for key in most_recent_keys}
        return recent_entries

    # Ensemble model will be abbreviated for simplicity


    def compute_error_metrics_cutoffs(
            df_, cutoffs:list, horizon:int, target:str, key_actual:str, key_fitted:str)->dict:
        ''' compute errror metrics for batches separated by cutoffs'''
        smard_metrics = {}
        for i, cutoff in enumerate(cutoffs):
            mask_ = (df_.index >= cutoff) & (df_.index < cutoff + pd.Timedelta(hours=horizon))
            actual = df_[key_actual][mask_]
            predicted = df_[key_fitted][mask_]
            df = pd.DataFrame({
                f'{target}_actual':actual.values,
                f'{target}_fitted': predicted.values,
                f'{target}_lower': np.zeros_like(actual.values),
                f'{target}_upper': np.zeros_like(actual.values)
            }, index=actual.index)
            smard_metrics[cutoff] = compute_error_metrics([target], df)
        return smard_metrics

    table = [] # to be shown in 'description'

    df_results = pd.DataFrame()

    df_entsoe = pd.read_parquet(database_dir + 'entsoe/' + 'history.parquet')

    # collect total values from all regions; compute error using average metric and ENTSO-E data
    for de_reg in de_regions:
        if de_reg['name'] in avail_regions:
            key = de_reg['TSO']
            suffix = de_reg['suffix']
            var = target + suffix

            # load sumamry metrics for different models and horizons for a given target varaible
            df = pd.read_csv( f'{results_root_dir}{var}/summary_metrics.csv' )
            res_models, res_drifts = analyze_model_performance(df, n_folds=n_folds, metric=metric)
            res_models = res_models[res_models['method'] == method_type]
            best_model:str = str(res_models['model_label'].values[0])
            if best_model.__contains__('ensemble'):
                best_model = convert_ensemble_string(best_model)

            # load timestamp when the model was last trained
            with open(f"{results_root_dir}{var}/{best_model}/{method_type}/datetime.json", "r") as file:
                train_time = pd.to_datetime(json.load(file)['datetime'])

            # load training properties (features etc.)
            with open(f"{results_root_dir}{var}/{best_model}/{method_type}/metadata.json", "r") as file:
                metadata:dict = json.load(file)

            cutoffs = df[df['method'] == method_type]['horizon'].unique()
            cutoffs = [pd.to_datetime(cutoff) for cutoff in cutoffs]
            horizon = int(metadata['horizon'])
            entsoe_metrics = compute_error_metrics_cutoffs(
                df_=df_entsoe,  cutoffs=cutoffs, horizon=horizon, target=var,
                key_actual=var, key_fitted=f"{target}_forecast{suffix}"
            )
            entsoe_metrics = {date:metric[var] for date, metric in entsoe_metrics.items()}

            smard_metrics = retain_most_recent_entries(entsoe_metrics, n_folds)
            ave_smard_metric = np.average([smard_metrics[time_s][metric] for time_s in smard_metrics.keys()])

            table.append({
                'TSO/Region':key,
                'Train Date':train_time.strftime('%Y-%m-%d'),
                'N Features':len(metadata['features']),
                'Best Model':best_model,
                'RMSE':float(res_models[f'avg_{metric}'].values[0]),
                'TSO RMSE': float(ave_smard_metric),
            })

            # LOAD results file and compute total (for SMARD comparison)
            df_res = pd.read_csv(
                f"{results_root_dir}{var}/{best_model}/{method_type}/result.csv",
                index_col=0, parse_dates=True)
            df_res.columns = [ col.replace(suffix, '') for col in df_res.columns ]


            if df_results.empty: df_results = df_res.copy()
            else: df_results += df_res.copy()


    # -------- FOR TOTAL COMPUTE ERROR OVER THE LAST N HORIZONS --------------- #


    df = pd.read_csv( f'{results_root_dir}{var}/summary_metrics.csv' ) # get cutoffs (general for all models)
    cutoffs = df[df['method'] == method_type]['horizon'].unique()
    cutoffs = [pd.to_datetime(cutoff) for cutoff in cutoffs]
    horizon = int(metadata['horizon'])

    total_metrics = compute_error_metrics_cutoffs(
        df_=df_results, cutoffs=cutoffs, horizon=horizon, target=target,
        key_actual=f'{target}_actual', key_fitted=f'{target}_fitted'
    )
    total_metrics = {date:metric[target] for date, metric in total_metrics.items()}

    total_metrics = retain_most_recent_entries(total_metrics, N=horizon)
    ave_total_metric = np.average([total_metrics[time_s][metric] for time_s in total_metrics.keys()])

    logger.info(f'For {target} average over {n_folds} total RMSE for {target} is {ave_total_metric}')


    # ----------- COMPUTE SMARD ERROR OVER THE LAST N HORIZONS ------------------ #

    df_smard = pd.read_parquet(database_dir + 'smard/' + 'history.parquet')

    df_smard.rename(columns={
        "total_grid_load_forecasted": "load_forecasted",
        "total_grid_load": "load",
    }, inplace=True)
    smard_metrics = compute_error_metrics_cutoffs(
        df_=df_smard,  cutoffs=cutoffs, horizon=horizon, target=target,
        key_actual=target, key_fitted=f"{target}_forecasted"
    )
    smard_metrics = {date:metric[target] for date, metric in smard_metrics.items()}
    smard_metrics = retain_most_recent_entries(smard_metrics, n_folds)
    ave_smard_metric = np.average([smard_metrics[time_s][metric] for time_s in smard_metrics.keys()])

    logger.info(f'For {target} Average over {n_folds} SMARD RMSE for {target} is {ave_smard_metric}')

    table = pd.DataFrame(table)

    # ----------- CONVERT FORECASTS CSV TO JSON FOR WEBPAGE ------------------ #

    possible_types = ['forecast.csv', 'result.csv']

    if not os.path.exists(output_dir):
        if verbose: logger.info(f'Creating output directory {output_dir}')
        os.makedirs(output_dir)

    # Convert .csv past and current forecasts into json files for each TSO
    for de_reg in de_regions:
        if de_reg['name'] in avail_regions:
            key = de_reg['TSO']
            suffix = de_reg['suffix']
            var = target + suffix

            output_dir_ = output_dir + var + '/'

            for ftype in possible_types:
                model_label = str(table[table['TSO/Region']==key]['Best Model'].values[0])
                df = pd.read_csv(
                    f'{results_root_dir}/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]

                convert_csv_to_json(
                    df = df,
                    target=var,
                    output_dir=output_dir_,
                    prefix = 'forecast_curr' if ftype == 'forecast.csv' else 'forecast_prev',
                    cols=[f"fitted", f"lower", f"upper"] if ftype == 'forecast.csv' else [f"actual", f"fitted", f"lower", f"upper"]#[f"actual", f"fitted", f"lower", f"upper"]
                )

    # Compute total from .csv past and current forecasts into json files for each TSO
    output_dir_ = output_dir + target + '/'
    for ftype in possible_types:
        df = pd.DataFrame
        for de_reg in de_regions:
            if de_reg['name'] in avail_regions:
                key = de_reg['TSO']
                suffix = de_reg['suffix']
                var = target + suffix

                model_label = str(table[table['TSO/Region']==key]['Best Model'].values[0])
                df1 = pd.read_csv(
                    f'{results_root_dir}/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]
                df1.columns = [ col.replace(f'{suffix}', '') for col in df1.columns ]

                if df.empty:df = df1.copy()
                else: df = df + df1.copy()

        convert_csv_to_json(
            df = df,
            target=target,
            output_dir=output_dir_,
            prefix = 'forecast_curr' if ftype == 'forecast.csv' else 'forecast_prev',
            cols=[f"fitted", f"lower", f"upper"] if ftype == 'forecast.csv' else [f"actual", f"fitted", f"lower", f"upper"]#[f"actual", f"fitted", f"lower", f"upper"]
        )

    ''' ---------- PREPARE RESULTS FOR SERVING ------------- '''

    # Rename values starting with 'meta_' to 'Ensemble'
    table["Best Model"] = table["Best Model"].apply(lambda x: "Ensemble" if x.startswith("meta_") else x)
    # Round floating point values to integers
    table[r"RMSE"] = table[r"RMSE"].round().astype(int)
    table[r"TSO RMSE"] = table[r"TSO RMSE"].round().astype(int)
    # Save as markdown
    summary_fpath = f'{output_dir}/{target}_notes_en.md'
    table.to_markdown(summary_fpath, index=False)

    # ------------- ENGLISH TEXT ---------------

    intro_sentences = \
        f"""
Our __week-ahead__ forecast has average RMSE of __{ave_total_metric:.0f}__.  
SMARD __day-ahead__ forecast has average accuracy of __{ave_smard_metric:.0f}__. 
    """

    # Reading the markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()
    # Prepending the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content
    # Saving the updated markdown content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)


    # -------------- GERMAN TEXT -------------------

    dictionary_en_de = {
        "Best Model": "Bestes Modell",
        "Average RMSE": "Durchschnittlicher RMSE",
        "TSO/Region": "ÜNB/Region",
        "Train Date": "Trainingsdatum",
        "N Features": "Anzahl der Merkmale"
    }
    table.rename(columns = dictionary_en_de, inplace = True)

    summary_fpath = f'{output_dir}/{target}_notes_de.md'
    table.to_markdown(summary_fpath, index=False)

    intro_sentences = f"""
Unsere __Wochenprognose__ hat einen durchschnittlichen RMSE von __{ave_total_metric:.0f}__.  
Die SMARD __Tagesprognose__ weist eine durchschnittliche Genauigkeit von __{ave_smard_metric:.0f}__ auf.
    """

    # Reading the markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()
    # Prepending the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content
    # Saving the updated markdown content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)


def publish_generation(
        run_label:str='wind_offshore',
        target='wind_offshore',
        avail_regions=('DE_50HZ', 'DE_TENNET'),
        n_folds = 3,
        metric = 'rmse',
        method_type = 'trained', # 'trained'
        results_root_dir = 'forecasting_modules/output/',
        database_dir = 'database/',
        output_dir = 'deploy/data/forecasts/',
        verbose:bool = True,
):

    def retain_most_recent_entries(data:dict, N:int):
        # Sort keys by timestamp (assuming keys are sortable timestamps)
        sorted_keys = sorted(data.keys(), reverse=True)
        # Select the most recent N keys
        most_recent_keys = sorted_keys[:N]
        # Create a new dictionary with only the most recent N entries
        recent_entries = {key: data[key] for key in most_recent_keys}
        return recent_entries

    # Ensemble model will be abbreviated for simplicity

    def get_actual_forecast_from_entsoe(df:pd.DataFrame, key_actual:str, key_fitted:str):
        if key_actual == 'energy_mix':
            actual = df[[key for key in entsoe_generation_type_mapping.keys() if key in df.columns]].sum(axis=1)
            predicted = df["generation_f"]
        else:
            actual = df[key_actual]
            predicted = df[key_fitted]
        return actual, predicted


    def compute_error_metrics_cutoffs(
            df_, cutoffs:list, horizon:int, target:str, key_actual:str, key_fitted:str)->dict:
        ''' compute errror metrics for batches separated by cutoffs'''
        smard_metrics = {}
        for i, cutoff in enumerate(cutoffs):
            mask_ = (df_.index >= cutoff) & (df_.index < cutoff + pd.Timedelta(hours=horizon))
            actual, predicted = get_actual_forecast_from_entsoe(df_, key_actual, key_fitted)
            actual = actual[mask_]
            predicted = predicted[mask_]
            df = pd.DataFrame({
                f'{target}_actual':actual.values,
                f'{target}_fitted': predicted.values,
                f'{target}_lower': np.zeros_like(actual.values),
                f'{target}_upper': np.zeros_like(actual.values)
            }, index=actual.index)
            smard_metrics[cutoff] = compute_error_metrics([target], df)
        return smard_metrics

    table = [] # to be shown in 'description'

    df_results = pd.DataFrame()

    df_entsoe = pd.read_parquet(database_dir + 'entsoe/' + 'history.parquet')

    # collect total values from all regions; compute error using average metric and ENTSO-E data
    for de_reg in de_regions:
        if de_reg['name'] in avail_regions:
            key = de_reg['TSO']
            suffix = de_reg['suffix']
            var = target + suffix

            # load sumamry metrics for different models and horizons for a given target varaible
            df = pd.read_csv( f'{results_root_dir}{var}/summary_metrics.csv' )
            res_models, res_drifts = analyze_model_performance(df, n_folds=n_folds, metric=metric)
            res_models = res_models[res_models['method'] == method_type]
            best_model:str = str(res_models['model_label'].values[0])
            if best_model.__contains__('ensemble'):
                best_model = convert_ensemble_string(best_model)

            # load timestamp when the model was last trained
            with open(f"{results_root_dir}{var}/{best_model}/{method_type}/datetime.json", "r") as file:
                train_time = pd.to_datetime(json.load(file)['datetime'])

            # load training properties (features etc.)
            with open(f"{results_root_dir}{var}/{best_model}/{method_type}/metadata.json", "r") as file:
                metadata:dict = json.load(file)

            cutoffs = df[df['method'] == method_type]['horizon'].unique()
            cutoffs = [pd.to_datetime(cutoff) for cutoff in cutoffs]
            horizon = int(metadata['horizon'])

            entsoe_metrics = compute_error_metrics_cutoffs(
                df_=df_entsoe,  cutoffs=cutoffs, horizon=horizon, target=target + suffix,
                key_actual=target + suffix, key_fitted=f"{target}_forecast{suffix}"
            )
            entsoe_metrics = {date:metric[var] for date, metric in entsoe_metrics.items()}

            smard_metrics = retain_most_recent_entries(entsoe_metrics, n_folds)
            ave_smard_metric = np.average([smard_metrics[time_s][metric] for time_s in smard_metrics.keys()])

            table.append({
                'TSO/Region':key,
                'Train Date':train_time.strftime('%Y-%m-%d'),
                'N Features':len(metadata['features']),
                'Best Model':best_model,
                'RMSE':float(res_models[f'avg_{metric}'].values[0]),
                'TSO RMSE': float(ave_smard_metric),
            })

            # LOAD results file and compute total (for SMARD comparison)
            df_res = pd.read_csv(
                f"{results_root_dir}{var}/{best_model}/{method_type}/result.csv",
                index_col=0, parse_dates=True)
            df_res.columns = [ col.replace(suffix, '') for col in df_res.columns ]


            if df_results.empty: df_results = df_res.copy()
            else: df_results += df_res.copy()


    # -------- FOR TOTAL COMPUTE ERROR OVER THE LAST N HORIZONS --------------- #


    df = pd.read_csv( f'{results_root_dir}{var}/summary_metrics.csv' ) # get cutoffs (general for all models)
    cutoffs = df[df['method'] == method_type]['horizon'].unique()
    cutoffs = [pd.to_datetime(cutoff) for cutoff in cutoffs]
    horizon = int(metadata['horizon'])

    total_metrics = compute_error_metrics_cutoffs(
        df_=df_results, cutoffs=cutoffs, horizon=horizon, target=target,
        key_actual=f'{target}_actual', key_fitted=f'{target}_fitted'
    )
    total_metrics = {date:metric[target] for date, metric in total_metrics.items()}

    total_metrics = retain_most_recent_entries(total_metrics, N=horizon)
    ave_total_metric = np.average([total_metrics[time_s][metric] for time_s in total_metrics.keys()])

    logger.info(f'For {target} average over {n_folds} total RMSE for {target} is {ave_total_metric}')


    # ----------- COMPUTE SMARD ERROR OVER THE LAST N HORIZONS ------------------ #

    df_smard = pd.read_parquet(database_dir + 'smard/' + 'history.parquet')

    df_smard.rename(columns={
        "total_grid_load_forecasted": "load_forecasted",
        "total_grid_load": "load",
    }, inplace=True)
    smard_metrics = compute_error_metrics_cutoffs(
        df_=df_smard,  cutoffs=cutoffs, horizon=horizon, target=target,
        key_actual=target, key_fitted=f"{target}_forecasted"
    )
    smard_metrics = {date:metric[target] for date, metric in smard_metrics.items()}
    smard_metrics = retain_most_recent_entries(smard_metrics, n_folds)
    ave_smard_metric = np.average([smard_metrics[time_s][metric] for time_s in smard_metrics.keys()])

    logger.info(f'For {target} Average over {n_folds} SMARD RMSE for {target} is {ave_smard_metric}')

    table = pd.DataFrame(table)

    # ----------- CONVERT FORECASTS CSV TO JSON FOR WEBPAGE ------------------ #

    possible_types = ['forecast.csv', 'result.csv']

    if not os.path.exists(output_dir):
        if verbose: logger.info(f'Creating output directory {output_dir}')
        os.makedirs(output_dir)

    # Convert .csv past and current forecasts into json files for each TSO
    for de_reg in de_regions:
        if de_reg['name'] in avail_regions:
            key = de_reg['TSO']
            suffix = de_reg['suffix']
            var = target + suffix

            output_dir_ = output_dir + var + '/'

            for ftype in possible_types:
                model_label = str(table[table['TSO/Region']==key]['Best Model'].values[0])
                df = pd.read_csv(
                    f'{results_root_dir}/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]

                convert_csv_to_json(
                    df = df,
                    target=var,
                    output_dir=output_dir_,
                    prefix = 'forecast_curr' if ftype == 'forecast.csv' else 'forecast_prev',
                    cols=[f"fitted", f"lower", f"upper"] if ftype == 'forecast.csv' else [f"actual", f"fitted", f"lower", f"upper"]#[f"actual", f"fitted", f"lower", f"upper"]
                )

    # Compute total from .csv past and current forecasts into json files for each TSO
    output_dir_ = output_dir + target + '/'
    for ftype in possible_types:
        df = pd.DataFrame
        for de_reg in de_regions:
            if de_reg['name'] in avail_regions:
                key = de_reg['TSO']
                suffix = de_reg['suffix']
                var = target + suffix

                model_label = str(table[table['TSO/Region']==key]['Best Model'].values[0])
                df1 = pd.read_csv(
                    f'{results_root_dir}/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]
                df1.columns = [ col.replace(f'{suffix}', '') for col in df1.columns ]

                if df.empty:df = df1.copy()
                else: df = df + df1.copy()

        convert_csv_to_json(
            df = df,
            target=target,
            output_dir=output_dir_,
            prefix = 'forecast_curr' if ftype == 'forecast.csv' else 'forecast_prev',
            cols=[f"fitted", f"lower", f"upper"] if ftype == 'forecast.csv' else [f"actual", f"fitted", f"lower", f"upper"]#[f"actual", f"fitted", f"lower", f"upper"]
        )

    ''' ---------- PREPARE RESULTS FOR SERVING ------------- '''

    # Rename values starting with 'meta_' to 'Ensemble'
    table["Best Model"] = table["Best Model"].apply(lambda x: "Ensemble" if x.startswith("meta_") else x)
    # Round floating point values to integers
    table[r"RMSE"] = table[r"RMSE"].round().astype(int)
    table[r"TSO RMSE"] = table[r"TSO RMSE"].round().astype(int)
    # Save as markdown
    summary_fpath = f'{output_dir}/{target}_notes_en.md'
    table.to_markdown(summary_fpath, index=False)

    # ------------- ENGLISH TEXT ---------------

    intro_sentences = \
        f"""
Our __week-ahead__ forecast has average RMSE of __{ave_total_metric:.0f}__.  
SMARD __day-ahead__ forecast has average accuracy of __{ave_smard_metric:.0f}__. 
    """

    # Reading the markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()
    # Prepending the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content
    # Saving the updated markdown content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)


    # -------------- GERMAN TEXT -------------------

    dictionary_en_de = {
        "Best Model": "Bestes Modell",
        "Average RMSE": "Durchschnittlicher RMSE",
        "TSO/Region": "ÜNB/Region",
        "Train Date": "Trainingsdatum",
        "N Features": "Anzahl der Merkmale"
    }
    table.rename(columns = dictionary_en_de, inplace = True)

    summary_fpath = f'{output_dir}/{target}_notes_de.md'
    table.to_markdown(summary_fpath, index=False)

    intro_sentences = f"""
Unsere __Wochenprognose__ hat einen durchschnittlichen RMSE von __{ave_total_metric:.0f}__.  
Die SMARD __Tagesprognose__ weist eine durchschnittliche Genauigkeit von __{ave_smard_metric:.0f}__ auf.
    """

    # Reading the markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()
    # Prepending the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content
    # Saving the updated markdown content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)







def publish_forecasts(db_path:str, target_settings:list[dict], verbose:bool):

    data_dir, data_dir_web, data_dir_api = create_output_dirs(verbose)

    for target_dict in target_settings:


        # publish data to webpage
        publish_generation(
            run_label=target_dict['label'],
            target=target_dict['target'],
            avail_regions=target_dict['regions'],
            n_folds = 3,
            metric = 'rmse',
            method_type = 'trained', # 'trained'
            results_root_dir = './output/forecasts/',
            database_dir = db_path,
            output_dir = data_dir_web
        )

    # for target_dict in target_settings:
    #     # publish to API folder
    #     publish_to_api(
    #         run_label=target_dict['label'],
    #         target=target_dict['target'],
    #         avail_regions=target_dict['regions'],
    #         method_type = 'forecast', # 'trained'
    #         results_root_dir = './output/forecasts/',
    #         output_dir = f'{data_dir_api}forecasts/'
    #     )

# def publish_generation_mix(verbose:bool):
#     data_dir, data_dir_web, data_dir_api = create_output_dirs(verbose)





class TargetData:

    def __init__(self, results_root_dir:str, verbose:bool):

        self.results_root_dir = results_root_dir

        self.df_train_res: pd.DataFrame = pd.DataFrame()
        self.df_forecast_res: pd.DataFrame = pd.DataFrame()
        self.df_forecast: pd.DataFrame = pd.DataFrame()
        self.finetune_time: Optional[datetime] = None
        self.train_time: Optional[datetime] = None
        self.forecast_time: Optional[datetime] = None
        self.metadata: dict = {}
        self.train_cutoffs: list = []
        self.forecast_cutoffs: list = []
        self.horizon: Optional[int] = None

        self.df_metrics : pd.DataFrame = pd.DataFrame()
        self.best_models : dict = {}
        self.best_model_label: str = "N/A"
        self.best_model:str = "N/A"

        self.verbose = verbose


    def load_summary(self, target_label:str, de_reg:dict):

        var = target_label+de_reg['suffix']
        self.target_label = target_label

        self.df_metrics = pd.read_csv( f'{self.results_root_dir}{var}/summary_metrics.csv' )
        logger.info(f"Results for target_label={target_label} TSO={de_reg['TSO']} are loaded successfully")
        # load best model name
        with open(f'{self.results_root_dir}{var}/best_model.json', 'r') as file:
            self.best_models : dict = json.load(file)
        self.targets = self.best_models.keys()
        if len(self.targets) == 0:
            raise ValueError("No targets defined in the best_model.json file")
        if len(self.targets) > 1:
            logger.info(f"Multiple targets ({len(self.targets)}) defined in the best_model.json file")


    def load_target_data(self, target_label:str, target:str, de_reg:dict, best_model:str or None):

        if self.df_metrics.empty:
            raise ValueError("df_metrics.empty is empty")
        if target_label != self.target_label:
            raise ValueError(f"Given target label {target_label} does not match class target label {self.target_label}")

        var = target_label+de_reg['suffix']
        suffix = de_reg['suffix']

        if best_model is None:
            with open(f'{self.results_root_dir}{var}/best_model.json', 'r') as file:
                best_models : dict = json.load(file)
            target_dict = best_models[target+suffix]
            best_model = target_dict['model_label']

        self.best_model_label = best_model

        logger.info(f"Loading data for target={target} with best_model={best_model}")
        if best_model.__contains__('ensemble'):
            best_model = convert_ensemble_string(best_model)

        self.best_model = best_model

        # load results and select columns only for this specific target (in case multi-target infernece was used)
        self.df_train_res = pd.read_csv(
            f"{self.results_root_dir}{var}/{best_model}/{'trained'}/result.csv",
            index_col=0, parse_dates=True)
        self.df_train_res.columns = [ col.replace(suffix, '') for col in self.df_train_res.columns ]
        self.df_train_res = self.df_train_res[[f'{target}_{key}' for key in ['actual', 'fitted', 'lower', 'upper']]]

        self.df_forecast_res = pd.read_csv(
            f"{self.results_root_dir}{var}/{best_model}/{'forecast'}/result.csv",
            index_col=0, parse_dates=True)
        self.df_forecast_res.columns = [ col.replace(suffix, '') for col in self.df_forecast_res.columns ]
        self.df_forecast_res = self.df_forecast_res[[
            f'{target}_{key}' for key in ['actual', 'fitted', 'lower', 'upper']
        ]]

        # load forecast
        self.df_forecast = pd.read_csv(
            f"{self.results_root_dir}{var}/{best_model}/{'forecast'}/forecast.csv",
            index_col=0, parse_dates=True)
        self.df_forecast.columns = [ col.replace(suffix, '') for col in self.df_forecast.columns ]
        self.df_forecast = self.df_forecast[[f'{target}_{key}' for key in ['actual', 'fitted', 'lower', 'upper']]]

        # load timestamp when the model was last trained and forecasted
        with open(f"{self.results_root_dir}{var}/{best_model}/{'finetuning'}/datetime.json", "r") as file:
            self.finetune_time = pd.to_datetime(json.load(file)['datetime'])
        with open(f"{self.results_root_dir}{var}/{best_model}/{'trained'}/datetime.json", "r") as file:
            self.train_time = pd.to_datetime(json.load(file)['datetime'])
        with open(f"{self.results_root_dir}{var}/{best_model}/{'forecast'}/datetime.json", "r") as file:
            self.forecast_time = pd.to_datetime(json.load(file)['datetime'])

        # load training properties (features etc.) Same in forecast
        with open(f"{self.results_root_dir}{var}/{best_model}/{'trained'}/metadata.json", "r") as file:
            self.metadata = json.load(file)

        self.train_cutoffs = self.df_metrics[
            (self.df_metrics['method'] == 'trained')
            & (self.df_metrics['target'] == target+de_reg['suffix'])
            & (self.df_metrics['model_label'] == self.best_model_label)
            ]['horizon'].unique()
        self.train_cutoffs = [pd.to_datetime(cutoff) for cutoff in self.train_cutoffs]
        assert len(self.train_cutoffs) > 1
        self.horizon = int(self.metadata['horizon'])

        self.forecast_cutoffs = self.df_metrics[
            (self.df_metrics['method'] == 'forecast')
            & (self.df_metrics['target'] == target+de_reg['suffix'])
            & (self.df_metrics['model_label'] == self.best_model_label)
            ]['horizon'].unique()
        self.forecast_cutoffs = [pd.to_datetime(cutoff) for cutoff in self.forecast_cutoffs]
        assert len(self.forecast_cutoffs) > 1

    def get_ave_metric(self, df:pd.DataFrame, target:str, key_actual:str, key_fitted:str, metric:str, n_folds:int)->np.floating:
        # compute forecast error over the past N horizons
        total_metrics = compute_error_metrics_cutoffs(
            df_=df, cutoffs=self.train_cutoffs, horizon=self.horizon, target=target,
            key_actual=key_actual, key_fitted=key_fitted
        )
        total_metrics = {date:metric[target] for date, metric in total_metrics.items()}
        total_metrics = retain_most_recent_entries(total_metrics, N=n_folds)
        ave_total_metric = np.average([total_metrics[time_s][metric] for time_s in total_metrics.keys()])
        return ave_total_metric

    def save_past_and_current_forecasts_json(self, output_dir_for_figs:str, target:str):
        if not os.path.isdir(output_dir_for_figs):
            os.makedirs(output_dir_for_figs)
            logger.info(f"Created {output_dir_for_figs}")
        if self.df_forecast_res.empty:
            raise ValueError('df_forecast_res is empty')

        convert_csv_to_json(
            df = self.df_forecast_res, target=target,
            output_dir=output_dir_for_figs, prefix = 'forecast_prev',
            cols= [f"actual", f"fitted", f"lower", f"upper"]
        )
        convert_csv_to_json(
            df = self.df_forecast[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)], target=target,
            output_dir=output_dir_for_figs, prefix = 'forecast_curr',
            cols=[f"fitted", f"lower", f"upper"]
        )

    def save_past_and_current_forecasts_api_json(self, output_dir_for_api:str, tso_region:str, target:str):
        if not os.path.isdir(output_dir_for_api):
            os.makedirs(output_dir_for_api)
            logger.info(f"Created {output_dir_for_api}")

        # Combine past and current forecasts
        df = pd.concat([self.df_forecast_res, self.df_forecast], axis=0)  # stack dataframes along index
        df.sort_index(inplace=True)  # Ensure the index is sorted
        df = df[~df.index.duplicated(keep='first')]
        df.drop(columns=[f"{target}_actual"], inplace=True)  # Drop the actual data column
        df.rename(columns={
            f"{target}_fitted": "forecast",
            f"{target}_lower": "ci_lower",
            f"{target}_upper": "ci_upper"
        }, inplace=True)

        # Convert the index (timestamps) to ISO 8601 strings and reset the index
        fname = f"{target.lower()}_{tso_region.lower()}.json"

        # generate metadata
        metadata = {
            "file": fname,
            "data_keys":list(df.columns.tolist()),
            "target_name": target,
            "tso_region": tso_region if tso_region != 'total' else 'DE',
            "model_label": self.best_model,
            "finetune_datetime":self.finetune_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "train_datetime": self.train_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "forecast_datetime": self.forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "source": "https://vsevolodnedora.github.io/energy_market_analysis/",
            "forecast_horizon_hours":self.horizon,
            "units": "MW",
            "notes": "None"
        }

        save_to_json(df, metadata, f"{output_dir_for_api}{fname}", self.verbose)


    def get_table_entry(self, tso_name:str, metric:np.floating, tso_metric:np.floating) -> dict:
        return {
            'TSO/Region':tso_name,
            'Train Date':self.train_time.strftime('%Y-%m-%d'),
            'N Features':len(self.metadata['features']),
            'Best Model':self.best_model,
            'RMSE':float(metric),
            'TSO RMSE': float(tso_metric),
        }



def add_datas(dt1:TargetData, dt2:TargetData, target1:str, target2:str, expect_same_dates:bool)->TargetData:

    if dt1.df_metrics.empty and not dt2.df_metrics.empty: dt1.df_metrics = copy.deepcopy(dt2.df_metrics)
    elif (not dt1.df_metrics.empty) and (not dt2.df_metrics.empty) and \
            (list(dt1.df_metrics.columns) != list(dt2.df_metrics.columns)):
        logging.error(f"Metrics for target1={target1} and target2={target2} have different columns!")

    if dt1.best_model_label == 'N/A' and dt2.best_model_label != 'N/A':
        dt1.best_model_label = dt2.best_model_label
    if dt1.best_model == 'N/A' and dt2.best_model != 'N/A':
        dt1.best_model = dt2.best_model

    # inicialization
    if dt1.horizon is None: dt1.horizon = copy.deepcopy(dt2.horizon)
    elif not dt1.horizon is None and not dt2.horizon is None and not dt1.horizon == dt2.horizon:
        raise ValueError(f"Horizon mismatch: dt1.horizon={dt1.horizon}, dt2.horizon={dt2.horizon}")

    if len(dt1.train_cutoffs) == 0: dt1.train_cutoffs = copy.deepcopy(dt2.train_cutoffs)
    elif len(dt1.train_cutoffs) > 0 and len(dt2.train_cutoffs) > 0 and list(dt1.train_cutoffs) != list(dt2.train_cutoffs):
        raise ValueError(
            f"Train cuttofs mismatch: {target1} vs {target2} | "
            f"\n{dt1.train_cutoffs},  \n{dt2.train_cutoffs}"
        )

    if len(dt1.forecast_cutoffs) == 0: dt1.forecast_cutoffs = copy.deepcopy(dt2.forecast_cutoffs)
    elif len(dt1.forecast_cutoffs) > 0 and len(dt2.forecast_cutoffs) > 0 and list(dt1.forecast_cutoffs) != list(dt2.forecast_cutoffs):
        raise ValueError(
            f"Forecast cuttofs mismatch: {target1} vs {target2} | "
            f"\n{dt1.forecast_cutoffs}, \n{dt2.forecast_cutoffs}"
        )

    if (dt1.finetune_time is None): dt1.finetune_time = copy.deepcopy(dt2.finetune_time)
    elif ((not dt1.finetune_time is None) and (not dt2.finetune_time is None)
          and  (dt2.finetune_time != dt1.finetune_time)
          and expect_same_dates):
        logging.error(f"Finetune time mismatch: {target1} vs {target2} | {dt1.finetune_time} vs {dt2.finetune_time} ")

    if (dt1.train_time is None): dt1.train_time = copy.deepcopy(dt2.train_time)
    elif ((not dt1.train_time is None) and (not dt2.train_time is None)
          and  (dt2.train_time != dt1.train_time)
          and expect_same_dates):
        raise ValueError(f"Train time mismatch: {target1} vs {target2} | {dt1.train_time} vs {dt2.train_time} ")

    if (dt1.forecast_time is None): dt1.forecast_time = copy.deepcopy(dt2.forecast_time)
    elif ((not dt1.forecast_time is None) and (not dt2.forecast_time is None)
          and  (dt2.forecast_time != dt1.forecast_time)
          and expect_same_dates):
        raise ValueError(f"Forecast time mismatch: {target1} vs {target2} | {dt1.forecast_time} vs {dt2.forecast_time} ")



    if len(list(dt1.metadata.keys())) == 0: dt1.metadata = copy.deepcopy(dt2.metadata)
    elif len(list(dt1.metadata.keys())) > 0 and len(list(dt2.metadata.keys())) > 0 and (dt1.metadata != dt2.metadata):
            if dt1.metadata.keys() != dt2.metadata.keys():
                logger.error(f"Metadata keys mismatch: {target1} vs {target2} | Overriding metadata!" )
            else:
                logger.warning(f"Metadata content mismatch: {target1} vs {target2} | Overriding metadata!" )
            dt1.metadata = copy.deepcopy(dt2.metadata)
    # combine dataframes (Training Results)
    if dt1.df_train_res.empty:
        dt1.df_train_res = dt2.df_train_res.copy()
        if target1 != target2:
            dt1.df_train_res.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_train_res[f"{target1}_{key}"] += dt2.df_train_res[f"{target2}_{key}"].values

    # combine dataframes (Training Results)
    if dt1.df_forecast_res.empty:
        dt1.df_forecast_res = dt2.df_forecast_res.copy()
        if target1 != target2:
            dt1.df_forecast_res.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_forecast_res[f"{target1}_{key}"] += dt2.df_forecast_res[f"{target2}_{key}"].values

    # combine dataframes (Training Results)
    if dt1.df_forecast.empty:
        dt1.df_forecast = dt2.df_forecast.copy()
        if target1 != target2:
            dt1.df_forecast.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_forecast[f"{target1}_{key}"] += dt2.df_forecast[f"{target2}_{key}"].values

    if len(dt1.df_train_res.columns) != 4:
        raise ValueError(f"Expect to have 4 columns for dataframe. (actual, fitted, lower, upper). "
                         f"Got {len(dt1.df_train_res.columns)} \n {dt1.df_train_res.columns.to_list()}")
    return dt1

def compute_error_metrics_cutoffs(
        df_, cutoffs:list, horizon:int, target:str, key_actual:str, key_fitted:str)->dict:
    if not key_actual in df_.columns:
        raise ValueError(f"key_actual {key_actual} not found in dataframe columns \n{df_.columns.tolist()}")
    if not key_fitted in df_.columns:
        raise ValueError(f"key_fitted {key_fitted} not found in dataframe columns \n{df_.columns.tolist()}")
    ''' compute errror metrics for batches separated by cutoffs'''
    metrics = {}
    for i, cutoff in enumerate(cutoffs):
        mask_ = (df_.index >= cutoff) & (df_.index < cutoff + pd.Timedelta(hours=horizon))
        actual = df_[key_actual]
        predicted = df_[key_fitted]

        df = pd.DataFrame({
            f'{target}_actual':actual[mask_].values,
            f'{target}_fitted': predicted[mask_].values,
            f'{target}_lower': np.zeros_like(actual[mask_].values),
            f'{target}_upper': np.zeros_like(actual[mask_].values)
        }, index=actual[mask_].index)

        metrics[cutoff] = compute_error_metrics([target], df)

    return metrics

def retain_most_recent_entries(data:dict, N:int):
    # Sort keys by timestamp (assuming keys are sortable timestamps)
    sorted_keys = sorted(data.keys(), reverse=True)
    # Select the most recent N keys
    most_recent_keys = sorted_keys[:N]
    # Create a new dictionary with only the most recent N entries
    recent_entries = {key: data[key] for key in most_recent_keys}
    return recent_entries

class PublishGenerationLoad:

    def __init__(self, db_path:str, results_root_dir:str, output_dir_for_figs:str, output_dir_for_api:str, verbose:bool):
        self.db_path = db_path
        self.verbose = verbose
        self.results_root_dir = results_root_dir # 'forecasting_modules/output/',

        self.load_entsoe_data()
        self.load_smard_data()

        self.metric = 'rmse'
        self.n_folds = 3

        self.tds = []

        for outdir in [output_dir_for_figs, output_dir_for_api]:
            if not os.path.isdir(outdir):
                os.makedirs(outdir, exist_ok=True)
                logger.info(f"Created directory {outdir}")

        self.output_dir_for_figs = output_dir_for_figs
        self.output_dir_for_api = output_dir_for_api

    def set_load_output_results(self, de_reg:dict, target_label:str):

        dt = TargetData(self.results_root_dir, self.verbose)
        dt.load_summary(target_label, de_reg) # load sumamry for target(s)
        dt_total = TargetData(self.results_root_dir, self.verbose)
        # df_train_total_res, df_forecast_total_train, df_forecast_total_forecast = \
        #     pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # allow different targets to have different best models (even if trained jointly)
        for target, target_dict in dt.best_models.items():
            target_dict = dt.best_models[target]
            target = target.replace(de_reg['suffix'], '')
            dt.load_target_data(target_label, target, de_reg=de_reg, best_model=target_dict['model_label']) # load target

            # todo add publish to api dt.publish_target_individual_plot()

            # todo add publish to deploy dt.publish_target_api()

            # summ all contributions as we can only compare the total energy generation with ENTSO-E
            if target_label == 'energy_mix':
                dt_total = add_datas(dt_total, dt, 'generation', target, expect_same_dates=True) # 'generation'
            else:
                dt_total = add_datas(dt_total, dt, target_label, target, expect_same_dates=True)


                # if df_train_total_res.empty:
                #     df_train_total_res:pd.DataFrame = dt.df_train_res.copy()
                #     df_train_total_res.rename(columns={
                #         f"{target}_{key}" : f"{'generation'}_{key}"
                #         for key in ['actual', 'fitted', 'lower', 'upper']
                #     }, inplace=True)
                # else:
                #     for key in ['actual', 'fitted', 'lower', 'upper']:
                #         df_train_total_res[f"{'generation'}_{key}"] += dt.df_train_res[f"{target}_{key}"].values
                #

        suffix = de_reg['suffix']
        # for energy mix we are only interested in the total generation for which there is ENTSO-E forecast
        if target_label == 'energy_mix':
            target = 'generation'
            # load best wind,solar
            for target_ in ['wind_offshore','wind_onshore','solar']:
                if not os.path.exists(f'{self.results_root_dir}{target_+suffix}/'):
                    logger.warning(f"Missing directory {self.results_root_dir}{target_+suffix}/")
                    continue
                logger.info(f"Adding {target_} to total {target}")
                dt.load_summary(target_, de_reg) # load sumamry for target(s)
                dt.load_target_data( target_, target_, de_reg=de_reg, best_model=None )
                add_datas(dt_total, dt, target, target_, expect_same_dates=False)


        # compute forecast error over the past N horizons
        ave_total_metric = dt_total.get_ave_metric(
            dt_total.df_train_res, target, f"{target}_actual",f"{target}_fitted",
            self.metric, self.n_folds
        )
        # compute ENTSO-E error
        ave_entsoe_metric = dt.get_ave_metric(
            self.df_entsoe, target, f"{target+suffix}",f"{target}_forecast"+suffix,
            self.metric, self.n_folds
        )

        logger.info(f'For {target} average over {self.n_folds} '
                    f'total RMSE is {ave_total_metric:.0f} | ENTSOE RMSE is {ave_entsoe_metric:.0f}')

        # publish generation per tso (LAST UPDATED FORECAST)
        dt_total.save_past_and_current_forecasts_json(
            self.output_dir_for_figs + '/' + target  + suffix + '/', target
        )
        dt_total.save_past_and_current_forecasts_api_json(
            self.output_dir_for_api + '/', de_reg['TSO'], target
        )

        # add summary to the summary dict
        metadata = dt_total.get_table_entry(de_reg['TSO'],ave_total_metric,ave_entsoe_metric)

        # meta = {"train_cutoffs": dt.train_cutoffs, "horizon": dt.horizon, "target": target}
        return dt_total, metadata

    def load_entsoe_data(self):
        self.df_entsoe = pd.read_parquet(self.db_path + 'entsoe/' + 'history.parquet')

        # compute total generation
        for de_reg in de_regions:
            suffix = de_reg['suffix']
            self.df_entsoe['generation'+suffix] = \
                self.df_entsoe[[
                    col+suffix for col in list(entsoe_generation_type_mapping.keys())
                        if col+suffix in self.df_entsoe.columns
                ]].sum(axis=1)

        logger.info(f'Loaded ENTSO-E with file {len(self.df_entsoe)} entries')

    def load_smard_data(self):
        self.df_smard = pd.read_parquet(self.db_path + 'smard/' + 'history.parquet')
        self.df_smard.rename(columns={
            "total_gen_forecasted":"generation_forecast",
            "wind_offshore_forecasted":"wind_offshore_forecast",
            "wind_onshore_forecasted":"wind_onshore_forecast",
            "solar_forecasted":"solar_forecast",
            "total_grid_load":"load",
            "total_grid_load_forecasted":"load_forecast"
        },inplace=True)
        # compute total_gen
        self.df_smard['generation'] = self.df_smard[[
            col for col in DataEnergySMARD.mapping_energy.values() if col in self.df_smard.columns
        ]].sum(axis=1)
        logger.info(f'Loaded SMARD with file {len(self.df_smard)} entries')

    def compute_entsoe_error_for_target(self, de_reg:dict, target:str):
        pass

    def process(self, target_label:str, avail_regions:tuple):
        # compute forecast error over the past N horizons
        if target_label == 'energy_mix':
            target = 'generation'
        else:
            target = target_label

        dt_total = TargetData(self.results_root_dir, self.verbose)
        table = []
        regions = [region for region in de_regions if region['name'] in avail_regions]
        for region_dict in regions:
            if not os.path.exists(f"{self.results_root_dir}/{target_label}{region_dict['suffix']}/"):
                logger.warning(f"Missing output directory: {self.results_root_dir}/{target_label}{region_dict['suffix']}/")
                continue

            dt_tso, metadata_tso = self.set_load_output_results(
                de_reg=region_dict, target_label=target_label
            )
            # add the data from dt_tso to dt_total
            add_datas(dt_total, dt_tso, target, target, expect_same_dates=False)
            table.append(metadata_tso)


        # publish the results (forecasts) to .json for ./deploy/...
        dt_total.save_past_and_current_forecasts_json(
            self.output_dir_for_figs + '/' + target + '/', target
        )
        dt_total.save_past_and_current_forecasts_api_json(
            self.output_dir_for_api  + '/', 'total', target
        )

        ave_total_metric=dt_total.get_ave_metric(
            dt_total.df_train_res, target, f"{target}_actual", f"{target}_fitted", self.metric, self.n_folds
        )

        # compute smard total error
        ave_smard_metric = dt_total.get_ave_metric(
            self.df_smard, target, target, f"{target}_forecast", self.metric, self.n_folds
        )

        logger.info(f"For {target} average over {self.n_folds} "
                    f"total RMSE is {ave_total_metric:.0f} | SMARD RMSE is {ave_smard_metric:.0f}")

        table = pd.DataFrame(table)
        # Rename values starting with 'meta_' to 'Ensemble'
        table["Best Model"] = table["Best Model"].apply(lambda x: "Ensemble" if x.startswith("meta_") else x)
        # Round floating point values to integers
        table[r"RMSE"] = table[r"RMSE"].round().astype(int)
        table[r"TSO RMSE"] = table[r"TSO RMSE"].round().astype(int)
        # Save as markdown
        summary_fpath = f'{self.output_dir_for_figs}/{target}_notes_en.md'
        table.to_markdown(summary_fpath, index=False)

        intro_sentences = \
        f"""
Our __week-ahead__ forecast has average RMSE of __{ave_total_metric:.0f}__.  
SMARD __day-ahead__ forecast has average accuracy of __{ave_smard_metric:.0f}__. 
    """

        # Reading the markdown content
        with open(summary_fpath, "r") as file:
            markdown_content = file.read()
        # Prepending the sentences to the markdown content
        updated_markdown_content = intro_sentences + "\n" + markdown_content
        # Saving the updated markdown content
        with open(summary_fpath, "w") as file:
            file.write(updated_markdown_content)


        # -------------- GERMAN TEXT -------------------

        dictionary_en_de = {
            "Best Model": "Bestes Modell",
            "Average RMSE": "Durchschnittlicher RMSE",
            "TSO/Region": "ÜNB/Region",
            "Train Date": "Trainingsdatum",
            "N Features": "Anzahl der Merkmale"
        }
        table.rename(columns = dictionary_en_de, inplace = True)

        summary_fpath = f'{self.output_dir_for_figs}/{target}_notes_de.md'
        table.to_markdown(summary_fpath, index=False)

        intro_sentences = f"""
Unsere __Wochenprognose__ hat einen durchschnittlichen RMSE von __{ave_total_metric:.0f}__.  
Die SMARD __Tagesprognose__ weist eine durchschnittliche Genauigkeit von __{ave_smard_metric:.0f}__ auf.
    """

        # Reading the markdown content
        with open(summary_fpath, "r") as file:
            markdown_content = file.read()
        # Prepending the sentences to the markdown content
        updated_markdown_content = intro_sentences + "\n" + markdown_content
        # Saving the updated markdown content
        with open(summary_fpath, "w") as file:
            file.write(updated_markdown_content)

def publish_main():

    publisher = PublishGenerationLoad(
        db_path='./database/',
        results_root_dir='./output/forecasts/',
        output_dir_for_figs = './deploy/data/forecasts/',
        output_dir_for_api = './deploy/data/api/forecasts/',
        verbose=True
    )
    target_settings = [
        {'label' : 'wind_offshore', 'target' : 'wind_offshore', "regions" : ('DE_50HZ', 'DE_TENNET')},
        {'label' : 'wind_onshore', 'target' : 'wind_onshore', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
        {'label' : 'solar', 'target' : 'solar', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
        {'label' : 'load', 'target' : 'load', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
        {'label' : 'energy_mix', 'target' : 'energy_mix', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')}
    ]
    for target_dict in target_settings:
        publisher.process(
            target_label=target_dict['label'],
            avail_regions=target_dict['regions']
        )


if __name__ == '__main__':
    db_path = './database/'
    #
    #
    # target_settings = [
    #     {'label' : 'wind_offshore', 'target' : 'wind_offshore', "regions" : ('DE_50HZ', 'DE_TENNET')},
    #     # {'label' : 'wind_onshore', 'target' : 'wind_onshore', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
    #     # {'label' : 'solar', 'target' : 'solar', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
    #     # {'label' : 'load', 'target' : 'load', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
    #     # {'label' : 'energy_mix', 'target' : 'energy_mix', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')}
    # ]
    #
    # # publish_forecasts(target_settings=target_settings, db_path=db_path, verbose=True)
    publish_main()



    logger.info(f"All tasks in update are completed successfully!")