'''
This script takes data from './output/' which is produced by forecasting and other models
and converts it into a format that is expected by JS code in './deploy/'
Created files are then saved into './deploy/data/
'''

import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta

from forecasting_modules import compute_error_metrics, analyze_model_performance, convert_ensemble_string
from data_collection_modules.german_locations import de_regions

from logger import get_logger
logger = get_logger(__name__)


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

def publish_forecasts(db_path:str, target_settings:list[dict], verbose:bool):

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

        # publish to API folder
        publish_to_api(
            run_label=target_dict['label'],
            target=target_dict['target'],
            avail_regions=target_dict['regions'],
            method_type = 'forecast', # 'trained'
            results_root_dir = './output/forecasts/',
            output_dir = f'{data_dir_api}forecasts/'
        )

if __name__ == '__main__':
    db_path = './database/'


    target_settings = [
        {'label' : 'wind_offshore', 'target' : 'wind_offshore', "regions" : ('DE_50HZ', 'DE_TENNET')},
        {'label' : 'wind_onshore', 'target' : 'wind_onshore', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
        {'label' : 'solar', 'target' : 'solar', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')},
        {'label' : 'load', 'target' : 'load', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET')}
    ]

    publish_forecasts(target_settings=target_settings, db_path=db_path, verbose=True)

    logger.info(f"All tasks in update are completed successfully!")