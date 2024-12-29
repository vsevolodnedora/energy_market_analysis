import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta

from forecasting_modules import compute_timeseries_split_cutoffs, compute_error_metrics


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

    # Prepare the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir = output_dir + target + '/'
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
            print(f"Column '{column}' not found in the CSV file. (columns={df.columns})")

def pick_best_models_and_check_drift(
        df: pd.DataFrame,
        n_foldsLint: int = 3,
        metric: str = 'mse'
):
    """
    Given a DataFrame containing error metrics for different models and methods
    ('trained' vs 'forecast'), selects the best model in each method by
    averaging the chosen metric over the last n_foldsLint horizons.
    Also checks if there is a model drift (i.e., forecast error > trained error).

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns:
        ['method', 'model_label', 'horizon', 'mse', 'rmse', 'mae', 'mape'].
    n_foldsLint : int
        How many of the most recent horizons to use when averaging the error.
    metric : str
        The error metric column to use for determining the best model
        (e.g., 'mse', 'rmse', 'mae', 'mape').

    Returns
    -------
    dict
        A dictionary with:
          {
            'best_trained_model': (model_label, avg_metric_value),
            'best_forecast_model': (model_label, avg_metric_value),
            'model_drift': bool
          }
        Where 'model_drift' is True if the forecast’s best average error
        is strictly greater than the trained best average error.
    """

    # 1) Ensure horizon is a proper datetime for correct sorting:
    df['horizon'] = pd.to_datetime(df['horizon'])

    # 2) Split into trained and forecast
    trained_df = df[df['method'] == 'trained'].copy()
    forecast_df = df[df['method'] == 'forecast'].copy()

    # 3) Function: for each model_label, pick the last n_foldsLint horizons
    #    (based on horizon descending), compute the mean of the chosen metric
    def compute_model_averages(data: pd.DataFrame) -> pd.DataFrame:
        """
        Groups by model_label, sorts by horizon descending, takes the
        last n_foldsLint records, and computes mean of the chosen metric.
        """
        # Sort by horizon descending
        data = data.sort_values('horizon', ascending=False)

        # We want to group by model_label, then "tail" n_foldsLint rows in each group
        # and compute the mean of the metric.
        # One approach is to use groupby.apply(...). Another approach is a custom aggregator.

        def tail_and_mean(group):
            tail_group = group.head(n_foldsLint)
            return tail_group[metric].mean()

        # Apply groupby
        avg_df = (
            data.groupby('model_label', as_index=False)
            .apply(tail_and_mean)
            .reset_index()
        )
        avg_df.columns = ['model_label', metric]
        return avg_df

    # Compute the mean metric for trained, forecast
    trained_avgs = compute_model_averages(trained_df)
    forecast_avgs = compute_model_averages(forecast_df)

    # 4) Find best model in trained (lowest metric)
    best_trained_row = trained_avgs.loc[trained_avgs[metric].idxmin()]
    best_trained_model = best_trained_row['model_label']
    best_trained_error = best_trained_row[metric]

    # 5) Find best model in forecast (lowest metric)
    best_forecast_row = forecast_avgs.loc[forecast_avgs[metric].idxmin()]
    best_forecast_model = best_forecast_row['model_label']
    best_forecast_error = best_forecast_row[metric]

    # 6) Check for model drift
    #    For simplicity, define drift = forecast metric > trained metric
    model_drift = (best_forecast_error > best_trained_error)

    return {
        'best_trained_model': (best_trained_model, best_trained_error),
        'best_forecast_model': (best_forecast_model, best_forecast_error),
        'model_drift': model_drift
    }

def get_error_metrics_for_smard_forecast(df_smard:pd.DataFrame, horizon:int, target:str) -> dict:
    cutoffs = compute_timeseries_split_cutoffs(
        df_hist.index,
        horizon=horizon,
        delta=horizon,
        folds=5,
    )
    smard_res = []
    results = {}
    smard_metrics = {}
    for i, cutoff in enumerate(cutoffs):
        mask_ = (df_smard.index >= cutoff) & (df_smard.index < cutoff + pd.Timedelta(hours=horizon))
        actual = df_smard[f"{target}"][mask_]
        predicted = df_smard[f"{target}_forecasted"][mask_]

        df = pd.DataFrame({
            f'{target}_actual':actual.values,
            f'{target}_fitted': predicted.values,
            f'{target}_lower': np.zeros_like(actual.values),
            f'{target}_upper': np.zeros_like(actual.values)
        }, index=actual.index)
        smard_res.append(copy.deepcopy(df))
        smard_metrics[cutoff] = compute_error_metrics(target, df)

    smard_metrics_ = [smard_metrics[key] for key in smard_metrics.keys()]
    ave_metrics = {
        key: np.mean( [smard_metrics_[i][key] for i in range(len((smard_metrics_)))] ) for key in list(smard_metrics_[0].keys())
    }
    smard_metrics_.append(ave_metrics)
    return smard_metrics_

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

def publish_offshore_wind_generation():

    def convert_ensemble_string(input_string):
        # Extract the ensemble name and the components
        ensemble_name = input_string.split('[')[1].split(']')[0]
        components = input_string.split('(')[1].split(')')[0].split(',')

        # Construct the desired format
        output_string = f"meta_{ensemble_name}_" + "_".join(components)
        return output_string

    table = []
    n_folds = 3
    metric = 'rmse'
    method_type = 'trained' # 'trained'
    df_results = pd.DataFrame()
    for var, region, key in zip(['wind_offshore_tenn', 'wind_offshore_50hz'],
                           ['TenneT', '50Hz'], ['_tenn','_50hz']):
        df = pd.read_csv( f'forecasting_modules/output/{var}/summary_metrics.csv' )
        res_models, res_drifts = analyze_model_performance(df, n_folds=n_folds, metric=metric)
        res_models = res_models[res_models['method'] == method_type]
        best_model:str = str(res_models['model_label'].values[0])
        if best_model.__contains__('ensemble'):
            best_model = convert_ensemble_string(best_model)
        with open(f"forecasting_modules/output/{var}/{best_model}/{method_type}/datetime.json", "r") as file:
            train_time = pd.to_datetime(json.load(file)['datetime'])

        with open(f"forecasting_modules/output/{var}/{best_model}/{method_type}/metadata.json", "r") as file:
            metadata:dict = json.load(file)

        table.append({
            'TSO/Region':region,
            'Train Date':train_time.strftime('%Y-%m-%d'),
            'N Features':len(metadata['features']),
            'Best Model':best_model,
            'Average RMSE':float(res_models[f'avg_{metric}'].values[0])
        })

        # compute total (for SMARD comparison)
        df_res = pd.read_csv(
            f"forecasting_modules/output/{var}/{best_model}/{method_type}/result.csv",
            index_col=0, parse_dates=True)
        df_res.columns = [ col.replace(key, '') for col in df_res.columns ]
        if df_results.empty: df_results = df_res.copy()
        else: df_results += df_res.copy()

    target = 'wind_offshore'

    def compute_error_metrics_cutoffs(df_, cutoffs:list, horizon:int, key_actual:str, key_fitted:str)->dict:
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
            smard_metrics[cutoff] = compute_error_metrics(target, df)
        return smard_metrics

    def retain_most_recent_entries(data:dict, N:int):
        # Sort keys by timestamp (assuming keys are sortable timestamps)
        sorted_keys = sorted(data.keys(), reverse=True)
        # Select the most recent N keys
        most_recent_keys = sorted_keys[:N]
        # Create a new dictionary with only the most recent N entries
        recent_entries = {key: data[key] for key in most_recent_keys}
        return recent_entries

    # -------- FOR TOTAL COMPUTE ERROR OVER THE LAST N HORIZONS --------------- #

    df = pd.read_csv( f'forecasting_modules/output/{var}/summary_metrics.csv' )
    cutoffs = df[df['method'] == method_type]['horizon'].unique()
    cutoffs = [pd.to_datetime(cutoff) for cutoff in cutoffs]
    horizon = int(metadata['horizon'])

    total_metrics = compute_error_metrics_cutoffs(
        df_=df_results, cutoffs=cutoffs, horizon=horizon,
        key_actual=f'{target}_actual', key_fitted=f'{target}_fitted'
    )
    total_metrics = retain_most_recent_entries(total_metrics, N=horizon)
    ave_total_metric = np.average([total_metrics[time_s][metric] for time_s in total_metrics.keys()])

    print(f'Average over {n_folds} total RMSE for {target} is {ave_total_metric}')


    # ----------- COMPUTE SMARD ERROR OVER THE LAST N HORIZONS ------------------ #

    df_smard = pd.read_parquet('database/' + 'smard/' + 'history.parquet')

    smard_metrics = compute_error_metrics_cutoffs(
        df_=df_smard,  cutoffs=cutoffs, horizon=horizon,
        key_actual=target, key_fitted=f"{target}_forecasted"
    )

    smard_metrics = retain_most_recent_entries(smard_metrics, n_folds)
    ave_smard_metric = np.average([smard_metrics[time_s][metric] for time_s in smard_metrics.keys()])

    print(f'Average over {n_folds} SMARD RMSE for {target} is {ave_smard_metric}')


    table = pd.DataFrame(table)

    for ftype in ['forecast.csv', 'result.csv']:
        for var, region in zip(['wind_offshore_tenn', 'wind_offshore_50hz', 'wind_offshore'],
                               ['TenneT', '50Hz', 'Total']):

            if region != 'Total':
                model_label = str(table[table['TSO/Region']==region]['Best Model'].values[0])
                df = pd.read_csv(
                    f'forecasting_modules/output/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]
            else:
                var, region = 'wind_offshore_50hz', '50Hz'
                model_label = str(table[table['TSO/Region']==region]['Best Model'].values[0])
                df1 = pd.read_csv(
                    f'forecasting_modules/output/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]
                df1.columns = [ col.replace('_50hz', '') for col in df1.columns ]

                var, region = 'wind_offshore_tenn', 'TenneT'
                model_label = str(table[table['TSO/Region']==region]['Best Model'].values[0])
                df2 = pd.read_csv(
                    f'forecasting_modules/output/{var}/{model_label}/forecast/{ftype}',
                    index_col=0,
                    parse_dates=True
                )[:pd.Timestamp(datetime.today(),tz='UTC') + timedelta(days=7)]
                df2.columns = [ col.replace('_tenn', '') for col in df2.columns ]
                df = df1 + df2
                var = 'wind_offshore'

            convert_csv_to_json(
                df = df,
                target=var,
                output_dir='deploy/data/forecasts/',
                prefix = 'XGBoost_curr' if ftype == 'forecast.csv' else 'XGBoost_prev',
                cols=[f"fitted", f"lower", f"upper"] if ftype == 'forecast.csv' else [f"actual", f"fitted"]#[f"actual", f"fitted", f"lower", f"upper"]
            )

    # print(table)
    # Rename values starting with 'meta_' to 'Ensemble'
    table["Best Model"] = table["Best Model"].apply(lambda x: "Ensemble" if x.startswith("meta_") else x)
    # Round floating point values to integers
    table["Average RMSE"] = table["Average RMSE"].round().astype(int)
    # Save as markdown
    summary_fpath = 'deploy/data/forecasts/wind_offshore_notes.md'
    table.to_markdown(summary_fpath, index=False)

    intro_sentences = \
    f"""
<h3>Summary</h3>
Our _week-ahead_ Offshore Wind Generation forecast has RMSE of __{ave_total_metric:.0f}__.  
SMARD _day-ahead_ forecast has average accuracy of __{ave_smard_metric:.0f}__. 
    
<ul>
    <li>Key properties of the forecasting pipeline</li>
    <ul>
        <li>raw and/or engineered weather features</li>
        <li>multiple windfarm locations</li>
        <li>hyperparameter for models and features (tuned with Optuna)</li>
        <li>multi-step single target forecasting (168 timesteps)</li>
        <li>ensemble models trained on OOS forecasts</li>
        <li>total wind power is obtained as a sum of contibutions from TSO regions</li>
    </ul>
</ul>
    
<h3> Forecast for each TSO </h3>
    """

    # Reading the markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()
    # Prepending the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content
    # Saving the updated markdown content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)

if __name__ == '__main__':
    publish_offshore_wind_generation()