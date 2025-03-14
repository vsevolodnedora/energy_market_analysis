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
import sys
import time

from forecasting_modules import compute_error_metrics, analyze_model_performance, convert_ensemble_string
from data_collection_modules.eu_locations import countries_metadata
from data_collection_modules.collect_data_entsoe import entsoe_generation_type_mapping
from data_collection_modules.collect_data_smard import DataEnergySMARD
from data_modules.utils import (
    validate_dataframe
)

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

def timeseries_to_json(df: pd.DataFrame, fname: str) -> None:
    """
    Convert a Pandas DataFrame with a Timestamp index into a JSON file formatted for ApexCharts.

    Parameters:
        df (pd.DataFrame): A DataFrame with a Timestamp index and multiple columns.
        fname (str): The filename to save the JSON output.
    """
    result = []
    df = df.copy()
    for column in df.columns:
        column_data = {
            "name": column,
            "data": [[ts.strftime("%Y-%m-%d %H:%M"), value] for ts, value in df[column].items()]
        }
        result.append(column_data)

    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

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

def analyze_energy_forecast(df: pd.DataFrame) -> None:
    """
    Analyze a week's worth of forecasted energy generation and consumption data for Germany.
    Focus on region='Total' and show how much each sub-region (Amprion, 50Hertz, TenneT, TransnetBW)
    contributes to the total.
    """

    # Convert date column if not in datetime
    df['date'] = pd.to_datetime(df['date'])

    # Define the sub-regions (all except 'Total')
    sub_regions = ['Amprion', '50Hertz', 'TenneT', 'TransnetBW']

    # Basic info: date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    num_days = (end_date - start_date).days + 1

    # Overall average load and generation (all rows, all regions)
    overall_avg_load = df['load'].mean()
    overall_avg_gen  = df['generation'].mean()

    # Group by region
    region_group = df.groupby('region')

    # -------------------------------------------------
    # 1) Basic Summary By Region (including 'Total')
    # -------------------------------------------------
    region_summary = region_group.agg({
        'load': 'mean',
        'generation': 'mean',
        'renewable_fraction': 'mean',
        'load_diff': 'mean',
        'residual_load': 'mean'
    }).rename(
        columns={
            'load': 'Avg Load',
            'generation': 'Avg Generation',
            'renewable_fraction': 'Avg RenewFrac',
            'load_diff': 'Avg (Load-Gen)',
            'residual_load': 'Avg Residual Load'
        }
    )

    # -------------------------------------------------
    # 2) Compare sub-regions (sum) vs. region='Total'
    # -------------------------------------------------
    # Sum across the four sub-regions for each hour
    df_sub = df[df['region'].isin(sub_regions)]
    df_sub_grouped_hourly = df_sub.groupby('date').agg({
        'load': 'sum',
        'generation': 'sum',
        'residual_load': 'sum',
        'load_diff': 'sum',
        'biomass': 'sum',
        'wind_onshore': 'sum',
        'wind_offshore': 'sum',
        'solar': 'sum'
    }).rename(columns=lambda c: c+"_subregions")

    # Merge the sub-region sum with the "Total" row for each hour
    # We'll also filter out only the "Total" rows from the original df
    df_total = df[df['region'] == 'Total'].set_index('date')
    merged_compare = df_total.join(df_sub_grouped_hourly, on='date', how='inner', rsuffix='_sub')
    # Calculate the difference (Total - sum_of_subregions) for load, generation, etc.
    for col in ['load', 'generation', 'residual_load', 'load_diff', 'biomass', 'wind_onshore', 'wind_offshore', 'solar']:
        merged_compare[col+'_diff'] = merged_compare[col] - merged_compare[col+'_subregions']

    # Get an average difference (it should ideally be near zero if 'Total' is truly the sum)
    avg_differences = merged_compare[[
        'load_diff', 'generation_diff', 'residual_load_diff',
        'load_diff_diff', 'biomass_diff', 'wind_onshore_diff',
        'wind_offshore_diff', 'solar_diff'
    ]].mean()
    avg_differences_df = pd.DataFrame(avg_differences, columns=['Avg (Total - SubRegions)'])

    # -------------------------------------------------
    # 3) Identify extremes in the "Total" region
    # -------------------------------------------------
    total_region_df = df[df['region'] == 'Total'].copy()

    # Helper function: find time of max/min for a column in total_region_df
    def find_extremes(subdf: pd.DataFrame, col: str):
        idx_max = subdf[col].idxmax()
        idx_min = subdf[col].idxmin()
        return (
            subdf.loc[idx_max, 'date'], subdf.loc[idx_max, col],
            subdf.loc[idx_min, 'date'], subdf.loc[idx_min, col]
        )

    (tmax_load, vmax_load, tmin_load, vmin_load) = find_extremes(total_region_df, 'load')
    (tmax_gen, vmax_gen, tmin_gen, vmin_gen)     = find_extremes(total_region_df, 'generation')

    # For renewable fraction extremes
    (tmax_rf, vmax_rf, tmin_rf, vmin_rf)         = find_extremes(total_region_df, 'renewable_fraction')

    # -------------------------------------------------
    # 4) Print the Report
    # -------------------------------------------------
    print("====================================================")
    print("ENERGY FORECAST ANALYSIS REPORT: Focus on 'Total'")
    print("====================================================")
    print(f"Date range: {start_date} to {end_date} (approx. {num_days} days)")
    print(f"Overall Avg Load (all regions): {overall_avg_load:.2f} MW")
    print(f"Overall Avg Generation (all regions): {overall_avg_gen:.2f} MW\n")

    print("--- Region Summary (Averages) ---")
    print(region_summary.to_string(float_format="%.2f"))

    print("\n--- Comparison: Sum of Sub-Regions vs. 'Total' (Averages of [Total - SubRegions]) ---")
    print("  (Ideally these should be close to zero if 'Total' = sum(Amprion, 50Hertz, TenneT, TransnetBW).)")
    print(avg_differences_df.to_string(float_format="%.2f"))

    print("\n--- Extremes in 'Total' Region ---")
    print(f"* Maximum Load:      {vmax_load:.2f} MW at {tmax_load}")
    print(f"* Minimum Load:      {vmin_load:.2f} MW at {tmin_load}")
    print(f"* Maximum Generation:{vmax_gen:.2f} MW at {tmax_gen}")
    print(f"* Minimum Generation:{vmin_gen:.2f} MW at {tmin_gen}")
    print(f"* Maximum RenewFrac: {vmax_rf:.3f} at {tmax_rf}")
    print(f"* Minimum RenewFrac: {vmin_rf:.3f} at {tmin_rf}\n")

    print("SHORT SUMMARY:")
    print("1) The 'Total' region represents the sum of the four TSOs: Amprion, 50Hertz, TenneT, and TransnetBW.")
    print("2) Above average-difference table shows how closely the official 'Total' aligns with the sum of sub-regions.")
    print("3) The extremes for the 'Total' region illustrate when Germany-wide load or generation peaked or bottomed out.")
    print("--- End of Report ---\n")

def create_carbon_intensity_markdown(df: pd.DataFrame,
                                     output_dir_for_figs: str,
                                     target: str = "carbon_intensity") -> None:
    """
    Compute daily carbon intensity for region='Total' from the given dataframe,
    convert the resulting table to markdown, and prepend an introductory text.
    """
    # --- 1) Filter for region='Total' only ---
    df_total = df.loc[df['region'] == 'Total'].copy()

    # --- 2) Define emission factors (ton CO2 per MWh), adjust values as appropriate ---
    emission_factors = {
        'biomass': 0.0,            # or a small positive factor, depending on your assumptions
        'coal_derived_gas': 0.4,   # example placeholder
        'gas': 0.2,                # example placeholder
        'hard_coal': 0.34,         # example placeholder
        'wind_onshore': 0.0,
        'wind_offshore': 0.0,
        'solar': 0.0
    }

    # --- 3) Create a helper column for each row's CO2 emissions ---
    #     Emissions_per_row (tCO2) = sum over each source (MW * factor) / 1000 if columns are in kW
    #     or directly (MWh * factor) if columns are hourly MWh
    #     Make sure to adjust if your columns are in different units (MW average for one hour, etc.)

    # If each column is the hourly generation in MWh, we can multiply directly:
    sources = list(emission_factors.keys())

    # Calculate per-row emissions:
    df_total['row_emissions_tCO2'] = 0.0
    for source in sources:
        if source in df_total.columns:
            df_total['row_emissions_tCO2'] += df_total[source] * emission_factors[source]

    # --- 4) Group by calendar day and compute total daily emissions & total daily generation ---
    # Convert date to daily period or floor the date to midnight
    df_total['day'] = df_total['date'].dt.floor('d')  # or df_total['date'].dt.date

    daily_summary = df_total.groupby('day').agg(
        daily_generation=('generation', 'sum'),
        daily_emissions_tCO2=('row_emissions_tCO2', 'sum')
    ).reset_index()

    # --- 5) Compute carbon intensity = total daily emissions / total daily generation ---
    # You can choose units: e.g. tCO2/MWh or gCO2/kWh (multiply by 1000)
    daily_summary['carbon_intensity_tCO2_per_MWh'] = (
            daily_summary['daily_emissions_tCO2'] / daily_summary['daily_generation']
    )

    # Optionally, for gCO2/kWh:
    daily_summary['carbon_intensity_gCO2_per_kWh'] = (
            daily_summary['carbon_intensity_tCO2_per_MWh'] * 1000
    )

    # --- 6) Convert to a small table with columns for final saving ---
    table = daily_summary[[
        'day',
        'daily_generation',
        'daily_emissions_tCO2',
        'carbon_intensity_tCO2_per_MWh',
        'carbon_intensity_gCO2_per_kWh'
    ]]

    # --- 7) Save as markdown ---
    summary_fpath = f'{output_dir_for_figs}/{target}_notes_en.md'
    table.to_markdown(summary_fpath, index=False)

    # Optionally prepend text to the markdown file
    intro_sentences = (
        f"""
TODO: add general analysis
        """
    )

    # Read the just-created markdown content
    with open(summary_fpath, "r") as file:
        markdown_content = file.read()

    # Prepend the sentences to the markdown content
    updated_markdown_content = intro_sentences + "\n" + markdown_content

    # Overwrite the file with the updated content
    with open(summary_fpath, "w") as file:
        file.write(updated_markdown_content)

    print(f"Daily carbon intensity file saved and updated at: {summary_fpath}")

def compute_carbon_intensities(dfs: dict[str, pd.DataFrame], suffix: str) -> pd.DataFrame:
    '''
    Given a dictionary with hourly energy generation forecasts and grid load (in MW) for about 7 days,
    returns a DataFrame with timeseries data for hourly carbon intensity and carbon cost.

    Parameters:
      dfs: dict[str, pd.DataFrame]
          A dictionary whose keys are energy generation types (e.g., 'wind_onshore', 'solar', etc.)
          and one key "load" for grid demand. Each generation DataFrame is assumed to have a column named
          "{key}_fitted" containing the forecast in MW.
      suffix: str
          A string suffix to append to the output column names (useful to distinguish scenarios or versions).

    Returns:
      pd.DataFrame:
          A DataFrame indexed by timestamp with columns:
          - carbon_intensity_gCO2_per_kWh{suffix}: Overall carbon intensity (gCO₂/kWh)
          - co2_cost_eur{suffix}: Carbon cost for the grid load (€/hour)
          - co2_price_eur_per_ton{suffix}: The CO₂ price used (€/ton)
    '''

    # Updated Carbon Intensity Values (gCO₂/kWh) for German Energy Mix
    carbon_intensity = {
        'wind_onshore': 0,         # No direct emissions
        'wind_offshore': 0,        # No direct emissions
        'solar': 0,                # No direct emissions
        'renewables': 0,           # Hydro and similar renewables assumed negligible emissions (~0 gCO₂/kWh)
        'biomass': 50,             # Biomass varies (typically 20-100 gCO₂/kWh), assumed near carbon-neutral
        'waste': 300,              # Waste-to-energy varies widely (~250-400 gCO₂/kWh), depending on composition
        'lignite': 1_100,          # Brown coal (lignite) is highly carbon-intensive (~1,000-1,200 gCO₂/kWh)
        'hard_coal': 850,          # Hard coal typically ranges from ~700-900 gCO₂/kWh
        'coal_derived_gas': 450,   # Coal-derived gas assumed similar to natural gas (~400-500 gCO₂/kWh)
        'gas': 400,                # Natural gas combustion ranges ~350-450 gCO₂/kWh, depending on efficiency
        'oil': 900,                # Oil-fired power ranges ~850-950 gCO₂/kWh
        'other_fossil': 700        # Other fossil fuels assumed similar to hard coal (~650-800 gCO₂/kWh)
    }


    # Fixed CO₂ price in Euros per ton
    co2_price_per_ton = 80.0  # €/ton

    # Identify keys for generation (exclude load)
    energy_gen_types = [key for key in dfs.keys() if key != 'load']

    # Build a single DataFrame for generation (in MW) with one column per energy type.
    # We rename each column to the energy generation type for clarity.
    df_generation_mw = pd.DataFrame()
    for target_, df_ in dfs.items():
        if target_ != 'load':
            col_name = f'{target_}_fitted'
            # Rename the series to match the energy type key
            series_gen = df_[col_name].rename(target_)
            if df_generation_mw.empty:
                df_generation_mw = series_gen.to_frame()
            else:
                # Join on the index (timestamps)
                df_generation_mw = df_generation_mw.join(series_gen, how='outer')

    # Extract grid load. Try to use a fitted column if available.
    df_load = dfs['load']
    if f'load_fitted' in df_load.columns:
        load_series = df_load[f'load_fitted']
    else:
        raise ValueError(f"Load column is not found")

    # Convert generation from MW to kWh (1 MW = 1000 kWh over one hour)
    df_generation_kWh = df_generation_mw * 1000

    # Calculate carbon emissions (gCO2) for each generation type:
    # Multiply energy (kWh) by its carbon intensity (gCO2/kWh)
    df_emissions = pd.DataFrame(index=df_generation_kWh.index)
    for gen_type in df_generation_kWh.columns:
        # Only compute if a carbon factor exists, else assume 0
        factor = carbon_intensity.get(gen_type, 0)
        df_emissions[gen_type] = df_generation_kWh[gen_type] * factor  # gCO2 per hour

    # Sum total emissions (gCO2) and total energy (kWh) per timestamp
    total_emissions_g = df_emissions.sum(axis=1)
    total_energy_kWh = df_generation_kWh.sum(axis=1)

    # Compute overall carbon intensity (gCO2/kWh)
    overall_carbon_intensity = total_emissions_g / total_energy_kWh

    # Convert overall carbon intensity to tons CO2 per MWh for cost calculation:
    # Explanation: 1 MWh = 1000 kWh, so total emissions per MWh = overall_intensity * 1000 (in gCO2).
    # Then convert grams to tons: (overall_intensity*1000) / 1e6 = overall_intensity/1000 ton/MWh.
    overall_intensity_ton_per_MWh = overall_carbon_intensity / 1000

    # Calculate carbon cost (€/hour) based on grid load.
    # For hourly data, a load in MW equals the energy consumption in MWh.
    carbon_cost_eur = overall_intensity_ton_per_MWh * load_series * co2_price_per_ton

    # Create output DataFrame with the computed timeseries
    result_df = pd.DataFrame({
        f'carbon_intensity_gCO2_per_kWh{suffix}': overall_carbon_intensity,
        f'co2_cost_eur{suffix}': carbon_cost_eur,
        f'co2_price_eur_per_ton{suffix}': co2_price_per_ton  # This is constant over time here.
    }, index=df_generation_mw.index)

    return result_df


def compute_carbon_intensity_simple(dfs: dict[str, pd.DataFrame], type:str, suffix:str) -> pd.DataFrame:
    """
    Computes the energy mix carbon intensity (gCO₂/kWh) for each timestamp.

    :param df: Pandas DataFrame with hourly generation data (MW) for each source.
    :return: DataFrame with added column "carbon_intensity" (gCO₂/kWh).
    """

    # Updated Carbon Intensity Values (gCO₂/kWh) for German Energy Mix
    carbon_intensity = {
        'nuclear': 0,
        'wind_onshore': 0,         # No direct emissions
        'wind_offshore': 0,        # No direct emissions
        'solar': 0,                # No direct emissions
        'renewables': 0,           # Hydro and similar renewables assumed negligible emissions (~0 gCO₂/kWh)
        'biomass': 50,             # Biomass varies (typically 20-100 gCO₂/kWh), assumed near carbon-neutral
        'waste': 300,              # Waste-to-energy varies widely (~250-400 gCO₂/kWh), depending on composition
        'lignite': 1_100,          # Brown coal (lignite) is highly carbon-intensive (~1,000-1,200 gCO₂/kWh)
        'hard_coal': 850,          # Hard coal typically ranges from ~700-900 gCO₂/kWh
        'coal_derived_gas': 450,   # Coal-derived gas assumed similar to natural gas (~400-500 gCO₂/kWh)
        'gas': 400,                # Natural gas combustion ranges ~350-450 gCO₂/kWh, depending on efficiency
        'oil': 900,                # Oil-fired power ranges ~850-950 gCO₂/kWh
        'other_fossil': 700        # Other fossil fuels assumed similar to hard coal (~650-800 gCO₂/kWh)
    }

    df_generation_mw = pd.DataFrame()
    for target_, df_ in dfs.items():
        if target_ != 'load':
            col_name = f'{target_}_{type}'
            if not col_name in df_.columns:
                raise KeyError(f"Column {col_name} not found")

            # Rename the series to match the energy type key
            series_gen = df_[col_name].rename(target_)
            if df_generation_mw.empty:
                df_generation_mw = series_gen.to_frame()
            else:
                # Join on the index (timestamps)
                df_generation_mw = df_generation_mw.join(series_gen, how='outer')

    # Convert MW to kWh (1 MW = 1000 kWh)
    df_kWh = df_generation_mw * 1000

    # Compute total electricity generation (kWh) per timestamp
    total_generation = df_kWh.sum(axis=1)

    # Compute weighted carbon emissions (gCO₂)
    total_emissions = sum(df_kWh[col] * carbon_intensity[col] for col in df_generation_mw.columns)

    # Compute carbon intensity (gCO₂/kWh)
    df = pd.DataFrame(index=df_generation_mw.index)

    target = "carbon_intensity"

    for type_ in ["fitted", "actual", "lower", "upper"]:
        if (type_ == type): df[f"{target}_{type_}"] = total_emissions / total_generation
        else: df[f"{target}_{type_}"] = np.full_like(df.index, 0)

    # Handle cases where total generation is zero (avoid NaN values)
    df = df.fillna(0)

    return df



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

# --- Class to hold data for a given target (train, forecast, dates, error metrics) ---
class TargetData:

    def __init__(self, results_root_dir:str, verbose:bool, positive_floor:float or None):

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

        self.additional_metrics: dict = {}

        self.verbose = verbose

        self.positive_floor = positive_floor

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
        if not self.positive_floor is None: self.df_train_res[self.df_train_res < 0.] = 0.

        # load past forecasts
        self.df_forecast_res = pd.read_csv(
            f"{self.results_root_dir}{var}/{best_model}/{'forecast'}/result.csv",
            index_col=0, parse_dates=True)
        self.df_forecast_res.columns = [ col.replace(suffix, '') for col in self.df_forecast_res.columns ]
        self.df_forecast_res = self.df_forecast_res[[
            f'{target}_{key}' for key in ['actual', 'fitted', 'lower', 'upper']
        ]]
        if not self.df_forecast_res is None: self.df_forecast_res[self.df_forecast_res < 0.] = 0.

        # load forecast
        self.df_forecast = pd.read_csv(
            f"{self.results_root_dir}{var}/{best_model}/{'forecast'}/forecast.csv",
            index_col=0, parse_dates=True)
        self.df_forecast.columns = [ col.replace(suffix, '') for col in self.df_forecast.columns ]
        self.df_forecast = self.df_forecast[[f'{target}_{key}' for key in ['actual', 'fitted', 'lower', 'upper']]]
        if not self.df_forecast is None: self.df_forecast[self.df_forecast < 0.] = 0.

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
            & (self.df_metrics['target'] == target+suffix)
            & (self.df_metrics['model_label'] == self.best_model_label)
            ]['horizon'].unique()
        self.train_cutoffs = [pd.to_datetime(cutoff) for cutoff in self.train_cutoffs]
        assert len(self.train_cutoffs) > 1
        self.horizon = int(self.metadata['horizon'])

        self.forecast_cutoffs = self.df_metrics[
            (self.df_metrics['method'] == 'forecast')
            & (self.df_metrics['target'] == target+suffix)
            & (self.df_metrics['model_label'] == self.best_model_label)
            ]['horizon'].unique()
        self.forecast_cutoffs = [pd.to_datetime(cutoff) for cutoff in self.forecast_cutoffs]
        assert len(self.forecast_cutoffs) > 1

    def get_ave_metric(self, df:pd.DataFrame, target:str, key_actual:str, key_fitted:str, metric:str, n_folds:int)->np.floating:
        if len(self.train_cutoffs)  == 0:
            raise ValueError("No training cutoff found")
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


# --- combine two TargetData classess
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
            f"Train cuttofs mismatch: {target1} ({dt1.best_model_label}) vs {target2} ({dt2.best_model_label}) | "
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
        dt1.df_train_res = copy.deepcopy(dt2.df_train_res)
        if target1 != target2:
            dt1.df_train_res.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        if target1 == target2 and not target1+'_'+"actual" in dt1.df_train_res.columns:
            for key in ['actual', 'fitted', 'lower', 'upper']:
                dt1.df_train_res[f"{target1}_{key}"] = copy.deepcopy(dt2.df_train_res[f"{target2}_{key}"])
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_train_res[f"{target1}_{key}"] = \
                dt1.df_train_res[f"{target1}_{key}"] + dt2.df_train_res[f"{target2}_{key}"]

    # combine dataframes (Training Results)
    if dt1.df_forecast_res.empty:
        dt1.df_forecast_res = dt2.df_forecast_res.copy()
        if target1 != target2:
            dt1.df_forecast_res.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        if target1 == target2 and not target1+'_'+"actual" in dt1.df_forecast_res.columns:
            for key in ['actual', 'fitted', 'lower', 'upper']:
                dt1.df_forecast_res[f"{target1}_{key}"] = copy.deepcopy(dt2.df_forecast_res[f"{target2}_{key}"])
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_forecast_res[f"{target1}_{key}"] = \
                dt1.df_forecast_res[f"{target1}_{key}"] + dt2.df_forecast_res[f"{target2}_{key}"]

    # combine dataframes (Training Results)
    if dt1.df_forecast.empty:
        dt1.df_forecast = dt2.df_forecast.copy()
        if target1 != target2:
            dt1.df_forecast.rename(columns={
                f"{target2}_{key}" : f"{target1}_{key}"
                for key in ['actual', 'fitted', 'lower', 'upper']
            }, inplace=True)
    else:
        if target1 == target2 and not target1+'_'+"actual" in dt1.df_forecast.columns:
            for key in ['actual', 'fitted', 'lower', 'upper']:
                dt1.df_forecast[f"{target1}_{key}"] = copy.deepcopy(dt2.df_forecast[f"{target2}_{key}"])
        for key in ['actual', 'fitted', 'lower', 'upper']:
            dt1.df_forecast[f"{target1}_{key}"] = \
                dt1.df_forecast[f"{target1}_{key}"] + dt2.df_forecast[f"{target2}_{key}"]

    if len(dt1.df_train_res.columns) % 4 != 0:
        raise ValueError(f"Expect to have 4 columns for dataframe. (actual, fitted, lower, upper). "
                         f"Got {len(dt1.df_train_res.columns)} \n {dt1.df_train_res.columns.to_list()}")
    return dt1

def save_carbon_intensity_json(dts:dict[str:TargetData],suffix:str,output_dir_for_figs:str):

    # --- CARBON INTENSITY: past actual ---
    df_carbon_forecast = compute_carbon_intensity_simple(
        dfs = { key : dt.df_forecast_res for key, dt in dts.items() if not key in [ 'generation', 'load' ] },
        type='actual',
        suffix = f'{suffix}'
    )
    convert_csv_to_json(
        df = df_carbon_forecast, target='carbon_intensity',
        output_dir=output_dir_for_figs + "carbon_intensity" + f'{suffix}' + '/', prefix = 'forecast_prev',
        cols= [f"actual"]
    )

    # --- CARBON INTENSITY: past actual ---
    df_carbon_forecast = compute_carbon_intensity_simple(
        dfs = { key : dt.df_forecast_res for key, dt in dts.items() if not key in [ 'generation' ] },
        type='fitted',
        suffix = f'{suffix}'
    )
    convert_csv_to_json(
        df = df_carbon_forecast, target='carbon_intensity',
        output_dir=output_dir_for_figs + "carbon_intensity" + f'{suffix}' + '/', prefix = 'forecast_prev',
        cols= [f"fitted"]
    )

    # --- CARBON INTENSITY: forecast ---
    df_carbon_forecast = compute_carbon_intensity_simple(
        dfs = { key : dt.df_forecast for key, dt in dts.items() if not key in [ 'generation' ] },
        type='fitted',
        suffix = f'{suffix}'
    )
    convert_csv_to_json(
        df = df_carbon_forecast, target='carbon_intensity',
        output_dir=output_dir_for_figs + "carbon_intensity" + f'{suffix}' + '/', prefix = 'forecast_curr',
        cols= [f"fitted"]
    )


class PublishGenerationLoad:

    def __init__(self, country_dict:dict, db_path:str, results_root_dir:str,
                 output_dir_for_figs:str, output_dir_for_api:str, verbose:bool):
        self.c_dict = country_dict
        self.db_path = db_path
        self.verbose = verbose
        self.results_root_dir = results_root_dir # 'forecasting_modules/output/',

        self.load_entsoe_data()
        if country_dict['code'] == 'DE':
            self.load_smard_data()
        else:
            self.df_smard = None

        self.metric = 'rmse'
        self.n_folds = 3

        self.tds = []

        for outdir in [output_dir_for_figs, output_dir_for_api]:
            if not os.path.isdir(outdir):
                os.makedirs(outdir, exist_ok=True)
                logger.info(f"Created directory {outdir}")

        self.output_dir_for_figs = output_dir_for_figs
        self.output_dir_for_api = output_dir_for_api

    def set_load_output_results(self, de_reg:dict, target_label:str, positive_floor:float):

        dt = TargetData(self.results_root_dir, self.verbose, positive_floor)
        dt.load_summary(target_label, de_reg) # load sumamry for target(s)
        # d = TargetData(self.results_root_dir, self.verbose)
        # dts_tarets = {}
        dts = {}
        # df_train_total_res, df_forecast_total_train, df_forecast_total_forecast = \
        #     pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # allow different targets to have different best models (even if trained jointly)
        for target, target_dict in dt.best_models.items():
            target_dict = dt.best_models[target]
            target = target.replace(de_reg['suffix'], '')
            dts[target] = TargetData(self.results_root_dir, self.verbose, positive_floor)
            dts[target].load_summary(target_label, de_reg) # load sumamry for target(s)
            dts[target].load_target_data(target_label, target, de_reg=de_reg, best_model=target_dict['model_label']) # load target
            logger.info(f"Added {target} data for {de_reg['suffix']} (model {target_dict['model_label']})")
            # dts_tarets[target] = copy.deepcopy(dt)

            # todo add publish to api dt.publish_target_individual_plot()

            # todo add publish to deploy dt.publish_target_api()

            # dts_tarets[target] = copy.deepcopy(dt)

            # # summ all contributions as we can only compare the total energy generation with ENTSO-E
            # if target_label == 'energy_mix':
            #     dt_total = add_datas(dt_total, dt, target, target, expect_same_dates=True) # 'generation'
            # else:
            #     dt_total = add_datas(dt_total, dt, target_label, target, expect_same_dates=True)


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
            # target = 'generation'
            # load best wind,solar
            for target_ in ['wind_offshore','wind_onshore','solar']:
                if not os.path.exists(f'{self.results_root_dir}{target_+suffix}/'):
                    logger.warning(f"Missing directory {self.results_root_dir}{target_+suffix}/")
                    continue
                # logger.info(f"Adding {target_} to total {target}")
                dts[target_] = TargetData( self.results_root_dir, self.verbose, positive_floor )
                dts[target_].load_summary( target_, de_reg ) # load sumamry for target(s)
                dts[target_].load_target_data( target_, target_, de_reg=de_reg, best_model=None )
                # dts_tarets[target_] = copy.deepcopy(dt)
                # add_datas(dt_total, dt, target_, target_, expect_same_dates=False)

        # add total generation
        if target_label == 'energy_mix':
            dts['generation'] = TargetData(self.results_root_dir, self.verbose, positive_floor)
            logger.info(f"Added generation data (summing all contributions) for {target_label}")
            for target, dt_ in dts.items():
                if target == 'generation': continue
                dts['generation'] = add_datas(dts['generation'], dt_, 'generation', target, False)
            target = 'generation'
        else:
            target = target_label

        if target_label == 'energy_mix':
            target_ = 'load'
            if not os.path.exists(f'{self.results_root_dir}{target_+suffix}/'):
                logger.warning(f"Missing directory {self.results_root_dir}{target_+suffix}/")
                raise KeyError("Missing load")
            dts['load'] = TargetData( self.results_root_dir, self.verbose, positive_floor)
            dts[target_].load_summary( target_, de_reg ) # load sumamry for target(s)
            dts[target_].load_target_data( target_, target_, de_reg=de_reg, best_model=None )

        # compute forecast error over the past N horizons
        ave_total_metric = dts[target].get_ave_metric(
            dts[target].df_train_res, target, f"{target}_actual",f"{target}_fitted",
            self.metric, self.n_folds
        )
        # compute ENTSO-E error
        ave_entsoe_metric = dts[target].get_ave_metric(
            self.df_entsoe, target, f"{target+suffix}",f"{target}_forecast"+suffix,
            self.metric, self.n_folds
        )

        logger.info(f'For {target} average over {self.n_folds} '
                    f'total RMSE is {ave_total_metric:.0f} | ENTSOE RMSE is {ave_entsoe_metric:.0f}')

        # publish generation per tso (LAST UPDATED FORECAST)
        dts[target].save_past_and_current_forecasts_json(
            self.output_dir_for_figs + '/' + target  + suffix + '/', target
        )
        dts[target].save_past_and_current_forecasts_api_json(
            self.output_dir_for_api + '/', de_reg['TSO'], target
        )
        if target_label == 'energy_mix':
            self.save_multiitarget_data_json(
                dts = { key:dt for key, dt in dts.items() if not key in ['generation','load'] }, suffix = suffix
            )
            save_carbon_intensity_json(dts, suffix, self.output_dir_for_figs)



        # if target_label == 'energy_mix':
        #     # ave energy mix
        #     self.save_multiitarget_data_json(de_reg['suffix'], dts_tarets)

        # add summary to the summary dict
        metadata = dts[target].get_table_entry(de_reg['TSO'], ave_total_metric, ave_entsoe_metric)

        # meta = {"train_cutoffs": dt.train_cutoffs, "horizon": dt.horizon, "target": target}
        return dt, dts, metadata

    def save_multiitarget_data_json(self, suffix:str, dts:dict[str:TargetData]):
        for target, dt in dts.items():
            if dt.df_forecast_res.empty:
                raise ValueError(f"No df_forecast_res found for {target}")
            if dt.df_forecast.empty:
                raise ValueError(f"No df_forecast found for {target}")

        outdir = self.output_dir_for_figs + f"energy_mix{suffix}/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            logger.info(f"Created directory {outdir}")

        # create dataframe with actual values
        df = pd.DataFrame({
            target:pd.Series(dt.df_forecast_res[f'{target}_actual']) for target, dt in dts.items()
        })
        timeseries_to_json( df=df, fname=outdir + "forecast_prev_actual.json" )

        df = pd.DataFrame({
            target:pd.Series(dt.df_forecast_res[f'{target}_fitted']) for target, dt in dts.items()
        })
        timeseries_to_json( df=df, fname=outdir + "forecast_prev_fitted.json" )

        df = pd.DataFrame({
            target:pd.Series(dt.df_forecast[f'{target}_fitted']) for target, dt in dts.items()
        })
        timeseries_to_json( df=df, fname=outdir + "forecast_curr_fitted.json" )

    def load_entsoe_data(self):
        self.df_entsoe = pd.read_parquet(self.db_path + 'entsoe/' + 'history_hourly.parquet')

        self.df_entsoe = validate_dataframe(self.df_entsoe, 'df_entsoe', logger.warning, self.verbose)

        # compute total generation
        for de_reg in self.c_dict['regions']:
            suffix = de_reg['suffix']
            self.df_entsoe['generation'+suffix] = \
                self.df_entsoe[[
                    col+suffix for col in list(entsoe_generation_type_mapping.keys())
                        if col+suffix in self.df_entsoe.columns
                ]].sum(axis=1)



        logger.info(f'Loaded ENTSO-E with file {len(self.df_entsoe)} entries')

    def load_smard_data(self):
        def print_nans(df):
            nan_counts = df.isna().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if not nan_cols.empty:
                logger.error(f"SMARD has nans in columns: {nan_cols}")
                raise ValueError("Nans in SMARD data")
        self.df_smard = pd.read_parquet(self.db_path + 'smard/' + 'history_hourly.parquet')

        if not isinstance(self.df_smard.index, pd.DatetimeIndex):
            self.df_smard = self.df_smard.set_index(pd.to_datetime(self.df_smard.index))

        self.df_smard = validate_dataframe(self.df_smard, 'df_entsoe', logger.warning, self.verbose)
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
        for target in ['generation', 'generation_forecast']:
            # Replace 0 values with NaN
            self.df_smard[target].replace(0, np.nan, inplace=True)
            # Interpolate NaN values using time series interpolation (backward direction)
            self.df_smard[target].interpolate(method='time', limit_direction='both', inplace=True)
        print_nans(self.df_smard)
        logger.info(f'Loaded SMARD with file {len(self.df_smard)} entries')

    def write_notes_for_energy_mix(
            self,dts_tso:dict[str:TargetData], dts_total:dict[str:TargetData], metadatas_tso:dict, total_error:dict, reference_error:dict
    ):
        '''

        :param dts_tso:
        :param dts_total:
        :return:
        '''
        summary_fpath = f"{self.output_dir_for_figs}/{'energy_mix'}_notes_en.md"

        dts_tso_ = dts_tso.copy()
        dts_total_ = dts_total.copy()
        total_error_ = total_error.copy()
        reference_error_ = reference_error.copy()
        metadatas_tso_ = pd.DataFrame(metadatas_tso).T

        ''' error section '''
        output_mrkdown_text = \
        f""" 
The total energy generation week-ahead forecast has a RMSE of **{int(total_error['generation'])}**, compared to **{int(reference_error['generation'])}** for the TSO day-ahead reference forecast.  
        """
        if len(list(dts_tso_.keys())) > 0:
            output_mrkdown_text+= \
        f"""
The largest contributor to the forecast error is **{metadatas_tso_.loc[metadatas_tso_['RMSE'].idxmax(), 'TSO/Region']}**, with an RMSE of **{int(metadatas_tso_.loc[metadatas_tso_['RMSE'].idxmax(), 'RMSE'])}**.  

On average, our forecast RMSE is **{(metadatas_tso_['RMSE'] / metadatas_tso_['TSO RMSE']).mean():.1f}** times the TSO forecast RMSE,   
and our forecast achieves lower error in the following regions:  **{", ".join(metadatas_tso_.loc[metadatas_tso_['RMSE'] < metadatas_tso_['TSO RMSE'], 'TSO/Region'].tolist())}**.  
        """
        output_mrkdown_text += \
        """
For a detailed breakdown of forecast error metrics, see the **'Individual Forecasts'** section.

📊 *Detailed analytics coming soon!*
        """
        summary_fpath = f"{self.output_dir_for_figs}/{'energy_mix'}_notes_en.md"
        with open(summary_fpath, "w") as file:
            file.write(output_mrkdown_text)


    def process(self, target_label:str, avail_regions:tuple,positive_floor:float or None):
        # compute forecast error over the past N horizons
        if target_label == 'energy_mix':
            target = 'generation'
        else:
            target = target_label

        # dt_total = TargetData(self.results_root_dir, self.verbose)
        dts_total = {}
        dts_tso = {}
        dt_tso = {}
        metadatas_tso = {}
        regions = [region for region in self.c_dict['regions'] if region['name'] in avail_regions]
        for region_dict in regions:
            if not os.path.exists(f"{self.results_root_dir}/{target_label}{region_dict['suffix']}/"):
                logger.warning(f"Missing output directory: {self.results_root_dir}/{target_label}{region_dict['suffix']}/")
                continue

            dt_tso[region_dict['TSO']], dts_tso[region_dict['TSO']], metadatas_tso[region_dict['TSO']] = \
                self.set_load_output_results(
                    de_reg=region_dict, target_label=target_label, positive_floor=positive_floor
                )

            for target_, dt in dts_tso[region_dict['TSO']].items():
                if not target_ in dts_total: dts_total[target_] = \
                    TargetData(self.results_root_dir, self.verbose, positive_floor)
                dts_total[target_] = add_datas(dts_total[target_], dt, target_, target_, False)


            # add the data from dt_tso to dt_total
            # add_datas(dt_total, dt_tso, target, target, expect_same_dates=False)
            # table.append(metadata_tso)
            # dts_total[region_dict['TSO']] = TargetData(self.results_root_dir, self.verbose)
            # for target_, dt in dts_tso.items():
            #     dts_total[target_] = add_datas(dts_tso[target_], dt, target_, target_, expect_same_dates=False)
                # dt_total = add_datas(dt_total, dt, 'generation', 'generation', expect_same_dates=False)

        # publish the results (forecasts) to .json for ./deploy/...
        dts_total[target].save_past_and_current_forecasts_json(
            self.output_dir_for_figs + '/' + target + '/', target
        )
        dts_total[target].save_past_and_current_forecasts_api_json(
            self.output_dir_for_api  + '/', 'total', target
        )
        if target_label == 'energy_mix':
            self.save_multiitarget_data_json(
                dts = {key:dt for key, dt in dts_total.items() if not key in ['generation','load']}, suffix=''
            )
            save_carbon_intensity_json(dts_total, '', self.output_dir_for_figs)

        ave_total_metric=dts_total[target].get_ave_metric(
            dts_total[target].df_train_res, target,
            f"{target}_actual", f"{target}_fitted", self.metric, self.n_folds
        )

        # compute smard total error
        if not self.df_smard is None:
            ave_smard_metric = dts_total[target].get_ave_metric(
                self.df_smard, target, target, f"{target}_forecast", self.metric, self.n_folds
            )
        else:
            if len(regions) != 1:
                raise ValueError("Expect one TSO for using ENTSOE average error as an overall error metric")
            ave_smard_metric = metadatas_tso[region_dict['TSO']]['TSO RMSE']

        logger.info(f"For {target} average over {self.n_folds} "
                    f"total RMSE is {ave_total_metric:.0f} | SMARD RMSE is {ave_smard_metric:.0f}")


        if target_label == 'energy_mix':
            self.write_notes_for_energy_mix(
                dts_tso, dts_total, metadatas_tso,
                {target:ave_total_metric}, {target:ave_smard_metric}
            )

        table = [val for key, val in metadatas_tso.items()]
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


def main(country_code:str, target:str, verboose:bool):

    countries = ['DE', 'FR', 'all']
    if not country_code in countries:
        raise ValueError(f'country_code must be in {countries}. Given: {country_code}')
    if country_code == 'all': country_code_ = countries[:-1] # all countries
    else: country_code_ = [country_code]

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load', 'energy_mix', 'all']
    if not target in targets:
        raise ValueError(f'target must be in {targets}. Given: {target}')
    if target == 'all': target_ = targets[:-1]
    else: target_ = [target]

    for country_code in country_code_:

        # check country
        c_dict:dict = [dict_ for dict_ in countries_metadata if dict_["code"] == country_code][0]
        if len(list(c_dict.keys())) == 0:
            raise KeyError(f"No country dict found for country code {country_code}. Check your country code.")
        regions = c_dict["regions"]
        if len(regions) == 0:
            logger.warning(f"No regions (TSOs) dicts found for country code {country_code}.")
        locations = list(c_dict['locations'].keys())
        if len(locations) == 0:
            logger.warning(f"No locations (for weather data) found for country code {country_code}.")

        # set database location

        start_time = time.time()  # Start the timer

        publisher = PublishGenerationLoad(
            country_dict=c_dict,
            db_path = f'./database/{country_code}/',
            results_root_dir = f'./output/{country_code}/forecasts/',
            output_dir_for_figs = f'./deploy/data/{country_code}/forecasts/',
            output_dir_for_api = f'./deploy/data/{country_code}/api/forecasts/',
            verbose=verboose
        )

        # target_settings = [
        #     {'label' : 'wind_offshore', 'target' : 'wind_offshore', "regions" : ('DE_50HZ', 'DE_TENNET'), 'positive_floor':0},
        #     {'label' : 'wind_onshore', 'target' : 'wind_onshore', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'), 'positive_floor':0},
        #     {'label' : 'solar', 'target' : 'solar', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'), 'positive_floor':0},
        #     {'label' : 'load', 'target' : 'load', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'), 'positive_floor':0},
        #     {'label' : 'energy_mix', 'target' : 'energy_mix', "regions" : ('DE_50HZ', 'DE_TENNET', 'DE_AMPRION', 'DE_TRANSNET'), 'positive_floor':0}
        # ]
        for target__ in target_:
            # target_dict = [t for t in target_settings if t["target"] == target__][0]
            regions = tuple([reg['name'] for reg in c_dict['regions'] if target__ in reg['available_targets']])
            publisher.process(
                target_label=target__,#target_dict['label'],
                avail_regions=regions,#target_dict['regions'],
                positive_floor=0.#target_dict['positive_floor']
            )


        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        hours, minutes = divmod(elapsed_time // 60, 60)

        logger.info(
            f"All tasks for country {country_code} are completed successfully! Execution time: "
            f"{int(hours)} hours and {int(minutes)} minutes."
        )

if __name__ == '__main__':

    print("launching publish_data.py")

    if len(sys.argv) != 3:
        raise KeyError("Usage: python update_database.py <country_code> <target>")
        # country_code = str( 'FR' )
        # target = str( 'all' )
        # freq = str( 'hourly' )
    else:
        country_code = str( sys.argv[1] )
        target = str( sys.argv[2] )
        # freq = str( sys.argv[3] )

    main(country_code, target, True)

