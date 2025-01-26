import copy, re, pandas as pd
import numpy as np
import os

from data_collection_modules.collect_data_openmeteo import OpenMeteo
from data_collection_modules.german_locations import (
    de_regions,
    loc_onshore_windfarms,
    loc_offshore_windfarms,
    loc_cities,
    loc_solarfarms
)
from data_modules.utils import (
    validate_dataframe
)
from forecasting_modules.utils import convert_ensemble_string


energy_mix_config = {
    'targets' : ["hard_coal", "lignite", "coal_derived_gas", "oil", "other_fossil", "gas", "renewables"],
    "aggregations": {"renewables": [
        "biomass","waste","geothermal","pumped_storage","run_of_river","water_reservoir","other_renewables"
    ]}
}

# TODO: USE SQL HERE!!!
def extract_from_database(target:str, db_path:str, outdir:str, tso_name:str, n_horizons:int, horizon:int, verbose:bool)\
        -> tuple[pd.DataFrame, pd.DataFrame]:

    # -------- laod database TODO move to SQLlite DB
    df_smard = pd.read_parquet(db_path + 'smard/' + 'history.parquet')
    df_om_offshore = pd.read_parquet(db_path + 'openmeteo/' + 'offshore_history.parquet')
    df_om_offshore_f = pd.read_parquet(db_path + 'openmeteo/' + 'offshore_forecast.parquet')
    df_om_onshore = pd.read_parquet(db_path + 'openmeteo/' + 'onshore_history.parquet')
    df_om_onshore_f = pd.read_parquet(db_path + 'openmeteo/' + 'onshore_forecast.parquet')
    df_om_solar = pd.read_parquet(db_path + 'openmeteo/' + 'solar_history.parquet')
    df_om_solar_f = pd.read_parquet(db_path + 'openmeteo/' + 'solar_forecast.parquet')
    df_om_cities = pd.read_parquet(db_path + 'openmeteo/' + 'cities_history.parquet')
    df_om_cities_f = pd.read_parquet(db_path + 'openmeteo/' + 'cities_forecast.parquet')
    df_es = pd.read_parquet(db_path + 'epexspot/' + 'history.parquet')
    df_entsoe = pd.read_parquet(db_path + 'entsoe/' + 'history.parquet')
    df_entsoe = df_entsoe.apply(pd.to_numeric, errors='coerce')
    # ----- CHECKS AND NOTES ----
    if verbose:
        print("---------- LOADING DATABASE DATA ----------")
        print(f"SMARD data shapes hist={df_smard.shape} (days={len(df_smard)/24}) start={df_smard.index[0]} end={df_smard.index[-1]}")
        print(f"ENTSOE data shapes hist={df_entsoe.shape} (days={len(df_entsoe)/24}) start={df_entsoe.index[0]} end={df_entsoe.index[-1]}")
        print(f"OM offshore data shapes hist={df_om_offshore.shape} (days={len(df_om_offshore)/24}) start={df_om_offshore.index[0]} end={df_om_offshore.index[-1]}")
        print(f"OM offshore data shapes forecast={df_om_offshore_f.shape} (days={len(df_om_offshore_f)/24}) start={df_om_offshore_f.index[0]} end={df_om_offshore_f.index[-1]}")
        print(f"OM onshore data shapes hist={df_om_onshore.shape} (days={len(df_om_onshore)/24}) start={df_om_onshore.index[0]} end={df_om_onshore.index[-1]}")
        print(f"OM onshore data shapes forecast={df_om_onshore_f.shape} (days={len(df_om_onshore_f)/24}) start={df_om_onshore_f.index[0]} end={df_om_onshore_f.index[-1]}")
        print(f"OM solar data shapes hist={df_om_solar.shape} (days={len(df_om_solar)/24}) start={df_om_solar.index[0]} end={df_om_solar.index[-1]}")
        print(f"OM solar data shapes forecast={df_om_solar_f.shape} (days={len(df_om_solar_f)/24}) start={df_om_solar_f.index[0]} end={df_om_solar_f.index[-1]}")
        print(f"EPEXSPOT data shapes hist={df_es.shape} (days={len(df_es)/24}) start={df_es.index[0]} end={df_es.index[-1]}")
        print("-------------------------------------------")

    if len(df_om_offshore_f) != len(df_om_offshore_f) or len(df_om_solar_f) != len(df_om_solar_f):
        raise IOError("The forecast DataFrames have different number of columns.")

    # check that horizon is correct and that start/end hours are correct
    if horizon < len(df_om_offshore_f):
        if verbose: print(
            f"Forecast dataframe has {len(df_om_offshore_f)} rows while {horizon} rows are requested. Trimming..."
        )
        df_om_offshore_f = df_om_offshore_f.iloc[:horizon]
        df_om_onshore_f = df_om_onshore_f.iloc[:horizon]
        df_om_solar_f = df_om_solar_f.iloc[:horizon]
        df_om_cities_f = df_om_cities_f.iloc[:horizon]
        assert len(df_om_offshore_f) == horizon
        assert (df_om_offshore_f.index[-1].hour == 23 and
                df_om_offshore_f.index[-1].minute == 0 and
                df_om_offshore_f.index[-1].second == 0)

    target_notso = ''
    tso_dict = {}
    if (('wind_offshore' in target) or ('wind_onshore' in target) or ('solar' in target) or ('load' in target) or ('energy_mix' in target)):

        for de_reg in de_regions:
            if target.__contains__(de_reg['suffix']) and tso_name != de_reg['name']:
                raise IOError(f"The region must be {de_reg} for target={target}")
            if target.endswith(de_reg['suffix']):
                target_notso = target.replace(de_reg['suffix'], '')
                tso_dict = de_reg
                break
        if target_notso == '':
            raise ValueError(f"target={target} does not contain {[de_reg['suffix'] for de_reg in de_regions]}")

        suffix = tso_dict['suffix']

        # get target column
        if target_notso == 'energy_mix':
            target_cols =  df_entsoe[
                [target_ + suffix for target_ in energy_mix_config['targets']
                    if not target_ in list(energy_mix_config['aggregations'].keys())]
            ]
            for key in energy_mix_config['targets']:
                if key in list(energy_mix_config['aggregations'].keys()):
                    keys_to_agg = [
                        col + suffix for col in energy_mix_config['aggregations'][key]
                            if col + suffix in list(df_entsoe.keys())
                    ]
                    if len(keys_to_agg) != len(energy_mix_config['aggregations'][key]):
                        print(f"Warning! Not all keys for aggregating {key} are found in entsoe dataframe. "
                              f"{len(keys_to_agg)} out of {len(energy_mix_config['aggregations'][key])} will be used ")
                    # aggregate for the required column
                    df_entsoe[key + suffix] = df_entsoe[keys_to_agg].sum(axis=1)
                    target_cols = pd.merge(
                        target_cols, df_entsoe[key + suffix], left_index=True, right_index=True, how='left'
                    )
        else:
            target_cols = df_entsoe[target_notso + suffix]

        target_cols.replace(['NaN', 'None', '', ' '], np.nan, inplace=True)
        if target_cols.isna().any().any():
            print("Warning! Nans in the target columns! Filling with 0")
        target_cols.fillna(0, inplace=True) #

        # get feature dataframe
        dataframe, dataframe_f = None, None
        if 'wind_offshore' in target: dataframe, dataframe_f = df_om_offshore, df_om_offshore_f
        elif 'wind_onshore' in target: dataframe, dataframe_f = df_om_onshore, df_om_onshore_f
        elif 'solar' in target: dataframe, dataframe_f = df_om_solar, df_om_solar_f
        elif 'load' in target: dataframe, dataframe_f = df_om_cities, df_om_cities_f
        elif 'energy_mix' in target: dataframe, dataframe_f = df_om_cities, df_om_cities_f
        else: raise NotImplementedError(f"No dataframe selection for target={target} tso_name={tso_name}")

        # get features specific to this location (TSO)
        locations = []
        if 'wind_offshore' in target: locations = loc_offshore_windfarms
        elif 'wind_onshore' in target: locations = loc_onshore_windfarms
        elif 'solar' in target: locations = loc_solarfarms
        elif 'load' in target: locations = loc_cities
        elif 'energy_mix' in target: locations = loc_cities
        else: raise NotImplementedError(f"Locations are not available for target={target} tso_name={tso_name}")

        om_suffixes = [loc['suffix'] for loc in locations if loc['TSO'] == tso_dict['TSO']]
        feature_col_names = dataframe.columns[dataframe.columns.str.endswith(tuple(om_suffixes))]

        if verbose:
            print(f"TARGET: {target} TSO: {tso_dict['TSO']} "
                  f"Locations: {len([loc for loc in locations if loc['TSO'] == tso_dict['TSO']])} "
                  f"OM suffixes: {len(om_suffixes)} Feature columns: {len(feature_col_names)}")
            print(f"Suffixes {om_suffixes}")
        # combine weather data and target column (by convention)
        df_hist = pd.merge(
            left=dataframe[feature_col_names], right=target_cols, left_index=True, right_index=True, how='left'
        )
        df_forecast = dataframe_f[feature_col_names]


    else:
        raise NotImplementedError(f"target={target} is not yet supported")

    # add additional quantities
    add_exog = []
    if 'load' in target: add_exog = ["wind_offshore", "wind_onshore", "solar"]
    elif 'energy_mix' in target: add_exog = ["wind_offshore", "wind_onshore", "solar", 'load', 'residual_load']

    if len(add_exog) > 0:
        if len(tso_dict.keys()) == 0: raise NotImplementedError(f"No TSO dict available for target={target}")
        for exog in add_exog:
            # load historic data
            exog_tso = exog + tso_dict['suffix']
            if not exog_tso in df_entsoe.columns.tolist():
                if verbose: print(f"Warning! Required exogenous feature {exog_tso} is not in ENTSO-E dataset. Skipping.")
                continue

            entsoe_col = df_entsoe[exog_tso]
            df_hist = pd.merge(left=df_hist, right=entsoe_col, left_index=True, right_index=True, how='left')

            # load forecast from current best forecast
            best_model = None
            fpath = outdir+exog_tso+'/'+'best_model.txt'
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Best model file not found {fpath}")
            with open(fpath, 'r') as f:
                best_model = f.read().strip()

            if best_model.__contains__('ensemble'):
                best_model = convert_ensemble_string(best_model)

            fpath = outdir+exog_tso+'/' + best_model + '/' + 'forecast/' + 'forecast.csv'
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Forecast file not found {fpath}")

            # load the latest forecast for exogenous variable and add it to df_forecast
            df_ = pd.read_csv(fpath, index_col=0)
            df_.index = pd.to_datetime(df_.index)
            df_.rename(columns={f'{exog_tso}_fitted':f'{exog_tso}'},inplace=True)

            # import matplotlib.pyplot as plt
            # plt.plot(df_.index, df_[f'{exog_tso}_fitted'])
            # plt.plot(df_forecast.index, df_forecast[df_forecast.columns.tolist()[0]])
            # plt.show()
            # exit(1)

            df_forecast = pd.merge(
                left=df_forecast,
                right=df_[f'{exog_tso}'],
                left_index=True,
                right_index=True,
                how='left'
            )


    # limit dataframe to the required max size
    df_hist = df_hist.tail(len(df_forecast)*n_horizons)
    if verbose:
        print(f"Limiting df_hist to {n_horizons} horizons. "
              f"From {len(dataframe)} entries do {len(df_hist)} ({len(df_hist)/len(dataframe)*100:.1f} %)")

    # check again the dataframe validity
    if not 'energy_mix' in target:
        if not len(df_hist.columns) == len(df_forecast.columns)+1:
            raise ValueError(f'The DataFrames have different columns. '
                             f'hist={df_hist.shape} forecast={df_forecast.shape}')

    if len(df_hist) <= 1 or len(df_forecast) <= 1:
        raise ValueError(f'The DataFrames must have >1 rows '
                         f'hist={df_hist.shape} forecast={df_forecast.shape}')
    # ------------------
    # if not validate_dataframe(df_hist):
    #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    # if not validate_dataframe(df_forecast):
    #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    if not df_forecast.index[0] == df_hist.index[-1]+pd.Timedelta(hours=1):
        raise ValueError(f"Forecast dataframe must have index[0] = historic index[-1] + 1 hour")

    expected_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq='h')
    if not df_hist.index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")

    if df_forecast.isna().any().any():
        raise ValueError(f"df_forecast contains NaN entries. df_forecast={df_forecast[df_forecast.isna()]}")

    return df_hist, df_forecast

    # # --- PREPARE DATA FOR WIND POWER FORECASTING ---
    # if (('wind_offshore' in target) or ('wind_onshore' in target) or ('solar' in target)):
    #     # quick sanity checks that regions is set correctly
    #     if (target.__contains__('_tenn') and tso_name != 'DE_TENNET'):
    #         raise IOError(f"The region must be 'DE_TENNE' for target={target}")
    #     if (target.__contains__('_50hz') and tso_name != 'DE_50HZ'):
    #         raise IOError(f"The region must be 'DE_50HZ' for target={target}")
    #     if (target.__contains__('_amp') and tso_name != 'DE_AMPRION'):
    #         raise IOError(f"The region must be 'DE_AMPRION' for target={target}")
    #     if (target.__contains__('_tran') and tso_name != 'DE_TRANSNET'):
    #         raise IOError(f"The region must be 'DE_TRANSNET' for target={target}")
    #
    #     if verbose: print(f"Target={target} Nans={df_entsoe[target].isna().sum().sum()}")
    #
    #     tsos = ['_tenn', '_50hz', '_amp', '_tran']
    #     regions = ['DE_TENNET', 'DE_50HZ', 'DE_AMPRION', 'DE_TRANSNET']
    #     target_general = ''
    #     for tso_suffix, reg in zip(tsos, regions):
    #         if target.__contains__(tso_suffix) and tso_name != reg:
    #             raise IOError(f"The region must be {reg} for target={target} (given region={tso_name}")
    #         if target.endswith(tso_suffix):
    #             target_general = target.replace(tso_suffix, '')
    #             break
    #     region_:dict = [reg for reg in de_regions if reg['name'] == tso_name][0]
    #     reg_suffix = region_['suffix']
    #
    #
    #     # get openmeteo data for locations within this region (for windmills associated with this TSO)
    #     region_:dict = [reg for reg in de_regions if reg['name'] == tso_name][0]
    #     reg_suffix = region_['suffix']
    #     entsoe_column = str('wind_offshore' if 'wind_offshore' in target else 'wind_onshore') + reg_suffix
    #     wind_farms = loc_offshore_windfarms if 'wind_offshore' in target else loc_onshore_windfarms
    #     om_suffixes = [wind_farm['suffix'] for wind_farm in wind_farms if wind_farm['TSO'] == region_['TSO']]
    #     dataframe = df_om_offshore if 'wind_offshore' in target else df_om_onshore
    #     columns_to_select = dataframe.columns[dataframe.columns.str.endswith(tuple(om_suffixes))]
    #
    #     # combine weather data and target column (convention)
    #     df_hist = pd.merge(left=dataframe[columns_to_select],right=df_entsoe[entsoe_column], left_index=True, right_index=True, how='left')
    #     dataframe_f = df_om_offshore_f if 'wind_offshore' in target else df_om_onshore_f
    #
    #     df_forecast = dataframe_f[columns_to_select]
    #
    #
    #     df_hist = df_hist.tail(len(df_forecast)*n_horizons) # crop the dataset if needed
    #
    #     if not len(df_hist.columns) == len(df_forecast.columns)+1:
    #         raise ValueError(f'The DataFrames have different columns. '
    #                          f'hist={df_hist.shape} forecast={df_forecast.shape}')
    #
    #     if len(df_hist) <= 1 or len(df_forecast) <= 1:
    #         raise ValueError(f'The DataFrames must have >1 rows '
    #                          f'hist={df_hist.shape} forecast={df_forecast.shape}')
    # else:
    #     raise NotImplementedError(f"Target {target} and region {tso_name} are not implemented.")
    #
    #
    #
    #
    # # ------------------
    # # if not validate_dataframe(df_hist):
    # #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    # # if not validate_dataframe(df_forecast):
    # #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    # if not df_forecast.index[0] == df_hist.index[-1]+pd.Timedelta(hours=1):
    #     raise ValueError(f"Forecast dataframe must have index[0] = historic index[-1] + 1 hour")
    #
    # expected_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq='h')
    # if not df_hist.index.equals(expected_range):
    #     raise ValueError("full_index must be continuous with hourly frequency.")
    #
    # return (df_hist, df_forecast)

def mask_outliers_and_unphysical_values(
        df_hist: pd.DataFrame,
        df_forecast: pd.DataFrame,
        target: str,
        verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Applies known physical limits to both df_hist and df_forecast,
       masking out (with NaNs) any values that violate these limits.
       Prints the number of replaced values if verbose=True.

    2) Performs a simple z-score based anomaly detection on the target
       column, trained on df_hist, and applies the same detection to
       df_forecast, replacing outliers with NaNs.

    Returns
    -------
    df_hist: pd.DataFrame
        Historical data with unphysical values and outliers masked.

    df_forecast: pd.DataFrame
        Forecast data with unphysical values and outliers masked.
    """

    # ----- 1) Mask unphysical values based on known physical limits -----
    for col in df_hist.columns:
        for feature, (phys_lower, phys_upper) in OpenMeteo.phys_limits.items():
            # Check if the feature name is contained in the column name
            if feature in str(col).lower():
                # Count how many values are out of physical bounds before masking
                unphys_hist = ((df_hist[col] < phys_lower) | (df_hist[col] > phys_upper)).sum()
                unphys_fore = ((df_forecast[col] < phys_lower) | (df_forecast[col] > phys_upper)).sum()

                if verbose and (unphys_hist > 0 or unphys_fore > 0):
                    print(
                        f"[Physical Limits] Column '{col}' | "
                        f"Replaced {unphys_hist} unphysical values in df_hist, "
                        f"{unphys_fore} in df_forecast."
                    )

                # Mask out-of-bounds values with NaN
                df_hist[col] = df_hist[col].where(
                    (df_hist[col] >= phys_lower) & (df_hist[col] <= phys_upper),
                    np.nan
                )
                df_forecast[col] = df_forecast[col].where(
                    (df_forecast[col] >= phys_lower) & (df_forecast[col] <= phys_upper),
                    np.nan
                )

    # ----- 2) Simple outlier (anomaly) detection on the target column -----
    # Train a z-score based detection on df_hist[target]
    # (If your target column is missing in any row, drop those temporarily to calculate stats)
    hist_nonan = df_hist[target].dropna()
    mean_val = hist_nonan.mean()
    std_val = hist_nonan.std()
    threshold = 3  # For example, 3 standard deviations

    if std_val == 0 or np.isnan(std_val):
        if verbose:
            print(f"[Anomaly Detection] Standard deviation is zero or NaN for '{target}'. Skipping outlier masking.")
    else:
        # Identify outliers in df_hist
        outliers_hist_mask = (df_hist[target] - mean_val).abs() > threshold * std_val
        outliers_hist_count = outliers_hist_mask.sum()
        if verbose and outliers_hist_count > 0:
            print(
                f"[Anomaly Detection] '{target}' | "
                f"{outliers_hist_count} outliers found in df_hist (z-score > {threshold})."
            )
        df_hist.loc[outliers_hist_mask, target] = np.nan

        # Identify outliers in df_forecast using the same mean, std
        # outliers_fore_mask = (df_forecast[target] - mean_val).abs() > threshold * std_val
        # outliers_fore_count = outliers_fore_mask.sum()
        # if verbose and outliers_fore_count > 0:
        #     print(
        #         f"[Anomaly Detection] '{target}' | "
        #         f"{outliers_fore_count} outliers found in df_forecast (z-score > {threshold})."
        #     )
        # df_forecast.loc[outliers_fore_mask, target] = np.nan

    # Return the cleaned dataframes
    return df_hist, df_forecast

def clean_and_impute(df_hist, df_forecast, target, verbose:bool)->tuple[pd.DataFrame, pd.DataFrame]:

    df_hist, df_forecast = mask_outliers_and_unphysical_values(df_hist, df_forecast, target, verbose)
    df_hist = validate_dataframe(df_hist, 'df_hist', verbose=verbose)
    df_forecast = validate_dataframe(df_forecast, 'df_forecast', verbose=verbose)
    expected_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq='h')
    if not df_hist.index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")

    # final check for nans. We must be certain that no nans enter training stage
    def check_for_nans_and_raise_error(df:pd.DataFrame):
        nan_counts = df.isna().sum()  # Get the count of NaNs per column
        columns_with_nans = nan_counts[nan_counts > 0]  # Filter columns with NaNs
        if not columns_with_nans.empty:
            error_message = "The following columns have NaN values:\n"
            for column, count in columns_with_nans.items():
                error_message += f"- {column}: {count} NaNs\n"
            raise ValueError(error_message)
        else:
            print("No NaNs found in the DataFrame.")
    check_for_nans_and_raise_error(df_hist)
    check_for_nans_and_raise_error(df_forecast)

    return df_hist, df_forecast