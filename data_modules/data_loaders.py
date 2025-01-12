import copy, re, pandas as pd
import numpy as np

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

def load_prepared_data(target:str,datapath:str, verbose:bool)  ->tuple[pd.DataFrame, pd.DataFrame]:

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

def extract_from_database(target:str, db_path:str, tso_name:str, n_horizons:int, horizon:int, verbose:bool) \
        -> tuple[pd.DataFrame, pd.DataFrame]:

    # -------- laod database TODO move to SQLlite DB
    df_smard = pd.read_parquet(db_path + 'smard/' + 'history.parquet')
    df_om_offshore = pd.read_parquet(db_path + 'openmeteo/' + 'offshore_history.parquet')
    df_om_offshore_f = pd.read_parquet(db_path + 'openmeteo/' + 'offshore_forecast.parquet')
    df_om_onshore = pd.read_parquet(db_path + 'openmeteo/' + 'onshore_history.parquet')
    df_om_onshore_f = pd.read_parquet(db_path + 'openmeteo/' + 'onshore_forecast.parquet')
    df_om_solar = pd.read_parquet(db_path + 'openmeteo/' + 'solar_history.parquet')
    df_om_solar_f = pd.read_parquet(db_path + 'openmeteo/' + 'solar_forecast.parquet')
    df_es = pd.read_parquet(db_path + 'epexspot/' + 'history.parquet')
    df_entsoe = pd.read_parquet(db_path + 'entsoe/' + 'history.parquet')

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
        if verbose: print("Forecast dataframe has {} rows while {} rows are requested. Trimming...")
        df_om_offshore_f = df_om_offshore_f.iloc[:horizon]
        df_om_onshore_f = df_om_onshore_f.iloc[:horizon]
        df_om_solar_f = df_om_solar_f.iloc[:horizon]
        assert len(df_om_offshore_f) == horizon
        assert (df_om_offshore_f.index[-1].hour == 23 and
                df_om_offshore_f.index[-1].minute == 0 and
                df_om_offshore_f.index[-1].second == 0)

    target_notso = ''
    tso_dict = {}
    if (('wind_offshore' in target) or ('wind_onshore' in target) or ('solar' in target)):

        for de_reg in de_regions:
            if target.__contains__(de_reg['suffix']) and tso_name != de_reg['name']:
                raise IOError(f"The region must be {de_reg} for target={target}")
            if target.endswith(de_reg['suffix']):
                target_notso = target.replace(de_reg['suffix'], '')
                tso_dict = de_reg
                break
        if target_notso == '':
            raise ValueError(f"target={target} does not contain {[de_reg['suffix'] for de_reg in de_regions]}")

        # get target column
        target_col = df_entsoe[target_notso + tso_dict['suffix']]

        # get feature dataframe
        dataframe, dataframe_f = None, None
        if 'wind_offshore' in target: dataframe, dataframe_f = df_om_offshore, df_om_offshore_f
        elif 'wind_onshore' in target: dataframe, dataframe_f = df_om_onshore, df_om_onshore_f
        elif 'solar' in target: dataframe, dataframe_f = df_om_solar, df_om_solar_f


        # get features specific to this location (TSO)
        locations = []
        if 'wind_offshore' in target: locations = loc_offshore_windfarms
        elif 'wind_onshore' in target: locations = loc_onshore_windfarms
        elif 'solar' in target: locations = loc_solarfarms
        om_suffixes = [loc['suffix'] for loc in locations if loc['TSO'] == tso_dict['TSO']]
        feature_col_names = dataframe.columns[dataframe.columns.str.endswith(tuple(om_suffixes))]

        if verbose:
            print(f"TARGET: {target} TSO: {tso_dict['TSO']} "
                  f"Locations: {len([loc for loc in locations if loc['TSO'] == tso_dict['TSO']])} "
                  f"OM suffixes: {len(om_suffixes)} Feature columns: {len(feature_col_names)}")
            print(f"Suffixes {om_suffixes}")
        # combine weather data and target column (by convention)
        df_hist = pd.merge(
            left=dataframe[feature_col_names], right=target_col, left_index=True, right_index=True, how='left'
        )
        df_forecast = dataframe_f[feature_col_names]
    else:
        raise NotImplementedError(f"target={target} is not yet supported")

    # limit dataframe to the required max size
    df_hist = df_hist.tail(len(df_forecast)*n_horizons)
    if verbose:
        print(f"Limiting df_hist to {n_horizons} horizons. "
              f"From {len(dataframe)} entries do {len(df_hist)} ({len(df_hist)/len(dataframe)*100:.1f} %)")

    # check again the dataframe validity
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

def OLD__mask_outliers_and_unphysical_values(df_hist: pd.DataFrame, df_forecast: pd.DataFrame, target:str, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mask unphysical values based on predefined physical limits and detect outliers
    using the IQR method. Outlier indices are replaced with NaNs for columns matching
    known weather features.

    Parameters
    ----------
    df_hist : pd.DataFrame
        Historical (training) dataset containing time-series weather features.
    df_forecast : pd.DataFrame
        Forecast (test) dataset containing time-series weather features.
    verbose : bool, optional
        If True, prints warnings when outliers are detected. Defaults to False.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (df_hist_clean, df_forecast_clean), each with outliers
        and unphysical values replaced by NaN.
    """



    # Iterate over historical DataFrame columns
    for col in df_hist.columns:

        # 1) Skip non-numeric columns entirely (no outlier masking for them)
        if not pd.api.types.is_numeric_dtype(df_hist[col]):
            if verbose:
                print(f"Skipping column '{col}' (non-numeric).")
            continue

        # 2) (Optional) check if we also want to skip columns that don't exist in df_forecast
        #    to avoid KeyErrors later
        if col not in df_forecast.columns:
            if verbose:
                print(f"Skipping column '{col}' (not found in df_forecast).")
            continue

        # 3) Attempt to match a known weather feature and apply physical constraints
        matched_feature = False
        for feature, (phys_lower, phys_upper) in OpenMeteo.phys_limits.items():
            if feature in str(col):
                matched_feature = True
                # If historical data is entirely NaN, skip
                hist_series_valid = df_hist[col].dropna()
                if hist_series_valid.empty:
                    if verbose:
                        print(f"Skipping column '{col}' (all NaN in df_hist).")
                    # We do a 'continue' to skip outlier detection for this column
                    # in the outer loop
                    continue

                # 3a) Apply physical constraints
                df_hist[col] = df_hist[col].where(
                    (df_hist[col] >= phys_lower) & (df_hist[col] <= phys_upper),
                    np.nan
                )
                df_forecast[col] = df_forecast[col].where(
                    (df_forecast[col] >= phys_lower) & (df_forecast[col] <= phys_upper),
                    np.nan
                )
                if verbose and df_hist[col].isnull().sum() > 0:
                    print(f"Found {df_hist[col].isnull().sum()} NaNs in df_hist[{col}]")
                if verbose and df_forecast[col].isnull().sum() > 0:
                    print(f"Found {df_forecast[col].isnull().sum()} NaNs in df_forecast[{col}].")
                # No need to check other features once matched
                break

        # If you ONLY want outlier masking for columns that match a known feature, do:
        if not matched_feature:
            if verbose:
                print(f"Skipping outlier detection for '{col}' (no matched feature).")
            continue

        # 4) Compute outlier bounds from the (cleaned) historical data
        # lower_bound, upper_bound = compute_outlier_bounds(df_hist[col].dropna())

        # # 5) Mask outliers in both dataframes
        # hist_outliers = df_hist[col][ (df_hist[col] < lower_bound) | (df_hist[col] > upper_bound) ]
        # # If col not in df_forecast.columns, we already continued above
        # forecast_outliers = df_forecast[col][ (df_forecast[col] < lower_bound) | (df_forecast[col] > upper_bound) ]
        #
        # # df_hist.loc[hist_outliers, col] = np.nan
        # # df_forecast.loc[forecast_outliers, col] = np.nan
        #
        # # 6) Verbose logging
        # if verbose:
        #     if hist_outliers.sum() > 0:
        #         n_hist_outliers = hist_outliers.sum()
        #         print(f"WARNING [HIST] {n_hist_outliers} outliers in '{col}'.")
        #     if forecast_outliers.sum() > 0:
        #         n_forecast_outliers = forecast_outliers.sum()
        #         print(f"WARNING [FORECAST] {n_forecast_outliers} outliers in '{col}'.")

    return df_hist, df_forecast


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