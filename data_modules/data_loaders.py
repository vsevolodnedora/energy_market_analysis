import copy, re, pandas as pd
import numpy as np

from data_collection_modules.collect_data_openmeteo import OpenMeteo
from data_collection_modules.locations import (
    locations, offshore_windfarms, onshore_windfarms, solarfarms, de_regions
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

def extract_from_database(target:str, datapath:str, verbose:bool, region:str, n_horizons:int) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    df_smard = pd.read_parquet(datapath + 'smard/' + 'history.parquet')
    df_om = pd.read_parquet(datapath + 'openmeteo/' + 'history.parquet')
    df_om_f = pd.read_parquet(datapath + 'openmeteo/' + 'forecast.parquet')
    df_es = pd.read_parquet(datapath + 'epexspot/' + 'history.parquet')
    df_entsoe = pd.read_parquet(datapath + 'entsoe/' + 'history.parquet')
    # -------------------
    if verbose:
        print(f"SMARD data shapes hist={df_smard.shape} (days={len(df_smard)/24}) start={df_smard.index[0]} end={df_smard.index[-1]}")
        print(f"ENTSOE data shapes hist={df_entsoe.shape} (days={len(df_entsoe)/24}) start={df_entsoe.index[0]} end={df_entsoe.index[-1]}")
        print(f"Openmeteo data shapes hist={df_om.shape} (days={len(df_om)/24}) start={df_om.index[0]} end={df_om.index[-1]}")
        print(f"Openmeteo data shapes forecast={df_om_f.shape} (days={len(df_om_f)/24}) start={df_om_f.index[0]} end={df_om_f.index[-1]}")
        print(f"EPEXSPOT data shapes hist={df_es.shape} (days={len(df_es)/24}) start={df_es.index[0]} end={df_es.index[-1]}")
    # -----------------
    if target in ['wind_offshore_tenn','wind_offshore_50hz']:
        if target == 'wind_offshore_tenn' and region != 'DE_TENNET':
            raise IOError(f"The region must be 'DE_TENNE' for target={target}")
        if target == 'wind_offshore_50hz' and region != 'DE_50HZ':
            raise IOError(f"The region must be 'DE_50HZ' for target={target}")

        if verbose: print(f"Target={target} Nans={df_entsoe[target].isna().sum().sum()}")

        # get openmeteo data for locations within this region (for windmills associated with this TSO)
        region_:dict = [reg for reg in de_regions if reg['name']==region][0]
        reg_suffix = region_['suffix']
        target = df_entsoe['wind_offshore' + reg_suffix]
        om_suffixes = [wind_farm['suffix'] for wind_farm in offshore_windfarms if wind_farm['TSO'] == region_['TSO']]
        columns_to_select = df_om.columns[df_om.columns.str.endswith(tuple(om_suffixes))]

        # combine weather data and target column (convention)
        df_hist = pd.merge(left=df_om[columns_to_select],right=target, left_index=True, right_index=True, how='left')
        df_forecast = df_om_f[columns_to_select]
        df_hist = df_hist.tail(len(df_forecast)*n_horizons) # crop the dataset if needed

        if not len(df_hist.columns) == len(df_forecast.columns)+1:
            raise ValueError(f'The DataFrames have different columns. '
                             f'hist={df_hist.shape} forecast={df_forecast.shape}')

        if len(df_hist) <= 1 or len(df_forecast) <= 1:
            raise ValueError(f'The DataFrames must have >1 rows '
                             f'hist={df_hist.shape} forecast={df_forecast.shape}')
    else:
        raise NotImplementedError(f"Target {target} and region {region} are not implemented.")

    # ------------------
    # if not validate_dataframe(df_hist):
    #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    # if not validate_dataframe(df_forecast):
    #     raise ValueError(f"History dataframe for target = {target} and region = {region} failed to validate.")
    if not df_forecast.index[0] == df_hist.index[-1]+pd.Timedelta(hours=1):
        raise ValueError(f"Forecast dataframe must have index[0] = historic index[-1] + 1 hour")

    return df_hist, df_forecast

def handle_nans_with_interpolation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Checks each column of the DataFrame for NaNs. If a column has more than 3 consecutive NaNs,
    it raises a ValueError. Otherwise, fills the NaNs using bi-directional interpolation.
    """

    df_copy = df.copy()

    def check_consecutive_nans(series: pd.Series):
        # Identify consecutive NaNs by grouping non-NaN segments and counting consecutive NaNs
        consecutive_nans = (series.isna().astype(int)
                            .groupby((~series.isna()).cumsum())
                            .cumsum())
        if consecutive_nans.max() > 3:
            raise ValueError(f"Column '{series.name}' in {name} contains more than 3 consecutive NaNs.")

    # Check all columns for consecutive NaNs first
    for col in df_copy.columns:
        check_consecutive_nans(df_copy[col])

    # Interpolate all columns at once after confirming they're valid
    df_copy = df_copy.interpolate(method='linear', limit_direction='both', axis=0)

    return df_copy

def fix_broken_periodicity_with_interpolation(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Fixes broken hourly periodicity by adding missing timestamps if fewer than 3 consecutive are missing.
    Raises an error if more than 3 consecutive timestamps are missing.
    Missing values are filled using time-based interpolation.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"The DataFrame {name} must have a datetime index.")

    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    missing_timestamps = expected_index.difference(df.index)

    if missing_timestamps.empty:
        print(f"The DataFrame {name} is already hourly with no missing segments.")
        return df

    # Convert to a Series to check consecutive missing timestamps
    missing_series = pd.Series(missing_timestamps)
    groups = (missing_series.diff() != pd.Timedelta(hours=1)).cumsum()

    # Check if any group has more than 3 missing points
    group_counts = groups.value_counts()
    if (group_counts > 3).any():
        bad_group = group_counts[group_counts > 3].index[0]
        raise ValueError(f"More than 3 consecutive missing timestamps detected: "
                         f"{missing_series[groups == bad_group].values} in {name}")

    # Reindex and interpolate
    fixed_df = df.reindex(expected_index)
    fixed_df = fixed_df.interpolate(method='time')

    print(f"Added and interpolated {len(missing_timestamps)} missing timestamps in {name}.")

    return fixed_df

def validate_dataframe(df: pd.DataFrame, name: str = '', verbose:bool=False) -> pd.DataFrame:
    """Check for NaNs, missing values, and periodicity in a time-series DataFrame."""

    # Check for NaNs
    if df.isnull().any().any():
        if verbose: print(f"ERROR! {name} DataFrame contains NaN values.")
        df = handle_nans_with_interpolation(df, name)

    # Check if index is sorted in ascending order
    if not df.index.is_monotonic_increasing:
        if verbose: print(f"ERROR! {name} The index is not in ascending order.")
        raise ValueError("Data is not in ascending order.")

    # Check for hourly frequency with no missing segments
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    if not full_range.equals(df.index):
        if verbose: print(f"ERROR! {name} The data is not hourly or has missing segments.")
        df = fix_broken_periodicity_with_interpolation(df, name)

    return df

def mask_outliers_and_unphysical_values(df_hist: pd.DataFrame, df_forecast: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    def compute_outlier_bounds(series: pd.Series) -> tuple[float, float]:
        """
        Returns the IQR-based lower and upper outlier bounds for a given series.
        Outliers are defined as values that fall outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        return lower_bound, upper_bound

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
        lower_bound, upper_bound = compute_outlier_bounds(df_hist[col].dropna())

        # 5) Mask outliers in both dataframes
        hist_outliers = (df_hist[col] < lower_bound) | (df_hist[col] > upper_bound)
        # If col not in df_forecast.columns, we already continued above
        forecast_outliers = (df_forecast[col] < lower_bound) | (df_forecast[col] > upper_bound)

        # df_hist.loc[hist_outliers, col] = np.nan
        # df_forecast.loc[forecast_outliers, col] = np.nan

        # 6) Verbose logging
        if verbose:
            if hist_outliers.any():
                n_hist_outliers = hist_outliers.sum()
                print(f"WARNING [HIST] {n_hist_outliers} outliers in '{col}'.")
            if forecast_outliers.any():
                n_forecast_outliers = forecast_outliers.sum()
                print(f"WARNING [FORECAST] {n_forecast_outliers} outliers in '{col}'.")

    return df_hist, df_forecast

def clean_and_impute(df_hist, df_forecast, target, verbose:bool)->tuple[pd.DataFrame, pd.DataFrame]:
    df_hist, df_forecast = mask_outliers_and_unphysical_values(df_hist, df_forecast, verbose)
    df_hist = validate_dataframe(df_hist, 'df_hist', verbose=verbose)
    df_forecast = validate_dataframe(df_forecast, 'df_forecast', verbose=verbose)
    return df_hist, df_forecast