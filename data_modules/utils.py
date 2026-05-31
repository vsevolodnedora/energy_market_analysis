from typing import Callable

import numpy as np
import pandas as pd

from logger import get_logger
logger = get_logger(__name__)

def handle_nans_with_interpolation(df: pd.DataFrame, name: str, log_func:Callable[..., None]) -> pd.DataFrame:
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
            msg = f"Column '{series.name}' in {name} contains {consecutive_nans.max()} consecutive NaNs."
            log_func(msg)
            # logger.warning(f"Column '{series.name}' in {name} contains {consecutive_nans.max()} "
            #                  f" consecutive NaNs.")

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
        logger.info(f"The DataFrame {name} is already hourly with no missing segments.")
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

    logger.info(f"Added and interpolated {len(missing_timestamps)} missing timestamps in {name}.")

    return fixed_df

def validate_dataframe(df: pd.DataFrame, name: str, log_func:Callable[...,None], verbose:bool=False) -> pd.DataFrame:
    """Check for NaNs, missing values, and periodicity in a time-series DataFrame."""

    # Check for NaNs
    if df.isnull().any().any():
        if verbose: logger.error(f"{name} DataFrame contains NaN values.")
        df = handle_nans_with_interpolation(df, name, log_func)

    # Check if index is sorted in ascending order
    if not df.index.is_monotonic_increasing:
        if verbose: logger.error(f"{name} The index is not in ascending order.")
        raise ValueError("Data is not in ascending order.")

    # Check for hourly frequency with no missing segments
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    if not full_range.equals(df.index):
        if verbose: logger.error(f"{name} The data is not hourly or has missing segments.")
        df = fix_broken_periodicity_with_interpolation(df, name)

    return df

