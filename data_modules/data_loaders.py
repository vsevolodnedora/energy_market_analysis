import copy, re, pandas as pd
import numpy as np

from data_collection_modules.collect_data_openmeteo import OpenMeteo
from data_collection_modules.locations import (
    locations, offshore_windfarms, onshore_windfarms, solarfarms, de_regions
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

def extract_from_database(target:str, datapath:str, verbose:bool, region:str, n_horizons:int, horizon:int) \
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
    if horizon < len(df_om_f):
        if verbose: print("Forecast dataframe has {} rows while {} rows are requested. Trimming...")
        df_om_f = df_om_f.iloc[:horizon]
        assert len(df_om_f) == horizon
        assert (df_om_f.index[-1].hour == 23 and
                df_om_f.index[-1].minute == 0 and
                df_om_f.index[-1].second == 0)
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

    expected_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq='h')
    if not df_hist.index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")

    return df_hist, df_forecast

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

def clean_and_impute(df_hist, df_forecast, target, verbose:bool)->tuple[pd.DataFrame, pd.DataFrame]:
    df_hist, df_forecast = mask_outliers_and_unphysical_values(df_hist, df_forecast, verbose)
    df_hist = validate_dataframe(df_hist, 'df_hist', verbose=verbose)
    df_forecast = validate_dataframe(df_forecast, 'df_forecast', verbose=verbose)
    expected_range = pd.date_range(start=df_hist.index.min(), end=df_hist.index.max(), freq='h')
    if not df_hist.index.equals(expected_range):
        raise ValueError("full_index must be continuous with hourly frequency.")
    return df_hist, df_forecast