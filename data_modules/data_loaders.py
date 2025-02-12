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

from logger import get_logger
logger = get_logger(__name__)

# TODO: refacto this into a proper ETL pipeline

# energy_mix_config = {
#     'targets' : ["hard_coal", "lignite", "coal_derived_gas", "oil", "other_fossil", "gas", "renewables"],
#     "aggregations": {"renewables": [
#         "biomass","waste","geothermal","pumped_storage","run_of_river","water_reservoir","other_renewables"
#     ]}
# }

def compute_residual_load(df:pd.DataFrame, suffix:str):
    load = copy.deepcopy(df['load{}'.format(suffix)])
    if f'wind_offshore{suffix}' in df.columns:
        load -= df['wind_offshore{}'.format(suffix)]
    if f'wind_onshore{suffix}' in df.columns:
        load -= df['wind_onshore{}'.format(suffix)]
    if f'solar{suffix}' in df.columns:
        load -= df['solar{}'.format(suffix)]
    return pd.Series(load.values, index=df.index, name=f'residual_load{suffix}')

# TODO: USE SQL HERE!!!
def extract_from_database(main_pars:dict, db_path:str, outdir:str, n_horizons:int, horizon:int, verbose:bool)\
        -> tuple[pd.DataFrame, pd.DataFrame]:

    tso_name = main_pars['region']
    targets = main_pars['targets']
    target_label = main_pars['label']

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
        logger.info("---------- LOADING DATABASE DATA ----------")
        logger.info(f"SMARD data shapes hist={df_smard.shape} (days={len(df_smard)/24}) start={df_smard.index[0]} end={df_smard.index[-1]}")
        logger.info(f"ENTSOE data shapes hist={df_entsoe.shape} (days={len(df_entsoe)/24}) start={df_entsoe.index[0]} end={df_entsoe.index[-1]}")
        logger.info(f"OM offshore data shapes hist={df_om_offshore.shape} (days={len(df_om_offshore)/24}) start={df_om_offshore.index[0]} end={df_om_offshore.index[-1]}")
        logger.info(f"OM offshore data shapes forecast={df_om_offshore_f.shape} (days={len(df_om_offshore_f)/24}) start={df_om_offshore_f.index[0]} end={df_om_offshore_f.index[-1]}")
        logger.info(f"OM onshore data shapes hist={df_om_onshore.shape} (days={len(df_om_onshore)/24}) start={df_om_onshore.index[0]} end={df_om_onshore.index[-1]}")
        logger.info(f"OM onshore data shapes forecast={df_om_onshore_f.shape} (days={len(df_om_onshore_f)/24}) start={df_om_onshore_f.index[0]} end={df_om_onshore_f.index[-1]}")
        logger.info(f"OM solar data shapes hist={df_om_solar.shape} (days={len(df_om_solar)/24}) start={df_om_solar.index[0]} end={df_om_solar.index[-1]}")
        logger.info(f"OM solar data shapes forecast={df_om_solar_f.shape} (days={len(df_om_solar_f)/24}) start={df_om_solar_f.index[0]} end={df_om_solar_f.index[-1]}")
        logger.info(f"EPEXSPOT data shapes hist={df_es.shape} (days={len(df_es)/24}) start={df_es.index[0]} end={df_es.index[-1]}")
        logger.info("-------------------------------------------")

    if len(df_om_offshore_f) != len(df_om_offshore_f) or len(df_om_solar_f) != len(df_om_solar_f):
        raise IOError("The forecast DataFrames have different number of columns.")

    # check that horizon is correct and that start/end hours are correct
    if horizon < len(df_om_offshore_f):
        if verbose: logger.info(
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

    target_label_notso = ''
    tso_dict = {}

    for de_reg in de_regions:
        if target_label.__contains__(de_reg['suffix']) and tso_name != de_reg['name']:
            raise IOError(f"The region must be {de_reg} for target_label={target_label}")
        if target_label.endswith(de_reg['suffix']):
            target_label_notso = target_label.replace(de_reg['suffix'], '')
            tso_dict = de_reg
            break
    if target_label_notso == '':
        raise ValueError(f"target_label={target_label} does not contain {[de_reg['suffix'] for de_reg in de_regions]}")


    suffix = tso_dict['suffix']

    ''' build df_hist and df_forecast '''

    # single target processing
    if len(targets) == 1:
        target = targets[0]
        # if target in df_entsoe.columns.tolist():
        #     raise KeyError(f"Target column {target} name already exists. This should not happen for aggregated columns.")
        if 'other_renewables_agg' in target:
            keys_to_agg = main_pars['aggregations'][target]
            keys_to_agg_ = [key for key in keys_to_agg if key in df_entsoe.columns.tolist()]
            if len(keys_to_agg_) != len(keys_to_agg):
                logger.warning(f"Not all keys for aggregating {target} are found in entsoe dataframe. "
                               f"{len(keys_to_agg_)} out of {len(keys_to_agg)} will be used ")
            # aggregate for the required column
            target_cols = pd.DataFrame(df_entsoe[keys_to_agg_].sum(axis=1), columns=[target])
        else:
            target_cols = df_entsoe[target]

        # select feature weather frame
        if 'wind_offshore' in target: dataframe, dataframe_f = df_om_offshore, df_om_offshore_f
        elif 'wind_onshore' in target: dataframe, dataframe_f = df_om_onshore, df_om_onshore_f
        elif 'solar' in target: dataframe, dataframe_f = df_om_solar, df_om_solar_f
        elif 'load' in target: dataframe, dataframe_f = df_om_cities, df_om_cities_f
        elif 'other_renewables_agg' in target: dataframe, dataframe_f = df_om_cities, df_om_cities_f
        else: raise NotImplementedError(f"No dataframe selection for target={target} tso_name={tso_name}")

        # select locations
        locations = []
        if 'wind_offshore' in target: locations = loc_offshore_windfarms
        elif 'wind_onshore' in target: locations = loc_onshore_windfarms
        elif 'solar' in target: locations = loc_solarfarms
        elif 'load' in target: locations = loc_cities
        elif 'other_renewables_agg' in target: locations = loc_cities
        else: raise NotImplementedError(f"Locations are not available for target={target} tso_name={tso_name}")
        # build df_hist and df_forecast
        om_suffixes = [loc['suffix'] for loc in locations if loc['TSO'] == tso_dict['TSO']]
        feature_col_names = dataframe.columns[dataframe.columns.str.endswith(tuple(om_suffixes))]

        if verbose:
            logger.info(f"TARGET LABEL: {target_label_notso} TSO: {tso_dict['TSO']} "
                  f"Locations: {len([loc for loc in locations if loc['TSO'] == tso_dict['TSO']])} "
                  f"OM suffixes: {len(om_suffixes)} Feature columns: {len(feature_col_names)}")
            logger.info(f"Suffixes {om_suffixes}")
        # combine weather data and target column (by convention)
        df_hist = pd.merge(
            left=dataframe[feature_col_names], right=target_cols, left_index=True, right_index=True, how='left'
        )
        df_forecast = dataframe_f[feature_col_names]

    # multitarget dataset
    else:
        if target_label_notso != 'energy_mix':
            raise NotImplementedError(f"Target label={target_label_notso} not implemented.")
        aggregations =  main_pars['aggregations'] if 'aggregations' in main_pars else {}
        # build target columns
        target_cols =  df_entsoe[[
            target_ for target_ in targets if not target_ in list(aggregations.keys()) and target_ in df_entsoe.columns
        ]]
        dropped_cols = []
        for col in target_cols:
            if len(df_entsoe[col].unique()) < len(df_entsoe[col])*.01:
                if verbose: logger.warning(
                    f"Dropping target column {col} as there are only { len(df_entsoe[col].unique()) } unique values. "
                    f"Sum={pd.Series(df_entsoe[col]).sum():.1e}."
                )
                target_cols = target_cols.drop(columns=[col])
                dropped_cols.append(col)
        dropped_data = {col: df_entsoe[col].sum()/target_cols.sum().sum() for col in dropped_cols}
        for col, val in dropped_data.items():
            if val > 0.1:
                logger.error(
                    f"Dropped column {col} has {val:.2f} fraction of power for TSO={tso_name} in all columns to forecast. "
                    f"Adding it back."
                )
                target_cols = pd.merge(target_cols, df_entsoe[col], left_index=True, right_index=True, how='left')

                # import matplotlib.pyplot as plt
                # plt.plot(target_cols.index, target_cols[col])
                # plt.show()

        # add aggregations if any
        for key in targets:
            if key in list(aggregations.keys()):
                keys_to_agg = [
                    col for col in aggregations[key]
                    if col in list(df_entsoe.keys())
                ]
                if len(keys_to_agg) != len(aggregations[key]):
                    logger.warning(f"Not all keys for aggregating {key} are found in entsoe dataframe. "
                          f"{len(keys_to_agg)} out of {len(aggregations[key])} will be used ")
                # aggregate for the required column
                df_entsoe[key] = df_entsoe[keys_to_agg].sum(axis=1)
                target_cols = pd.merge(
                    target_cols, df_entsoe[key], left_index=True, right_index=True, how='left'
                )

        # fixed weather dataframes and locations
        dataframe, dataframe_f = df_om_cities, df_om_cities_f
        locations = loc_cities
        # build df_hist and df_forecast
        om_suffixes = [loc['suffix'] for loc in locations if loc['TSO'] == tso_dict['TSO']]
        feature_col_names = dataframe.columns[dataframe.columns.str.endswith(tuple(om_suffixes))]
        feature_col_names = [
            col for col in feature_col_names if (
                    col.startswith('temperature') or
                    col.startswith('wind_speed') or
                    col.startswith('precipitation') or
                    col.startswith('cloud_cover') or
                    col.startswith('shortwave_radiation')
            )
        ]
        # build df_hist and df_forecast
        if verbose:
            logger.info(f"TARGET LABEL: {target_label_notso} TSO: {tso_dict['TSO']} "
                  f"Locations: {len([loc for loc in locations if loc['TSO'] == tso_dict['TSO']])} "
                  f"OM suffixes: {len(om_suffixes)} Feature columns: {len(feature_col_names)}")
            logger.info(f"Suffixes {om_suffixes}")
        # combine weather data and target column (by convention)
        df_hist = pd.merge(
            left=dataframe[feature_col_names], right=target_cols, left_index=True, right_index=True, how='left'
        )
        df_forecast = dataframe_f[feature_col_names]


    ''' add forecasted quantities as features (ALL TSOs) '''

    # add additional quantities
    if len(targets) == 1 and 'load' in targets[0]: add_exog = ["wind_offshore", "wind_onshore", "solar"]
    elif len(targets) == 1 and 'other_renewables_agg' in targets[0]: add_exog = ["wind_offshore", "wind_onshore", "solar", 'load', 'residual_load']
    elif len(targets) > 1 and 'energy_mix' in target_label: add_exog = ["wind_offshore", "wind_onshore", "solar", 'load', 'residual_load']
    else: add_exog = []

    if len(add_exog) > 0:
        if len(tso_dict.keys()) == 0: raise NotImplementedError(
            f"No TSO dict available for target_label={target_label} tso_name={tso_name}"
        )

        for exog in add_exog:
            for tso_dict in de_regions:
                # load historic data
                exog_tso_ = exog + tso_dict['suffix']
                suffix_ = tso_dict['suffix']

                logger.info(f"Adding exogenous feature {exog_tso_} to dataframe")

                if not exog_tso_ in df_entsoe.columns.tolist() and not exog == 'residual_load':
                    if verbose: logger.warning(f"Required exogenous feature {exog_tso_} is not in ENTSO-E dataset. Skipping.")
                    continue

                if exog == 'residual_load':
                    res_load = compute_residual_load(df_hist, suffix_)
                    df_hist = pd.merge(
                        left=df_hist, right=res_load, left_index=True, right_index=True, how='left'
                    )

                    df_forecast = pd.merge(
                        left=df_forecast,
                        right=compute_residual_load(df_forecast, suffix_),
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    continue


                entsoe_col = df_entsoe[exog_tso_]
                df_hist = pd.merge(left=df_hist, right=entsoe_col, left_index=True, right_index=True, how='left')

                # load forecast from current best forecast
                best_model = None
                fpath = outdir+exog_tso_+'/'+'best_model.txt'
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"Best model file not found {fpath}")
                with open(fpath, 'r') as f:
                    best_model = f.read().strip()

                if best_model.__contains__('ensemble'):
                    best_model = convert_ensemble_string(best_model)

                fpath = outdir+exog_tso_+'/' + best_model + '/' + 'forecast/' + 'forecast.csv'
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"Forecast file not found {fpath}")

                # load the latest forecast for exogenous variable and add it to df_forecast
                df_ = pd.read_csv(fpath, index_col=0)
                df_.index = pd.to_datetime(df_.index)
                df_.rename(columns={f'{exog_tso_}_fitted':f'{exog_tso_}'},inplace=True)

                # import matplotlib.pyplot as plt
                # plt.plot(df_.index, df_[f'{exog_tso}_fitted'])
                # plt.plot(df_forecast.index, df_forecast[df_forecast.columns.tolist()[0]])
                # plt.show()
                # exit(1)

                df_forecast = pd.merge(
                    left=df_forecast,
                    right=df_[f'{exog_tso_}'],
                    left_index=True,
                    right_index=True,
                    how='left'
                )


    # limit dataframe to the required max size
    df_hist = df_hist.tail(len(df_forecast)*n_horizons)
    if verbose:
        logger.info(f"Limiting df_hist to {n_horizons} horizons. "
              f"From {len(dataframe)} entries do {len(df_hist)} ({len(df_hist)/len(dataframe)*100:.1f} %)")

    # check again the dataframe validity
    if len(targets) == 1:
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
                    logger.info(
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

    targets = [str(col) for col in df_hist.columns if col not in df_forecast.columns]
    for target in targets:

        # ----- 2) Simple outlier (anomaly) detection on the target column -----
        # Train a z-score based detection on df_hist[target]
        # (If your target column is missing in any row, drop those temporarily to calculate stats)
        hist_nonan = df_hist[target].dropna()
        mean_val = hist_nonan.mean()
        std_val = hist_nonan.std()
        threshold = 3  # For example, 3 standard deviations

        if std_val == 0 or np.isnan(std_val):
            if verbose:
                logger.info(f"[Anomaly Detection] Standard deviation is zero or NaN for '{target}'. Skipping outlier masking.")
        else:
            # Identify outliers in df_hist
            outliers_hist_mask = (df_hist[target] - mean_val).abs() > threshold * std_val
            outliers_hist_count = outliers_hist_mask.sum()
            if verbose and outliers_hist_count > 0:
                logger.info(
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

def clean_and_impute(df_hist, df_forecast, verbose:bool)->tuple[pd.DataFrame, pd.DataFrame]:

    df_hist, df_forecast = mask_outliers_and_unphysical_values(df_hist, df_forecast, verbose)
    df_hist = validate_dataframe(df_hist, 'df_hist', log_func=logger.warning, verbose=verbose)
    df_forecast = validate_dataframe(df_forecast, 'df_forecast', log_func=logger.warning, verbose=verbose)
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
            logger.info("No NaNs found in the DataFrame.")
    check_for_nans_and_raise_error(df_hist)
    check_for_nans_and_raise_error(df_forecast)

    return df_hist, df_forecast