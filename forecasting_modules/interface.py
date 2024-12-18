
import copy, re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import os.path; from datetime import datetime, timedelta

from data_collection_modules import OpenMeteo
from forecasting_modules import ForecastingTaskSingleTarget

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

def preprocess_openmeteo_for_offshore_wind(df:pd.DataFrame, location_suffix="_hsee")->pd.DataFrame:
    """
    Preprocesses weather data for forecasting offshore wind energy generation.
    Focuses on critical physical features and includes turbulence_intensity, wind_ramp, and wind_shear.

    df contains the following columns with suffixes for different locations:
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_speed_100m",
        "wind_direction_10m",
        "wind_direction_100m",
        # additional
        "precipitation",
        "wind_gusts_10m",
        "cloud_cover",
        "shortwave_radiation"

    """

    # 1. Filter for the offshore wind farm location only
    cols_to_keep = [c for c in df.columns if c.endswith(location_suffix)]
    df = df[cols_to_keep].copy()

    # 2. Define key variable columns
    wind_speed_10m_col = f"wind_speed_10m{location_suffix}"
    wind_speed_100m_col = f"wind_speed_100m{location_suffix}"
    wind_dir_100m_col = f"wind_direction_100m{location_suffix}"
    temp_col = f"temperature_2m{location_suffix}"
    press_col = f"surface_pressure{location_suffix}"

    # 3. Compute Air Density (ρ)
    if temp_col in df.columns and press_col in df.columns:
        temp_K = df[temp_col] + 273.15
        R_specific = 287.05  # J/(kg·K) for dry air
        # convert pressure from hPa to Pa
        df["air_density"] = np.array( (df[press_col] * 100.) / (R_specific * temp_K) )

    # 4. Compute Wind Power Density (if wind_speed_100m and air_density are available)
    if wind_speed_100m_col in df.columns and "air_density" in df.columns:
        df["wind_power_density"] = np.array( 0.5 * df["air_density"] * (df[wind_speed_100m_col] ** 3) )

    # 5. Encode Wind Direction (Cyclic)
    if wind_dir_100m_col in df.columns:
        df["wind_dir_sin"] = np.sin(np.deg2rad(df[wind_dir_100m_col]))
        df["wind_dir_cos"] = np.cos(np.deg2rad(df[wind_dir_100m_col]))
        df.drop(columns=[wind_dir_100m_col], inplace=True)

    # 6. Wind Shear (Requires both 10m and 100m wind speeds)
    if wind_speed_10m_col in df.columns and wind_speed_100m_col in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df["wind_shear"] = np.log(df[wind_speed_100m_col] / df[wind_speed_10m_col]) / np.log(100/10)
        # Replace infinities or NaNs if they occur
        df["wind_shear"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # 7. Turbulence Intensity (using a short rolling window on 100m wind speed)
    if wind_speed_100m_col in df.columns:
        rolling_std = df[wind_speed_100m_col].rolling(window=3, min_periods=1).std()
        rolling_mean = df[wind_speed_100m_col].rolling(window=3, min_periods=1).mean()
        df["turbulence_intensity"] = np.array( rolling_std / rolling_mean )

    # 8. Wind Ramp (difference in 100m wind speed over 1 timestep)
    if wind_speed_100m_col in df.columns:
        df["wind_ramp"] = df[wind_speed_100m_col].diff(1)

    # 9. Lag Features for Wind Speed at 100m
    if wind_speed_100m_col in df.columns:
        for lag in [1, 6, 12, 24]:
            df[f"wind_speed_lag_{lag}"] = df[wind_speed_100m_col].shift(lag)

    # 11. Drop Irrelevant Columns
    # Decide which columns to drop. For model simplicity, consider dropping raw weather inputs
    # that have been transformed into more physical parameters.
    # However, keep wind speeds if you think they add value.
    # For now, we keep the wind speeds since other derived features depend on them.
    drop_vars = [
        temp_col, press_col, "air_density",
        f"precipitation{location_suffix}",
        f"cloud_cover{location_suffix}",
        f"shortwave_radiation{location_suffix}",
        f"relative_humidity_2m{location_suffix}",
        f"wind_direction_10m{location_suffix}",
        f"wind_gusts_10m{location_suffix}"
    ]
    drop_cols = [c for c in drop_vars if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Handle missing values introduced by lagging and other computations
    # df.dropna(inplace=True)

    return df

def get_raw_weather_features(df_history:pd.DataFrame):
    # list of openmeteo feature names (excluding suffix added for different locations)
    # patterns = [
    #     'cloud_cover', 'precipitation', 'relative_humidity_2m', 'shortwave_radiation',
    #     'surface_pressure', 'temperature_2m', 'wind_direction_10m', 'wind_gusts_10m', 'wind_speed_10m'
    # ]
    patterns = OpenMeteo.variables_standard

    # Use the filter method to get columns that match the regex pattern
    weather_columns = df_history.filter(
        regex='|'.join([f"{pattern}_(fran|hsee|mun|solw|stut)" for pattern in patterns])
    ).columns.tolist()
    # Display or use the filtered columns as needed
    print(f"Weather features found {len(weather_columns)}")
    return weather_columns

def input_preprocessing_pipeline_wind_offshore(datapath:str, verbose:bool, features:list, target:str) \
        -> tuple[pd.DataFrame, pd.DataFrame]:

    target = 'wind_offshore'
    # df = pd.read_parquet(data_dir + 'latest.parquet')
    df_smard = pd.read_parquet(datapath + 'smard/' + 'history.parquet') # energy generation and consumption
    df_om = pd.read_parquet(datapath + 'openmeteo/' + 'history.parquet') # raw weather quantities for different locations
    df_om_f = pd.read_parquet(datapath + 'openmeteo/' + 'forecast.parquet') # weather forecasts for all locations
    df_es = pd.read_parquet(datapath + 'epexspot/' + 'history.parquet') # energy prices

    if verbose:
        print(f"SMARD data shapes hist={df_smard.shape} start={df_smard.index[0]} end={df_smard.index[-1]}")
        print(f"Openmeteo data shapes hist={df_om.shape} start={df_om.index[0]} end={df_om.index[-1]}")
        print(f"Openmeteo data shapes forecast={df_om_f.shape} start={df_om_f.index[0]} end={df_om_f.index[-1]}")
        print(f"EPEXSPOT data shapes hist={df_es.shape} start={df_es.index[0]} end={df_es.index[-1]}")

        print(f"Target={target} Nans={df_smard[target].isna().sum().sum()}")


    # set how to split the dataset
    cutoff = df_om_f.index[0]
    if cutoff == cutoff.normalize():
        if verbose:
            print(f"The cutoff timestamp corresponds to the beginning of the day {cutoff.normalize()}")
    else:
        raise ValueError(f"The cutoff timestamp does not correspond to the beginning of the day {cutoff}")
    if verbose:
        print(f"Dataset is split into ({len(df_om[:cutoff])}) before and "
          f"({len(df_om_f[cutoff:])}) ({int(len(df_om_f[cutoff:])/24)} days) after {cutoff}.")

    # combine historic and forecasted data
    df_om = df_om.combine_first(df_om_f)
    if df_om.isna().any().any():
        raise ValueError("ERROR! Nans in the dataframe")

    # Extract data for the offshore wind farm location and create new features
    df_om_prep = preprocess_openmeteo_for_offshore_wind(df=df_om, location_suffix="_hsee")
    df_om_prep.dropna(inplace=True) # drop nans formed when lagged features were added

    horizon = 7 * 24

    # merger with SMRD target column
    df_om_prep.dropna(inplace=True, how='any')
    df_hist = pd.merge(
        df_om_prep[:cutoff-timedelta(hours=1)],
        df_smard[:cutoff-timedelta(hours=1)][target],
        left_index=True, right_index=True, how="inner"
    )
    df_forecast = df_om_prep[cutoff : cutoff+timedelta(hours=horizon - 1)]
    df_hist = validate_dataframe(df_hist, 'df_hist', verbose=verbose)
    df_forecast = validate_dataframe(df_forecast, 'df_forecast', verbose=verbose)
    df_hist = df_hist[df_hist.index[-1]-pd.Timedelta(hours = 100 * horizon - 1):]

    if verbose:
        print(f"Features {len(df_hist.columns)-1} hist.shape={df_hist.shape} ({int(len(df_hist)/horizon)}) forecast.shape={df_forecast.shape}")
        print(f"Hist: from {df_hist.index[0]} to {df_hist.index[-1]} ({len(df_hist)/horizon})")
        print(f"Fore: from {df_forecast.index[0]} to {df_forecast.index[-1]} ({len(df_forecast)/horizon})")
        print(f"Given dataset has features: {df_hist.columns.tolist()}")

    # restrict to required features
    features_to_restrict : list = []
    for feature in features:
        # preprocess some features
        if feature  == 'weather':
            # TODO IMPROVE (USE OPENMETEO CLASS HERE)
            weather_features:list = get_raw_weather_features(df_hist)
            features_to_restrict += weather_features
    if not features:
        if verbose: print(f"No features selected for {target}. Using all features: \n{df_forecast.columns.tolist()}")
        features_to_restrict = df_forecast.columns.tolist()

    # remove unnecessary features from the dataset
    if verbose:
        print(f"Restricting dataframe from {len(df_hist.columns)} features to {len(features_to_restrict)}")

    df_hist = df_hist[features_to_restrict + [target]]
    df_forecast = df_forecast[features_to_restrict]
    # if not validate_dataframe(df_forecast,'df_forecast', verbose=verbose):
    #     raise ValueError("Nans in the df_forecast after restricting. Cannot continue.")


    return df_hist, df_forecast

def main():
    verbose = True
    outdir = './output/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    tasks = [
        {
            "target": "wind_offshore",
            "label": "Wind off-shore [MW]",
            "input_preprocessing_pipeline": {
                'func':'input_preprocessing_pipeline_wind_offshore',
                'kwargs':{'datapath':'../database/', 'features':[]}
            },
            "task_fine_tuning":[
                # {'model':'Prophet',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'feature_pca_pars' : None,
                #      'limit_pca_to_features':None,#'weather',
                #      'fourier_features' : {'period':24, 'order':3},
                #      'add_cyclical_time_features':True,
                #      'lags_target':None,'log_target':True,
                #      'copy_input':True
                #  },
                # 'finetuning_pars':{'n_trials':120,'optim_metric':'rmse','cv_folds':3}},

                # {'model':'XGBoost',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'limit_pca_to_features':None,#'weather',
                #      'feature_pca_pars':None,#{'n_components':0.95},
                #      'fourier_features': {},
                #      'add_cyclical_time_features':True,
                #      'lags_target':24,'log_target':True,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':5,'optim_metric':'rmse','cv_folds':3}},
                #
                # {'model':'ElasticNet',
                #  'dataset_pars':{
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'limit_pca_to_features':None,#'weather',
                #      'feature_pca_pars':None,#{'n_components':0.95},
                #      'fourier_features': {},
                #      'add_cyclical_time_features':True,
                #      'lags_target':24,'log_target':True,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':5,'optim_metric':'rmse','cv_folds':3}},
                #
                # {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
                #  'dataset_pars': {
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'limit_pca_to_features':None,#'weather',
                #      'feature_pca_pars':None,#{'n_components':0.95},
                #      'add_cyclical_time_features':False,
                #      'fourier_features': {},
                #      'ensemble_features': 'cyclic_time',
                #      'log_target':True,
                #      'lags_target': None,
                #      'copy_input':True
                #
                #  },
                #  'finetuning_pars':{'n_trials':5,
                #                     'optim_metric':'rmse',
                #                     'cv_folds':3,
                #                     'cv_folds_base':35,
                #                     'use_base_models_pred_intervals':False}}
            ],
            "task_training":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                # {'model':'XGBoost', 'pars':{'cv_folds':5}},
                # {'model':'ElasticNet', 'pars':{'cv_folds':5}},
                # {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':5}}
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','pars':{'cv_folds':5}}
            ],
            "task_forecasting":[
                # {'model':'Prophet'},
                {'model':'XGBoost', 'past_folds':5},
                {'model':'ElasticNet', 'past_folds':5},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','past_folds':5},
                {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','past_folds':5}
            ],
            "task_plot":[
                # {'model':'Prophet', 'n':2,
                #  'name':"Prophet",'lw':0.7,'color':"red",'ci_alpha':0.0},
                {'model':'XGBoost','n':2,
                 'name':'XGBoost','lw':0.7,'color':"green",'ci_alpha':0.0,
                 'train_forecast':'train'},
                {'model':'ElasticNet','n':2,
                 'name':'ElasticNet','lw':0.7,'color':"blue",'ci_alpha':0.0,
                 'train_forecast':'train'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','n':2,
                 'name':'Ensemble','lw':1.0,'color':"purple",'ci_alpha':0.2,
                 'train_forecast':'train'},
                {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','n':2,
                 'name':'Ensemble','lw':1.0,'color':"magenta",'ci_alpha':0.2,
                 'train_forecast':'train'},
            ],
            "task_summarize":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                {'model':'XGBoost', 'summary_metric':'rmse'},
                {'model':'ElasticNet', 'summary_metric':'rmse'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
                {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
            ]
        }
    ]

    for task in tasks:
        target = task['target']
        input_preprocessing_pipeline_ = task['input_preprocessing_pipeline']
        if input_preprocessing_pipeline_['func'] == 'input_preprocessing_pipeline_wind_offshore':
            df_hist, df_forecast = input_preprocessing_pipeline_wind_offshore(
                **(input_preprocessing_pipeline_['kwargs'] | {'target':target, 'verbose':verbose})
            )
        else:
            raise NotImplementedError(
                f"Input preprocessing pipeline {input_preprocessing_pipeline_['func']} not implemented."
            )

        processor = ForecastingTaskSingleTarget(
            df_history=df_hist,df_forecast=df_forecast,task=task,outdir=outdir,verbose=verbose
        )

        # process task to fine-tune the forecasting model
        if task['task_fine_tuning']:
            for ft_task in task['task_fine_tuning']:
                if ft_task['model'].__contains__('ensemble'):
                    processor.process_finetuning_task_ensemble(ft_task)
                else:
                    processor.process_finetuning_task_base(ft_task)

        # train forecasting model on full dataset assuming hyperparameters are in finetuning dir
        if task['task_training']:
            for t_task in task['task_training']:
                if t_task['model'].__contains__('ensemble'):
                    processor.process_training_task_ensemble(t_task)
                else:
                    processor.process_training_task_base(t_task)

        # forecast with trained model
        if task['task_forecasting']:
            for t_task in task['task_forecasting']:
                if t_task['model'].__contains__('ensemble'):
                    processor.process_forecasting_task_ensemble(t_task)
                else:
                    processor.process_forecasting_task_base(t_task)

        if task['task_plot']:
            processor.process_task_plot_predict_forecast(task)

        if task['task_summarize']:
            processor.process_task_determine_the_best_model(task, outdir=outdir+target+'/')


if __name__ == '__main__':
    main()