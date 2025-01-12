import copy, re, pandas as pd, numpy as np, matplotlib.pyplot as plt
import os.path; from datetime import datetime, timedelta

from data_collection_modules import OpenMeteo
from forecasting_modules.tasks import ForecastingTaskSingleTarget
from data_collection_modules.german_locations import (
    loc_offshore_windfarms,
    loc_onshore_windfarms,
    loc_solarfarms,
    loc_cities,
    de_regions
)
from data_modules.data_loaders import (
    extract_from_database,
    clean_and_impute
)


def OLD__preprocess_openmeteo_for_offshore_wind(df:pd.DataFrame, location_suffix="_hsee")->pd.DataFrame:
    """
    Preprocesses weather data for forecasting offshore wind energy generation.
    Focuses on critical physical features and includes turbulence_intensity, wind_ramp, and wind_shear.

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

    return df

def OLD__get_raw_weather_features(df_history:pd.DataFrame):
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

def OLD__input_preprocessing_pipeline_wind_offshore(datapath:str, verbose:bool, features:list, target:str) \
        -> tuple[pd.DataFrame, pd.DataFrame]:

    target = 'wind_offshore'
    # df = pd.read_parquet(data_dir + 'latest.parquet')
    df_smard = pd.read_parquet(datapath + 'smard/' + 'history.parquet') # energy generation and consumption
    df_entsoe = pd.read_parquet(datapath + 'entsoe/' + 'history.parquet') # energy generation by TSO
    df_om = pd.read_parquet(datapath + 'openmeteo/' + 'history.parquet') # raw weather quantities for different locations
    df_om_f = pd.read_parquet(datapath + 'openmeteo/' + 'forecast.parquet') # weather forecasts for all locations
    df_es = pd.read_parquet(datapath + 'epexspot/' + 'history.parquet') # energy prices

    if verbose:
        print(f"SMARD data shapes hist={df_smard.shape} start={df_smard.index[0]} end={df_smard.index[-1]}")
        print(f"ENTSOE data shapes hist={df_entsoe.shape} start={df_entsoe.index[0]} end={df_entsoe.index[-1]}")
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
    df_om_prep = OLD__preprocess_openmeteo_for_offshore_wind(df=df_om, location_suffix="_woff_enbw")
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
            weather_features:list = OLD__get_raw_weather_features(df_hist)
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


def main_forecasting_pipeline(task_list:list, outdir:str, database:str, verbose:bool):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for task in task_list:
        target = task['target']
        region = task['region']

        # get features + target (historic) and features (forecast) from database
        df_hist, df_forecast = extract_from_database(
            target=target, db_path=database, verbose=verbose, tso_name=region, n_horizons=100, horizon=7*24
        )

        # clean data from nans and outliers
        df_hist, df_forecast = clean_and_impute(df_hist=df_hist,df_forecast=df_forecast,target=target,verbose=verbose)

        # initialize the processor for tasks
        processor = ForecastingTaskSingleTarget(
            df_history=df_hist,df_forecast=df_forecast,task=task,outdir=outdir,verbose=verbose
        )

        # process task to fine-tune the forecasting model. Note: ensemble tasks require base models to be processed first
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





def update_forecast_production(database:str, outdir:str, variable:str, verbose:bool):
    cv_folds_ft = 3
    cv_folds_eval = 5
    task_list = [
        {
            "target": "wind_offshore_tenn",
            "region": "DE_TENNET",
            "label": "Offshore Wind Power Generation (TenneT) [MW]",
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
                # 'finetuning_pars':{'n_trials':120,'optim_metric':'rmse','cv_folds':cv_folds_ft}},

                # {'model':'XGBoost',
                #  'dataset_pars':{
                #      'log_target':True,
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'copy_input':True,
                #      'locations':[loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                #      'add_cyclical_time_features':True,
                #      'feature_engineer':'WeatherWindPowerFE'
                #  },
                #  'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
                #
                # {'model':'ElasticNet',
                #  'dataset_pars':{
                #      'log_target':True,
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'copy_input':True,
                #      'locations':[loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                #      'add_cyclical_time_features':True,
                #      'feature_engineer':'WeatherWindPowerFE'
                #  },
                #  'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
                #
                # {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
                #  'dataset_pars': {
                #      'log_target':True,
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'add_cyclical_time_features':True,
                #      'locations':[loc for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                #      'feature_engineer': None,#'WeatherWindPowerFE',
                #      'lags_target': None,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':25,
                #                     'optim_metric':'rmse',
                #                     'cv_folds':cv_folds_ft,
                #                     'cv_folds_base':35, # at least cv_folds_eval + 1
                #                     'use_base_models_pred_intervals':False}}
            ],
            "task_training":[
                # {'model':'Prophet', 'pars':{'cv_folds':cv_folds_eval}},
                # {'model':'XGBoost', 'pars':{'cv_folds':cv_folds_eval}},
                # {'model':'ElasticNet', 'pars':{'cv_folds':cv_folds_eval}},
                # {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}}
            ],
            "task_forecasting":[
                # {'model':'Prophet'},
                {'model':'XGBoost', 'past_folds':cv_folds_eval},
                {'model':'ElasticNet', 'past_folds':cv_folds_eval},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','past_folds':cv_folds_eval},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','past_folds':cv_folds_eval}
            ],
            "task_plot":[
                # {'model':'Prophet', 'n':2,
                #  'name':"Prophet",'lw':0.7,'color':"red",'ci_alpha':0.0},
                # {'model':'XGBoost','n':2,
                #  'name':'XGBoost','lw':0.7,'color':"green",'ci_alpha':0.0,
                #  'train_forecast':'train'},
                # {'model':'ElasticNet','n':2,
                #  'name':'ElasticNet','lw':0.7,'color':"blue",'ci_alpha':0.0,
                #  'train_forecast':'train'},
                # {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','n':2,
                #  'name':'Ensemble','lw':1.0,'color':"purple",'ci_alpha':0.2,
                #  'train_forecast':'train'},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','n':2,
                #  'name':'Ensemble','lw':1.0,'color':"magenta",'ci_alpha':0.2,
                #  'train_forecast':'train'},
            ],
            "task_summarize":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                {'model':'XGBoost', 'summary_metric':'rmse'},
                {'model':'ElasticNet', 'summary_metric':'rmse'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
            ]
        }
    ]

    ''' -------------- OFFSHORE WIND POWER GENERATION (2 TSOs) ------------- '''

    if variable == "wind_offshore":
        avail_regions = ["DE_TENNET", "DE_50HZ"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"Offshore Wind Power Generation ({tso_reg['name']}) [MW]"
                    t['target'] = f"wind_offshore{tso_reg['suffix']}"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherWindPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)


    ''' -------------- ONSHORE WIND POWER GENERATION (4 TSOs) ------------- '''

    if variable == "wind_onshore":
        avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"Onshore Wind Power Generation ({tso_reg['name']}) [MW]"
                    t['target'] = f"wind_onshore{tso_reg['suffix']}"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherWindPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- SOLAR POWER GENERATION (4 TSOs) ------------- '''

    if variable == "solar":
        avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"Solar Power Generation ({tso_reg['name']}) [MW]"
                    t['target'] = f"solar{tso_reg['suffix']}"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)





    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Offshore Wind Power Generation (TenneT) [MW]"
    #     t['target'] = "wind_offshore_tenn"
    #     t['region'] = "DE_TENNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Offshore Wind Power Generation (50Hertz) [MW]"
    #     t['target'] = "wind_offshore_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='50Hertz']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)
    #
    #
    # ''' -------------- ONSHORE WIND GENERATION (4 TSOs) ------------- '''
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Onshore Wind Power Generation (50Hertz) [MW]"
    #     t['target'] = "wind_onshore_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='50Hertz']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Onshore Wind Power Generation (TransnetBW) [MW]"
    #     t['target'] = "wind_onshore_tran"
    #     t['region'] = "DE_TRANSNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='TransnetBW']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Onshore Wind Power Power Generation (Amprion) [MW]"
    #     t['target'] = "wind_onshore_ampr"
    #     t['region'] = "DE_AMPRION"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='Amprion']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Onshore Wind Power Generation (TenneT) [MW]"
    #     t['target'] = "wind_onshore_tenn"
    #     t['region'] = "DE_TENNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='TenneT']
    # main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- SOLAR GENERATION (4 TSOs) ------------- '''

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Solar Power Generation (50Hz) [MW]"
    #     t['target'] = "solar_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='50Hertz']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['label'] = "Solar Power Generation (TenneT) [MW]"
    #     t['target'] = "solar_tran"
    #     t['region'] = "DE_TRANSNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='TransnetBW']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_ampr"
    #     t['region'] = "DE_AMPRION"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='Amprion']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_tenn"
    #     t['region'] = "DE_TENNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='TenneT']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)


def main():

    cv_folds_ft = 3
    cv_folds_eval = 5
    task_list = [
        {
            "target": "wind_offshore_tenn",
            "region": "DE_TENNET",
            "label": "Wind off-shore [MW]",
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
                # 'finetuning_pars':{'n_trials':120,'optim_metric':'rmse','cv_folds':cv_folds_ft}},

                {'model':'XGBoost',
                 'dataset_pars':{
                     'log_target':True,
                     'forecast_horizon':None,
                     'target_scaler':'StandardScaler',
                     'feature_scaler':'StandardScaler',
                     'copy_input':True,
                     'locations':[loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                     'add_cyclical_time_features':True,
                     'feature_engineer':'WeatherWindPowerFE'
                 },
                 'finetuning_pars':{'n_trials':20,'optim_metric':'rmse','cv_folds':cv_folds_ft}},

                {'model':'ElasticNet',
                 'dataset_pars':{
                     'log_target':True,
                     'forecast_horizon':None,
                     'target_scaler':'StandardScaler',
                     'feature_scaler':'StandardScaler',
                     'copy_input':True,
                     'locations':[loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                     'add_cyclical_time_features':True,
                     'feature_engineer':'WeatherWindPowerFE'
                 },
                 'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
                #
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
                 'dataset_pars': {
                     'log_target':True,
                     'forecast_horizon':None,
                     'target_scaler':'StandardScaler',
                     'feature_scaler':'StandardScaler',
                     'add_cyclical_time_features':True,
                     'locations':[loc for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                     'feature_engineer': None,#'WeatherWindPowerFE',
                     'lags_target': None,
                     'copy_input':True
                 },
                 'finetuning_pars':{'n_trials':25,
                                    'optim_metric':'rmse',
                                    'cv_folds':cv_folds_ft,
                                    'cv_folds_base':35, # at least cv_folds_eval + 1
                                    'use_base_models_pred_intervals':False}}
            ],
            "task_training":[
                # {'model':'Prophet', 'pars':{'cv_folds':cv_folds_eval}},
                {'model':'XGBoost', 'pars':{'cv_folds':cv_folds_eval}},
                {'model':'ElasticNet', 'pars':{'cv_folds':cv_folds_eval}},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}}
            ],
            "task_forecasting":[
                # {'model':'Prophet'},
                {'model':'XGBoost', 'past_folds':cv_folds_eval},
                {'model':'ElasticNet', 'past_folds':cv_folds_eval},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','past_folds':cv_folds_eval},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','past_folds':cv_folds_eval}
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
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','n':2,
                #  'name':'Ensemble','lw':1.0,'color':"magenta",'ci_alpha':0.2,
                #  'train_forecast':'train'},
            ],
            "task_summarize":[
                # {'model':'Prophet', 'pars':{'cv_folds':5}},
                {'model':'XGBoost', 'summary_metric':'rmse'},
                {'model':'ElasticNet', 'summary_metric':'rmse'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
            ]
        }
    ]

    ''' -------------- OFFSHORE WIND GENERATION (2 TSOs) ------------- '''

    # task_list_ = copy.deepcopy(task_list)
    # process_task_list(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "wind_offshore_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in offshore_windfarms if loc['TSO']=='50Hertz']
    # process_task_list(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    ''' -------------- ONSHORE WIND GENERATION (4 TSOs) ------------- '''

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "wind_onshore_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='50Hertz']
    # process_task_list(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "wind_onshore_tran"
    #     t['region'] = "DE_TRANSNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='TransnetBW']
    # process_task_list(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    task_list_ = copy.deepcopy(task_list)
    for t in task_list_:
        t['target'] = "wind_onshore_ampr"
        t['region'] = "DE_AMPRION"
        for tt in t['task_fine_tuning']:
            tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='Amprion']
    main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "wind_onshore_tenn"
    #     t['region'] = "DE_TENNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']=='TenneT']
    # process_task_list(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)

    ''' -------------- SOLAR GENERATION (4 TSOs) ------------- '''

    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_50hz"
    #     t['region'] = "DE_50HZ"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='50Hertz']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_tran"
    #     t['region'] = "DE_TRANSNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='TransnetBW']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_ampr"
    #     t['region'] = "DE_AMPRION"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='Amprion']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)
    #
    # task_list_ = copy.deepcopy(task_list)
    # for t in task_list_:
    #     t['target'] = "solar_tenn"
    #     t['region'] = "DE_TENNET"
    #     for tt in t['task_fine_tuning']:
    #         tt['dataset_pars']['locations'] = [loc['name'] for loc in loc_solarfarms if loc['TSO']=='TenneT']
    #         tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
    # main_forecasting_pipeline(task_list=task_list_, outdir='./output/', database='../database/', verbose=True)


if __name__ == '__main__':
    main()