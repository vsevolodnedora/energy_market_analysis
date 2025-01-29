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

from logger import get_logger
logger = get_logger(__name__)

def main_forecasting_pipeline(task_list:list, outdir:str, database:str, verbose:bool):

    if not os.path.isdir(outdir):
        if verbose: logger.info("Creating output directory {}".format(outdir))
        os.mkdir(outdir)

    for task in task_list:
        targets = task['targets']
        region = task['region']
        run_label = task['label']

        # get features + target (historic) and features (forecast) from database
        df_hist, df_forecast = extract_from_database(
            main_pars=task, db_path=database, outdir=outdir, verbose=verbose, n_horizons=100, horizon=7*24
        )

        # clean data from nans and outliers
        df_hist, df_forecast = clean_and_impute(df_hist=df_hist,df_forecast=df_forecast,verbose=verbose)

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
            processor.process_task_determine_the_best_model(task, outdir=outdir+run_label+'/')

def update_forecast_production(database:str, outdir:str, variable:str, verbose:bool):
    cv_folds_ft = 3
    cv_folds_eval = 5
    task_list = [
        {
            "label":["wind_offshore_tenn"],
            "targets": "wind_offshore_tenn",
            "region": "DE_TENNET",
            "plot_label": "Offshore Wind Power Generation (TenneT) [MW]",
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
                #      'log_target':False,
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
                #      'log_target':False,
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
                #      'log_target':False,
                #      'forecast_horizon':None,
                #      'target_scaler':'StandardScaler',
                #      'feature_scaler':'StandardScaler',
                #      'add_cyclical_time_features':True,
                #      'locations':[loc for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
                #      'feature_engineer': None,#'WeatherWindPowerFE',
                #      'lags_target': None,
                #      'copy_input':True
                #  },
                #  'finetuning_pars':{'n_trials':20,
                #                     'optim_metric':'rmse',
                #                     'cv_folds':cv_folds_ft,
                #                     'cv_folds_base':40,#35, # at least cv_folds_eval + 1
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
                {'model':'XGBoost', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
                {'model':'ElasticNet', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
                {'model':'ensemble[XGBoost](XGBoost,ElasticNet)', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
                # {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)', 'summary_metric':'rmse'},
            ]
        }
    ]

    ''' -------------- OFFSHORE WIND POWER GENERATION (2 TSOs) ------------- '''

    if variable == "wind_offshore":
        avail_regions = ["DE_50HZ", "DE_TENNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"wind_offshore{tso_reg['suffix']}"
                    t['targets'] = [t['label']]
                    t['plot_label'] = f"Offshore Wind Power Generation ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer'] = 'WeatherWindPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_offshore_windfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- ONSHORE WIND POWER GENERATION (4 TSOs) ------------- '''

    if variable == "wind_onshore":
        avail_regions = ["DE_AMPRION","DE_TENNET", "DE_50HZ", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"wind_onshore{tso_reg['suffix']}"
                    t['targets'] = [t['label']]
                    t['plot_label'] = f"Onshore Wind Power Generation ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherWindPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_onshore_windfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- SOLAR POWER GENERATION (4 TSOs) ------------- '''

    if variable == "solar":
        avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"solar{tso_reg['suffix']}"
                    t['targets'] = [t['label']]
                    t['plot_label'] = f"Solar Power Generation ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['log_target'] = False
                        tt['dataset_pars']['feature_engineer']  = 'WeatherSolarPowerFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_solarfarms if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- LOAD (4 TSOs) ------------- '''

    if variable == "load":
        avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"load{tso_reg['suffix']}"
                    t['targets'] = [t['label']]
                    t['plot_label'] = f"Load ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherLoadFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_cities if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    ''' -------------- MIX (4 TSOs) ------------- '''

    task_list_ = copy.deepcopy(task_list)
    task_list_[0]['task_fine_tuning'] = [
        {'model':'CatBoost',
         'dataset_pars':{
             'log_target':False,
             'lags_target': None,
             'forecast_horizon':None,
             'target_scaler':'StandardScaler',
             'feature_scaler':'StandardScaler',
             'copy_input':True,
             'locations':[loc['name'] for loc in loc_offshore_windfarms if loc['TSO']=='TenneT'],
             'add_cyclical_time_features':True,
             'feature_engineer':'WeatherLoadPowerFE',
             'spatial_agg_method': 'mean' # fix

         },
         'finetuning_pars':{'n_trials':5,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
    ]

    if variable == "energy_mix":
        avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list_)
                for t in task_list_:
                    t['label'] = f"energy_mix{tso_reg['suffix']}"
                    t['targets'] = ["hard_coal", "lignite", "coal_derived_gas", "oil", "other_fossil", "gas", "renewables"]
                    t['aggregations'] = {"renewables": [
                        "biomass","waste","geothermal","pumped_storage","run_of_river","water_reservoir","other_renewables"
                    ]}
                    t['plot_label'] = f"Energy Mix ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        tt['dataset_pars']['feature_engineer']  = 'WeatherLoadFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in loc_cities if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(task_list=task_list_, outdir=outdir, database=database, verbose=verbose)

    logger.critical('BYE')

if __name__ == '__main__':
    # todo add tests
    pass