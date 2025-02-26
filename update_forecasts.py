import copy
import os, time, sys

from data_collection_modules.eu_locations import (
    countries_metadata
)
from forecasting_modules.interface import (
    main_forecasting_pipeline
)

from logger import get_logger
logger = get_logger(__name__)


single_target_model_list = [
    'LightGBM','XGBoost','ElasticNet',
    'ensemble[XGBoost](XGBoost,ElasticNet)',
    'ensemble[LightGBM](LightGBM,ElasticNet)'
]
multi_target_model_list = [
    "MultiTargetCatBoost",
    "MultiTargetLGBM",
    "MultiTargetElasticNet",
    "ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)"
]

available_models = {
    "wind_offshore":single_target_model_list,
    "wind_onshore":single_target_model_list,
    "solar":single_target_model_list,
    "load":single_target_model_list,
    "energy_mix":multi_target_model_list
}

def create_task_list(country_dict:str,target:str,freq:str,models:list,tasks:list) -> list:


    for model in models:
        if not model in available_models[target]:
            raise Exception(
                f"Model {model} for target {target} is not available. Use: {available_models[target]}"
            )

    # meta-values for all runs/targets
    cv_folds_ft = 3
    cv_folds_eval = 5

    default_task_original = {
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

            {'model':'LightGBM',
             'dataset_pars':{
                 'log_target':False,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'copy_input':True,
                 'locations':[],
                 'add_cyclical_time_features':True,
                 'feature_engineer':'WeatherWindPowerFE'
             },
             'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
            {'model':'XGBoost',
             'dataset_pars':{
                 'log_target':False,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'copy_input':True,
                 'locations':[],
                 'add_cyclical_time_features':True,
                 'feature_engineer':'WeatherWindPowerFE'
             },
             'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
            {'model':'ElasticNet',
             'dataset_pars':{
                 'log_target':False,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'copy_input':True,
                 'locations':[],
                 'add_cyclical_time_features':True,
                 'feature_engineer':'WeatherWindPowerFE'
             },
             'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},

            {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
             'dataset_pars': {
                 'log_target':False,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'add_cyclical_time_features':True,
                 'locations':[],
                 'feature_engineer': None,#'WeatherWindPowerFE',
                 'lags_target': None,
                 'copy_input':True
             },
             'finetuning_pars':{'n_trials':20,
                                'optim_metric':'rmse',
                                'cv_folds':cv_folds_ft,
                                'cv_folds_base':40,#35, # at least cv_folds_eval + 1
                                'use_base_models_pred_intervals':False}},
            {'model':'ensemble[LightGBM](LightGBM,ElasticNet)',
             'dataset_pars': {
                 'log_target':False,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'add_cyclical_time_features':True,
                 'locations':[],
                 'feature_engineer': None,#'WeatherWindPowerFE',
                 'lags_target': None,
                 'copy_input':True
             },
             'finetuning_pars':{'n_trials':20,
                                'optim_metric':'rmse',
                                'cv_folds':cv_folds_ft,
                                'cv_folds_base':40,#35, # at least cv_folds_eval + 1
                                'use_base_models_pred_intervals':False}},

            # ---------------- MULTITARGET ------------------

            # {'model':'MultiTargetCatBoost',
            #  'dataset_pars':{
            #      'log_target':False,
            #      # 'lags_target': None,
            #      'forecast_horizon':None,
            #      'target_scaler':'StandardScaler',
            #      'feature_scaler':'StandardScaler',
            #      'copy_input':True,
            #      'locations':[],
            #      'add_cyclical_time_features':True,
            #      'feature_engineer':None,#'WeatherLoadPowerFE',
            #      'spatial_agg_method': 'mean' # fix
            #
            #  },
            #  'finetuning_pars':{'n_trials':20,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
            {'model':'MultiTargetLGBM',
             'dataset_pars':{
                 'log_target':False,
                 # 'lags_target': None,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'copy_input':True,
                 'locations':[],
                 'add_cyclical_time_features':True,
                 'feature_engineer':None,#'WeatherLoadPowerFE',
                 'spatial_agg_method': 'mean' # fix

             },
             'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
            {'model':'MultiTargetElasticNet',
             'dataset_pars':{
                 'log_target':False,
                 # 'lags_target': None,
                 'forecast_horizon':None,
                 'target_scaler':'StandardScaler',
                 'feature_scaler':'StandardScaler',
                 'copy_input':True,
                 'locations':[],
                 'add_cyclical_time_features':True,
                 'feature_engineer':None,#'WeatherLoadPowerFE',
                 'spatial_agg_method': 'mean' # fix

             },
             'finetuning_pars':{'n_trials':50,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
            # {'model':'ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)',
            #  'dataset_pars': {
            #      'log_target':False,
            #      'forecast_horizon':None,
            #      'target_scaler':'StandardScaler',
            #      'feature_scaler':'StandardScaler',
            #      'add_cyclical_time_features':True,
            #      'locations':[],
            #      'feature_engineer': None,#'WeatherWindPowerFE',
            #      'lags_target': None,
            #      'copy_input':True
            #  },
            #  'finetuning_pars':{'n_trials':40,
            #                     'optim_metric':'rmse',
            #                     'cv_folds':cv_folds_ft,
            #                     'cv_folds_base':40,#35, # at least cv_folds_eval + 1
            #                     'use_base_models_pred_intervals':False}}

        ],
        "task_training":[
            # {'model':'Prophet', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'XGBoost', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'LightGBM', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'ElasticNet', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}},
            {'model':'ensemble[LightGBM](LightGBM,ElasticNet)','pars':{'cv_folds':cv_folds_eval}},
            # --- MULTITARGET ---
            # {'model':'MultiTargetCatBoost', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'MultiTargetLGBM', 'pars':{'cv_folds':cv_folds_eval}},
            {'model':'MultiTargetElasticNet', 'pars':{'cv_folds':cv_folds_eval}},
            # {'model':'ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)', 'pars':{'cv_folds':cv_folds_eval}}
        ],
        "task_forecasting":[
            # {'model':'Prophet'},
            {'model':'XGBoost', 'past_folds':cv_folds_eval},
            {'model':'LightGBM', 'past_folds':cv_folds_eval},
            {'model':'ElasticNet', 'past_folds':cv_folds_eval},
            {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','past_folds':cv_folds_eval},
            {'model':'ensemble[LightGBM](LightGBM,ElasticNet)','past_folds':cv_folds_eval},
            # ----
            # {'model':'MultiTargetCatBoost', 'past_folds':cv_folds_eval},
            {'model':'MultiTargetLGBM', 'past_folds':cv_folds_eval},
            {'model':'MultiTargetElasticNet', 'past_folds':cv_folds_eval},
            # {'model':'ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)', 'past_folds':cv_folds_eval}
        ],
        "task_plot":[
            {'model':'Prophet', 'n':2, 'name':"Prophet",'lw':0.7,'color':"red",'ci_alpha':0.0},
            {'model':'XGBoost','n':2, 'name':'XGBoost','lw':0.7,'color':"green",'ci_alpha':0.0,
             'train_forecast':'train'},
            {'model':'LightGBM','n':2, 'name':'LightGBM','lw':0.7,'color':"orange",'ci_alpha':0.0,
             'train_forecast':'train'},
            {'model':'ElasticNet','n':2, 'name':'ElasticNet','lw':0.7,'color':"blue",'ci_alpha':0.0,
             'train_forecast':'train'},
            {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','n':2,
             'name':'Ensemble','lw':1.0,'color':"purple",'ci_alpha':0.2, 'train_forecast':'train'},
            {'model':'ensemble[LightGBM](LightGBM,ElasticNet)','n':2, 'name':'ensemble','lw':0.7,'color':"purple",'ci_alpha':0.0,
             'train_forecast':'train'},
            {'model':'ensemble[ElasticNet](XGBoost,ElasticNet)','n':2,
             'name':'Ensemble','lw':1.0,'color':"magenta",'ci_alpha':0.2, 'train_forecast':'train'},
            # ----
            # {'model':'MultiTargetCatBoost','n':2,  'name':'MultiTargetCatBoost','lw':1.0,
            #  'color':"blue", 'ci_alpha':0.2, 'train_forecast':'train'},
            {'model':'MultiTargetLGBM','n':2,  'name':'MultiTargetLGBM','lw':1.0,
             'color':"green", 'ci_alpha':0.2, 'train_forecast':'train'},
            {'model':'MultiTargetElasticNet','n':2,  'name':'MultiTargetElasticNet','lw':1.0,
             'color':"red", 'ci_alpha':0.2, 'train_forecast':'train'},
            # {'model':'ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)','n':2,  'name':'ensemble','lw':1.0,
            #  'color':"magenta", 'ci_alpha':0.2, 'train_forecast':'train'}
        ],
        "task_summarize":[
            # {'model':'Prophet', 'pars':{'cv_folds':5}},
            {'model':'XGBoost', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            {'model':'LightGBM', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            {'model':'ElasticNet', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            {'model':'ensemble[XGBoost](XGBoost,ElasticNet)', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            {'model':'ensemble[LightGBM](LightGBM,ElasticNet)', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            # ----
            # { 'model':'MultiTargetCatBoost', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            { 'model':'MultiTargetLGBM', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            { 'model':'MultiTargetElasticNet', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
            # { 'model':'ensemble[MultiTargetLGBM](MultiTargetLGBM,MultiTargetCatBoost,MultiTargetElasticNet)', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'}
        ]
    }

    default_task = copy.deepcopy(default_task_original)


    # select tasks (remove unwanted)
    default_task["task_fine_tuning"] = []
    if "finetune" in tasks:
        for i, task in enumerate(default_task_original["task_fine_tuning"]):
            if task['model'] in models:
                default_task["task_fine_tuning"].append(copy.deepcopy(task))

    default_task["task_training"] = []
    if "train" in tasks:
        for i, task in enumerate(default_task_original["task_training"]):
            if task['model'] in models:
                default_task["task_training"].append(copy.deepcopy(task))

    default_task["task_forecasting"] = []
    if "forecast" in tasks:
        for i, task in enumerate(default_task_original["task_forecasting"]):
            if task['model'] in models:
                default_task["task_forecasting"].append(copy.deepcopy(task))

    default_task["task_plot"] = []
    if "plot" in tasks:
        for i, task in enumerate(default_task_original["task_plot"]):
            if task['model'] in models:
                default_task["task_plot"].append(copy.deepcopy(task))

    default_task["task_summarize"] = []
    if "summarize" in tasks:
        for i, task in enumerate(default_task_original["task_summarize"]):
            if task['model'] in models:
                default_task["task_summarize"].append(copy.deepcopy(task))

    return [default_task]


    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # create task
    # task = dict()
    #
    # task["label"] = target
    # task["targets"] = None
    # task["region"] = None
    # task["plot_label"] = None
    #
    # # empty lists for different tasks
    # task_fine_tuning = []
    # task_training = []
    # task_forecasting = []
    # task_plot = []
    # task_summarize = []
    #
    # lgbm_setups = {
    #     "task_fine_tuning":
    #         {'model':'LightGBM',
    #          'dataset_pars':{
    #              'log_target':False,
    #              'forecast_horizon':None,
    #              'target_scaler':'StandardScaler',
    #              'feature_scaler':'StandardScaler',
    #              'copy_input':True,
    #              'locations':[],
    #              'add_cyclical_time_features':True,
    #              'feature_engineer':'WeatherWindPowerFE'
    #          },
    #          'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
    #     "task_training":{'model':'LightGBM', 'pars':{'cv_folds':cv_folds_eval}},
    #     "task_forecasting": {'model':'LightGBM', 'past_folds':cv_folds_eval},
    #     "task_plot":
    #         {'model':'LightGBM','n':2, 'name':'LightGBM','lw':0.7,'color':"orange",'ci_alpha':0.0,
    #          'train_forecast':'train'},
    #     "task_summarize":
    #         {'model':'LightGBM', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'},
    # }
    # xdg_setups = {
    #
    # }
    #
    #
    # # add tasks based on model to be used
    # if "LightGBM":
    #     if "finetune" in tasks:
    #         task_fine_tuning_ = \
    #             {'model':'LightGBM',
    #              'dataset_pars':{
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'copy_input':True,
    #                  'locations':[],
    #                  'add_cyclical_time_features':True,
    #                  'feature_engineer':'WeatherWindPowerFE'
    #              },
    #              'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
    #         task_fine_tuning.append(task_fine_tuning_)
    #     if "train" in tasks:
    #         task_training.append({'model':'XGBoost', 'pars':{'cv_folds':cv_folds_eval}})
    #     if "forecast" in tasks:
    #         task_forecasting.append({'model':'XGBoost', 'past_folds':cv_folds_eval})
    #     if "plot" in tasks:
    #         task_plot.append(
    #             {'model':'XGBoost','n':2, 'name':'XGBoost','lw':0.7,'color':"green",'ci_alpha':0.0,
    #              'train_forecast':'train'}
    #         )
    #     if "summarize":
    #         task_summarize.append(
    #             {'model':'XGBoost', 'summary_metric':'rmse', 'n_folds_best':3, 'method_for_best':'trained'}
    #         )
    #
    # task["task_fine_tuning"] = copy.deepcopy(task_fine_tuning)
    # task["task_training"] = copy.deepcopy(task_training)
    # task["task_forecasting"] = copy.deepcopy(task_forecasting)
    # task["task_plot"] = copy.deepcopy(task_plot)
    # task["task_summarize"] = copy.deepcopy(task_summarize)
    #
    # ''' ----------------------- FINETUNING TASKS --------------------- '''
    #
    # task["task_fine_tuning"] = []
    # if "finetune" in tasks:
    #     # ---
    #     if "LightGBM" in models:
    #         task_fine_tuning = \
    #         {'model':'LightGBM',
    #              'dataset_pars':{
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'copy_input':True,
    #                  'locations':[],
    #                  'add_cyclical_time_features':True,
    #                  'feature_engineer':'WeatherWindPowerFE'
    #              },
    #              'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}},
    #         task["task_fine_tuning"].append(task_fine_tuning)
    #     # ---
    #     if 'XGBoost' in models:
    #         task_fine_tuning = \
    #             {'model':'XGBoost',
    #              'dataset_pars':{
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'copy_input':True,
    #                  'locations':[],
    #                  'add_cyclical_time_features':True,
    #                  'feature_engineer':'WeatherWindPowerFE'
    #              },
    #              'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}}
    #         task["task_fine_tuning"].append(task_fine_tuning)
    #     # ---
    #     if 'ElasticNet' in models:
    #         task_fine_tuning = \
    #             {'model':'ElasticNet',
    #              'dataset_pars':{
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'copy_input':True,
    #                  'locations':[],
    #                  'add_cyclical_time_features':True,
    #                  'feature_engineer':'WeatherWindPowerFE'
    #              },
    #              'finetuning_pars':{'n_trials':30,'optim_metric':'rmse','cv_folds':cv_folds_ft}}
    #         task["task_fine_tuning"].append(task_fine_tuning)
    #     # ---
    #     if 'ensemble[XGBoost](XGBoost,ElasticNet)' in models:
    #         task_fine_tuning = \
    #             {'model':'ensemble[XGBoost](XGBoost,ElasticNet)',
    #              'dataset_pars': {
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'add_cyclical_time_features':True,
    #                  'locations':[],
    #                  'feature_engineer': None,#'WeatherWindPowerFE',
    #                  'lags_target': None,
    #                  'copy_input':True
    #              },
    #              'finetuning_pars':{'n_trials':20,
    #                                 'optim_metric':'rmse',
    #                                 'cv_folds':cv_folds_ft,
    #                                 'cv_folds_base':40,#35, # at least cv_folds_eval + 1
    #                                 'use_base_models_pred_intervals':False}}
    #         task["task_fine_tuning"].append(task_fine_tuning)
    #     # ---
    #     if 'ensemble[LightGBM](LightGBM,ElasticNet)' in models:
    #         task_fine_tuning = \
    #             {'model':'ensemble[LightGBM](LightGBM,ElasticNet)',
    #              'dataset_pars': {
    #                  'log_target':False,
    #                  'forecast_horizon':None,
    #                  'target_scaler':'StandardScaler',
    #                  'feature_scaler':'StandardScaler',
    #                  'add_cyclical_time_features':True,
    #                  'locations':[],
    #                  'feature_engineer': None,#'WeatherWindPowerFE',
    #                  'lags_target': None,
    #                  'copy_input':True
    #              },
    #              'finetuning_pars':{'n_trials':20,
    #                                 'optim_metric':'rmse',
    #                                 'cv_folds':cv_folds_ft,
    #                                 'cv_folds_base':40,#35, # at least cv_folds_eval + 1
    #                                 'use_base_models_pred_intervals':False}}
    #         task["task_fine_tuning"].append(task_fine_tuning)
    #
    # ''' ----------------------- FULL MODEL TRAINING TASKS ------------------- '''
    #
    # task["task_training"] = []
    # if "training" in task:
    #     if "XGBoost" in models:
    #         task["task_training"].append({'model':'XGBoost', 'pars':{'cv_folds':cv_folds_eval}})
    #     if "LightGBM" in models:
    #         task["task_training"].append({'model':'LightGBM', 'pars':{'cv_folds':cv_folds_eval}})
    #     if "ElasticNet" in models:
    #         task["task_training"].append({'model':'ElasticNet', 'pars':{'cv_folds':cv_folds_eval}})
    #     if "ensemble[XGBoost](XGBoost,ElasticNet)" in models:
    #         task["task_training"].append(
    #             {'model':'ensemble[XGBoost](XGBoost,ElasticNet)','pars':{'cv_folds':cv_folds_eval}}
    #         )
    #     if "ensemble[LightGBM](LightGBM,ElasticNet)" in models:
    #         task["task_training"].append(
    #             {'model':'ensemble[LightGBM](LightGBM,ElasticNet)','pars':{'cv_folds':cv_folds_eval}}
    #         )
    #
    # ''' ------------------- FORECASTING WITH FULLY TRAINED MODEL TASKS --------------------- '''
    # task['task_forecasting'] = []
    # if 'forecasting' in tasks:


def adjust_and_run_for_tasklist(database:str,c_dict:dict, task_list:list, variable:str,outdir:str,verbose:bool):

    ''' pass '''

    de_regions = c_dict['regions']

    ''' -------------- OFFSHORE WIND POWER GENERATION (2 TSOs) ------------- '''
    if variable == "wind_offshore":
        avail_regions = [tso['name'] for tso in c_dict['regions'] if variable in tso['available_targets']]#["DE_50HZ", "DE_TENNET"]
        locations = c_dict['locations']['offshore']
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
                            [loc['name'] for loc in locations if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(
                    c_dict=c_dict,task_list=task_list_, outdir=outdir, database=database, freq=freq, verbose=verbose
                )

    ''' -------------- ONSHORE WIND POWER GENERATION (4 TSOs) ------------- '''

    if variable == "wind_onshore":
        # avail_regions = ["DE_AMPRION","DE_TENNET", "DE_50HZ", "DE_TRANSNET"]
        avail_regions = [tso['name'] for tso in c_dict['regions'] if variable in tso['available_targets']]
        locations = c_dict['locations']['onshore']
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
                            [loc['name'] for loc in locations if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(
                    c_dict=c_dict,task_list=task_list_, outdir=outdir, database=database, freq=freq, verbose=verbose
                )

    ''' -------------- SOLAR POWER GENERATION (4 TSOs) ------------- '''

    if variable == "solar":
        # avail_regions = ["DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET"]
        avail_regions = [tso['name'] for tso in c_dict['regions'] if variable in tso['available_targets']]
        locations = c_dict['locations']['solar']
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
                            [loc['name'] for loc in locations if loc['TSO']==tso_reg['TSO']]
                main_forecasting_pipeline(
                    c_dict=c_dict,task_list=task_list_, outdir=outdir, database=database, freq=freq, verbose=verbose
                )

    ''' -------------- LOAD (4 TSOs) ------------- '''

    if variable == "load":
        # avail_regions = [ "DE_TENNET", "DE_50HZ", "DE_AMPRION", "DE_TRANSNET" ]
        avail_regions = [tso['name'] for tso in c_dict['regions'] if variable in tso['available_targets']]
        locations = c_dict['locations']['cities']
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
                            [loc['name'] for loc in locations if loc['TSO'] == tso_reg['TSO']]
                main_forecasting_pipeline(
                    c_dict=c_dict,task_list=task_list_, outdir=outdir, database=database, freq=freq, verbose=verbose
                )

    ''' -------------- ENERGY MIX (4 TSOs) ------------- '''

    if variable == "energy_mix":

        # avail_regions = [ "DE_50HZ", "DE_TENNET", "DE_AMPRION", "DE_TRANSNET" ] # [ "DE_50HZ", "DE_TENNET", "DE_AMPRION", "DE_TRANSNET" ]
        avail_regions = [tso['name'] for tso in c_dict['regions'] if variable in tso['available_targets']]
        locations = c_dict['locations']['cities']
        for tso_reg in de_regions:
            if tso_reg['name'] in avail_regions:
                task_list_ = copy.deepcopy(task_list)
                for t in task_list_:
                    t['label'] = f"energy_mix{tso_reg['suffix']}"
                    t['targets'] = [key+tso_reg['suffix'] for key in [
                        "hard_coal", "lignite", "coal_derived_gas", "oil", "other_fossil", "gas", "renewables","biomass","waste"]
                                    ]
                    t['aggregations'] = {f"renewables{tso_reg['suffix']}": [
                        key+tso_reg['suffix'] for key in
                        ["geothermal","pumped_storage","run_of_river","water_reservoir","other_renewables"]
                    ]}

                    t['plot_label'] = f"Energy Mix ({tso_reg['name']}) [MW]"
                    t['region'] = tso_reg['name']
                    for tt in t['task_fine_tuning']:
                        # tt['dataset_pars']['feature_engineer']  = 'WeatherLoadFE'
                        tt['dataset_pars']['locations'] = \
                            [loc['name'] for loc in locations if loc['TSO']==tso_reg['TSO']]

                start_time = time.time()  # Start the timer
                main_forecasting_pipeline(
                    c_dict=c_dict,task_list=task_list_, outdir=outdir, database=database, freq=freq, verbose=verbose
                )
                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time
                hours, minutes = divmod(elapsed_time // 60, 60)
                _tasks = [key for key in task_list_[0].keys() if len(task_list_[0][key]) > 0 and key.startswith('task')]
                logger.info(
                    f"Tasks: {_tasks} region: {tso_reg['name']} \n"
                    f"\tRuntime: {int(hours)} hr. & {int(minutes)} min."
                )

def main(country_code:str, target:str, model:str, mode:str, freq:str,verbose:bool):

    countries = ['DE', 'FR', 'all']
    if not country_code in countries:
        raise ValueError(f'country_code must be in {countries}. Given: {country_code}')
    if country_code == 'all': country_code = countries[:-1] # all countries
    else: country_code = [country_code]

    targets = ['wind_offshore', 'wind_onshore', 'solar', 'load', 'energy_mix', 'all']
    if not target in targets:
        raise ValueError(f'target must be in {targets}. Given: {target}')
    if target == 'all': target = targets[:-1]
    else: target = [target]

    # models = single_target_model_list + multi_target_model_list + ['all']
    # if not model in models:
    #     raise ValueError(f'model must be in {models}. Given: {model}')
    # if model == 'all': model = models[:-1]
    # else: model = [model]

    modes = ['finetune', 'train', 'forecast', 'plot', 'summarize', 'all']
    if not mode in modes:
        raise ValueError(f'mode must be in {modes}. Given: {mode}')
    if mode == 'all': mode = modes[1:-1]
    else: mode = [mode]

    freqs = ['hourly', 'minutely_15']
    if not freq in freqs:
        raise ValueError(f'freq must be in {freqs}. Given: {freq}')

    for country in country_code:

        # check country
        c_dict:dict = [dict_ for dict_ in countries_metadata if dict_["code"] == country][0]
        if len(list(c_dict.keys())) == 0:
            raise KeyError(f"No country dict found for country code {country_code}. Check your country code.")
        regions = c_dict["regions"]
        if len(regions) == 0:
            logger.warning(f"No regions (TSOs) dicts found for country code {country_code}.")
        locations = list(c_dict['locations'].keys())
        if len(locations) == 0:
            logger.warning(f"No locations (for weather data) found for country code {country_code}.")

        # set database location
        db_path = f'./database/{country}/'
        outdir = f'./output/{country}/forecasts/'


        start_time = time.time()  # Start the timer

        # run for target
        for target_ in target:

            if model == 'all':
                model_ = available_models[target_]
            else:
                if not model in available_models[target_]:
                    raise ValueError(f'model must be in {available_models[target_]}. Given: {model}')
                else:
                    model_ = [model]

            logger.info(f'Updating forecasts for country {country} and target {target_}...')
            tasks = create_task_list(country,target_,freq,model_,mode)

            adjust_and_run_for_tasklist(
                database=db_path, c_dict=c_dict, task_list=tasks,variable=target_,outdir=outdir,verbose=verbose
            )

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        hours, minutes = divmod(elapsed_time // 60, 60)

        logger.info(
            f"All tasks for country {country_code} are completed successfully! Execution time: "
            f"{int(hours)} hours and {int(minutes)} minutes."
        )



    # # freq='hourly'
    # # freq='minutely_15'
    #
    # # db_path = './database/'
    # # db_path_15min = './database_15min/'
    #
    # # targets = ['wind_offshore', 'wind_onshore', 'solar', 'load', 'energy_mix']
    # targets = ['wind_offshore', 'wind_onshore', 'solar', 'load', 'energy_mix']
    # # targets = ['energy_mix']
    #
    # # update_forecast_production(
    # #     database=db_path, variable='energy_mix', outdir='./output/forecasts/', verbose=True
    # # )
    #
    # start_time = time.time()  # Start the timer
    #
    #
    # for target in targets:
    #     update_forecast_production(
    #         database='./database/DE/',
    #         variable=target,
    #         outdir='./output/forecasts/',
    #         freq='hourly',
    #         verbose=True
    #     )
    #     # update_forecast_production(
    #     #     database='./database_15min/',
    #     #     variable=target,
    #     #     outdir='./output_15min/forecasts/',
    #     #     freq='minutely_15',
    #     #     verbose=True
    #     # )
    #
    # end_time = time.time()  # End the timer
    # elapsed_time = end_time - start_time
    # hours, minutes = divmod(elapsed_time // 60, 60)
    #
    # logger.info(
    #     f"All tasks in update are completed successfully! Execution time: "
    #     f"{int(hours)} hours and {int(minutes)} minutes."
    # )



if __name__ == '__main__':

    print("launching update_forecasts.py")

    if len(sys.argv) != 6:
        # raise KeyError("Usage: python update_database.py <country_code> <task> <freq>")
        country_code = str( 'FR' )
        target = str( 'wind_offshore' )
        model = str( 'LightGBM' )
        mode = str( 'finetune' )
        freq = str( 'hourly' )
    else:
        country_code = str( sys.argv[1] )
        target = str( sys.argv[2] )
        model = str( sys.argv[3] )
        mode = str( sys.argv[4] )
        freq = str( sys.argv[5] )

    main(country_code, target, model, mode, freq, True)
