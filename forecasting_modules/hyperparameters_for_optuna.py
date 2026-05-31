import optuna

def get_parameters_for_optuna_trial(model_name, trial:optuna.trial):

    if model_name == 'XGBoost':
        param = {
            # 'objective': self.optim_pars['objective'],#'reg:squarederror',
            # 'eval_metric': self.optim_pars['eval_metric'],#'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        }
    elif model_name == 'MultiTargetCatBoost':
        param = {
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 1.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            # 'loss_function': 'MultiRMSE',  # Fixed since we're using multi-target regression
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 10),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        }
    elif model_name == 'ElasticNet' or model_name == 'MultiTargetElasticNet':
        param = {
            'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 1.0, log=False),
            'alpha': trial.suggest_float('alpha', 0.01, 1.0, log=False)
        }
    elif model_name == 'Prophet':
        param = {
            # 'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
            # 'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
            # 'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
            'n_changepoints': trial.suggest_int('n_changepoints', 5, 100),
            'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
            'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False]),
            'daily_seasonality': trial.suggest_categorical('daily_seasonality', [True, False]),
        }
    elif model_name == 'MultiTargetLGBM' or model_name == 'LightGBM':
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            # 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            # 'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf'])
        }
    else:
        raise NotImplementedError(f"Fine-tuning parameter set for {model_name} not implemented")

    return param
