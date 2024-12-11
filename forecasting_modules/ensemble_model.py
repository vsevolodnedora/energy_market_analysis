import copy
import pandas as pd

from .utils import (
    compute_timeseries_split_cutoffs,
    compute_error_metrics
)
from data_modules import (
    HistForecastDatasetBase, HistForecastDataset
)
from .base_models import (
    BaseForecaster,XGBoostMapieRegressor,ElasticNetMapieRegressor,ProphetForecaster
)

class EnsembleForecaster_OLD(BaseForecaster):
    def __init__(
            self,
            target: str,
            models: list[BaseForecaster],
            datas: list[HistForecastDatasetBase],
            ensemble_model:BaseForecaster,
            alpha:float=0.05,
            base_cv_folds:int=10,
            meta_cv_folds:int =3,
            delta_between_model_fits:int or None = None,
            use_base_models_pred_intervals:bool=False
    ):
        '''
        :param target: str name of the target variable
        :param models: list of base forecasting models to be fitted and their forecasts to be used to train meta-model
        :param datas: list od dataclasses
        :param ensemble_model: regressor to be used as an esnemble model
        :param alpha: credibility interval to compute with Mapieregressors that are wrapping base predictors
        :param cv_folds: number of fits to perform with base regressors (their forecasts are used to train meta-model)
        :param meta_cv_folds: number of fits to perform with meta-model to evaluate its performance should be << cv_folds
        :param delta_between_model_fits: number of hours between forecasting windows.
            By default is the same as horizon (non-overlapping windows)
        :param use_base_models_pred_intervals: add prediction intervals from base models as features for meta-model
        '''
        super().__init__(
            target, 0, alpha
        )
        self.name = 'ensemble'
        self.target = target
        self.models = models
        self.meta_model = ensemble_model
        assert len(models) == len(datas), 'Number of models must equal number of datasets'
        self.dss = datas
        self.min_train_size = 3 * 30 * 24  # hours (3 months)
        self.base_cv_folds = base_cv_folds # number of times to fit-test base models to create train data for ensemble model
        self.meta_cv_folds = meta_cv_folds # number of times to fit-test ensemble model to test it
        self.horizon = len(self.dss[0].get_forecast_index())

        # -------------------------------------------------- #
        self.delta = delta_between_model_fits if not delta_between_model_fits is None else self.horizon
        self.cutoffs = compute_timeseries_split_cutoffs(
            self.dss[0].get_index(),
            horizon=self.horizon,
            delta=self.delta,
            folds=self.base_cv_folds,
            min_train_size=self.min_train_size
        )

        # how to split the data for training base models
        self.X_meta_model_train:pd.DataFrame
        self.y_meta_model_train:pd.Series

        # store all base model forecasts (+1 for the actual forecast obtained by calling .forecast() on the class obj.
        self.all_forecasts:dict[str:list[pd.DataFrame]] = {
            str(model.name):[ pd.DataFrame() for i in range(self.base_cv_folds+1) ] for model in self.models
        }
        self.all_forecasts[self.name] = [ pd.DataFrame() for i in range(self.meta_cv_folds+1) ]

        # store all performance metrics
        self.all_model_metrics:dict[str:list[dict]] = {
            str(model.name):[ {} for i in range(self.base_cv_folds) ] for model in self.models
        }
        self.all_model_metrics[self.name] = [{} for i in range(self.meta_cv_folds)]
        # if true, use also prediction intervals from each trained base predictor
        if use_base_models_pred_intervals: self.features_from_base_models = ['fitted','lower','upper']
        else: self.features_from_base_models = ['fitted']


    def get_base_model_train_cutoffs(self)->list[pd.Timestamp]:
        if len( self.cutoffs ) == 0:
            raise ValueError(f"Time-series cuttoffs are not yet initialized")
        return self.cutoffs

    def fit_base_models_for_all_folds(self, X_meta_scaled:pd.DataFrame or None = None):
        '''
        Train base models `cv_folds` times by shifted forecasting window forward untill the end of the dataset is reached.
        At each iteration the forecast is made and is compared with actual available data to compute error metrics
        All forecasts are collected in self.all_forecasts:dict for future use.
        Forecasts are also collected into a single dataframe as features for meta-model.
        :param X_meta_scaled: scaled features for meta-model (may differ from base models but should have the same index)
        :return:
        '''
        ''' train the base forecasters get their forecast for out of sample data and train the metamodel on the result '''
        # Check that X_scaled_meta has the same index as the training data
        if ((not X_meta_scaled is None) and (not X_meta_scaled.index.equals(self.dss[0].get_index()))):
            raise ValueError(f"X_meta_scaled must have the same index as the training data. "
                             f"Given X_meta_scaled={len(X_meta_scaled)} and expected {len(self.dss[0].get_index())}")

        print(f"Will forecast {len(self.cutoffs)} times {self.delta}-hour-horizons"
              f"with N={len(self.models)} base models to train 1 meta_model {self.meta_model.name}")
        # target column should be the same for all base models
        y_scaled = self.dss[0].get_target_transformed()
        X_meta_train_list, y_meta_train_list = [], []
        # for each split train base models, evaluate on the test part of the window and store these forecasts
        for idx, cutoff in enumerate(self.cutoffs):
            train_mask = y_scaled.index <= cutoff
            test_mask = (y_scaled.index > cutoff) & (y_scaled.index <= cutoff + pd.Timedelta(hours=self.horizon))
            forecasts_df = pd.DataFrame(index=y_scaled[test_mask].index)
            # train base models
            for i in range(len(self.models)):
                name = str(self.models[i].name)
                X_for_model: pd.DataFrame = self.dss[i].get_exogenous_trasnformed()
                y_for_model: pd.Series = self.dss[i].get_target_transformed()

                self.models[i].fit(X_for_model[train_mask], y_for_model[train_mask])

                # predict values for out of sample features (forecast, essentially)
                self.all_forecasts[name][idx]:pd.DataFrame = self.models[i].forecast_window( X_for_model[test_mask] )
                self.all_forecasts[name][idx][f'{self.target}_actual'] = y_for_model[test_mask]
                # compute model performance metrics and store them for future aggregation (undo transformation)
                self.all_model_metrics[name][idx] = compute_error_metrics(
                    self.target, self.all_forecasts[name][idx].apply(self.dss[0].inv_transform_target_series)
                )
                print(f"Fold {idx}/{len(self.cutoffs)} cutoff={cutoff} | {name} "
                      f"| RMSE={self.all_model_metrics[name][idx]['rmse']:.1f} "
                      f"sMAPE={self.all_model_metrics[name][idx]['smape']:.2f}"
                      f"| X_train={X_for_model[train_mask].shape} y_train={y_for_model[train_mask].shape}")

                # store predictions and prediction intervals for later training the meta model
                for key in self.features_from_base_models:
                    forecasts_df[f'base_model_{i}_{key}'] = self.all_forecasts[name][idx][f'{self.target}_{key}']

            # add meta features that might help with model performance
            if not X_meta_scaled is None:
                forecasts_df = forecasts_df.merge(X_meta_scaled[test_mask], left_index=True, right_index=True)

            X_meta_train_list.append(forecasts_df)
            y_meta_train_list.append(y_scaled[test_mask])

        # Concatenate all folds and check shapes
        self.X_meta_model_train = pd.concat(X_meta_train_list)
        self.y_meta_model_train = pd.concat(y_meta_train_list)
        if not X_meta_scaled is None:
            if not (len(self.X_meta_model_train.columns) ==
                    len(X_meta_scaled.columns) + len(self.models)*len(self.features_from_base_models)):
                raise ValueError(f"Expected {len(X_meta_scaled.columns)+len(self.models)} columns in "
                                 f"X_meta_model_train. Got={len(self.X_meta_model_train.columns)}")
        else:
            if not (len(self.X_meta_model_train.columns) == len(self.models)):
                raise ValueError(f"Expected {len(self.models)} columns in "
                                 f"X_meta_model_train. Got={len(self.X_meta_model_train.columns)}")

    def fit(self, X_meta_scaled:pd.DataFrame or None = None, y_meta_scaled:pd.Series or None = None):
        '''
            Fit base models `cv_folds` times shifting the forecast horizon forward by `delta_between_model_fits`
            until the end of the dataset is reached.
            Use the forecasts by base models and actual targets to train a meta or ensemble model.
            The model is retrained `meta_cv_folds` times to allow for average error estimation.
            In the end both base models and meta model are trained using all available data.
        '''

        # fit base models 'folds' time and store their predictions for unseen data
        self.fit_base_models_for_all_folds(X_meta_scaled)

        # train and evaluate ensemble model for different amount of training data (to get error metrics)
        if not self.meta_cv_folds is None:
            if self.meta_cv_folds > self.base_cv_folds - 1:
                raise ValueError(f"Number of cv_runs for meta_model cannot be less than number of base model evaluations. "
                                 f"Given cv_runs={self.meta_cv_folds} while base models were evaluated {self.base_cv_folds} times")
            elif self.meta_cv_folds == 1: cv_folds = [self.cutoffs[-1]]
            else: cv_folds = self.cutoffs[-self.meta_cv_folds:]

            for idx, cutoff in enumerate(cv_folds):
                train_mask = self.X_meta_model_train.index <= cutoff
                test_mask = (self.X_meta_model_train.index > cutoff) & \
                            (self.X_meta_model_train.index <= cutoff + pd.Timedelta(hours=self.horizon))

                X_train, X_test = self.X_meta_model_train[train_mask], self.X_meta_model_train[test_mask]
                y_train, y_test = self.y_meta_model_train[train_mask], self.y_meta_model_train[test_mask]

                self.meta_model.fit(X_train, y_train)

                self.all_forecasts[self.name][idx] = copy.deepcopy( self.meta_model.forecast_window( X_test ) )
                self.all_forecasts[self.name][idx][f'{self.target}_actual'] = y_test

                # compute error metrics
                self.all_model_metrics[self.name][idx] = compute_error_metrics(
                        self.target, self.all_forecasts[self.name][idx].apply(self.dss[0].inv_transform_target_series)
                )

                print(f"Fold={idx}/{len(cv_folds)} cutoff={cutoff} | meta {self.name} "
                      f"| RMSE={self.all_model_metrics[self.name][idx]['rmse']:.1f} "
                      f"sMAPE={self.all_model_metrics[self.name][idx]['smape']:.2f} "
                      f"| X_train={X_train.shape} y_train={y_train.shape}")

        # train base forecasters on the entire dataset
        print(f"Training base forecasters on the entire dataset")
        for i in range(len(self.models)):
            X_for_model: pd.DataFrame = self.dss[i].get_exogenous_trasnformed()
            y_for_model: pd.Series = self.dss[i].get_target_transformed()
            self.models[i].fit(X_for_model, y_for_model)

        # fit the meta-model on the collected dataset
        print(f"Training meta_model on the all available data "
              f"X_train={self.X_meta_model_train.shape} "
              f"y_train={pd.Series(self.y_meta_model_train).shape}")
        self.meta_model.fit(self.X_meta_model_train, pd.Series(self.y_meta_model_train))
    def forecast_window(self, X_meta_scaled:pd.DataFrame or None)->pd.DataFrame:
        '''
            First forecast every base model for their respective X_scaled and then use their predictions
            as well as provided X_meta_scaled to generate a prediction with meta-model
        '''
        forecasts_df = pd.DataFrame(index=X_meta_scaled.index)
        # collect forecasts from base models
        for i in range(len(self.models)):
            name = self.models[i].name
            X_for_model: pd.DataFrame = self.dss[i].get_forecast_exogenous()
            y_model_forecast = self.models[i].forecast_window( X_for_model )
            self.all_forecasts[name][-1] = copy.deepcopy( y_model_forecast )
            for key in self.features_from_base_models:
                forecasts_df[f'base_model_{i}_{key}'] = \
                    copy.deepcopy( self.all_forecasts[name][-1][f'{self.target}_{key}'] )

        # append meta exogenous quantities
        if not X_meta_scaled is None:
            forecasts_df = forecasts_df.merge(X_meta_scaled, left_index=True, right_index=True)

        # predict with model
        last_forecast = self.meta_model.forecast_window( forecasts_df )
        self.all_forecasts[self.name][-1] = copy.deepcopy(last_forecast)
        return last_forecast


    # def predict(self, X_meta_scaled:pd.DataFrame or None, y_meta_scaled:pd.Series or None)->pd.DataFrame:
    #     ''' predict target for the train data or part of it '''
    #     predict_df = pd.DataFrame(index=X_meta_scaled.index)
    #     # collect forecasts from base models (X_scaled for the meta_model)
    #     for i in range(len(self.models)):
    #         name = self.models[i].name
    #         X_for_model: pd.DataFrame = self.dss[i].get_exogenous_trasnformed()
    #         y_for_model: pd.Series = self.dss[i].get_target_transformed()
    #         y_model_predict = self.models[i].predict( X_for_model, y_for_model )
    #         for key in self.features_from_base_models:
    #             predict_df[f'base_model_{i}_{key}'] = (
    #                 copy.deepcopy( y_model_predict[f'{self.target}_{key}']))
    #
    #     # add the external data if needed
    #     if not X_meta_scaled is None:
    #         predict_df = predict_df.merge(X_meta_scaled, left_index=True, right_index=True)
    #
    #     if y_meta_scaled is None:
    #         y_meta_scaled = self.dss[0].get_target_transformed()
    #
    #     # call the metamodel for prediction
    #     return self.meta_model.predict(predict_df, y_meta_scaled)


class EnsembleForecaster(BaseForecaster):
    def __init__(
            self,
            target: str,
            models: list[BaseForecaster],
            datas: list[HistForecastDatasetBase],
            ensemble_model:BaseForecaster,
            alpha:float=0.05,
            base_cv_folds:int=10,
            meta_cv_folds:int =3,
            delta_between_model_fits:int or None = None,
            use_base_models_pred_intervals:bool=False
    ):
        '''
        :param target: str name of the target variable
        :param models: list of base forecasting models to be fitted and their forecasts to be used to train meta-model
        :param datas: list od dataclasses
        :param ensemble_model: regressor to be used as an esnemble model
        :param alpha: credibility interval to compute with Mapieregressors that are wrapping base predictors
        :param cv_folds: number of fits to perform with base regressors (their forecasts are used to train meta-model)
        :param meta_cv_folds: number of fits to perform with meta-model to evaluate its performance should be << cv_folds
        :param delta_between_model_fits: number of hours between forecasting windows.
            By default is the same as horizon (non-overlapping windows)
        :param use_base_models_pred_intervals: add prediction intervals from base models as features for meta-model
        '''
        super().__init__(
            target, 0, alpha
        )
        self.name = 'ensemble'
        self.target = target
        self.models = models
        self.meta_model = ensemble_model
        assert len(models) == len(datas), 'Number of models must equal number of datasets'
        self.dss = datas
        self.min_train_size = 3 * 30 * 24  # hours (3 months)
        self.base_cv_folds = base_cv_folds # number of times to fit-test base models to create train data for ensemble model
        self.meta_cv_folds = meta_cv_folds # number of times to fit-test ensemble model to test it
        self.horizon = len(self.dss[0].get_forecast_index())

        # -------------------------------------------------- #
        self.delta = delta_between_model_fits if not delta_between_model_fits is None else self.horizon
        self.cutoffs = compute_timeseries_split_cutoffs(
            self.dss[0].get_index(),
            horizon=self.horizon,
            delta=self.delta,
            folds=self.base_cv_folds,
            min_train_size=self.min_train_size
        )

        # how to split the data for training base models
        self.X_meta_model_train:pd.DataFrame
        self.y_meta_model_train:pd.Series

        # store all base model forecasts (+1 for the actual forecast obtained by calling .forecast() on the class obj.
        self.all_forecasts:dict[str:list[pd.DataFrame]] = {
            str(model.name):[ pd.DataFrame() for i in range(self.base_cv_folds+1) ] for model in self.models
        }
        self.all_forecasts[self.name] = [ pd.DataFrame() for i in range(self.meta_cv_folds+1) ]

        # store all performance metrics
        self.all_model_metrics:dict[str:list[dict]] = {
            str(model.name):[ {} for i in range(self.base_cv_folds) ] for model in self.models
        }
        self.all_model_metrics[self.name] = [{} for i in range(self.meta_cv_folds)]
        # if true, use also prediction intervals from each trained base predictor
        if use_base_models_pred_intervals: self.features_from_base_models = ['fitted','lower','upper']
        else: self.features_from_base_models = ['fitted']


    def get_base_model_train_cutoffs(self)->list[pd.Timestamp]:
        if len( self.cutoffs ) == 0:
            raise ValueError(f"Time-series cuttoffs are not yet initialized")
        return self.cutoffs

    def fit_base_models_for_all_folds(self, X_meta_scaled:pd.DataFrame or None = None):
        '''
        Train base models `cv_folds` times by shifted forecasting window forward untill the end of the dataset is reached.
        At each iteration the forecast is made and is compared with actual available data to compute error metrics
        All forecasts are collected in self.all_forecasts:dict for future use.
        Forecasts are also collected into a single dataframe as features for meta-model.
        :param X_meta_scaled: scaled features for meta-model (may differ from base models but should have the same index)
        :return:
        '''
        ''' train the base forecasters get their forecast for out of sample data and train the metamodel on the result '''
        # Check that X_scaled_meta has the same index as the training data
        if ((not X_meta_scaled is None) and (not X_meta_scaled.index.equals(self.dss[0].get_index()))):
            raise ValueError(f"X_meta_scaled must have the same index as the training data. "
                             f"Given X_meta_scaled={len(X_meta_scaled)} and expected {len(self.dss[0].get_index())}")

        print(f"Will forecast {len(self.cutoffs)} times {self.delta}-hour-horizons"
              f"with N={len(self.models)} base models to train 1 meta_model {self.meta_model.name}")
        # target column should be the same for all base models
        y_scaled = self.dss[0].get_target_transformed()
        X_meta_train_list, y_meta_train_list = [], []
        # for each split train base models, evaluate on the test part of the window and store these forecasts
        for idx, cutoff in enumerate(self.cutoffs):
            train_mask = y_scaled.index <= cutoff
            test_mask = (y_scaled.index > cutoff) & (y_scaled.index <= cutoff + pd.Timedelta(hours=self.horizon))
            forecasts_df = pd.DataFrame(index=y_scaled[test_mask].index)
            # train base models
            for i in range(len(self.models)):
                name = str(self.models[i].name)
                X_for_model: pd.DataFrame = self.dss[i].get_exogenous_trasnformed()
                y_for_model: pd.Series = self.dss[i].get_target_transformed()

                self.models[i].fit(X_for_model[train_mask], y_for_model[train_mask])

                # predict values for out of sample features (forecast, essentially)
                self.all_forecasts[name][idx]:pd.DataFrame = self.models[i].forecast_window( X_for_model[test_mask] )
                self.all_forecasts[name][idx][f'{self.target}_actual'] = y_for_model[test_mask]
                # compute model performance metrics and store them for future aggregation (undo transformation)
                self.all_model_metrics[name][idx] = compute_error_metrics(
                    self.target, self.all_forecasts[name][idx].apply(self.dss[0].inv_transform_target_series)
                )
                print(f"Fold {idx}/{len(self.cutoffs)} cutoff={cutoff} | {name} "
                      f"| RMSE={self.all_model_metrics[name][idx]['rmse']:.1f} "
                      f"sMAPE={self.all_model_metrics[name][idx]['smape']:.2f}"
                      f"| X_train={X_for_model[train_mask].shape} y_train={y_for_model[train_mask].shape}")

                # store predictions and prediction intervals for later training the meta model
                for key in self.features_from_base_models:
                    forecasts_df[f'base_model_{i}_{key}'] = self.all_forecasts[name][idx][f'{self.target}_{key}']

            # add meta features that might help with model performance
            if not X_meta_scaled is None:
                forecasts_df = forecasts_df.merge(X_meta_scaled[test_mask], left_index=True, right_index=True)

            X_meta_train_list.append(forecasts_df)
            y_meta_train_list.append(y_scaled[test_mask])

        # Concatenate all folds and check shapes
        self.X_meta_model_train = pd.concat(X_meta_train_list)
        self.y_meta_model_train = pd.concat(y_meta_train_list)
        if not X_meta_scaled is None:
            if not (len(self.X_meta_model_train.columns) ==
                    len(X_meta_scaled.columns) + len(self.models)*len(self.features_from_base_models)):
                raise ValueError(f"Expected {len(X_meta_scaled.columns)+len(self.models)} columns in "
                                 f"X_meta_model_train. Got={len(self.X_meta_model_train.columns)}")
        else:
            if not (len(self.X_meta_model_train.columns) == len(self.models)):
                raise ValueError(f"Expected {len(self.models)} columns in "
                                 f"X_meta_model_train. Got={len(self.X_meta_model_train.columns)}")

    def fit(self, X_meta_scaled:pd.DataFrame or None = None, y_meta_scaled:pd.Series or None = None):
        '''
            Fit base models `cv_folds` times shifting the forecast horizon forward by `delta_between_model_fits`
            until the end of the dataset is reached.
            Use the forecasts by base models and actual targets to train a meta or ensemble model.
            The model is retrained `meta_cv_folds` times to allow for average error estimation.
            In the end both base models and meta model are trained using all available data.
        '''

        # fit base models 'folds' time and store their predictions for unseen data
        self.fit_base_models_for_all_folds(X_meta_scaled)

        # train and evaluate ensemble model for different amount of training data (to get error metrics)
        if not self.meta_cv_folds is None:
            if self.meta_cv_folds > self.base_cv_folds - 1:
                raise ValueError(f"Number of cv_runs for meta_model cannot be less than number of base model evaluations. "
                                 f"Given cv_runs={self.meta_cv_folds} while base models were evaluated {self.base_cv_folds} times")
            elif self.meta_cv_folds == 1: cv_folds = [self.cutoffs[-1]]
            else: cv_folds = self.cutoffs[-self.meta_cv_folds:]

            for idx, cutoff in enumerate(cv_folds):
                train_mask = self.X_meta_model_train.index <= cutoff
                test_mask = (self.X_meta_model_train.index > cutoff) & \
                            (self.X_meta_model_train.index <= cutoff + pd.Timedelta(hours=self.horizon))

                X_train, X_test = self.X_meta_model_train[train_mask], self.X_meta_model_train[test_mask]
                y_train, y_test = self.y_meta_model_train[train_mask], self.y_meta_model_train[test_mask]

                self.meta_model.fit(X_train, y_train)

                self.all_forecasts[self.name][idx] = copy.deepcopy( self.meta_model.forecast_window( X_test ) )
                self.all_forecasts[self.name][idx][f'{self.target}_actual'] = y_test

                # compute error metrics
                self.all_model_metrics[self.name][idx] = compute_error_metrics(
                    self.target, self.all_forecasts[self.name][idx].apply(self.dss[0].inv_transform_target_series)
                )

                print(f"Fold={idx}/{len(cv_folds)} cutoff={cutoff} | meta {self.name} "
                      f"| RMSE={self.all_model_metrics[self.name][idx]['rmse']:.1f} "
                      f"sMAPE={self.all_model_metrics[self.name][idx]['smape']:.2f} "
                      f"| X_train={X_train.shape} y_train={y_train.shape}")

        # train base forecasters on the entire dataset
        print(f"Training base forecasters on the entire dataset")
        for i in range(len(self.models)):
            X_for_model: pd.DataFrame = self.dss[i].get_exogenous_trasnformed()
            y_for_model: pd.Series = self.dss[i].get_target_transformed()
            self.models[i].fit(X_for_model, y_for_model)

        # fit the meta-model on the collected dataset
        print(f"Training meta_model on the all available data "
              f"X_train={self.X_meta_model_train.shape} "
              f"y_train={pd.Series(self.y_meta_model_train).shape}")
        self.meta_model.fit(self.X_meta_model_train, pd.Series(self.y_meta_model_train))
    def forecast_window(self, X_meta_scaled:pd.DataFrame or None)->pd.DataFrame:
        '''
            First forecast every base model for their respective X_scaled and then use their predictions
            as well as provided X_meta_scaled to generate a prediction with meta-model
        '''
        forecasts_df = pd.DataFrame(index=X_meta_scaled.index)
        # collect forecasts from base models
        for i in range(len(self.models)):
            name = self.models[i].name
            X_for_model: pd.DataFrame = self.dss[i].get_forecast_exogenous()
            y_model_forecast = self.models[i].forecast_window( X_for_model )
            self.all_forecasts[name][-1] = copy.deepcopy( y_model_forecast )
            for key in self.features_from_base_models:
                forecasts_df[f'base_model_{i}_{key}'] = \
                    copy.deepcopy( self.all_forecasts[name][-1][f'{self.target}_{key}'] )

        # append meta exogenous quantities
        if not X_meta_scaled is None:
            forecasts_df = forecasts_df.merge(X_meta_scaled, left_index=True, right_index=True)

        # predict with model
        last_forecast = self.meta_model.forecast_window( forecasts_df )
        self.all_forecasts[self.name][-1] = copy.deepcopy(last_forecast)
        return last_forecast
