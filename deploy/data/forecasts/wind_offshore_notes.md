
### Total Offshore Wind Power Forecast Performance

Our __week-ahead__ forecast has average RMSE of __1022__ 
(last week it was 1220). 
 
SMARD __day-ahead__ forecast has average accuracy of __834__ 
(last week it was 622). 


### Key properties of the forecasting pipeline
- raw and/or engineered weather features;
- multiple windfarm locations for each TSO region;
- hyperparameter for models and features (tuned with Optuna);
- multi-step single target forecasting (168 timesteps);
- ensemble models trained on OOS forecasts;
- total wind power is a sum of contributions from TSO regions.


### Forecast for each TSO
    
| TSO/Region   | Train Date   |   N Features | Best Model   |   Average RMSE |
|:-------------|:-------------|-------------:|:-------------|---------------:|
| 50Hz         | 2024-12-29   |            7 | Ensemble     |            277 |
| TenneT       | 2024-12-29   |           83 | XGBoost      |            869 |