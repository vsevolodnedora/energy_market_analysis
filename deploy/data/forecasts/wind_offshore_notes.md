
<h3>Summary</h3>
Our _week-ahead_ Offshore Wind Generation forecast has RMSE of __1022__.  
SMARD _day-ahead_ forecast has average accuracy of __834__. 
    
<ul>
    <li>Key properties of the forecasting pipeline</li>
    <ul>
        <li>raw and/or engineered weather features</li>
        <li>multiple windfarm locations</li>
        <li>hyperparameter for models and features (tuned with Optuna)</li>
        <li>multi-step single target forecasting (168 timesteps)</li>
        <li>ensemble models trained on OOS forecasts</li>
        <li>total wind power is obtained as a sum of contibutions from TSO regions</li>
    </ul>
</ul>
    
<h3> Forecast for each TSO </h3>
    
| TSO/Region   | Train Date   |   N Features | Best Model   |   Average RMSE |
|:-------------|:-------------|-------------:|:-------------|---------------:|
| TenneT       | 2024-12-29   |           83 | XGBoost      |            869 |
| 50Hz         | 2024-12-29   |            7 | Ensemble     |            277 |