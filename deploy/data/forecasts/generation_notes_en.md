
Our __week-ahead__ forecast has average RMSE of __4079__.  
SMARD __day-ahead__ forecast has average accuracy of __3538__. 
    
| TSO/Region   | Train Date   |   N Features | Best Model            |   RMSE |   TSO RMSE |
|:-------------|:-------------|-------------:|:----------------------|-------:|-----------:|
| Amprion      | 2025-02-13   |           44 | MultiTargetCatBoost   |   1764 |       1632 |
| 50Hertz      | 2025-02-13   |           32 | MultiTargetCatBoost   |   1885 |       4083 |
| TenneT       | 2025-02-13   |           33 | MultiTargetElasticNet |   1940 |       1930 |
| TransnetBW   | 2025-02-13   |           70 | MultiTargetLGBM       |    822 |       1390 |