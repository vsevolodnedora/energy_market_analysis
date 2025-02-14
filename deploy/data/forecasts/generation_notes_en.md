
Our __week-ahead__ forecast has average RMSE of __4214__.  
SMARD __day-ahead__ forecast has average accuracy of __3494__. 
    
| TSO/Region   | Train Date   |   N Features | Best Model            |   RMSE |   TSO RMSE |
|:-------------|:-------------|-------------:|:----------------------|-------:|-----------:|
| Amprion      | 2025-02-13   |           44 | MultiTargetCatBoost   |   1833 |       1632 |
| 50Hertz      | 2025-02-13   |           32 | MultiTargetCatBoost   |   1885 |       4086 |
| TenneT       | 2025-02-13   |           37 | MultiTargetElasticNet |   1987 |       1930 |
| TransnetBW   | 2025-02-13   |           70 | MultiTargetLGBM       |    806 |       1423 |