
---

The project was initiated as a _personal portfolio project_ in the Summer of 2024 with a goal to gain experience working with time-series data, forecasting, API and data engineering, machine learning and predictive analytics, MLOps and DataOps and front-end development (HTML, JavaScript).  

__Proof of Concept (POC)__, a multi-step electricity load forecasting model, was released on [Kaggle](https://www.kaggle.com/code/vsevolodnedora/multi-output-electrical-load-forecasting) in September 2024.  

__Minimum Viable Product (MVP)__ featuring fully automatic data collection, preprocessing, forecasting and serving pipeline as well as front-end deployed on GitHub Pages was released in January 2025.  

### Stage 1: [Data Collection](https://medium.com/@vsevolod.nedora/mlops-electricity-price-forecasting-project-2-ad1012350067)
In this stage I prototyped data collection and scraping scripts and pipelines. 
- About 10 years of hourly data was collected from SMARD, openmeteo and EPEXSPOT. For technical reasons, in addition to APIs, web-scraping techniques were used, including [energy-charts](https://energy-charts.info/?l=en&c=DE) for which I developed a [scraper](https://github.com/vsevolodnedora/energy_charts_collector) that goes through ApexCharts and pulls the data.
- At the same time additional scrapers were developed, e.g., [nordpool_collector](https://github.com/vsevolodnedora/nordpool_collector) and [eex_collector](https://github.com/vsevolodnedora/eex_collector) in case I decide to expand the project to other countries besides Germany.
- Several weather data providors were considered e.g., [openweather](https://openweathermap.org/) and [openmeteo](https://open-meteo.com/). Due to price considerations and ease of collecting past and present forecasts, openmeteo was eventually chosen. 
  - Openmeteo has separate APIs for past data, past forecasts and future forecasts. To assure continuity of data for model training and also for ease of use, all three APIs are used to construct a _continuous_ time-series. 
  - Every time the data collection is called, past forecasts are being updated with actually observed data _to preserve data quality_.

### Stage 2: [Baseline Forecasting](https://medium.com/@vsevolod.nedora/mlops-electricity-price-forecasting-project-2-ad1012350067)
Once I had regularly updated energy and weather data I begun constructing forecasting models. The following models were employed:
- __SARIMAX__ -- classical model for forecasting. It was applied to forecast _electricity load_ using raw weather features as exogneous variables. I quickly discovered, that training is prohibetively slow while performance is moderate at best.
- __LSTM__ -- advanced deep-learning model. Applied to the same data it showed promising results, but hyperparamter and architecture tuning was time consuming and it was hard to assess whether there is ROI for this R&D. 
- __Multi-step XGBoost__ -- gradient boosting model. Again, applied to the same data it showed a very good (visual) performance but training time was extenive. 

After developing __Multi-step XGBoost__ I was confident that the project goal can be achieved and knew how to proceed so I released the POC on [Kaggle](https://www.kaggle.com/code/vsevolodnedora/multi-output-electrical-load-forecasting).

The main roadblock was the absence of reference forecasts. Then, I discovered, that SMARD (and ENTSO-E) provides TSO day-ahead forecasts for multiple quantities including renewable generation and load. It was only natural to compare my forecasts with theirs.  
Thus, I extended the data collection pipeline to include ENTSO-E using their [python API](https://github.com/EnergieID/entsoe-py) and started comparing my week-ahead forecasts with theirs.

After discovering that even extensively trained single model systematically underperforms with respect to TSO forecast, I invested some time in developing an Ensemble model. 
In my implementation, Ensemble model is just a base model, such as XGBoost or Elastic Net but trained on OOS forecasts by other base models. In this way I was aiming to create a model that learns on systematic errors of base models and overcomes them. 
However, I discovered that for an ensemble model to outperform individual models, advance feature engineering and large number of models is required. 

### Stage 3: [Automated Fine-Tuning and Training](https://medium.com/@vsevolod.nedora/building-a-modular-forecasting-framework-fine-tuning-and-predicting-offshore-wind-generation-c668e343f6c2)

Forecasting different targets with various models I needed a systematic way to fine-tune, train and apply the models. 
Moreover, analyzing best practicies and talking to industry experts (virtual coffies) I found that feature sets should also be tailored for each target.  
For instance, wind generation requires features such as wind power density and wind sheer as well as using advance spatial aggregation for a set of windfarms.
However, while industry knowledge provides a general direction, the best set of transformations and aggregations can only be obtained statistically. 
- __Method__: Iterating over various combinations of model and dataset parameters and assessing different models performance metrics to determin the best model and dataset setup.
- __Technology__: Optuna optimization trials. 

At this stage the codebase had to be refactored multiple times to achieve an optimal balance between the size, complexity, extendability and reusability. 
For instance, when optimizing, training and performing inference with multiple models, I found it advantageous to have a base forecasting class with the API shared by all models. 
Furthermore, I found it advantageous to have a single dataset class that, while containing different features, provides a unified access to train and test data.  
Additionally, due to a large number of features I need to forecast, I opted for a task-based pipeline, where the following stages can be scheduled for each forecasting model:
- __finetune__: that runs a Optuna optimization trial sampling model and dataset parameters. The best trial and corresponding parameters are then saved in a specific directory. 
- __train__: that loads the dataset and model parameters from optimization run directory and trains the model on the full dataset. 
- __forecast__: that loads the trained model and dataset attributes (e.g., fitted scalers) and performs inference only. 

This split allows to finetune and train models on premises, where computational cost is not as limited or expensive and perform inferences on the cloud (GitHub actions) to update the current forecasts. 


### Stage 4: Static Webpage Display (MVP Release) [_In progress_]

At this point I had database that was updated daily as well as forecasting pipeline that could perform perform inferences on the cloud using pre-trained models. 
Serving of the data to the downstream user was initially planned to be with a static webpage. Making data presentable and also providing several options to view the data required a degree of HTML and JavaScript engineering due to the limitaitons of static web pages. 
Also, having in mind that various quantities, their forecasts as well as their statistical and market analysis will be eventually added, I opted for a modular webpage structure, where different drop-down sections serve as independent dashboards that can be opened and closed and interacted with independently.
Currently, the webpage is under active development and might change, but the overall structure should remain the same. 

### Future Enhancements [_Planned_]
In the next stages I plan to first forecast more features such as onshore wind and solar energy generations, load and residual load, as well as aggregates of cross-border flows. Forecasting these features I will use other forecasts in a waterfall-like structure. I view it as requried due to entangled nature of energy system and energy market. 
Once all ancillary features are forecasted, electricy prices will forecasted using a variety of methods. At this stage I plan to introduce advance forecasting techniques such as transformer based deep neural networks. 

A separate endavour planned is the automatic market report generation using historic and forecasted data and large language models (most likely called through an API due to computational cost of running an LLM locally).  

If a project grows to the stage where GitHub actions and Pages are no longer sufficient, I plan to move to one of the cloud platforms. I am currently analyzing varios options such as GCP, AWS and Azure. The final choice will depend on cost, complexity and estimated transition time. 

Another future enhancment for the project is addition of other European countries and, thus, more accurate modeling of cross-border interactions. While this is not technically difficult, since most of the codebase can be reused, the computational cost will most likely demand a move to a larger cloud infrastructure. 

Finally, the project direction can be changed by you. Should you be interested in collaboration, additional features or functionalities, -- please reach out. 

