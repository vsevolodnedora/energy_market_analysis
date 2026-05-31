#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME="ds11"

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Move to the project directory if needed
# cd /path/to/your/project

# Run the update scripts --country --task --frequency
python update_database.py all update_entsoe hourly
python update_database.py DE update_smard hourly
python update_database.py all update_epexspot hourly
python update_database.py all update_openmeteo_windfarms_offshore hourly
python update_database.py all update_openmeteo_windfarms_onshore hourly
python update_database.py all update_openmeteo_solarfarms hourly
python update_database.py all update_openmeteo_cities hourly

# Run update forecasts --country --targets --models --mode --frequency
python update_forecasts.py DE all all forecast hourly
python update_forecasts.py DE all all summarize hourly

# publish result to webpage --country --targets
python publish_data.py DE all

# Deactivate the Conda environment
conda deactivate
