#!/bin/bash

# Define the Conda environment name
CONDA_ENV_NAME="ds11"

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Move to the project directory if needed
# cd /path/to/your/project

# Run the update scripts
python update_database.py all update_entsoe hourly
python update_database.py DE update_smard hourly
python update_database.py all update_epexspot hourly
python update_database.py all update_openmeteo_windfarms_offshore hourly
python update_database.py all update_openmeteo_windfarms_onshore hourly
python update_database.py all update_openmeteo_solarfarms hourly
python update_database.py all update_openmeteo_cities hourly

# Run update forecasts
python update_forecasts.py DE all all forecast hourly
python update_forecasts.py DE all all summarize hourly

# publish result to webpage

# Deactivate the Conda environment
conda deactivate
