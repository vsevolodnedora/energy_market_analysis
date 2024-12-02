'''
    Main file for data collection
    See notebooks/SAME_FILE_NAME.ipynb for intercative / manual option
'''
import os.path
from datetime import datetime, timedelta
from glob import glob
import pandas as pd
try: pd.set_option('future.no_silent_downcasting', True)
except: print("Failed to set future.no_silent_downcasting=True")
import numpy as np
import gc

from .collect_data_smard import update_smard_from_api
from .collect_data_openmeteo import update_openmeteo_from_api
from .collect_data_epexspot import update_epexspot_from_files
from .locations import locations

def update_database(today:pd.Timestamp,data_dir:str):

    update_smard_from_api(today, data_dir + 'smard/')

    update_epexspot_from_files(today, data_dir + 'epexspot/')

    update_openmeteo_from_api(today, data_dir + 'openmeteo/')


if __name__ == '__main__':
    # TODO add tests
    pass