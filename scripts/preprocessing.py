from sensex.core.preprocess import *
from sensex.utils import io_utils
from sensex.utils import config
import os


# base path for SensexPointsPrediction
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'data.yaml')
DATA_SAVE_PATH = os.path.join(BASE_PATH, 'data', 'main.csv')

# load data configs
data_config = config.load_config(DATA_CONFIG_PATH)

dfs = []
# special handling for 'points' and 'us_inr' datasets, which require date parsing
for df, path in data_config['paths'].items():
    if df == 'points':
        dfs.append(io_utils.load_sensex_with_date(os.path.join(BASE_PATH, path)))
    elif df == 'us_inr':
        dfs.append(io_utils.load_us_inr_with_date(os.path.join(BASE_PATH, path)))
    else:
        dfs.append(io_utils.load_data(os.path.join(BASE_PATH, path)))

# merging all the dataframes
df = consolidate_data(dfs)

# rename columns and adjust their order to standardize the structure of the final dataframe
df = rename_and_rearrange_columns(df)

# saving dataframe
io_utils.save_data(df, DATA_SAVE_PATH)
print('Data saved successfully')
