from sensex.core.train import *
from sensex.core.preprocess import *
from sensex.utils.io_utils import load_data
from sensex.utils import config
import os


# base path for sensex points prediction
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_PATH, 'data', 'main.csv')
DATA_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'data.yaml')
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'model.yaml')

# loads the merged dataframe
df = load_data(DATA_PATH)

# load data and model configs
data_config = config.load_config(DATA_CONFIG_PATH)
model_config = config.load_config(MODEL_CONFIG_PATH)

# preprocessing for deep learning modeling
# splitting data to train, test
features, target = df, df[:, -1]
x_train, x_test, y_train, y_test = split_data(features, target)

# scaling features and target
