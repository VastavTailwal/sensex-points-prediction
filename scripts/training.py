from sensex.core.train import *
from sensex.core.preprocess import *
from sensex.utils.io_utils import *
from sensex.utils.model_utils import evaluate_model
from sensex.utils import config
import os


# base path for SensexPointsPrediction
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_PATH = os.path.join(BASE_PATH, 'data', 'main.csv')
DATA_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'data.yaml')
MODEL_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'model.yaml')

# loads the merged dataframe
df = load_data(DATA_PATH)

# load data and model configs
data_config = config.load_config(DATA_CONFIG_PATH)
model_config = config.load_config(MODEL_CONFIG_PATH)

# loading model hyperparameters
length = data_config['preprocess']['length']
batch_size = data_config['preprocess']['batch_size']
model_hps = model_config['model_hps']
lstm_hps = model_config['lstm_hps']
dense_hps = model_config['dense_hps']
trainer_hps = model_config['trainer_hps']
early_stopping_hps = model_config['early_stopping_hps']

# model and scalers save paths
MODEL_SAVE_PATH = os.path.join(BASE_PATH, model_config['paths']['model'])
FEATURE_SCALER_SAVE_PATH = os.path.join(BASE_PATH, model_config['paths']['feature_scaler'])
TARGET_SCALER_SAVE_PATH = os.path.join(BASE_PATH, model_config['paths']['target_scaler'])

# splitting data to train, test
features, target = df.iloc[:, 1:], df.iloc[:, -1]
x_train, x_test, y_train, y_test = split_data(features, target)

# scaling features and target
x_train, x_test, feature_scaler = scale_features(x_train, x_test)
y_train, y_test, target_scaler = scale_target(y_train, y_test)

# converting numpy arrays to time series
training_data = convert_to_time_series(x_train, y_train, length, batch_size)
val_data = convert_to_time_series(x_test, y_test, length, batch_size)

# training model
model = get_model(model_hps, lstm_hps, dense_hps, trainer_hps)
history = train_model(model, training_data, val_data, trainer_hps, early_stopping_hps)

# saving artifacts
save_model(MODEL_SAVE_PATH, model)
save_scaler(FEATURE_SCALER_SAVE_PATH, feature_scaler)
save_scaler(TARGET_SCALER_SAVE_PATH, target_scaler)

# model evaluation
evaluate_model(model, target_scaler, val_data)
