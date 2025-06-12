import streamlit as st
import plotly.express as px
from sensex.utils import config
from sensex.utils.logger import get_logger
from sensex.utils.io_utils import *
from sensex.core.preprocess import *
from sensex.apis.predict import get_next_month_forecast
import os


def get_forecast() -> pd.DataFrame:
    """
    Called by streamlit app to get next month forecast.

    Returns
    -------
    next_month_data : pd.DataFrame
    """
    # base path for SensexPointsPrediction
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_LOAD_PATH = os.path.join(BASE_PATH, 'data', 'main.csv')
    MODEL_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'model.yaml')
    DATA_CONFIG_PATH = os.path.join(BASE_PATH, 'configs', 'data.yaml')
    LOG_PATH = os.path.join(BASE_PATH, 'logs', 'app.log')

    # initiate logger
    app_logger = get_logger(name="sensex", log_file=LOG_PATH)

    # load model configs
    model_config = config.load_config(MODEL_CONFIG_PATH)
    data_config = config.load_config(DATA_CONFIG_PATH)
    # load paths for scalers and model
    feature_scaler_path = model_config['paths']['feature_scaler']
    target_scaler_path = model_config['paths']['target_scaler']
    model_path = model_config['paths']['model']
    # load other parameters
    length = data_config['preprocess']['length']
    batch_size = data_config['preprocess']['batch_size']
    window_length = model_config['model_hps']['window_length']
    BIAS = model_config['bias']
    app_logger.info("loaded config")

    # load scalers and model
    feature_scaler = load_scaler(os.path.join(str(BASE_PATH), feature_scaler_path))
    target_scaler = load_scaler(os.path.join(str(BASE_PATH), target_scaler_path))
    model = load_model(os.path.join(str(BASE_PATH), model_path))
    app_logger.info("loaded model and scalers")

    # load data
    df = load_data(DATA_LOAD_PATH)
    app_logger.info("loaded data")
    # extract the latest window
    window = get_prev_window_df(df, window_length)
    # splitting data to features and target
    features, target = window.iloc[:, 1:], window.iloc[:, -1]
    # scale features and target
    features, target = feature_scaler.transform(features.values), target_scaler.transform(target.values.reshape(-1, 1))
    app_logger.info("transformed data")
    # converting numpy arrays to time series
    data_to_forecast = convert_to_time_series(features, target, length, batch_size)
    app_logger.info("transformed data to time series")

    next_month_data = get_next_month_forecast(window, window_length, target_scaler, model, data_to_forecast)
    next_month_data['points'] = next_month_data['points'] - BIAS
    app_logger.info("next month data forecasted")

    return next_month_data


if __name__ == "__main__":
    # streamlit app title
    st.title("Sensex Prediction for May 2024")
    # on button click
    if st.button("Predict"):
        st.header("Prediction")
        fig = px.line(get_forecast(), y='points')
        st.plotly_chart(fig)
