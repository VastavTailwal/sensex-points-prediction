import numpy as np
import pandas as pd
from tensorflow.keras import Model
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler


def predict(
        prev_window: TimeseriesGenerator,
        features: np.ndarray,
        target_scaler: MinMaxScaler,
        model: Model
) -> pd.DataFrame:
    """
    Predicts next day sensex points.

    Parameters
    ----------
    prev_window : TimeseriesGenerator
        Last window of the data.
    features : np.ndarray
        Scaled features to predict.
    target_scaler : MinMaxScaler
        Scaler for target values.
    model : Model

    Returns
    -------
    prediction : pd.DataFrame
    """
    columns = [
        'usinr', 'gdp', 'inflation', 'interest', 'leap',
        'election', 'dow_jones', 'gold', 'oil', 'points'
    ]
    target_pred = target_scaler.inverse_transform(model.predict(prev_window))
    prediction = pd.DataFrame(np.concatenate((features[-1, :-1].reshape((1, 9)), target_pred), axis=1), columns=columns)
    return prediction


def get_next_month_data(
        window: TimeseriesGenerator,
        window_length: int,
        target_scaler: MinMaxScaler,
        model: Model,
        data_to_predict: np.ndarray
) -> TimeseriesGenerator:
    """
    Get next month predicted sensex points.

    Parameters
    ----------
    window : TimeseriesGenerator
        Last window of the data.
    window_length : int
    target_scaler : MinMaxScaler
    model : Model
    data_to_predict : np.ndarray
        Scaled features to predict.
    Returns
    -------
    next_month_data : TimeseriesGenerator
    """
    for _ in range(window_length):
        next_row = predict(window, data_to_predict, target_scaler, model)
        window = pd.concat([window, next_row], ignore_index=True)
        window = window.iloc[1:, :].reset_index(drop=True)

    return window
