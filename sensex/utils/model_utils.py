import numpy as np
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from matplotlib import pyplot as plt


def evaluate_model(model: Model, scaler: MinMaxScaler, x_test: TimeseriesGenerator, y_test: TimeseriesGenerator) -> dict[str, float]:
    """
    Evaluate a trained LSTM model and metrics.

    Parameters
    ----------
    model : Model
    scaler : MinMaxScaler
        Scaler used during training for inverse transformation.
    x_test : TimeseriesGenerator
        Test input data (shape: [samples, timesteps, features])
    y_test : TimeseriesGenerator
        Validation target data (shape: [samples, 1])

    Returns
    -------
    metrics : dict[str, float]
        Metrics - RMSE, MSE, MAE, R2_score.
    """
    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)

    metrics = {
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    return metrics


def plot_history(history: History) -> None:
    """
    Plot training and validation history.
    Parameters
    ----------
    history : History
        Training history of the model.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.show()


def plot_predictions(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot actual vs. predicted values.

    Parameters
    ----------
    y_test : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values corresponding to `y_test`.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax.plot(y_test, label='Actual')
    ax.plot(y_pred, label='Predicted')
    ax.set_title('Validation Set: Actual vs Predicted')
    ax.set_xlabel('Sensex Points')
    ax.set_ylabel('Day')
    ax.legend()
    plt.show()
