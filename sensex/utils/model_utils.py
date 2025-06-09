import numpy as np
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from matplotlib import pyplot as plt


def evaluate_model(model: Model, scaler: MinMaxScaler, val_data: TimeseriesGenerator) -> None:
    """
    Evaluate a trained LSTM model and print metrics.

    Parameters
    ----------
    model : Model
    scaler : MinMaxScaler
        Scaler used during training for inverse transformation.
    val_data : TimeseriesGenerator
        Test data used for evaluation.

    Returns
    -------
    None
    """
    y_pred = model.predict(val_data)
    y_pred = scaler.inverse_transform(y_pred)
    y_true = []
    for i in range(len(val_data)):
        _, y_batch = val_data[i]
        y_true.append(y_batch)
    y_true = np.concatenate(y_true)

    print(f'RMSE: {root_mean_squared_error(y_true, y_pred)}')
    print(f'MSE: {mean_squared_error(y_true, y_pred)}')
    print(f'MAE: {mean_absolute_error(y_true, y_pred)}')
    print(f'R2: {r2_score(y_true, y_pred)}')


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
