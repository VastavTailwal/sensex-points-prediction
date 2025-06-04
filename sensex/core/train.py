from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, History
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from typing import Any


def get_model(
        model_hps: dict[str, Any],
        lstm_hps: list[dict[str, Any]],
        dense_hps: list[dict[str, Any]],
        trainer_hps: dict[str, Any]
) -> Model:
    """
    Builds a model using the specified configurations for LSTM layers,
    dense layers, and other training-specific parameters.

    Parameters
    ----------
    model_hps : dict[str, Any]
        General model hyperparameters such as input shape,
        output size, activation functions, dropout rates, etc.
    lstm_hps : list[dict[str, Any]]
        A list of hyperparameter dictionaries, each defining
        the configuration for an LSTM layer (e.g., units, return_sequences, activation).
    dense_hps : list[dict[str, Any]]
        A list of hyperparameter dictionaries for configuring
        dense layers (e.g., units, activation, use_bias).
    trainer_hps : dict[str, Any]
        Hyperparameters related to model training, such as
        optimizer settings, loss function, and metrics.

    Returns
    -------
    model : Model
        A compiled Keras Model instance ready for training.
    """
    model = Sequential()

    model.add(Input(shape=(model_hps['window_length'], model_hps['n_features'])))

    for hps in lstm_hps:
        model.add(LSTM(
            units=hps['units'],
            activation=hps['activation'],
            return_sequences=hps['return_sequences'])
        )
        model.add(Dropout(hps['dropout']))

    for hps in dense_hps:
        model.add(Dense(hps['units'], activation=hps['activation']))

    model.compile(
        optimizer=trainer_hps['optimizer'],
        loss=trainer_hps['loss']
    )

    return model


def train_model(
        train: TimeseriesGenerator,
        val: TimeseriesGenerator,
        model: Model,
        trainer_hps: dict[str, Any],
        early_stopping_hps: dict[str, Any]
) -> History:
    """
    Trains a Keras model using the provided training and validation data,
    training hyperparameters, and early stopping configuration.

    Parameters
    ----------
    train : TimeseriesGenerator
        Training data.
    val : TimeseriesGenerator
        Validation data.
    model : Model
        A compiled Keras Model instance to be trained.
    trainer_hps : dict[str, Any]
        Model training hyperparameters.
    early_stopping_hps : dict[str, Any]
        Early stopping hyperparameters.

    Returns
    -------
    history : History
        A Keras History object containing training and validation metrics
        recorded at the end of each epoch.
    """
    early_stopping = EarlyStopping(monitor=early_stopping_hps['monitor'], patience=early_stopping_hps['patience'])
    history = model.fit(
        train,
        validation_data=val,
        epochs=trainer_hps['epochs'],
        batch_size=trainer_hps['batch_size'],
        shuffle=trainer_hps['shuffle'],
        callbacks=early_stopping,
        verbose=trainer_hps['verbose']
    )
    return history
