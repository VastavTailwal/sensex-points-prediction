from pandas import DataFrame, read_csv
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras import Model
import joblib


def load_data(df_path: str) -> DataFrame:
    """
    Loads dataframe from df_path.

    Parameters
    ----------
    df_path : str
        Path to load the dataframe.

    Returns
    -------
    df : pd.DataFrame
    """
    df = read_csv(df_path)
    return df


def load_sensex_with_date(df_path: str) -> DataFrame:
    """
    Loads dataframe from Sensex df_path.

    Parameters
    ----------
    df_path : str
        Path to load the Sensex dataframe.

    Returns
    -------
    df : pd.DataFrame
    """
    df = read_csv(df_path, parse_dates=[0], date_format='%d-%b-%y')
    return df


def load_us_inr_with_date(df_path: str) -> DataFrame:
    """
    Loads dataframe from US/INR df_path.

    Parameters
    ----------
    df_path : str
        Path to load the US/INR dataframe.

    Returns
    -------
    df : pd.DataFrame
    """
    df = read_csv(df_path, parse_dates=[0], date_format='%d-%m-%y')
    return df


def save_data(df: DataFrame, df_path: str) -> None:
    """
    Saves dataframe to df_path.

    Parameters
    ----------
    df : pd.dataframe.
    df_path : str
        Path to save the dataframe.

    Returns
    -------
    None
    """
    df.to_csv(df_path, index=False)


def load_model(model_path: str) -> Model:
    """
    Loads model from model_path.

    Parameters
    ----------
    model_path : str
        Path to load the model.

    Returns
    -------
    model : keras.Model
    """
    model = load_keras_model(model_path)
    return model


def save_model(model_path: str, model: Model) -> None:
    """
    Saves model to model_path.

    Parameters
    ----------
    model_path : str
        Path to save model.
    model : keras.Model

    Returns
    -------
    None
    """
    model.save(model_path)


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """
    Loads scaler from scaler_path.

    Parameters
    ----------
    scaler_path : str
        Path to load the scaler.

    Returns
    -------
    scaler : MinMaxScaler
    """
    scaler = joblib.load(scaler_path)
    return scaler

def save_scaler(scaler_path: str, scaler: MinMaxScaler) -> None:
    """
    Saves scaler to scaler_path.

    Parameters
    ----------
    scaler_path : str
        Path to save the scaler.
    scaler : MinMaxScaler

    Returns
    -------
    None
    """
    joblib.dump(scaler, scaler_path)
