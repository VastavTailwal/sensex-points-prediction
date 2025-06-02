from pandas import DataFrame, read_csv
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras import Model


def load_data(df_path: str) -> DataFrame:
    """
    Loads dataframe from df_path.

    Params
    df_path: path to load dataframe

    Returns
    df: pandas dataframe
    """
    df = read_csv(df_path)
    return df


def save_data(df: DataFrame, df_path: str) -> None:
    """
    Saves dataframe to df_path.

    Params
    df: pandas dataframe
    df_path: path to save dataframe
    """
    df.to_csv(df_path, index=False)


def load_model(model_path: str) -> Model:
    """
    Loads model from model_path.

    Params
    model_path: path to load model

    Returns
    model: loaded model
    """
    model = load_keras_model(model_path)
    return model


def save_model(model_path: str, model: Model) -> None:
    """
    Saves model to model_path.

    Params
    model_path: path to save model
    model: trained model
    """
    model.save(model_path)
