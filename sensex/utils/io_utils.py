from pandas import DataFrame, read_csv
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras import Model


def load_data(df_path: str) -> DataFrame:
    """
    Loads dataframe from df_path.

    Parameters
    ----------
    df_path : str
        Path to load dataframe.

    Returns
    -------
    pd.DataFrame
    """
    df = read_csv(df_path)
    return df


def save_data(df: DataFrame, df_path: str) -> None:
    """
    Saves dataframe to df_path.

    Parameters
    ----------
    df : pd.dataframe.
    df_path : str
        Path to save dataframe.

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
    keras.Model
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
