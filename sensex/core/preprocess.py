import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Any


def consolidate_data(dfs: tuple[pd.DataFrame, ...]) -> pd.DataFrame:
    """
    Merge multiple DataFrames into a single consolidated DataFrame.

    Parameters
    ----------
    dfs : tuple of pd.DataFrames
        Order of dataframes - points, us_inr, gdp, inflation, interest_rate, leap_election, dow_jones, gold, oil.

    Returns
    -------
    df : pd.DataFrame
    """
    if len(dfs) != 9:
        raise ValueError(f"Expected 9 DataFrames, but got {len(dfs)}")

    points, us_inr, gdp, inflation, interest_rate, leap_election, dow_jones, gold, oil = dfs

    df = pd.merge(points, us_inr, on='date', how='inner', suffixes=('_ssx', '_usinr'))
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    df = df.merge(
        gdp, on='year', how='inner'
    ).merge(
        inflation, on='year', how='inner'
    ).merge(
        interest_rate, on='year', how='inner', suffixes=('_inf', '_intr')
    ).merge(
        leap_election, on='year', how='inner'
    ).merge(
        dow_jones, on='year', how='inner'
    ).merge(
        gold, on='year', how='inner'
    ).merge(
        oil, on='year', how='inner'
    )
    return df


def rename_and_rearrange_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns of DataFrame and rearrange them as features first and target variable last.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
    """
    df.columns = [
        'date', 'points', 'usinr', 'year', 'month', 'day', 'gdp', 'inflation',
        'interest', 'leap', 'election', 'dow_jones', 'gold', 'oil'
    ]
    df = df.iloc[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 13, 1]]
    df.set_index('date')
    return df


def split_data(features: pd.DataFrame, target: pd.Series, test_size: float = 0.2) -> tuple[np.ndarray, ...]:
    """
    Splits features and target into training and testing sets without shuffling.

    Parameters
    ----------
    features : pd.DataFrame
        Input feature set.
    target : pd.DataFrame
        Target variable.
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.

    Returns
    -------
    x_train : np.ndarray
    x_test : np.ndarray
    y_train : np.ndarray
    y_test : np.ndarray
    """
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)
    return x_train, x_test, y_train, y_test


def scale_features(x_train: np.ndarray, x_test: np.ndarray) -> tuple[Any, ...]:
    """
    Scales features using MinMaxScaler.

    Parameters
    ----------
    x_train : np.ndarray
        Train features before scaling.
    x_test: np.ndarray
        Test features before scaling.

    Returns
    -------
    x_train : np.ndarray
    x_test : np.ndarray
    feature_scaler : MinMaxScaler
    """
    feature_scaler = MinMaxScaler()
    x_train = feature_scaler.fit_transform(X=x_train)
    x_test = feature_scaler.transform(X=x_test)
    return x_train, x_test, feature_scaler


def scale_target(y_train: np.ndarray, y_test: np.ndarray) -> tuple[Any, ...]:
    """
    Scales target variable using MinMaxScaler.

    Parameters
    ----------
    y_train : np.ndarray
        Train target variable before scaling.
    y_test : np.ndarray
        Test target variable before scaling.

    Returns
    -------
    y_train : np.ndarray
    y_test : np.ndarray
    target_scaler : MinMaxScaler
    """
    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(X=y_train)
    y_test = target_scaler.transform(X=y_test)
    return y_train, y_test, target_scaler


def convert_to_time_series(x: np.ndarray,y: np.ndarray, length: int, batch_size: int) -> TimeseriesGenerator:
    """
    Converts data into time series.

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    length : int
        Length of the time series window.
    batch_size : int

    Returns
    -------
    time_series_gen : TimeseriesGenerator
    """
    time_series_gen = TimeseriesGenerator(x, y, length=length, sampling_rate=1, batch_size=batch_size)
    return time_series_gen


def get_prev_window_df(df: pd.DataFrame, window_length: int) -> pd.DataFrame:
    """
    Extracts the last window from dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    window_length : int
        Length of the time series window.

    Returns
    -------
    df : pd.DataFrame
    """
    return df.iloc[-window_length - 1:, :]
