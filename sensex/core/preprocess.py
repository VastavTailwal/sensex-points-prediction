from pandas import DataFrame, merge


def consolidate_data(dfs: tuple[DataFrame]) -> DataFrame:
    """
    Merge multiple DataFrames into a single consolidated DataFrame.

    Params:
    dataframes: tuple of dataframes

    Returns:
    df: single consolidated dataframe
    """
    if len(dfs) != 9:
        raise ValueError(f"Expected 9 DataFrames, but got {len(dfs)}")

    points, us_inr, gdp, inflation, interest_rate, leap_election, dow_jones, gold, oil = dfs

    df = merge(points, us_inr, on='date', how='inner', suffixes=('_ssx', '_usinr'))
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


def rename_and_rearrange_columns(df: DataFrame) -> DataFrame:
    """
    Rename columns of DataFrame and rearrange them as features first and target variable last.

    Params
    df: DataFrame

    Returns
    df: DataFrame after renaming and rearranging columns
    """
    df.columns = [
        'date', 'points', 'usinr', 'year', 'month', 'day', 'gdp', 'inflation',
        'interest', 'leap', 'election', 'dow_jones', 'gold', 'oil'
    ]
    df = df.iloc[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 13, 1]]
    df.set_index('date')
    return df
