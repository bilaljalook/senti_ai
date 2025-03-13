import pandas as pd

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling standard deviation and moving averages for Nasdaq and BTC for the latest row.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'NASDAQ_Close' and 'BTC_Close' columns.

    Returns:
    pd.DataFrame: DataFrame with added columns for volatility and moving averages.
    """
    last_row_index = df.index[-1]

    df.loc[last_row_index, 'NASDAQ_Volatility'] = df['NASDAQ_Close'].rolling(window=30).std().iloc[-1]
    df.loc[last_row_index, 'BTC_Volatility'] = df['BTC_Close'].rolling(window=30).std().iloc[-1]
    df.loc[last_row_index, 'BTC_Close_MA30'] = df['BTC_Close'].rolling(window=30).mean().iloc[-1]
    df.loc[last_row_index, 'NASDAQ_Close_MA30'] = df['NASDAQ_Close'].rolling(window=30).mean().iloc[-1]

    return df
