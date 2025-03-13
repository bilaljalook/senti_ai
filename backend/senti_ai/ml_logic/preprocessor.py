import pandas as pd

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling standard deviation and moving averages for Nasdaq and BTC.
    """
    df['NASDAQ_Volatility'] = df['NASDAQ_Close'].rolling(window=30).std()
    df['BTC_Volatility'] = df['BTC_Close'].rolling(window=30).std()
    df['BTC_Close_MA'] = df['BTC_Close'].rolling(window=30).mean()
    df['NASDAQ_Close_MA'] = df['NASDAQ_Close'].rolling(window=30).mean()
    return df
