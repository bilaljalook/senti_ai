import pandas as pd

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=False)

    btc_volatility = df['BTC_Close'].rolling(window=30).std()
    nasdaq_volatility = df['NASDAQ_Close'].rolling(window=30).std()
    btc_ma30 = df['BTC_Close'].rolling(window=30).mean()
    nasdaq_ma30 = df['NASDAQ_Close'].rolling(window=30).mean()

    last_row_index = df.index[0]

    df.loc[last_row_index, 'BTC_Volatility'] = btc_volatility.iloc[29] if len(btc_volatility) >= 30 else None
    df.loc[last_row_index, 'NASDAQ_Volatility'] = nasdaq_volatility.iloc[29] if len(nasdaq_volatility) >= 30 else None
    df.loc[last_row_index, 'BTC_Close_MA30'] = btc_ma30.iloc[29] if len(btc_ma30) >= 30 else None
    df.loc[last_row_index, 'NASDAQ_Close_MA30'] = nasdaq_ma30.iloc[29] if len(nasdaq_ma30) >= 30 else None

    return df
