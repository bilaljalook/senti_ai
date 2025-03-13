import pandas as pd
import requests
import yfinance as yf
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime
from senti_ai.interface.notification import send_pushover_notification

def save_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

def load_data_from_bq(
        gcp_project:str,
        bq_dataset:str,
        table: str,
    ) -> pd.DataFrame:

    #query = f"""SELECT * FROM `{gcp_project}`.{bq_dataset}.{table}"""
    query = f"""SELECT * FROM `{gcp_project}`.{bq_dataset}.{table}"""
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    return df


def fetch_daily_data():

    df = []

    #### CNN Fear and Greed ## START
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        current_data = data.get("fear_and_greed", {})
        stock_fear_greed_df = int(current_data.get('score'))

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except ValueError as e:
        print(f"Failed to parse JSON: {e}")
    #### CNN Fear and Greed ## END


    #### Crypto Fear and Greed ## START
    url = "https://api.alternative.me/fng/?limit=1&date_format=us"
    response = requests.get(url)
    data = response.json()
    crypto_fear_greed_df = data['data']
    #crypto_fear_greed_df['value'] = int(crypto_fear_greed_df['value'])
    #crypto_fear_greed_df['timestamp'] = pd.to_datetime(crypto_fear_greed_df['timestamp'])
    #### Crypto Fear and Greed ## END


    #### BTC from yfinance ## START
    df_btc = yf.download(
    "BTC-USD",
    start=datetime.today().date(),
    interval="1d"
    )
    #### BTC from yfinance ## END

    #### Nasdaq ## START
    df_nasdaq = yf.download(
    "^IXIC",
    start=datetime.today().date(),
    interval="1d"
    )
    #### Nasdaq ## END

    #### Construct final dataframe START

    df_result = pd.DataFrame({
        "date": datetime.today().date(),
        "BTC_Close":df_btc['Close']['BTC-USD'],
        "BTC_High":df_btc['High']['BTC-USD'],
        "BTC_Low":df_btc['Low']['BTC-USD'],
        "BTC_Open":df_btc['Open']['BTC-USD'],
        "BTC_Volume":"",
        "BTC_sentiment_score":int(crypto_fear_greed_df[0]['value']),
        "NASDAQ_Close":df_nasdaq["Close"]['^IXIC'],
        "NASDAQ_High":df_nasdaq["Close"]['^IXIC'],
        "NASDAQ_Low":df_nasdaq["Low"]['^IXIC'],
        "NASDAQ_Open":df_nasdaq["Open"]['^IXIC'],
        "NASDAQ_Volume":"",
        "NASDAQ_sentiment_score":stock_fear_greed_df
        })

    df_result.index = [1]

    send_pushover_notification('Daily fetching successful')

    return df_result
