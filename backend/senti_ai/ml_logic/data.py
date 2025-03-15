import pandas as pd
import requests
import yfinance as yf
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
from datetime import datetime, timedelta
from senti_ai.interface.notification import send_pushover_notification
from senti_ai.ml_logic.preprocessor import calculate_indicators
from senti_ai.params import *

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
    pushover_user_keys = [AHMET_USER_KEY, KSENIA_USER_KEY, BILL_USER_KEY, PHILIP_USER_KEY]
    for key in pushover_user_keys:
        send_pushover_notification(f'Data saved to bigquery, with shape {data.shape}', user_key=key)

def load_data_from_bq(
        gcp_project:str,
        bq_dataset:str,
        table: str,
    ) -> pd.DataFrame:

    query = f"""SELECT * FROM `{gcp_project}`.{bq_dataset}.{table}"""
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()
    df = df.sort_values('date', ascending=False)

    return df

def combine_all_data_and_save():
    pushover_user_keys = [AHMET_USER_KEY, KSENIA_USER_KEY, BILL_USER_KEY, PHILIP_USER_KEY]

    df_small = fetch_daily_data()

    print(f"✅ df_small: {df_small}")
    if df_small is None:
        print(f"❌ Combine failed because daily fetch did not return a correct dataframe.")
        return None

    df_small["BTC_Close_MA30"] = None
    df_small["NASDAQ_Close_MA30"] = None
    df_small["BTC_Volatility"] = None
    df_small["NASDAQ_Volatility"] = None

    #print(f"✅ Fetch finished")
    df_historical = load_data_from_bq(GCP_PROJECT_AHMET,BQ_DATASET,"raw")
    #print(f"✅ Load finished")
    #print(f"✅ {BQ_DATASET}")

    df_combined = pd.concat([df_small, df_historical], ignore_index=True)
    print(f"✅ Before calculate indicators method:")
    print(df_combined[['BTC_Volatility','date']])
    df_combined = calculate_indicators(df_combined)
    print(f"✅ After calculate indicators method:")
    print(df_combined[['BTC_Volatility','date']])
    #print(f"✅ {df_combined.head(5)}")

    has_nan = df_combined.isna().any()

    if (df_combined.shape[1] == 17) & (not has_nan.any()):

        #print(df_combined.head(5))
        #print(df_combined.info())
        #print(f"✅ {df_combined.head(5)}")
        try:
            save_data_to_bq(df_combined,GCP_PROJECT_AHMET,BQ_DATASET,"raw", truncate=True)
        except:
            [send_pushover_notification('Daily save to BQ failed.', user_key=key) for key in pushover_user_keys]
        else:
            [send_pushover_notification('Daily save to BQ succeeded.', user_key=key) for key in pushover_user_keys]

        try:
            df_combined.to_csv(f"./all_data_{datetime.today().date()}.csv")
        except:
            [send_pushover_notification('Daily save to CSV succeeded.', user_key=key) for key in pushover_user_keys]
        else:
            [send_pushover_notification('Daily save to CSV succeeded.', user_key=key) for key in pushover_user_keys]

    return None

def fetch_daily_data():
    print(f"✅ Fetch started")
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
    print(f"✅ CNN Fear and Greed collected")

    #### Crypto Fear and Greed ## START
    url = "https://api.alternative.me/fng/?limit=0&date_format=us"
    response = requests.get(url)
    data = response.json()
    crypto_fear_greed_df = data['data']
    #### Crypto Fear and Greed ## END
    print(f"✅ Crypto Fear and Greed collected")

    #### BTC and NASDAQ from yfinance ## START

    today = datetime.today().date()
    tomorrow = today + timedelta(days=1)

    tickers = ["BTC-USD", "^IXIC"]
    df_combined = yf.download(
    tickers,
    start=today,
    end=tomorrow,
    progress=False
    )

    df_today = df_combined.iloc[-1:]
    btc_close_today = df_today['Close']['BTC-USD']
    nasdaq_close_today = df_today['Close']['^IXIC']

    #### BTC and NASDAQ from yfinance ## END


    #### Construct final dataframe START

    df_result = pd.DataFrame({
        "date": datetime.today().date(),
        "BTC_Close":df_today['Close']['BTC-USD'],
        "BTC_High":df_today['High']['BTC-USD'],
        "BTC_Low":df_today['Low']['BTC-USD'],
        "BTC_Open":df_today['Open']['BTC-USD'],
        "BTC_Volume":df_today['Volume']['BTC-USD'],
        "BTC_sentiment_score":int(crypto_fear_greed_df[0]['value']),
        "NASDAQ_Close":df_today['Close']['^IXIC'],
        "NASDAQ_High":df_today['High']['^IXIC'],
        "NASDAQ_Low":df_today['Low']['^IXIC'],
        "NASDAQ_Open":df_today['Open']['^IXIC'],
        "NASDAQ_Volume":df_today['Volume']['^IXIC'],
        "NASDAQ_sentiment_score":stock_fear_greed_df,
        }, index=[0])

    # print(f"✅ Putting together mock df_result")
    # df_result = pd.DataFrame({
    #     "date": datetime.today().date(),
    #     "BTC_Close":81461.62,
    #     "BTC_High":81961.26,
    #     "BTC_Low":76808.10,
    #     "BTC_Open":78582.16,
    #     "BTC_Volume":58110119936,
    #     "BTC_sentiment_score":24,
    #     "NASDAQ_Close":17436.10,
    #     "NASDAQ_High":17687.40,
    #     "NASDAQ_Low":17238.24,
    #     "NASDAQ_Open":17443.09,
    #     "NASDAQ_Volume":9177320000,
    #     "NASDAQ_sentiment_score":13,
    #     }, index=[0])

    print(f"✅ df_result.shape[0] should be 1: {df_result.shape[0]}")
    print(f"✅ df_result.shape[1] should be 13: {df_result.shape[1]}")

    pushover_user_keys = [AHMET_USER_KEY, KSENIA_USER_KEY, BILL_USER_KEY, PHILIP_USER_KEY]

    has_nan = df_result.isna().any()

    if (df_result.shape[1] == 13) & (not has_nan.any()) & (df_result.shape[0] == 1):
        #for key in pushover_user_keys:
        #    send_pushover_notification('Daily fetching successful.', user_key=key)
        #df_result.index = [1]
        print(f"✅ Returning df_result with shape:{df_result.shape}")
        return df_result

    else:
        for key in pushover_user_keys:
            send_pushover_notification('Daily fetching failed.', user_key=key)
        return None
