from fastapi import FastAPI
import pandas as pd
from senti_ai.ml_logic.data import load_data_from_bq
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/get_historical")
def get_historical(
        gcp_project:str,
        bq_dataset:str,
        table: str,
    ) -> pd.DataFrame:
    df = load_data_from_bq(gcp_project, bq_dataset, table)
    return df

@app.get("/get_pred")
def get_pred() -> pd.DataFrame:

    ## MOCK PREDICTIONS START
    tomorrow = datetime.now() + timedelta(days=1)
    start_date = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    date_range = [start_date + timedelta(days=i) for i in range(30)]
    base_price = 68000
    np.random.seed(42)
    daily_changes = np.random.normal(0, 0.02, 30)
    cumulative_changes = np.cumprod(1 + daily_changes)
    prices = base_price * cumulative_changes
    prices = np.round(prices, 2)
    df = pd.DataFrame({
        'date': date_range,
        'close': prices
    })
    ## MOCK PREDICTIONS END

    return df
