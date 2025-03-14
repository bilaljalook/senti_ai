import streamlit as st
import requests
import streamlit.components.v1 as components
from app2 import load_historical_data_api
from app2 import load_prediction_data_api
from app2 import main
#from app2 import plot_market_chart
#from app2 import plot
#from app2 import load_all_data, plot_combined_chart, plot_prediction_chart

import numpy as np
import pandas as pd
import datetime

st.set_page_config(layout='wide')

#variables
url = "http://web.senti-ai.net:5367/get_historical?gcp_project=supple-folder-448412-n9&bq_dataset=sentiai&table=raw"
bitcoin_chart_url="https://s3.tradingview.com/tv.js"
logo_large='/Users/kk/code/sentiai-09.png'
#all_data = load_all_data()

historical_datasets, _ = load_historical_data_api()
prediction_datasets, _ = load_prediction_data_api()

if historical_datasets is None or prediction_datasets is None:
    st.error("Failed to load data from APIs.")

    # Extract specific datasets
btc_price_historical = historical_datasets.get("BTC Price (Daily)")
btc_fear_greed_historical = historical_datasets.get("Fear & Greed Index (BTC)")
nasdaq_price_historical = historical_datasets.get("Nasdaq Price (Daily)")
nasdaq_fear_greed_historical = historical_datasets.get("Fear & Greed Index (Nasdaq)")
btc_prediction = prediction_datasets.get("BTC Predictions")
nasdaq_prediction = prediction_datasets.get("Nasdaq Predictions")


#layout

st.subheader('Historical overview')

st.logo(logo_large, size='large')

days = st.slider(label='slide me', min_value=1, max_value=365, value=365, step=1)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True, height=400):
        st.subheader("Bitcoin", divider='orange')
        if btc_price_historical is not None:
                # Simplify to single line chart with only "Close" prices
            max_date = btc_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_filtered = btc_price_historical[btc_price_historical.index >= min_date].tail(days)
            st.line_chart(btc_filtered[["Close"]], use_container_width=True)
        else:
            st.error("Bitcoin historical data not available.")
        #bitcoin_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_chart)
        #st.image("https://g.co/finance/BTC-EUR", width=200)
        #components.html(
        """
        <div class="tradingview-widget-container">
            <div id="tradingview_74048"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({
                    "autosize": true,
                    "symbol": "COINBASE:BTCEUR",
                    "interval": "D",
                    "timezone": "Europe/Zurich",
                    "theme": "light",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "container_id": "tradingview_74048"
                });
            </script>
        </div>
        """,
        #height=200,


with col2:
    with st.container(border=True, height=400):
        st.subheader("Bitcoin Fear and Greed Index", divider='orange')
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        #components.iframe(url, width=300, height=200)
        if btc_fear_greed_historical is not None:
                # Simplify to single line chart with "sentiment score"
            max_date = btc_fear_greed_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_fear_greed_filtered = btc_fear_greed_historical[btc_fear_greed_historical.index >= min_date].tail(days)
            st.line_chart(btc_fear_greed_filtered[["sentiment score"]], use_container_width=True)
        else:
            st.error("Bitcoin fear and greed data not available.")

with col3:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq", divider='orange')
        if nasdaq_price_historical is not None:
            max_date = nasdaq_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            nasdaq_filtered = nasdaq_price_historical[nasdaq_price_historical.index >= min_date].tail(days)
            st.line_chart(nasdaq_filtered[["Close"]], use_container_width=True)
        else:
            st.error("Nasdaq historical data not available.")
        #nasdaq_chart = pd.DataFrame(xxx)
        #st.line_chart(nasdaq_chart)
        st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with col4:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq Fear and Greed Index", divider='orange')
        if nasdaq_fear_greed_historical is not None:
            max_date = nasdaq_fear_greed_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            nasdaq_fear_greed_filtered = nasdaq_fear_greed_historical[nasdaq_fear_greed_historical.index >= min_date].tail(days)
            st.line_chart(nasdaq_fear_greed_filtered[["sentiment score"]], use_container_width=True)
        else:
            st.error("Nasdaq fear and greed data not available.")
        #nasdaq_fag_chart = pd.DataFrame(xxx)
        # #st.line_chart(nasdaq_fag_chart)


st.subheader('Overlapping charts')

with st.container(border=True, height=500):
    #combined_chart=plot_combined_chart(all_data)
    #st.plotly_chart(combined_chart, use_container_width=True)
    #nasdaq_chart = pd.DataFrame(xxx)
    #st.line_chart(nasdaq_chart)
    st.slider(label='days', min_value=1, max_value=365, step=1)
    if btc_price_historical is not None and nasdaq_price_historical is not None:
            # Normalize data for overlapping chart
            try:
                first_btc_close = btc_price_historical["Close"].dropna().iloc[0]
                first_nasdaq_close = nasdaq_price_historical["Close"].dropna().iloc[0]
            except IndexError:
                st.error("Insufficient data for normalization.")
            else:
                combined_index = pd.Index.union(btc_price_historical.index, nasdaq_price_historical.index)
                btc_price_df = btc_price_historical.reindex(combined_index)
                nasdaq_price_df = nasdaq_price_historical.reindex(combined_index)
                btc_normalized = (btc_price_df["Close"] / first_btc_close) * 100
                nasdaq_normalized = (nasdaq_price_df["Close"] / first_nasdaq_close) * 100
                # Create a DataFrame with two columns for overlapping display
                overlap_df = pd.DataFrame({
                    "BTC Price (% change from start)": btc_normalized,
                    "NASDAQ Price (% change from start)": nasdaq_normalized
                })
                st.line_chart(overlap_df, use_container_width=True)
    else:
        st.error("Data for overlapping charts not available.")

st.subheader('Our Predictions')

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True, height=550):
        st.write("Bitcoin Prediction")
        #bitcoin_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_chart)
        #st.image("https://g.co/finance/BTC-EUR", width=200)
        st.slider(label='days', min_value=1, max_value=30, step=1)
        st.line_chart(np.random.randn(50, 3))

with col2:
    with st.container(border=True, height=550):
        st.write("Stock Market Prediction")
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        st.slider(label='Days', min_value=1, max_value=30, step=1)
        st.line_chart(np.random.randn(50, 3))

st.subheader('The accuracy of our model')

with st.container(border=True, height=550):
    #nasdaq_chart = pd.DataFrame(xxx)
    #st.line_chart(nasdaq_chart)
    st.slider(label='', min_value=1, max_value=10, step=1)
    st.line_chart(np.random.randn(50, 2))

st.subheader('How we were improving each day')

with st.container(border=True, height=550):
    st.slider(label='day', min_value=1, max_value=10, step=1)
    st.line_chart(np.random.randn(50, 2), color=['#5340A9', '#00AF54'])
