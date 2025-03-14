import streamlit as st
import requests
import streamlit.components.v1 as components
from app2 import load_historical_data_api
from app2 import load_prediction_data_api
from app2 import main
import altair as alt
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
        st.subheader("Bitcoin", divider='grey')
        if btc_price_historical is not None:
                # Simplify to single line chart with only "Close" prices
            max_date = btc_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_filtered = btc_price_historical[btc_price_historical.index >= min_date].tail(days)
            st.line_chart(btc_filtered[["Close"]], use_container_width=True, color='#F7931A')
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
        st.subheader("Bitcoin F&G Index", divider='grey')
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        #components.iframe(url, width=300, height=200)
        if btc_fear_greed_historical is not None:
                # Simplify to single line chart with "sentiment score"
            max_date = btc_fear_greed_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_fear_greed_filtered = btc_fear_greed_historical[btc_fear_greed_historical.index >= min_date].tail(days)
            st.line_chart(btc_fear_greed_filtered[["sentiment score"]], use_container_width=True, color='#0092BC')
        else:
            st.error("Bitcoin fear and greed data not available.")

with col3:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq", divider='grey')
        if nasdaq_price_historical is not None:
            max_date = nasdaq_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            nasdaq_filtered = nasdaq_price_historical[nasdaq_price_historical.index >= min_date].tail(days)
            st.line_chart(nasdaq_filtered[["Close"]], use_container_width=True, color='#0092BC')
        else:
            st.error("Nasdaq historical data not available.")
        #nasdaq_chart = pd.DataFrame(xxx)
        #st.line_chart(nasdaq_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with col4:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq F&G Index", divider='grey')
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

with st.container(border=True, height=700):
    #combined_chart=plot_combined_chart(all_data)
    #st.plotly_chart(combined_chart, use_container_width=True)
    #nasdaq_chart = pd.DataFrame(xxx)
    #st.line_chart(nasdaq_chart)
    st.slider(label='days', min_value=1, max_value=365, step=1)
    if btc_price_historical is not None and nasdaq_price_historical is not None:
            max_date = btc_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_filtered = btc_price_historical[btc_price_historical.index >= min_date].tail(days)
            nasdaq_filtered = nasdaq_price_historical[nasdaq_price_historical.index >= min_date].tail(days)

            overlap_df = pd.merge(btc_filtered[["Close"]], nasdaq_filtered[["Close"]], left_index=True, right_index=True, suffixes=("_BTC", "_NASDAQ"))
            overlap_df.reset_index(inplace=True)
            overlap_df.columns = ["Date", "BTC Price", "NASDAQ Price"]

            nasdaq_min = overlap_df["NASDAQ Price"].min()
            nasdaq_max = overlap_df["NASDAQ Price"].max()
            btc_min = overlap_df["BTC Price"].min()
            btc_max = overlap_df["BTC Price"].max()

            base = alt.Chart(overlap_df).encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m-%d', tickCount=12)),
                tooltip=['Date:T', 'BTC Price:Q', 'NASDAQ Price:Q']
            )

            color_scale = alt.Scale(
                domain=['NASDAQ Price', 'BTC Price'],
                range=['blue', 'orange']
            )

            nasdaq_chart = base.transform_fold(
                ['NASDAQ Price'],
                as_=['variable', 'value']
            ).mark_line().encode(
                y=alt.Y('value:Q', scale=alt.Scale(domain=[nasdaq_min, nasdaq_max]), title='NASDAQ Price', axis=alt.Axis(orient='left')),
                color=alt.Color('variable:N', scale=color_scale, legend=alt.Legend(title="Price Legend"))
            )
            btc_chart = base.transform_fold(
                ['BTC Price'],
                as_=['variable', 'value']
            ).mark_line().encode(
                y=alt.Y('value:Q', scale=alt.Scale(domain=[btc_min, btc_max]), title='BTC Price', axis=alt.Axis(orient='right')),
                color=alt.Color('variable:N', scale=color_scale, legend=alt.Legend(title="Price Legend"))
            )
            chart = (nasdaq_chart + btc_chart).resolve_scale(y='independent').properties(width='container', height=600)
            st.altair_chart(chart, use_container_width=True)

            # Add note
            st.write("**Note**: The y-axes have different scales (left for NASDAQ, right for BTC). To compare trends, observe the shape and direction of the lines.")
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
        if btc_prediction is not None:
                # Simplify to single line chart with "Close" prices
            st.line_chart(btc_prediction[["Close"]], use_container_width=True)
        else:
            st.error("Bitcoin prediction data not available.")

with col2:
    with st.container(border=True, height=550):
        st.write("Stock Market Prediction")
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        st.slider(label='Days', min_value=1, max_value=30, step=1)
        if nasdaq_prediction is not None:
                # Simplify to single line chart with "Close" prices
            st.line_chart(nasdaq_prediction[["Close"]], use_container_width=True)
        else:
            st.error("Nasdaq prediction data not available.")

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
