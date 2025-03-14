import streamlit as st
import requests
import streamlit.components.v1 as components
#from app2 import load_historical_data_api
#from app2 import load_prediction_data_api
#from app2 import plot_market_chart
#from app2 import plot
#from charts import load_all_data, plot_combined_chart, plot_prediction_chart

import numpy as np
import pandas as pd
import datetime

st.set_page_config(layout='wide')

#variables
url = "http://web.senti-ai.net:5367/get_historical?gcp_project=supple-folder-448412-n9&bq_dataset=sentiai&table=raw"
bitcoin_chart_url="https://s3.tradingview.com/tv.js"
logo_large='/Users/kk/code/sentiai-09.png'
#all_data = load_all_data()

#layout

st.title('Historical overview')

#st.logo(logo_large, size='large')

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True, height=400):
        st.subheader("Bitcoin", divider='orange')
        #bitcoin_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_chart)
        #st.image("https://g.co/finance/BTC-EUR", width=200)
        components.html(
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
        height=200,
    )

with col2:
    with st.container(border=True, height=400):
        st.subheader("Bitcoin Fear and Greed Index", divider='orange')
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        components.iframe(url, width=300, height=200)

with col3:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq", divider='orange')
        #nasdaq_chart = pd.DataFrame(xxx)
        #st.line_chart(nasdaq_chart)
        st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with col4:
    with st.container(border=True, height=400):
        st.subheader("Nasdaq Fear and Greed Index", divider='orange')
        #nasdaq_fag_chart = pd.DataFrame(xxx)
        # #st.line_chart(nasdaq_fag_chart)
        st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with st.container(border=True, height=500):
    st.subheader('Overlapping charts', divider=True)
    #combined_chart=plot_combined_chart(all_data)
    #st.plotly_chart(combined_chart, use_container_width=True)
    #nasdaq_chart = pd.DataFrame(xxx)
    #st.line_chart(nasdaq_chart)
    st.slider(label='days', min_value=1, max_value=365, step=1)
    st.line_chart(np.random.randn(50, 3))

st.title('Our Predictions')

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

with st.container(border=True, height=550):
    st.subheader('The accuracy of our model', divider=True)
    #nasdaq_chart = pd.DataFrame(xxx)
    #st.line_chart(nasdaq_chart)
    st.slider(label='', min_value=1, max_value=10, step=1)
    st.line_chart(np.random.randn(50, 2))

with st.container(border=True, height=550):
    st.subheader('How we were improving each day')
    st.slider(label='day', min_value=1, max_value=10, step=1)
    st.line_chart(np.random.randn(50, 2), color=['#5340A9', '#00AF54'])
