import streamlit as st
import requests
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import datetime

#variables
url = "http://web.senti-ai.net:5367/get_historical?gcp_project=supple-folder-448412-n9&bq_dataset=sentiai&table=raw"
chart_url='https://g.co/finance/BTC-EUR'
logo_large='/Users/kk/code/logo.png'

#layout
st.set_page_config(layout='wide')

st.title('Historical overview')

st.logo(logo_large, size='large')

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True, height=400):
        st.write("Bitcoin Historical Data")
        #bitcoin_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_chart)
        #st.image("https://g.co/finance/BTC-EUR", width=200)
        components.iframe(chart_url, width=300, height=200)

with col2:
    with st.container(border=True, height=400):
        st.subheader("Bitcoin Historical Fear and Greed Index", divider='blue')
        #bitcoin_fag_chart = pd.DataFrame(xxx)
        #st.line_chart(bitcoin_fag_chart)
        #st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)
        components.iframe(url, width=300, height=200)

with col3:
    with st.container(border=True, height=400):
        st.write("Nasdaq Historical Data")
        #nasdaq_chart = pd.DataFrame(xxx)
        #st.line_chart(nasdaq_chart)
        st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with col4:
    with st.container(border=True, height=400):
        st.write("Nasdaq Historical Fear and Greed Index")
        #nasdaq_fag_chart = pd.DataFrame(xxx)
        # #st.line_chart(nasdaq_fag_chart)
        st.image("https://images.app.goo.gl/5eN51YRNPVwEXMpn9", width=200)

with st.container(border=True, height=500):
    st.subheader('Overlapping charts', divider=True)
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
