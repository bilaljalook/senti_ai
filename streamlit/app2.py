import streamlit as st
import requests
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import datetime

# Set page config as the first Streamlit command
st.set_page_config(page_title='Historical overview', layout='wide')

# Add custom CSS for spacing, styling, and slider thumb color
st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem;
        max-width: 1200px;
    }
    .element-container, .stMarkdown, .stDataFrame {
        margin-bottom: 2rem;
        padding: 0 !important;  /* Remove default padding */
    }
    .stSelectbox, .stSlider, .stCheck {
        margin-bottom: 1.5rem;
    }

    /* Custom slider thumb styling for all browsers with specific sidebar targeting */
    .stSideBar .stSlider [type="range"]::-webkit-slider-thumb {
        background: red !important;
        -webkit-appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        cursor: pointer;
        border: none !important;
        box-shadow: none !important;
    }
    .stSideBar .stSlider [type="range"]::-moz-range-thumb {
        background: red !important;
        -moz-appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        cursor: pointer;
        border: none !important;
        box-shadow: none !important;
    }
    .stSideBar .stSlider [type="range"]::-ms-thumb {
        background: red !important;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        cursor: pointer;
        border: none !important;
        box-shadow: none !important;
    }

    /* Optional: Style the unselected portion of the track */
    .stSideBar .stSlider [type="range"]::-webkit-slider-runnable-track {
        background: #ddd;
    }
    .stSideBar .stSlider [type="range"]::-moz-range-track {
        background: #ddd;
    }
    .stSideBar .stSlider [type="range"]::-ms-track {
        background: #ddd;
    }

    /* Pure CSS hack to override the "filled" portion of the slider in Streamlit */
    .stSideBar [data-testid="stSlider"] > div [data-baseweb="slider"] > div:nth-child(2) {
        background-color: red !important;
    }

    /* Enforce label color to black */
    .stSideBar .stSlider label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Data loading functions with updated caching
@st.cache_data
def load_historical_data_api():
    url = "http://web.senti-ai.net:5367/get_historical?gcp_project=supple-folder-448412-n9&bq_dataset=sentiai&table=raw"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return None, []

    datasets = {
        "BTC Price (Daily)": None,
        "Fear & Greed Index (BTC)": None,
        "Nasdaq Price (Daily)": None,
        "Fear & Greed Index (Nasdaq)": None
    }

    if not isinstance(data, list) or not data:
        return None, []

    df = pd.DataFrame(data)
    columns = df.columns.tolist()  # Capture columns to return

    btc_columns = {
        "date": "Date",
        "BTC_Open": "Open",
        "BTC_High": "High",
        "BTC_Low": "Low",
        "BTC_Close": "Close",
        "BTC_Volume": "Volume",
        "BTC_sentiment_score": "sentiment score"
    }
    nasdaq_columns = {
        "date": "Date",
        "NASDAQ_Open": "Open",
        "NASDAQ_High": "High",
        "NASDAQ_Low": "Low",
        "NASDAQ_Close": "Close",
        "NASDAQ_Volume": "Volume",
        "NASDAQ_sentiment_score": "sentiment score"
    }

    btc_available_columns = [col for col in btc_columns.keys() if col in df.columns]
    if btc_available_columns:
        btc_df = df[btc_available_columns].rename(columns=btc_columns)
        btc_df["Date"] = pd.to_datetime(btc_df["Date"], errors="coerce")
        btc_df = btc_df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
        if not btc_df.empty:
            datasets["BTC Price (Daily)"] = btc_df.drop(columns=["sentiment score"], errors="ignore")
            if "sentiment score" in btc_df.columns:
                btc_df["sentiment score"] = pd.to_numeric(btc_df["sentiment score"], errors="coerce")
                datasets["Fear & Greed Index (BTC)"] = btc_df[["sentiment score"]].dropna()

    nasdaq_available_columns = [col for col in nasdaq_columns.keys() if col in df.columns]
    if nasdaq_available_columns:
        nasdaq_df = df[nasdaq_available_columns].rename(columns=nasdaq_columns)
        nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"], errors="coerce")
        nasdaq_df = nasdaq_df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
        if not nasdaq_df.empty:
            datasets["Nasdaq Price (Daily)"] = nasdaq_df.drop(columns=["sentiment score"], errors="ignore")
            if "sentiment score" in nasdaq_df.columns:
                nasdaq_df["sentiment score"] = pd.to_numeric(nasdaq_df["sentiment score"], errors="coerce")
                datasets["Fear & Greed Index (Nasdaq)"] = nasdaq_df[["sentiment score"]].dropna()

    return datasets, columns

@st.cache_data
def load_prediction_data_api():
    url = "http://web.senti-ai.net:5367/get_pred"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return None, []

    datasets = {
        "BTC Predictions": None,
        "Nasdaq Predictions": None
    }

    if not isinstance(data, list) or not data:
        return None, []

    df = pd.DataFrame(data)
    columns = df.columns.tolist()  # Capture columns to return

    # Process BTC Predictions
    btc_df = df.copy()
    btc_df["Date"] = pd.to_datetime(btc_df["date"], errors="coerce")
    btc_df = btc_df.drop(columns=["date"], errors="ignore").rename(columns={"close": "Close"})
    btc_df = btc_df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")
    if not btc_df.empty:
        datasets["BTC Predictions"] = btc_df

    # Generate Nasdaq Predictions by scaling BTC's prediction data
    if not btc_df.empty:
        nasdaq_current_price = 20000  # Set first predicted close to 20,000 as per user
        first_date = btc_df.index[0]
        btc_first_close = btc_df.loc[first_date, "Close"]
        ratios = btc_df["Close"] / btc_first_close  # Get relative changes from first predicted close
        nasdaq_pred_df = pd.DataFrame({"Close": nasdaq_current_price * ratios}, index=btc_df.index)
        datasets["Nasdaq Predictions"] = nasdaq_pred_df

    return datasets, columns

# Main application
def main():
    # Load data
    historical_datasets, _ = load_historical_data_api()
    prediction_datasets, _ = load_prediction_data_api()

    if historical_datasets is None or prediction_datasets is None:
        st.error("Failed to load data from APIs.")
        return

    # Extract specific datasets
    btc_price_historical = historical_datasets.get("BTC Price (Daily)")
    btc_fear_greed_historical = historical_datasets.get("Fear & Greed Index (BTC)")
    nasdaq_price_historical = historical_datasets.get("Nasdaq Price (Daily)")
    nasdaq_fear_greed_historical = historical_datasets.get("Fear & Greed Index (Nasdaq)")
    btc_prediction = prediction_datasets.get("BTC Predictions")
    nasdaq_prediction = prediction_datasets.get("Nasdaq Predictions")

    # Title
    st.title('Historical overview')

    # Columns for historical data
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("<h4 style='margin-bottom: 0px;'>Bitcoin Historical Data</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if btc_price_historical is not None:
                # Simplify to single line chart with only "Close" prices
                st.line_chart(btc_price_historical[["Close"]], use_container_width=True)
            else:
                st.error("Bitcoin historical data not available.")

    with col2:
        st.write("<h4 style='margin-bottom: 0px;'>Bitcoin Historical Fear and Greed Index</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if btc_fear_greed_historical is not None:
                # Simplify to single line chart with "sentiment score"
                st.line_chart(btc_fear_greed_historical[["sentiment score"]], use_container_width=True)
            else:
                st.error("Bitcoin fear and greed data not available.")

    with col3:
        st.write("<h4 style='margin-bottom: 0px;'>Nasdaq Historical Data</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if nasdaq_price_historical is not None:
                # Simplify to single line chart with only "Close" prices
                st.line_chart(nasdaq_price_historical[["Close"]], use_container_width=True)
            else:
                st.error("Nasdaq historical data not available.")

    with col4:
        st.write("<h4 style='margin-bottom: 0px;'>Nasdaq Historical Fear and Greed Index</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if nasdaq_fear_greed_historical is not None:
                # Simplify to single line chart with "sentiment score"
                st.line_chart(nasdaq_fear_greed_historical[["sentiment score"]], use_container_width=True)
            else:
                st.error("Nasdaq fear and greed data not available.")

    # Overlapping charts section
    st.subheader('Overlapping charts', divider=True)
    with st.container(border=True, height=700):
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
    st.slider(label='days', min_value=1, max_value=365, step=1)

    # Predictions section
    st.title('Our Predictions')

    col1, col2 = st.columns(2)

    with col1:
        st.write("Bitcoin Prediction")
        with st.container(border=True, height=800):
            if btc_prediction is not None:
                # Simplify to single line chart with "Close" prices
                st.line_chart(btc_prediction[["Close"]], use_container_width=True)
            else:
                st.error("Bitcoin prediction data not available.")
        st.slider(label='days', min_value=1, max_value=30, step=1)

    with col2:
        st.write("Stock Market Prediction")
        with st.container(border=True, height=800):
            if nasdaq_prediction is not None:
                # Simplify to single line chart with "Close" prices
                st.line_chart(nasdaq_prediction[["Close"]], use_container_width=True)
            else:
                st.error("Nasdaq prediction data not available.")
        st.slider(label='Days', min_value=1, max_value=30, step=1)

    # Accuracy section
    st.subheader('The accuracy of our model', divider=True)
    with st.container(border=True, height=800):
        # Placeholder for accuracy chart (already simple)
        st.line_chart(np.random.randn(50, 3), use_container_width=True)
    st.slider(label='', min_value=1, max_value=10, step=1)

if __name__ == "__main__":
    main()
