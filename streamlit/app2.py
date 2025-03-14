import streamlit as st
import pandas as pd
import altair as alt
import requests

# Set page config as the first Streamlit command
st.set_page_config(page_title='Historical overview', layout='wide')

# Data loading function with updated caching
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
    columns = df.columns.tolist()

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

# Main application (Historical overview section only)
def main():
    # Load data
    historical_datasets, _ = load_historical_data_api()

    if historical_datasets is None:
        st.error("Failed to load data from APIs.")
        return

    # Extract specific datasets
    btc_price_historical = historical_datasets.get("BTC Price (Daily)")
    btc_fear_greed_historical = historical_datasets.get("Fear & Greed Index (BTC)")
    nasdaq_price_historical = historical_datasets.get("Nasdaq Price (Daily)")
    nasdaq_fear_greed_historical = historical_datasets.get("Fear & Greed Index (Nasdaq)")

    # Title
    st.title('Historical overview')

    # Slider for selecting the number of days to display (within one year)
    days = st.slider(label='days', min_value=1, max_value=365, value=365, step=1)

    # Columns for historical data in two rows for staggered layout
    col1, col2 = st.columns(2)
    with col1:
        st.write("<h4 style='margin-bottom: 0px;'>Bitcoin Historical Data</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if btc_price_historical is not None:
                # Filter data to the last year (365 days) and then apply slider
                max_date = btc_price_historical.index.max()
                min_date = max_date - pd.Timedelta(days=365)
                btc_filtered = btc_price_historical[btc_price_historical.index >= min_date].tail(days)
                st.line_chart(btc_filtered[["Close"]], use_container_width=True)
            else:
                st.error("Bitcoin historical data not available.")
    with col2:
        st.write("<h4 style='margin-bottom: 0px;'>Bitcoin Historical Fear and Greed Index</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=500):
            if btc_fear_greed_historical is not None:
                # Filter data to the last year (365 days) and then apply slider
                max_date = btc_fear_greed_historical.index.max()
                min_date = max_date - pd.Timedelta(days=365)
                btc_fear_greed_filtered = btc_fear_greed_historical[btc_fear_greed_historical.index >= min_date].tail(days)
                st.line_chart(btc_fear_greed_filtered[["sentiment score"]], use_container_width=True)
            else:
                st.error("Bitcoin fear and greed data not available.")

    col3, col4 = st.columns(2)
    with col3:
        st.write("<h4 style='margin-bottom: 0px;'>Nasdaq Historical Data</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=600):
            if nasdaq_price_historical is not None:
                # Filter data to the last year (365 days) and then apply slider
                max_date = nasdaq_price_historical.index.max()
                min_date = max_date - pd.Timedelta(days=365)
                nasdaq_filtered = nasdaq_price_historical[nasdaq_price_historical.index >= min_date].tail(days)
                st.line_chart(nasdaq_filtered[["Close"]], use_container_width=True)
            else:
                st.error("Nasdaq historical data not available.")
    with col4:
        st.write("<h4 style='margin-bottom: 0px;'>Nasdaq Historical Fear and Greed Index</h4>", unsafe_allow_html=True)
        with st.container(border=True, height=500):
            if nasdaq_fear_greed_historical is not None:
                # Filter data to the last year (365 days) and then apply slider
                max_date = nasdaq_fear_greed_historical.index.max()
                min_date = max_date - pd.Timedelta(days=365)
                nasdaq_fear_greed_filtered = nasdaq_fear_greed_historical[nasdaq_fear_greed_historical.index >= min_date].tail(days)
                st.line_chart(nasdaq_fear_greed_filtered[["sentiment score"]], use_container_width=True)
            else:
                st.error("Nasdaq fear and greed data not available.")

    # Overlapping charts section with corrected Dual Y-Axes and enhanced tooltips
    st.subheader('Overlapping charts', divider=True)

    with st.container(border=True, height=700):
        if btc_price_historical is not None and nasdaq_price_historical is not None:
            # Filter data to the last year (365 days) and then apply slider
            max_date = btc_price_historical.index.max()
            min_date = max_date - pd.Timedelta(days=365)
            btc_filtered = btc_price_historical[btc_price_historical.index >= min_date].tail(days)
            nasdaq_filtered = nasdaq_price_historical[nasdaq_price_historical.index >= min_date].tail(days)

            # Merge data for dual y-axis chart
            overlap_df = pd.merge(btc_filtered[["Close"]], nasdaq_filtered[["Close"]], left_index=True, right_index=True, suffixes=("_BTC", "_NASDAQ"))
            overlap_df.reset_index(inplace=True)
            overlap_df.columns = ["Date", "BTC Price", "NASDAQ Price"]

            # Determine dynamic domains
            nasdaq_min = overlap_df["NASDAQ Price"].min()
            nasdaq_max = overlap_df["NASDAQ Price"].max()
            btc_min = overlap_df["BTC Price"].min()
            btc_max = overlap_df["BTC Price"].max()

            # Create a selection for nearest point interaction
            nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Date'], empty='none')

            # Create base chart for dual y-axes
            base = alt.Chart(overlap_df).encode(
                x=alt.X('Date:T', axis=alt.Axis(format='%Y-%m-%d', tickCount=12))
            )

            # Define color scale to ensure NASDAQ is blue and Bitcoin is orange
            color_scale = alt.Scale(
                domain=['NASDAQ Price', 'BTC Price'],
                range=['blue', 'orange']
            )

            # NASDAQ chart (left y-axis)
            nasdaq_chart = base.mark_line().encode(
                y=alt.Y('NASDAQ Price:Q', scale=alt.Scale(domain=[nasdaq_min, nasdaq_max]), title='NASDAQ Price', axis=alt.Axis(orient='left')),
                color=alt.ColorValue('blue')
            )

            # Bitcoin chart (right y-axis)
            btc_chart = base.mark_line().encode(
                y=alt.Y('BTC Price:Q', scale=alt.Scale(domain=[btc_min, btc_max]), title='BTC Price', axis=alt.Axis(orient='right')),
                color=alt.ColorValue('orange')
            )

            # Add points for tooltip triggering
            points = alt.Chart(overlap_df).mark_point().encode(
                x='Date:T',
                y=alt.Y('NASDAQ Price:Q', scale=alt.Scale(domain=[nasdaq_min, nasdaq_max])),
                color=alt.ColorValue('blue'),
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            ).transform_filter(
                'isValid(datum["NASDAQ Price"])'
            ) + alt.Chart(overlap_df).mark_point().encode(
                x='Date:T',
                y=alt.Y('BTC Price:Q', scale=alt.Scale(domain=[btc_min, btc_max])),
                color=alt.ColorValue('orange'),
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            ).transform_filter(
                'isValid(datum["BTC Price"])'
            )

            # Create tooltip rule lines
            rules = alt.Chart(overlap_df).mark_rule(color='gray').encode(
                x='Date:T',
                opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
                tooltip=[
                    alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
                    alt.Tooltip('NASDAQ Price:Q', title='NASDAQ Price', format=',.0f'),
                    alt.Tooltip('BTC Price:Q', title='BTC Price', format=',.0f')
                ]
            ).add_selection(
                nearest
            )

            # Combine the chart elements
            chart = (nasdaq_chart + btc_chart + points + rules).resolve_scale(y='independent').properties(width='container', height=700)
            st.altair_chart(chart, use_container_width=True)

            # Add manual legend
            st.write("**Price Legend**:")
            st.write("- Blue line represents NASDAQ Price")
            st.write("- Orange line represents BTC Price")

            # Add note
            st.write("**Note**: The y-axes have different scales (left for NASDAQ, right for BTC). To compare trends, observe the shape and direction of the lines.")
        else:
            st.error("Data for overlapping charts not available.")

if __name__ == "__main__":
    main()
