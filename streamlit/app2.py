import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add custom CSS for spacing and styling
st.markdown(
    """
    <style>
    .block-container {
        padding: 2rem;
        max-width: 1200px;
    }
    .element-container, .stMarkdown, .stDataFrame {
        margin-bottom: 2rem;
    }
    .stSelectbox, .stSlider, .stCheckbox {
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- API-based Data Loading Functions ---

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

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"close": "Close"}).drop(columns=["date"])
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=["Date"]).set_index("Date")

    if not df.empty:
        datasets["BTC Predictions"] = df  # Assuming BTC predictions

    return datasets, columns

# --- Plotting Function ---

def plot_market_chart(price_df, fear_df, market_name, ma_window=20):
    # Initialize a list to collect debug outputs
    debug_outputs = []

    # Validate and prepare data
    if "Close" not in price_df.columns or price_df["Close"].isna().all():
        debug_outputs.append(("Error", "No valid price data available for plotting. Check the raw data."))
        debug_outputs.append(("Price Data", price_df))
        return go.Figure(), debug_outputs

    # Calculate Moving Average
    ma = price_df["Close"].rolling(window=ma_window, min_periods=1).mean()

    # Debug: Show raw Fear & Greed data before merging
    if fear_df is not None and not fear_df.empty and "sentiment score" in fear_df.columns:
        debug_outputs.append(("Debug: Raw Fear & Greed Data Sample (Before Merge)", fear_df.head(10)))
        debug_outputs.append(("Debug: Fear & Greed Data Statistics", fear_df["sentiment score"].describe()))

    # Merge datasets for aligned dates using a left join to preserve price data
    merged_df = price_df.join(fear_df, how="left").sort_index()

    # Debug: Print merged data to verify
    debug_outputs.append(("Debug: Merged Data Sample", merged_df.head(10)))
    if "sentiment score" in merged_df.columns:
        debug_outputs.append(("Debug: Sentiment Data Statistics (After Merge)", merged_df["sentiment score"].describe()))

    # Create subplots: one for main chart (Price, MA, Fear & Greed), one for Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=["Price, MA, and Fear & Greed Index", "Volume"],
        specs=[[{"secondary_y": True}], [{}]]  # Enable secondary y-axis for the first row
    )

    # Main chart (Price, MA, Fear & Greed) - Row 1
    # Price (Close) - Added first with increased visibility, on primary left y-axis (y)
    fig.add_trace(go.Scatter(
        x=merged_df.index,
        y=merged_df["Close"],
        mode="lines",
        name=f"{market_name} Price",
        line=dict(color="blue", width=3),
        showlegend=True
    ), row=1, col=1)

    # Moving Average - Added after price, on primary left y-axis (y)
    fig.add_trace(go.Scatter(
        x=merged_df.index,
        y=ma,
        mode="lines",
        name=f"{ma_window}-Day MA",
        line=dict(color="orange", width=2),
        showlegend=True
    ), row=1, col=1)

    # Fear & Greed Index - Secondary right y-axis (y2) with range 0-100
    sentiment_col = "sentiment score" if "sentiment score" in merged_df.columns else None
    if sentiment_col and not merged_df[sentiment_col].isna().all():
        # Identify non-NaN segments
        valid_indices = merged_df.index[merged_df[sentiment_col].notna()]
        segment_count = 1
        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            for i in range(1, len(valid_indices)):
                if valid_indices[i] != valid_indices[i-1] + pd.Timedelta(days=1):  # Detect gap
                    end_idx = valid_indices[i-1]
                    segment_df = merged_df.loc[start_idx:end_idx].copy()
                    if not segment_df.empty:
                        debug_outputs.append((f"Debug: Segment {segment_count} from {start_idx} to {end_idx}", segment_df[[sentiment_col]]))
                        debug_outputs.append((f"Debug: Segment {segment_count} Statistics from {start_idx} to {end_idx}", segment_df[sentiment_col].describe()))
                        fig.add_trace(go.Scatter(
                            x=segment_df.index,
                            y=segment_df[sentiment_col],
                            mode="lines+markers",
                            name=f"Fear & Greed Index Segment {segment_count}",
                            line=dict(color="red", width=1.5),
                            marker=dict(size=5),
                            showlegend=True
                        ), row=1, col=1, secondary_y=True)  # Use secondary y-axis for Fear & Greed Index
                        segment_count += 1
                    start_idx = valid_indices[i]
            # Handle the last segment
            end_idx = valid_indices[-1]
            segment_df = merged_df.loc[start_idx:end_idx].copy()
            if not segment_df.empty:
                debug_outputs.append((f"Debug: Segment {segment_count} from {start_idx} to {end_idx}", segment_df[[sentiment_col]]))
                debug_outputs.append((f"Debug: Segment {segment_count} Statistics from {start_idx} to {end_idx}", segment_df[sentiment_col].describe()))
                fig.add_trace(go.Scatter(
                    x=segment_df.index,
                    y=segment_df[sentiment_col],
                    mode="lines+markers",
                    name=f"Fear & Greed Index Segment {segment_count}",
                    line=dict(color="red", width=1.5),
                    marker=dict(size=5),
                    showlegend=True
                ), row=1, col=1, secondary_y=True)  # Use secondary y-axis for Fear & Greed Index
        else:
            debug_outputs.append(("Warning", "No valid (non-NaN) Fear & Greed Index data available to plot."))

    # Volume - Row 2, on primary y-axis for the second subplot
    if "Volume" in merged_df.columns and not merged_df["Volume"].isna().all():
        fig.add_trace(go.Bar(
            x=merged_df.index,
            y=merged_df["Volume"],
            name="Volume",
            marker=dict(color="#4B0082", opacity=1.0),
            showlegend=True
        ), row=2, col=1)

    # Update layout with multiple y-axes
    fig.update_layout(
        title={
            "text": f"{market_name} Price History with MA, Fear & Greed, and Volume",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24}
        },
        height=1000,
        legend=dict(
            x=0.5,
            y=-0.3,  # Adjusted to position legend below the Volume subplot's date axis
            xanchor="center",
            yanchor="top",
            orientation="h",
            font={"size": 16},
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1,
            groupclick="toggleitem"
        ),
        hovermode="x unified",
        clickmode="event+select",
        margin=dict(l=80, r=80, t=150, b=150),
        template="plotly_white"
    )

    # Update axes
    # Primary left y-axis for Price and MA (y)
    fig.update_yaxes(
        title_text=f"{market_name} Price (USD)",
        title_font={"size": 16},
        tickfont={"size": 14},
        showgrid=False,
        side="left",
        row=1,
        col=1,
        autorange=True
    )

    # Secondary right y-axis for Fear & Greed Index (y2)
    fig.update_yaxes(
        title_text="Fear & Greed Index",
        title_font={"size": 16},
        tickfont={"size": 14},
        range=[0, 100],
        showgrid=False,
        side="right",
        row=1,
        col=1,
        secondary_y=True
    )

    # Primary y-axis for Volume (second subplot)
    fig.update_yaxes(
        title_text="Volume",
        title_font={"size": 16},
        tickfont={"size": 14},
        showgrid=False,
        side="left",
        row=2,
        col=1,
        autorange=True
    )

    # X-axis for Date
    fig.update_xaxes(
        title_text="Date",
        title_font={"size": 16},
        tickfont={"size": 14},
        row=2,
        col=1
    )

    debug_outputs.append(("Debug: Plotly Traces", [trace.name for trace in fig.data]))

    return fig, debug_outputs

# --- Main Application ---

def main():
    # Initialize session state for button control
    if 'show_api_debug' not in st.session_state:
        st.session_state.show_api_debug = False

    # Sidebar
    st.sidebar.title("Market Selection")
    market = st.sidebar.selectbox("Choose Market", ["Bitcoin (BTC)", "Nasdaq"])

    # Load data from both APIs
    historical_datasets, historical_columns = load_historical_data_api()
    prediction_datasets, prediction_columns = load_prediction_data_api()

    # Handle errors
    if historical_datasets is None:
        st.error("Error loading historical data from API.")
        return
    if prediction_datasets is None:
        st.error("Error loading prediction data from API.")
        return

    all_datasets = {**historical_datasets, **prediction_datasets}

    if not all_datasets:
        st.error("No data loaded from either API.")
        return

    # Move description to the top
    if market == "Bitcoin (BTC)":
        st.title("Crypto Market Data Visualization")
        st.markdown(
            """
            **Chart Overview:**
            - **Blue Line**: {market} Price (Close) in USD.
            - **Orange Line**: {ma_window}-Day Moving Average (MA) of the price.
            - **Red Line**: Fear & Greed Index (0-100), showing market sentiment.
            - **Purple Bars**: Trading Volume (in a separate chart below).
            Use the legend to toggle traces on/off for better focus.
            """.format(market=market, ma_window=20)  # Default ma_window for display
        )
        st.write("")  # Add spacing
    elif market == "Nasdaq":
        st.title("Nasdaq Market Data Visualization")
        st.markdown(
            """
            **Chart Overview:**
            - **Blue Line**: {market} Price (Close) in USD.
            - **Orange Line**: {ma_window}-Day Moving Average (MA) of the price.
            - **Red Line**: Fear & Greed Index (0-100), showing market sentiment.
            - **Purple Bars**: Trading Volume (in a separate chart below).
            Use the legend to toggle traces on/off for better focus.
            """.format(market=market, ma_window=20)  # Default ma_window for display
        )
        st.write("")  # Add spacing

    # Display chart and selection methods
    # Dataset selection and sliders
    st.sidebar.subheader("Chart Options")
    if market == "Bitcoin (BTC)":
        price_options = [k for k in all_datasets.keys() if "BTC" in k and "Fear" not in k]
        fear_options = [k for k in all_datasets.keys() if "Fear & Greed Index (BTC)" in k]
    elif market == "Nasdaq":
        price_options = [k for k in all_datasets.keys() if "Nasdaq" in k and "Fear" not in k]
        fear_options = [k for k in all_datasets.keys() if "Fear & Greed Index (Nasdaq)" in k]

    selected_price_dataset = st.sidebar.selectbox("Select Price Dataset", price_options)
    selected_fear_dataset = st.sidebar.selectbox("Select Fear & Greed Dataset", fear_options, disabled=len(fear_options) == 0)
    ma_window = st.sidebar.slider("Moving Average Window (days)", 5, 50, 20)
    time_frame = st.sidebar.slider("Time Frame (Years)", 2015, 2025, (2015, 2025), step=1)

    price_df = all_datasets.get(selected_price_dataset)
    fear_df = all_datasets.get(selected_fear_dataset) if selected_fear_dataset else pd.DataFrame()

    if price_df is None:
        st.error(f"No data available for {selected_price_dataset}.")
        return

    # Filter data based on selected time frame
    start_year, end_year = time_frame
    if price_df is not None and not price_df.empty:
        price_df = price_df[(price_df.index.year >= start_year) & (price_df.index.year <= end_year)]
    if fear_df is not None and not fear_df.empty:
        fear_df = fear_df[(fear_df.index.year >= start_year) & (fear_df.index.year <= end_year)]

    # Plot chart with filtered data and capture debug outputs
    fig, debug_outputs = plot_market_chart(price_df, fear_df, market, ma_window)
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data
    if st.sidebar.checkbox("Show Raw Data"):
        if market == "Bitcoin (BTC)":
            st.subheader(f"Price Data - {selected_price_dataset}")
            st.dataframe(price_df.style.set_table_styles(
                [{'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center')]},
                 {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right')]}]
            ))
            if not fear_df.empty:
                st.subheader(f"Fear & Greed Data - {selected_fear_dataset}")
                st.dataframe(fear_df.style.set_table_styles(
                    [{'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center')]},
                     {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right')]}]
                ))
        elif market == "Nasdaq":
            st.subheader(f"Price Data - {selected_price_dataset}")
            st.dataframe(price_df.style.set_table_styles(
                [{'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center')]},
                 {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right')]}]
            ))
            if not fear_df.empty:
                st.subheader(f"Fear & Greed Data - {selected_fear_dataset}")
                st.dataframe(fear_df.style.set_table_styles(
                    [{'selector': 'th', 'props': [('font-size', '14px'), ('text-align', 'center')]},
                     {'selector': 'td', 'props': [('font-size', '12px'), ('text-align', 'right')]}]
                ))
        st.write("")  # Add spacing

    # Show API and debug info behind a button at the bottom
    if st.sidebar.button("Show API & Debug Info"):
        st.session_state.show_api_debug = not st.session_state.show_api_debug

    if st.session_state.show_api_debug:
        with st.expander("API & Debug Info", expanded=True):
            st.write("**Historical API returned columns:**")
            st.write(historical_columns)
            st.write("**Prediction API returned columns:**")
            st.write(prediction_columns)
            st.write("**Available Historical Datasets:**")
            st.write(list(historical_datasets.keys()))
            st.write("**Available Prediction Datasets:**")
            st.write(list(prediction_datasets.keys()))
            # Display debug outputs from plot_market_chart
            for label, output in debug_outputs:
                st.write(f"**{label}**")
                st.write(output)

if __name__ == "__main__":
    main()
