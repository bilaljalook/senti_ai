import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from google.cloud import bigquery
import argparse
import os

from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.model import (
    load_and_preprocess_data,
    create_sequences,
    evaluate_forecast,
    plot_predictions
)
from senti_ai.ml_logic.registry import load_model

def preprocess_raw_data(df, input_seq_length, forecast_horizon):
    """
    Preprocess raw data for LSTM model prediction.

    Args:
        df: Raw DataFrame
        input_seq_length: Number of days to use as input
        forecast_horizon: Number of days to forecast

    Returns:
        Preprocessed data ready for prediction
    """
    temp_csv_path = "temp_prediction_data.csv"
    df.to_csv(temp_csv_path, index=False)

    # Load and preprocess
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler_y, feature_names = load_and_preprocess_data(temp_csv_path)

    # Create sequences
    X_seq, y_seq = create_sequences(X_test, y_test, input_seq_length, forecast_horizon)

    # Clean up temp file
    os.remove(temp_csv_path)

    return X_seq, y_seq, dates_test, scaler_y


def predict(
    input_seq_length=LSTM_INPUT_SEQ_LENGTH,
    forecast_horizon=LSTM_FORECAST_HORIZON,
    data_source=DATA_SOURCE,
    data_path=DATA_PATH,
    generate_plot=True
):
    """
    Make predictions using the trained LSTM model.

    Args:
        input_seq_length: Number of days to use as input
        forecast_horizon: Number of days to forecast
        data_source: Source of data ("bigquery" or "local")
        data_path: Path to local CSV file (used if data_source is "local")
        generate_plot: Whether to generate visualization plot
    """
    print(Fore.BLUE + "\n⭐️ Using LSTM model for prediction ⭐️" + Style.RESET_ALL)

    # Load model
    model = load_model(model_type="lstm")
    if model is None:
        print(Fore.RED + "❌ Failed to load LSTM model" + Style.RESET_ALL)
        return None

    # Get data
    if data_source == "bigquery":
        print(Fore.BLUE + "\nLoading data from BigQuery..." + Style.RESET_ALL)
        query = f"""SELECT * FROM `{GCP_PROJECT_AHMET}.{BQ_DATASET}.raw`"""
        client = bigquery.Client(project=GCP_PROJECT_AHMET)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()
        df = df.sort_values('date', ascending=True)
        print(f"✅ Data loaded from BigQuery, shape: {df.shape}")
    else:
        print(Fore.BLUE + f"\nLoading data from {data_path}..." + Style.RESET_ALL)
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded from CSV, shape: {df.shape}")

    # Preprocess data
    try:
        X_seq, y_seq, dates, scaler_y = preprocess_raw_data(df, input_seq_length, forecast_horizon)
        print(f"✅ Data preprocessed for prediction")

        # Make prediction
        y_pred = model.predict(X_seq)
        print(f"✅ Predictions generated for {len(y_pred)} sequences")

        # Evaluate predictions
        evaluation = evaluate_forecast(y_seq, y_pred, forecast_horizon, scaler_y)

        print(Fore.GREEN + "\n📊 Prediction Results:" + Style.RESET_ALL)
        print(f"MSE: {evaluation['mse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"Avg Direction Accuracy: {np.mean(evaluation['direction_accuracy']):.4f}")

        # Generate plots if requested
        if generate_plot and scaler_y is not None:
            # Get actual and predicted values (inverse transform to original scale)
            y_true_orig = scaler_y.inverse_transform(y_seq.reshape(-1, 1)).reshape(y_seq.shape)
            y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

            # Reshape to get just the first step predictions for each sequence
            y_true_first_step = y_true_orig[:, 0]
            y_pred_first_step = y_pred_orig[:, 0]

            # Get corresponding dates
            prediction_dates = dates[-len(y_true_first_step):]

            # Plot using only a subset of points to avoid overcrowding
            step = max(1, len(prediction_dates) // 50)  # show max 50 points

            plt_obj = plot_predictions(
                y_true_first_step[::step],
                y_pred_first_step[::step],
                prediction_dates[::step],
                title="Bitcoin Price Prediction (LSTM Model)"
            )

            # Save plot
            output_file = f"bitcoin_prediction_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt_obj.savefig(output_file)
            plt_obj.close()
            print(f"✅ Prediction plot saved to {output_file}")

        # Generate future predictions
        last_sequence = X_seq[-1]

        # Create dates for future predictions
        last_date = dates.iloc[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]

        # Make future prediction
        future_pred = model.predict(np.expand_dims(last_sequence, axis=0))[0]

        # Inverse transform future predictions
        if scaler_y is not None:
            future_pred_orig = scaler_y.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        else:
            future_pred_orig = future_pred

        # Create DataFrame with future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_BTC_Close': future_pred_orig
        })

        print(Fore.GREEN + "\n🔮 Future Predictions:" + Style.RESET_ALL)
        for i, row in future_df.iterrows():
            print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Predicted_BTC_Close']:.2f}")

        return evaluation, future_df

    except Exception as e:
        print(Fore.RED + f"\n❌ Error during prediction: {e}" + Style.RESET_ALL)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using LSTM model for Bitcoin price')
    parser.add_argument('--input_seq_length', type=int, default=30, help='Number of days to use as input')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--data_source', type=str, default="bigquery", choices=["bigquery", "local"], help='Source of data')
    parser.add_argument('--data_path', type=str, default=None, help='Path to local CSV file')
    parser.add_argument('--no_plot', action='store_true', help='Disable plot generation')

    args = parser.parse_args()

    # Validate arguments
    if args.data_source == "local" and args.data_path is None:
        print(Fore.RED + "❌ Error: You must specify --data_path when using --data_source=local" + Style.RESET_ALL)
        exit(1)

    # Make predictions
    predict(
        input_seq_length=args.input_seq_length,
        forecast_horizon=args.forecast_horizon,
        data_source=args.data_source,
        data_path=args.data_path,
        generate_plot=not args.no_plot
    )
