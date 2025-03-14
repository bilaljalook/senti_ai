import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from google.cloud import bigquery
from pathlib import Path

from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.model import (
    run_bitcoin_prediction,
    plot_predictions,
    plot_feature_importance,
    plot_error_by_forecast_step
)
from senti_ai.ml_logic.registry import save_model, save_results

def train(
    input_seq_length=LSTM_INPUT_SEQ_LENGTH,  # Fixed variable name
    forecast_horizon=LSTM_FORECAST_HORIZON,  # Fixed variable name
    epochs=LSTM_MAX_EPOCHS,  # Fixed variable name
    data_source=DATA_SOURCE,
    data_path=DATA_PATH
):
    """
    Train the LSTM model using data from BigQuery or a local CSV file.

    Args:
        input_seq_length: Number of days to use as input for prediction
        forecast_horizon: Number of days to forecast
        epochs: Maximum number of training epochs
        data_source: Source of data ("bigquery" or "local")
        data_path: Path to local CSV file (used if data_source is "local")
    """
    print(Fore.BLUE + "\n⭐️ Using LSTM model ⭐️" + Style.RESET_ALL)

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

    # Create temp CSV file for model training
    temp_csv_path = "temp_bitcoin_data.csv"
    df.to_csv(temp_csv_path, index=False)
    print(f"✅ Temporary CSV file created at {temp_csv_path}")

    # Train model
    try:
        model, evaluation, feature_importance, history = run_bitcoin_prediction(
            temp_csv_path,
            input_seq_length=input_seq_length,
            forecast_horizon=forecast_horizon,
            epochs=epochs
        )

        # Clean up temp file
        os.remove(temp_csv_path)

        # Save model
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(model, model_type="lstm")

        # Save results
        eval_metrics = {
            "mse": evaluation["mse"],
            "mae": evaluation["mae"],
            "rmse": evaluation["rmse"],
            "avg_direction_accuracy": np.mean(evaluation["direction_accuracy"])
        }
        save_results(eval_metrics, model_version, model_type="lstm")

        print(Fore.GREEN + "\n✅ LSTM model training completed successfully!" + Style.RESET_ALL)
        print(f"Model version: {model_version}")
        print(f"MSE: {evaluation['mse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"Avg Direction Accuracy: {np.mean(evaluation['direction_accuracy']):.4f}")

        return model, evaluation, feature_importance, history

    except Exception as e:
        print(Fore.RED + f"\n❌ Error during LSTM model training: {e}" + Style.RESET_ALL)

        # Clean up temp file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        return None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model for Bitcoin price prediction')
    parser.add_argument('--input_seq_length', type=int, default=30, help='Number of days to use as input')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--data_source', type=str, default="bigquery", choices=["bigquery", "local"], help='Source of data')
    parser.add_argument('--data_path', type=str, default=None, help='Path to local CSV file')

    args = parser.parse_args()

    # Validate arguments
    if args.data_source == "local" and args.data_path is None:
        print(Fore.RED + "❌ Error: You must specify --data_path when using --data_source=local" + Style.RESET_ALL)
        exit(1)

    # Train model
    train(
        input_seq_length=args.input_seq_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        data_source=args.data_source,
        data_path=args.data_path
    )
