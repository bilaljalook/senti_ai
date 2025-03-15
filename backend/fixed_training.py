#!/usr/bin/env python

import sys
import os
# Add the parent directory to path so Python can find the senti_ai module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.model import build_model, evaluate_forecast, plot_predictions
from senti_ai.ml_logic.baseline_models import create_baseline_models, get_best_baseline_model
from senti_ai.ml_logic.registry import save_model, save_results

def fixed_train_model(X_train, y_train, X_test, y_test, dates_test,
                      input_seq_length=10, forecast_horizon=1, epochs=50):
    """
    Train a Bitcoin price prediction model with proper sequence handling.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        dates_test: Dates for test data
        input_seq_length: Number of timesteps for input
        forecast_horizon: Number of timesteps to predict
        epochs: Training epochs
    """
    print(Fore.BLUE + "\n🔧 Running fixed training pipeline..." + Style.RESET_ALL)

    # Run baseline models
    print(Fore.BLUE + "\nRunning baseline models..." + Style.RESET_ALL)
    baseline_results = create_baseline_models(X_train, y_train, X_test, y_test, forecast_horizon=1)
    best_baseline = get_best_baseline_model(baseline_results)
    print(f"✅ Best baseline model: {best_baseline}")

    # Create sequences with fixed approach
    X_train_seq, y_train_seq = create_fixed_sequences(X_train, y_train, input_seq_length, forecast_horizon)
    X_test_seq, y_test_seq = create_fixed_sequences(X_test, y_test, input_seq_length, forecast_horizon)

    print(f"Sequence shapes - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")
    print(f"Sequence shapes - X_test: {X_test_seq.shape}, y_test: {y_test_seq.shape}")

    # Build model
    model = build_model(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        output_length=forecast_horizon
    )

    # Train the model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ]

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = model.predict(X_test_seq)
    evaluation = evaluate_forecast(y_test_seq, y_pred, forecast_horizon)

    print(Fore.GREEN + "\n📊 Model Evaluation Results:" + Style.RESET_ALL)
    print(f"MSE: {evaluation['mse']:.4f}")
    print(f"MAE: {evaluation['mae']:.4f}")
    print(f"RMSE: {evaluation['rmse']:.4f}")

    # Compare with baseline
    lstm_mae = evaluation['mae']
    baseline_mae = baseline_results[best_baseline]['mae']
    improvement = (baseline_mae - lstm_mae) / baseline_mae * 100

    print(Fore.GREEN + f"\nLSTM vs Best Baseline ({best_baseline}):" + Style.RESET_ALL)
    print(f"LSTM MAE: {lstm_mae:.4f}")
    print(f"Best Baseline MAE: {baseline_mae:.4f}")
    print(f"Improvement: {improvement:.2f}%")

    # Save the model
    save_model(model, model_type="lstm_fixed")

    return model, evaluation, history

def create_fixed_sequences(X, y, input_seq_length, forecast_horizon):
    """
    Create sequences with proper alignment for time series forecasting.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Target array of shape (n_samples,)
        input_seq_length: Number of time steps to use as input
        forecast_horizon: Number of time steps to predict

    Returns:
        X_seq: Input sequences
        y_seq: Target sequences
    """
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, but have shapes {X.shape} and {y.shape}")

    # Force forecast_horizon to 1 if there are issues
    if forecast_horizon > 1 and len(X) < input_seq_length + forecast_horizon:
        print(f"Warning: Not enough data for forecast_horizon={forecast_horizon}. Setting to 1.")
        forecast_horizon = 1

    # Calculate the number of valid samples we can create
    n_samples = len(X) - input_seq_length - forecast_horizon + 1

    if n_samples <= 0:
        print("Warning: Not enough data to create sequences. Reducing input sequence length.")
        input_seq_length = max(1, len(X) // 2)
        n_samples = len(X) - input_seq_length - forecast_horizon + 1

    print(f"Creating {n_samples} sequences with input_length={input_seq_length}, horizon={forecast_horizon}")

    X_seq, y_seq = [], []

    for i in range(n_samples):
        # Get the input sequence
        X_window = X[i:i + input_seq_length]

        # Get the output sequence
        y_window = y[i + input_seq_length:i + input_seq_length + forecast_horizon]

        # Make sure we have complete sequences
        if len(X_window) == input_seq_length and len(y_window) == forecast_horizon:
            X_seq.append(X_window)
            y_seq.append(y_window)

    X_seq_array = np.array(X_seq)
    y_seq_array = np.array(y_seq)

    return X_seq_array, y_seq_array

def run_fixed_training():
    """
    Run the fixed training pipeline with simpler parameters that will work.
    """
    # Load data from file if available
    try:
        from senti_ai.ml_logic.model import load_and_preprocess_data

        # Use a shorter forecast horizon to avoid issues
        input_seq_length = 15
        forecast_horizon = 1

        print(Fore.BLUE + "\nLoading data from BigQuery..." + Style.RESET_ALL)
        # Use BigQuery data
        from google.cloud import bigquery
        query = f"""SELECT * FROM `{GCP_PROJECT_AHMET}.{BQ_DATASET}.raw` ORDER BY date ASC"""
        client = bigquery.Client(project=GCP_PROJECT_AHMET)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        print(f"✅ Data loaded, shape: {df.shape}")

        # Save to temporary CSV
        temp_csv_path = "temp_bitcoin_data.csv"
        df.to_csv(temp_csv_path, index=False)

        # Load and preprocess
        X_train, X_test, y_train, y_test, dates_train, dates_test, scaler_y, feature_names = load_and_preprocess_data(temp_csv_path)

        # Run fixed training
        model, evaluation, history = fixed_train_model(
            X_train, y_train, X_test, y_test, dates_test,
            input_seq_length=input_seq_length,
            forecast_horizon=forecast_horizon
        )

        # Clean up temp file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        print(Fore.GREEN + "\n✅ Training completed successfully!" + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"\n❌ Error during fixed training: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    run_fixed_training()
