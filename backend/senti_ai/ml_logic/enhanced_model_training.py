import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from google.cloud import bigquery
from pathlib import Path

from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.model import (
    run_bitcoin_prediction,
    build_model,
    load_and_preprocess_data,
    create_sequences,
    evaluate_forecast,
    plot_predictions,
    plot_feature_importance,
    plot_error_by_forecast_step
)
from senti_ai.ml_logic.registry import save_model, save_results
from senti_ai.ml_logic.baseline_models import (
    create_baseline_models,
    plot_baseline_comparisons,
    get_best_baseline_model
)
from senti_ai.ml_logic.time_series_cv import (
    time_series_cross_validation,
    plot_cv_error_distribution,
    expanding_window_cv
)

def enhanced_train(
    input_seq_length=LSTM_INPUT_SEQ_LENGTH,
    forecast_horizon=LSTM_FORECAST_HORIZON,
    epochs=LSTM_MAX_EPOCHS,
    data_source=DATA_SOURCE,
    data_path=DATA_PATH,
    run_baselines=True,
    run_cv=True,
    n_cv_splits=5,
    cv_type='time_series_split',  # 'time_series_split' or 'expanding_window'
    save_model_and_results=True,
    generate_plots=True
):
    """
    Enhanced training pipeline that includes baseline models and cross-validation.

    Args:
        input_seq_length: Number of days to use as input for prediction
        forecast_horizon: Number of days to forecast
        epochs: Maximum number of training epochs
        data_source: Source of data ("bigquery" or "local")
        data_path: Path to local CSV file (used if data_source is "local")
        run_baselines: Whether to run baseline models
        run_cv: Whether to run cross-validation
        n_cv_splits: Number of CV splits
        cv_type: Type of cross-validation ('time_series_split' or 'expanding_window')
        save_model_and_results: Whether to save the model and results
        generate_plots: Whether to generate plots
    """
    print(Fore.BLUE + "\n⭐️ Enhanced training with baselines and cross-validation ⭐️" + Style.RESET_ALL)

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

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test, dates_train, dates_test, scaler_y, feature_names = load_and_preprocess_data(temp_csv_path)
        print(f"✅ Data preprocessed: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Run baseline models if requested
        baseline_results = None
        if run_baselines:
            print(Fore.BLUE + "\nRunning baseline models..." + Style.RESET_ALL)
            baseline_results = create_baseline_models(X_train, y_train, X_test, y_test, forecast_horizon)

            # Plot baseline comparisons if requested
            if generate_plots:
                plt_baseline = plot_baseline_comparisons(
                    y_test,
                    baseline_results,
                    dates_test,
                    title="Bitcoin Price Forecast: Baseline Models Comparison"
                )
                baseline_plot_path = f"baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt_baseline.savefig(baseline_plot_path)
                plt_baseline.close()
                print(f"✅ Baseline comparison plot saved to {baseline_plot_path}")

            # Identify best baseline model
            best_baseline = get_best_baseline_model(baseline_results, metric='mae')
            print(f"✅ Best baseline model: {best_baseline}")

        # Run cross-validation if requested
        cv_results = None
        if run_cv:
            print(Fore.BLUE + "\nRunning cross-validation..." + Style.RESET_ALL)

            # Combine all data for cross-validation
            X_all = np.vstack((X_train, X_test))
            y_all = np.concatenate((y_train, y_test))

            if cv_type == 'time_series_split':
                cv_results = time_series_cross_validation(
                    X=X_all,
                    y=y_all,
                    build_model_fn=build_model,
                    input_seq_length=input_seq_length,
                    forecast_horizon=forecast_horizon,
                    create_sequences_fn=create_sequences,
                    evaluate_fn=evaluate_forecast,
                    n_splits=n_cv_splits,
                    verbose=True
                )
            elif cv_type == 'expanding_window':
                cv_results = expanding_window_cv(
                    X=X_all,
                    y=y_all,
                    build_model_fn=build_model,
                    input_seq_length=input_seq_length,
                    forecast_horizon=forecast_horizon,
                    create_sequences_fn=create_sequences,
                    evaluate_fn=evaluate_forecast,
                    n_windows=n_cv_splits,
                    verbose=True
                )

            # Plot CV error distribution if requested
            if generate_plots and cv_results is not None:
                plt_cv = plot_cv_error_distribution(cv_results)
                cv_plot_path = f"cv_error_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt_cv.savefig(cv_plot_path)
                plt_cv.close()
                print(f"✅ Cross-validation error distribution plot saved to {cv_plot_path}")

        # Train the full LSTM model
        print(Fore.BLUE + "\nTraining full LSTM model..." + Style.RESET_ALL)

        # Create sequences for LSTM
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, input_seq_length, forecast_horizon)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, input_seq_length, forecast_horizon)

        # Build and train model
        model = build_model(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
            output_length=forecast_horizon
        )

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint('best_bitcoin_model.h5', monitor='val_loss', save_best_only=True)
        ]

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model on test set
        y_pred = model.predict(X_test_seq)
        evaluation = evaluate_forecast(y_test_seq, y_pred, forecast_horizon, scaler_y)

        print(Fore.GREEN + "\n📊 LSTM Model Evaluation Results:" + Style.RESET_ALL)
        print(f"MSE: {evaluation['mse']:.4f}")
        print(f"MAE: {evaluation['mae']:.4f}")
        print(f"RMSE: {evaluation['rmse']:.4f}")
        print(f"Avg Direction Accuracy: {np.mean(evaluation['direction_accuracy']):.4f}")

        # Compare with best baseline if available
        if baseline_results is not None and best_baseline is not None:
            lstm_mae = evaluation['mae']
            baseline_mae = baseline_results[best_baseline]['mae']
            improvement = (baseline_mae - lstm_mae) / baseline_mae * 100

            print(Fore.GREEN + f"\nLSTM vs Best Baseline ({best_baseline}):" + Style.RESET_ALL)
            print(f"LSTM MAE: {lstm_mae:.4f}")
            print(f"Best Baseline MAE: {baseline_mae:.4f}")
            print(f"Improvement: {improvement:.2f}%")

        # Generate plots if requested
        if generate_plots:
            # Plot predictions
            if scaler_y is not None:
                y_true_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(y_test_seq.shape)
                y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

                # Only use first step predictions for visualization
                y_true_first_step = y_true_orig[:, 0]
                y_pred_first_step = y_pred_orig[:, 0]

                # Plot only a subset to avoid overcrowding
                step = max(1, len(y_true_first_step) // 50)

                plt_pred = plot_predictions(
                    y_true_first_step[::step],
                    y_pred_first_step[::step],
                    dates_test[::step],
                    title="Bitcoin Price Prediction (LSTM Model)"
                )
                pred_plot_path = f"lstm_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt_pred.savefig(pred_plot_path)
                plt_pred.close()
                print(f"✅ LSTM predictions plot saved to {pred_plot_path}")

            # Plot training history
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('MAE During Training')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()

            plt.tight_layout()
            history_plot_path = f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(history_plot_path)
            plt.close()
            print(f"✅ Training history plot saved to {history_plot_path}")

        # Save model and results if requested
        if save_model_and_results:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare metrics for saving
            metrics = {
                "mse": evaluation["mse"],
                "mae": evaluation["mae"],
                "rmse": evaluation["rmse"],
                "avg_direction_accuracy": np.mean(evaluation["direction_accuracy"])
            }

            # Add baseline comparison if available
            if baseline_results is not None and best_baseline is not None:
                metrics["best_baseline"] = best_baseline
                metrics["best_baseline_mae"] = baseline_results[best_baseline]['mae']
                metrics["improvement_over_baseline"] = improvement

            # Add cross-validation results if available
            if cv_results is not None and 'summary' in cv_results:
                for key, value in cv_results['summary'].items():
                    metrics[f"cv_{key}"] = value

            # Save results
            save_results(metrics, model_version, model_type="lstm_enhanced")

            # Save model
            save_model(model, model_type="lstm_enhanced")

            print(Fore.GREEN + f"\n✅ Enhanced LSTM model (version {model_version}) saved successfully!" + Style.RESET_ALL)

        return {
            'model': model,
            'evaluation': evaluation,
            'baseline_results': baseline_results,
            'best_baseline': best_baseline if baseline_results is not None else None,
            'cv_results': cv_results
        }

    except Exception as e:
        print(Fore.RED + f"\n❌ Error during enhanced training: {e}" + Style.RESET_ALL)

        # Clean up temp file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        return None

    finally:
        # Clean up temp file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"✅ Temporary CSV file removed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced training for Bitcoin price prediction with baselines and cross-validation')
    parser.add_argument('--input_seq_length', type=int, default=30, help='Number of days to use as input')
    parser.add_argument('--forecast_horizon', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--data_source', type=str, default="bigquery", choices=["bigquery", "local"], help='Source of data')
    parser.add_argument('--data_path', type=str, default=None, help='Path to local CSV file')
    parser.add_argument('--no_baselines', action='store_true', help='Skip baseline models')
    parser.add_argument('--no_cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--cv_type', type=str, default='time_series_split', choices=['time_series_split', 'expanding_window'], help='Type of cross-validation')
    parser.add_argument('--no_save', action='store_true', help='Skip saving model and results')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    # Validate arguments
    if args.data_source == "local" and args.data_path is None:
        print(Fore.RED + "❌ Error: You must specify --data_path when using --data_source=local" + Style.RESET_ALL)
        exit(1)

    # Run enhanced training
    enhanced_train(
        input_seq_length=args.input_seq_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        data_source=args.data_source,
        data_path=args.data_path,
        run_baselines=not args.no_baselines,
        run_cv=not args.no_cv,
        n_cv_splits=args.cv_splits,
        cv_type=args.cv_type,
        save_model_and_results=not args.no_save,
        generate_plots=not args.no_plots
    )
