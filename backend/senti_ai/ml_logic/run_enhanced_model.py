#!/usr/bin/env python

import os
import click
from colorama import Fore, Style
from senti_ai.params import *
from senti_ai.ml_logic.enhanced_model_training import enhanced_train

@click.command()
@click.option('--input_seq_length', default=LSTM_INPUT_SEQ_LENGTH, help='Number of days to use as input')
@click.option('--forecast_horizon', default=LSTM_FORECAST_HORIZON, help='Number of days to forecast')
@click.option('--epochs', default=LSTM_MAX_EPOCHS, help='Maximum number of training epochs')
@click.option('--data_source', default=DATA_SOURCE, type=click.Choice(['bigquery', 'local']), help='Source of data')
@click.option('--data_path', default=DATA_PATH, help='Path to local CSV file')
@click.option('--skip_baselines', is_flag=True, help='Skip baseline models')
@click.option('--skip_cv', is_flag=True, help='Skip cross-validation')
@click.option('--cv_splits', default=5, help='Number of cross-validation splits')
@click.option('--cv_type', default='time_series_split', type=click.Choice(['time_series_split', 'expanding_window']), help='Type of cross-validation')
@click.option('--skip_save', is_flag=True, help='Skip saving model and results')
@click.option('--skip_plots', is_flag=True, help='Skip generating plots')
def run(input_seq_length, forecast_horizon, epochs, data_source, data_path,
        skip_baselines, skip_cv, cv_splits, cv_type, skip_save, skip_plots):
    """
    Run enhanced Bitcoin price prediction model with baseline comparison and cross-validation.
    """
    print(Fore.GREEN + "\n🚀 Running Enhanced Bitcoin Price Prediction Model 🚀" + Style.RESET_ALL)

    # Validate arguments
    if data_source == "local" and data_path is None:
        print(Fore.RED + "❌ Error: You must specify --data_path when using --data_source=local" + Style.RESET_ALL)
        return

    print(Fore.BLUE + "\nConfiguration:" + Style.RESET_ALL)
    print(f"- Input sequence length: {input_seq_length} days")
    print(f"- Forecast horizon: {forecast_horizon} days")
    print(f"- Max epochs: {epochs}")
    print(f"- Data source: {data_source}")
    print(f"- Run baseline models: {'No' if skip_baselines else 'Yes'}")
    print(f"- Run cross-validation: {'No' if skip_cv else 'Yes'}")
    if not skip_cv:
        print(f"- CV splits: {cv_splits}")
        print(f"- CV type: {cv_type}")
    print(f"- Save model and results: {'No' if skip_save else 'Yes'}")
    print(f"- Generate plots: {'No' if skip_plots else 'Yes'}")

    # Run enhanced training
    results = enhanced_train(
        input_seq_length=input_seq_length,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        data_source=data_source,
        data_path=data_path,
        run_baselines=not skip_baselines,
        run_cv=not skip_cv,
        n_cv_splits=cv_splits,
        cv_type=cv_type,
        save_model_and_results=not skip_save,
        generate_plots=not skip_plots
    )

    if results is not None:
        print(Fore.GREEN + "\n✅ Enhanced model training completed successfully!" + Style.RESET_ALL)

        # Display some key results
        evaluation = results['evaluation']
        print(Fore.BLUE + "\nLSTM Model Performance:" + Style.RESET_ALL)
        print(f"- MAE: {evaluation['mae']:.4f}")
        print(f"- RMSE: {evaluation['rmse']:.4f}")
        print(f"- Direction Accuracy: {np.mean(evaluation['direction_accuracy']):.4f}")

        if results['baseline_results'] is not None and results['best_baseline'] is not None:
            best_baseline = results['best_baseline']
            baseline_mae = results['baseline_results'][best_baseline]['mae']
            lstm_mae = evaluation['mae']
            improvement = (baseline_mae - lstm_mae) / baseline_mae * 100

            print(Fore.BLUE + "\nComparison to Best Baseline:" + Style.RESET_ALL)
            print(f"- Best baseline model: {best_baseline}")
            print(f"- Baseline MAE: {baseline_mae:.4f}")
            print(f"- Improvement: {improvement:.2f}%")

        if results['cv_results'] is not None and 'summary' in results['cv_results']:
            print(Fore.BLUE + "\nCross-Validation Results:" + Style.RESET_ALL)
            if 'mae_mean' in results['cv_results']['summary']:
                print(f"- CV MAE: {results['cv_results']['summary']['mae_mean']:.4f} ± {results['cv_results']['summary']['mae_std']:.4f}")
            if 'rmse_mean' in results['cv_results']['summary']:
                print(f"- CV RMSE: {results['cv_results']['summary']['rmse_mean']:.4f} ± {results['cv_results']['summary']['rmse_std']:.4f}")
    else:
        print(Fore.RED + "\n❌ Enhanced model training failed!" + Style.RESET_ALL)

if __name__ == '__main__':
    run()
