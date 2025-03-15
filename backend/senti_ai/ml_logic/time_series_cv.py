import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from colorama import Fore, Style
from typing import Dict, List, Tuple, Union, Callable

def create_robust_sequences(X, y, input_seq_length, forecast_horizon):
    """
    Create sequences with validation and error handling for cross-validation.
    """
    # Validation
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, but have shapes {X.shape} and {y.shape}")

    # Check if we have enough data
    if len(X) < input_seq_length + forecast_horizon:
        raise ValueError(
            f"Not enough data points ({len(X)}) for input_seq_length={input_seq_length} + "
            f"forecast_horizon={forecast_horizon}. Need at least {input_seq_length + forecast_horizon}."
        )

    # Calculate valid samples
    valid_samples = len(X) - input_seq_length - forecast_horizon + 1

    X_seq, y_seq = [], []

    for i in range(valid_samples):
        X_seq.append(X[i:i + input_seq_length])
        y_seq.append(y[i + input_seq_length:i + input_seq_length + forecast_horizon])

    return np.array(X_seq), np.array(y_seq)

def time_series_cross_validation(
    X, y, build_model_fn, input_seq_length, forecast_horizon, create_sequences_fn, evaluate_fn,
    n_splits=5, max_train_size=None, gap=0, verbose=True
):
    print(Fore.BLUE + f"\nPerforming time series cross-validation with {n_splits} splits..." + Style.RESET_ALL)

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, gap=gap)

    # Initialize results dictionary
    cv_results = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'direction_accuracy': []
    }

    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ]

    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if verbose:
            print(f"\nFold {fold+1}/{n_splits}")
            print(f"Train indices: {train_idx[0]} to {train_idx[-1]}, Test indices: {test_idx[0]} to {test_idx[-1]}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Ensure train set has enough data for sequences
        if len(X_train) < input_seq_length + forecast_horizon:
            print(f"Skipping fold {fold+1}: train set too small for sequences")
            continue

        # Ensure test set has enough data for evaluation
        if len(X_test) < input_seq_length + forecast_horizon:
            print(f"Skipping fold {fold+1}: test set too small for evaluation")
            continue

        try:
            # Create sequences
            X_train_seq, y_train_seq = create_robust_sequences(X_train, y_train, input_seq_length, forecast_horizon)
            X_test_seq, y_test_seq = create_robust_sequences(X_test, y_test, input_seq_length, forecast_horizon)
            # Skip if sequence creation failed
            if len(X_train_seq) == 0 or len(y_train_seq) == 0 or len(X_test_seq) == 0 or len(y_test_seq) == 0:
                print(f"Skipping fold {fold+1}: failed to create sequences")
                continue

            if verbose:
                print(f"Sequence shapes - X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")
                print(f"Sequence shapes - X_test: {X_test_seq.shape}, y_test: {y_test_seq.shape}")

            # Verify matching dimensions
            if X_train_seq.shape[0] != y_train_seq.shape[0]:
                print(f"Shape mismatch in training sequences: X={X_train_seq.shape}, y={y_train_seq.shape}")
                continue
            if X_test_seq.shape[0] != y_test_seq.shape[0]:
                print(f"Shape mismatch in test sequences: X={X_test_seq.shape}, y={y_test_seq.shape}")
                continue
        except Exception as e:
            print(f"Error creating sequences for fold {fold+1}: {e}")
            continue

        # Skip fold if not enough data for sequences
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Skipping fold {fold+1} due to insufficient data for sequences")
            continue

        # Build and train model
        model = build_model_fn(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                              output_length=forecast_horizon)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,  # Reduced epochs for CV
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0 if not verbose else 1
        )

        # Evaluate model
        y_pred = model.predict(X_test_seq, verbose=0)
        fold_evaluation = evaluate_fn(y_test_seq, y_pred, forecast_horizon)

        # Record metrics
        cv_results['mae'].append(fold_evaluation['mae'])
        cv_results['mse'].append(fold_evaluation['mse'])
        cv_results['rmse'].append(fold_evaluation['rmse'])

        if 'direction_accuracy' in fold_evaluation:
            cv_results['direction_accuracy'].append(np.mean(fold_evaluation['direction_accuracy']))

        if verbose:
            print(f"Fold {fold+1} - MAE: {fold_evaluation['mae']:.4f}, RMSE: {fold_evaluation['rmse']:.4f}")

    # Calculate mean and std for each metric
    cv_summary = {}
    for metric in cv_results:
        if cv_results[metric]:  # Check if we have any values for this metric
            cv_summary[f'{metric}_mean'] = np.mean(cv_results[metric])
            cv_summary[f'{metric}_std'] = np.std(cv_results[metric])

    if verbose:
        print("\nCross-validation results:")
        for metric in ['mae', 'mse', 'rmse', 'direction_accuracy']:
            if f'{metric}_mean' in cv_summary:
                print(f"{metric.upper()}: {cv_summary[f'{metric}_mean']:.4f} ± {cv_summary[f'{metric}_std']:.4f}")

    return {'fold_results': cv_results, 'summary': cv_summary}

def plot_cv_error_distribution(cv_results: Dict, metric: str = 'mae'):
    """
    Plot the distribution of errors across CV folds.

    Args:
        cv_results: Dictionary with cross-validation results
        metric: Metric to plot ('mae', 'mse', or 'rmse')
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))

    if metric in cv_results['fold_results'] and len(cv_results['fold_results'][metric]) > 0:
        sns.boxplot(y=cv_results['fold_results'][metric])
        plt.title(f'Distribution of {metric.upper()} across CV folds')
        plt.ylabel(metric.upper())
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt

def expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    build_model_fn: Callable,
    input_seq_length: int,
    forecast_horizon: int,
    create_sequences_fn: Callable,
    evaluate_fn: Callable,
    initial_train_size: float = 0.5,
    stride: int = 30,
    n_windows: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Perform expanding window cross-validation for a deep learning model.

    Args:
        X: Feature array
        y: Target array
        build_model_fn: Function to build the model
        input_seq_length: Length of input sequences
        forecast_horizon: Number of steps to forecast
        create_sequences_fn: Function to create sequences
        evaluate_fn: Function to evaluate the model
        initial_train_size: Initial proportion of data to use for training
        stride: Number of steps between each window
        n_windows: Number of expanding windows
        verbose: Whether to print progress

    Returns:
        Dictionary with cross-validation results
    """
    print(Fore.BLUE + f"\nPerforming expanding window cross-validation with {n_windows} windows..." + Style.RESET_ALL)

    data_size = len(X)
    initial_train_end = int(data_size * initial_train_size)

    # Initialize results dictionary
    cv_results = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'direction_accuracy': []
    }

    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    ]

    # Perform expanding window cross-validation
    for window in range(n_windows):
        train_end = initial_train_end + window * stride
        test_start = train_end
        test_end = min(test_start + forecast_horizon, data_size)

        # Skip if we've reached the end of the data
        if test_start >= data_size - 1:
            break

        if verbose:
            print(f"\nWindow {window+1}/{n_windows}")
            print(f"Train: [0:{train_end}], Test: [{test_start}:{test_end}]")

        # Split data
        X_train, X_test = X[:train_end], X[test_start:test_end]
        y_train, y_test = y[:train_end], y[test_start:test_end]

        # Create sequences
        X_train_seq, y_train_seq = create_sequences_fn(X_train, y_train, input_seq_length, forecast_horizon)

        # For test data, we only need one sequence
        X_test_seq = X_test[:input_seq_length].reshape(1, input_seq_length, X_test.shape[1])
        y_test_seq = y_test[:forecast_horizon].reshape(1, forecast_horizon)

        if verbose:
            print(f"Sequence shapes - X_train: {X_train_seq.shape}, X_test: {X_test_seq.shape}")

        # Skip window if not enough data for sequences
        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Skipping window {window+1} due to insufficient data for sequences")
            continue

        # Build and train model
        model = build_model_fn(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                              output_length=forecast_horizon)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,  # Reduced epochs for CV
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0 if not verbose else 1
        )

        # Evaluate model
        y_pred = model.predict(X_test_seq, verbose=0)

        # Ensure y_test_seq and y_pred have the same shape
        min_length = min(len(y_test_seq[0]), len(y_pred[0]))
        y_test_seq_trimmed = y_test_seq[:, :min_length]
        y_pred_trimmed = y_pred[:, :min_length]

        window_evaluation = evaluate_fn(y_test_seq_trimmed, y_pred_trimmed, min_length)

        # Record metrics
        cv_results['mae'].append(window_evaluation['mae'])
        cv_results['mse'].append(window_evaluation['mse'])
        cv_results['rmse'].append(window_evaluation['rmse'])

        if 'direction_accuracy' in window_evaluation:
            cv_results['direction_accuracy'].append(np.mean(window_evaluation['direction_accuracy']))

        if verbose:
            print(f"Window {window+1} - MAE: {window_evaluation['mae']:.4f}, RMSE: {window_evaluation['rmse']:.4f}")

    # Calculate mean and std for each metric
    cv_summary = {}
    for metric in cv_results:
        if cv_results[metric]:  # Check if we have any values for this metric
            cv_summary[f'{metric}_mean'] = np.mean(cv_results[metric])
            cv_summary[f'{metric}_std'] = np.std(cv_results[metric])

    if verbose:
        print("\nExpanding window cross-validation results:")
        for metric in ['mae', 'mse', 'rmse', 'direction_accuracy']:
            if f'{metric}_mean' in cv_summary:
                print(f"{metric.upper()}: {cv_summary[f'{metric}_mean']:.4f} ± {cv_summary[f'{metric}_std']:.4f}")

    return {'window_results': cv_results, 'summary': cv_summary}
