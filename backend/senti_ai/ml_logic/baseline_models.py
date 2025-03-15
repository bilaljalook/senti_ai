import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from colorama import Fore, Style
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

def naive_forecast(y_train: np.ndarray, forecast_horizon: int) -> np.ndarray:
    """
    Generate a naive forecast by using the last observed value.

    Args:
        y_train: Training target values
        forecast_horizon: Number of steps to forecast

    Returns:
        Forecasted values
    """
    # Use the last value in training set for all forecasts
    last_value = y_train[-1]
    return np.full(forecast_horizon, last_value)

def seasonal_naive_forecast(y_train: np.ndarray, forecast_horizon: int, season_length: int = 7) -> np.ndarray:
    """
    Generate a seasonal naive forecast by using the value from the same time in the previous cycle.

    Args:
        y_train: Training target values
        forecast_horizon: Number of steps to forecast
        season_length: Length of the seasonal cycle (e.g., 7 for weekly patterns)

    Returns:
        Forecasted values
    """
    # Make sure we have enough data for at least one complete season
    if len(y_train) < season_length:
        return naive_forecast(y_train, forecast_horizon)

    forecast = []
    for i in range(forecast_horizon):
        # Use value from the same position in the previous season
        idx = len(y_train) - season_length + (i % season_length)
        forecast.append(y_train[idx])

    return np.array(forecast)

def moving_average_forecast(y_train: np.ndarray, forecast_horizon: int, window: int = 7) -> np.ndarray:
    """
    Generate a forecast using simple moving average.

    Args:
        y_train: Training target values
        forecast_horizon: Number of steps to forecast
        window: Size of the moving average window

    Returns:
        Forecasted values
    """
    # Use the average of the last window values for all forecasts
    window = min(window, len(y_train))
    avg_value = np.mean(y_train[-window:])
    return np.full(forecast_horizon, avg_value)

def linear_regression_forecast(X_train: np.ndarray, y_train: np.ndarray,
                               X_forecast: np.ndarray) -> np.ndarray:
    """
    Generate a forecast using linear regression.

    Args:
        X_train: Training feature values
        y_train: Training target values
        X_forecast: Feature values for forecasting period

    Returns:
        Forecasted values
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_forecast)

def arima_forecast(y_train: np.ndarray, forecast_horizon: int, order: Tuple[int, int, int] = (5, 1, 0)) -> np.ndarray:
    """
    Generate a forecast using ARIMA model.

    Args:
        y_train: Training target values
        forecast_horizon: Number of steps to forecast
        order: ARIMA order (p, d, q)

    Returns:
        Forecasted values
    """
    try:
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast
    except Exception as e:
        print(f"ARIMA model failed with error: {e}")
        return naive_forecast(y_train, forecast_horizon)

def exponential_smoothing_forecast(y_train: np.ndarray, forecast_horizon: int) -> np.ndarray:
    """
    Generate a forecast using simple exponential smoothing.

    Args:
        y_train: Training target values
        forecast_horizon: Number of steps to forecast

    Returns:
        Forecasted values
    """
    try:
        model = ExponentialSmoothing(y_train)
        model_fit = model.fit()
        forecast = model_fit.forecast(forecast_horizon)
        return forecast
    except Exception as e:
        print(f"Exponential smoothing failed with error: {e}")
        return naive_forecast(y_train, forecast_horizon)

def create_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           forecast_horizon: int = 1) -> Dict:
    """
    Create and evaluate several baseline models.

    Args:
        X_train: Training feature values
        y_train: Training target values
        X_test: Test feature values
        y_test: Test target values
        forecast_horizon: Number of steps to forecast

    Returns:
        Dictionary with evaluation results for each model
    """
    print(Fore.BLUE + "\nCreating baseline models..." + Style.RESET_ALL)

    results = {}

    # Naive forecast
    y_pred_naive = naive_forecast(y_train, len(y_test))
    results['naive'] = {
        'mse': mean_squared_error(y_test, y_pred_naive),
        'mae': mean_absolute_error(y_test, y_pred_naive),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_naive)),
        'predictions': y_pred_naive
    }
    print(f"✅ Naive forecast: MSE = {results['naive']['mse']:.4f}, MAE = {results['naive']['mae']:.4f}")

    # Seasonal naive forecast (assuming weekly seasonality)
    y_pred_seasonal = seasonal_naive_forecast(y_train, len(y_test), season_length=7)
    results['seasonal_naive'] = {
        'mse': mean_squared_error(y_test, y_pred_seasonal),
        'mae': mean_absolute_error(y_test, y_pred_seasonal),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_seasonal)),
        'predictions': y_pred_seasonal
    }
    print(f"✅ Seasonal naive forecast: MSE = {results['seasonal_naive']['mse']:.4f}, MAE = {results['seasonal_naive']['mae']:.4f}")

    # Moving average
    y_pred_ma = moving_average_forecast(y_train, len(y_test))
    results['moving_average'] = {
        'mse': mean_squared_error(y_test, y_pred_ma),
        'mae': mean_absolute_error(y_test, y_pred_ma),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ma)),
        'predictions': y_pred_ma
    }
    print(f"✅ Moving average forecast: MSE = {results['moving_average']['mse']:.4f}, MAE = {results['moving_average']['mae']:.4f}")

    # Linear Regression (if features are available)
    if X_train is not None and X_test is not None:
        try:
            y_pred_lr = linear_regression_forecast(X_train, y_train, X_test)
            results['linear_regression'] = {
                'mse': mean_squared_error(y_test, y_pred_lr),
                'mae': mean_absolute_error(y_test, y_pred_lr),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                'predictions': y_pred_lr
            }
            print(f"✅ Linear regression forecast: MSE = {results['linear_regression']['mse']:.4f}, MAE = {results['linear_regression']['mae']:.4f}")
        except Exception as e:
            print(f"❌ Linear regression failed: {e}")

    # ARIMA model (only if time series is not too long)
    if len(y_train) < 5000:  # Avoid running ARIMA on very long series
        try:
            y_pred_arima = arima_forecast(y_train, len(y_test))
            results['arima'] = {
                'mse': mean_squared_error(y_test, y_pred_arima),
                'mae': mean_absolute_error(y_test, y_pred_arima),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_arima)),
                'predictions': y_pred_arima
            }
            print(f"✅ ARIMA forecast: MSE = {results['arima']['mse']:.4f}, MAE = {results['arima']['mae']:.4f}")
        except Exception as e:
            print(f"❌ ARIMA model failed: {e}")

    # Exponential smoothing
    try:
        y_pred_es = exponential_smoothing_forecast(y_train, len(y_test))
        results['exp_smoothing'] = {
            'mse': mean_squared_error(y_test, y_pred_es),
            'mae': mean_absolute_error(y_test, y_pred_es),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_es)),
            'predictions': y_pred_es
        }
        print(f"✅ Exponential smoothing forecast: MSE = {results['exp_smoothing']['mse']:.4f}, MAE = {results['exp_smoothing']['mae']:.4f}")
    except Exception as e:
        print(f"❌ Exponential smoothing failed: {e}")

    return results

def plot_baseline_comparisons(y_test: np.ndarray, baseline_results: Dict, dates_test=None, title="Baseline Model Comparison"):
    """
    Plot the true vs predicted values for baseline models.

    Args:
        y_test: True values from test set
        baseline_results: Dictionary with baseline model results
        dates_test: Optional dates for x-axis
        title: Plot title
    """
    plt.figure(figsize=(14, 8))

    # Plot actual values
    if dates_test is not None:
        plt.plot(dates_test, y_test, label='Actual', color='black', linewidth=2)
    else:
        plt.plot(y_test, label='Actual', color='black', linewidth=2)

    # Plot predictions for each model
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    color_idx = 0

    for model_name, result in baseline_results.items():
        if 'predictions' in result:
            if dates_test is not None:
                plt.plot(dates_test, result['predictions'], label=f'{model_name} (MAE={result["mae"]:.2f})',
                         color=colors[color_idx % len(colors)], linestyle='--')
            else:
                plt.plot(result['predictions'], label=f'{model_name} (MAE={result["mae"]:.2f})',
                         color=colors[color_idx % len(colors)], linestyle='--')
            color_idx += 1

    plt.title(title)
    plt.xlabel('Date' if dates_test is not None else 'Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # If we have many data points, rotate date labels
    if dates_test is not None and len(dates_test) > 20:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return plt

def get_best_baseline_model(baseline_results: Dict, metric: str = 'mae') -> str:
    """
    Get the name of the best performing baseline model.

    Args:
        baseline_results: Dictionary with baseline model results
        metric: Metric to use for comparison ('mae', 'mse', or 'rmse')

    Returns:
        Name of the best model
    """
    if not baseline_results:
        return None

    best_model = None
    best_score = float('inf')

    for model_name, result in baseline_results.items():
        if metric in result and result[metric] < best_score:
            best_score = result[metric]
            best_model = model_name

    return best_model
