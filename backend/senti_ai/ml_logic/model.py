import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from colorama import Fore, Style
from typing import Tuple, Dict, List

def load_and_preprocess_data(file_path, normalize=True):
    """
    Load and preprocess data for time series forecasting.

    Args:
        file_path: Path to the CSV data file
        normalize: Whether to normalize the data

    Returns:
        Preprocessed training and testing data
    """
    data = pd.read_csv(file_path)

    # Convert date to datetime and set as index
    data['date'] = pd.to_datetime(data['date'])

    # Handle outliers using IQR method
    numeric_columns = data.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        # Identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers at bounds (winsorization)
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])

    # Split data into features and target
    target_column = 'BTC_Close'
    date_column = 'date'

    columns_to_drop = [target_column, date_column]
    if 'Unnamed: 0' in data.columns:
        columns_to_drop.append('Unnamed: 0')
    X = data.drop(columns_to_drop, axis=1)
    y = data[target_column]
    dates = data[date_column]

    # Train-test split (keeping time order)
    train_size = int(len(data) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    # Scale the data
    if normalize:
        scaler_X = RobustScaler()  # Using RobustScaler to be less sensitive to outliers
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = RobustScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    else:
        X_train_scaled, X_test_scaled = X_train.values, X_test.values
        y_train_scaled, y_test_scaled = y_train.values, y_test.values
        scaler_y = None

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            dates_train, dates_test, scaler_y, X.columns)

def create_sequences(X, y, input_sequence_length, forecast_horizon):
    """
    Create sequences for multi-step forecasting.

    Args:
        X: Features array
        y: Target array
        input_sequence_length: Number of time steps to use as input
        forecast_horizon: Number of time steps to predict

    Returns:
        X_seq: Input sequences
        y_seq: Target sequences for forecasting
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - input_sequence_length - forecast_horizon + 1):
        X_seq.append(X[i:i + input_sequence_length])
        y_seq.append(y[i + input_sequence_length:i + input_sequence_length + forecast_horizon])

    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape, output_length, dropout_rate=0.2):
    """
    Build a bidirectional LSTM model for multi-step time series forecasting.

    Args:
        input_shape: Shape of input sequences (sequence_length, features)
        output_length: Length of the forecast horizon
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled keras model
    """
    print(Fore.BLUE + "\nBuilding LSTM model..." + Style.RESET_ALL)

    model = Sequential([
        # First LSTM layer
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=input_shape),
        Dropout(dropout_rate),

        # Second LSTM layer
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(dropout_rate),

        # Dense output layer for multi-step prediction
        Dense(output_length)
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae'])

    print("✅ LSTM Model built and compiled")

    return model

def analyze_feature_importance(X, y, feature_names):
    """
    Analyze feature importance using Random Forest

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: Names of features

    Returns:
        DataFrame with feature importances
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Create DataFrame of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    return feature_importance

def train_model(X_train_seq, y_train_seq, X_val_seq=None, y_val_seq=None,
                     input_shape=None, forecast_horizon=30, epochs=100, batch_size=32,
                     patience=15):
    """
    Train the LSTM model for time series forecasting.

    Args:
        X_train_seq: Training input sequences
        y_train_seq: Training target sequences
        X_val_seq: Validation input sequences (if None, will use validation_split)
        y_val_seq: Validation target sequences
        input_shape: Shape of input sequences
        forecast_horizon: Length of forecast horizon
        epochs: Number of training epochs
        batch_size: Training batch size
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    print(Fore.BLUE + "\nTraining LSTM model..." + Style.RESET_ALL)

    if input_shape is None:
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    # Build model
    model = build_model(
        input_shape=input_shape,
        output_length=forecast_horizon
    )

    # Callbacks for training
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    model_checkpoint = ModelCheckpoint(
        'best_bitcoin_model.h5', monitor='val_loss', save_best_only=True
    )

    # Determine validation approach
    validation_data = None
    validation_split = 0.2

    if X_val_seq is not None and y_val_seq is not None:
        validation_data = (X_val_seq, y_val_seq)
        validation_split = None

    # Train the model
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    print(f"✅ LSTM Model trained with min val_loss: {min(history.history['val_loss']):.4f}")

    return model, history

def evaluate_forecast(y_true, y_pred, forecast_horizon, scaler=None):
    """
    Evaluate multi-step forecast with various metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        forecast_horizon: Number of steps in the forecast
        scaler: Scaler used on the target variable (for inverse transform)

    Returns:
        Dictionary with evaluation metrics
    """
    # If data was scaled, inverse transform to original scale
    if scaler is not None:
        y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    results = {}

    # Overall metrics across all forecasting steps
    results['mse'] = mean_squared_error(y_true_orig.flatten(), y_pred_orig.flatten())
    results['mae'] = mean_absolute_error(y_true_orig.flatten(), y_pred_orig.flatten())
    results['rmse'] = np.sqrt(results['mse'])

    # Per-step metrics
    step_mse = []
    step_mae = []

    for i in range(forecast_horizon):
        step_mse.append(mean_squared_error(y_true_orig[:, i], y_pred_orig[:, i]))
        step_mae.append(mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i]))

    results['step_mse'] = step_mse
    results['step_mae'] = step_mae

    # Direction accuracy (for financial time series, direction of movement is crucial)
    def direction_accuracy(y_true, y_pred, step):
        """Calculate how often the direction of movement is correctly predicted"""
        true_direction = np.sign(y_true[:, step] - y_true[:, step-1] if step > 0 else y_true[:, 0])
        pred_direction = np.sign(y_pred[:, step] - y_pred[:, step-1] if step > 0 else y_pred[:, 0])
        return np.mean(true_direction == pred_direction)

    direction_acc = [direction_accuracy(y_true_orig, y_pred_orig, i) for i in range(forecast_horizon)]
    results['direction_accuracy'] = direction_acc

    return results

def recursive_forecast(model, initial_sequence, n_steps, scaler=None):
    """
    Generate a multi-step forecast recursively, one step at a time.

    Args:
        model: Trained model that predicts a single step
        initial_sequence: The starting sequence for prediction
        n_steps: Number of steps to forecast
        scaler: Scaler for inverse_transform

    Returns:
        Array of predictions
    """
    forecast = []
    current_sequence = initial_sequence.copy()

    for _ in range(n_steps):
        # Reshape for model (batch_size, timesteps, features)
        current_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])

        # Predict next step
        next_step = model.predict(current_input, verbose=0)[0]
        forecast.append(next_step)

        # Update sequence for next prediction
        # We remove the oldest timestep and add our prediction as the newest timestep
        new_sequence = np.append(current_sequence[1:], np.array([[next_step]]), axis=0)

        # Create a new input sequence with updated values
        current_sequence = new_sequence

    forecast = np.array(forecast)

    # Inverse transform if scaler provided
    if scaler is not None:
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).reshape(forecast.shape)

    return forecast

# Visualization functions
def plot_predictions(true_values, predicted_values, dates=None, title="Bitcoin Price Prediction"):
    """
    Plot the true vs predicted values.

    Args:
        true_values: Array of true values
        predicted_values: Array of predicted values
        dates: Optional dates for x-axis
        title: Plot title
    """
    plt.figure(figsize=(12, 6))

    if dates is not None:
        plt.plot(dates, true_values, label='Actual', color='blue')
        plt.plot(dates, predicted_values, label='Predicted', color='red', linestyle='--')
    else:
        plt.plot(true_values, label='Actual', color='blue')
        plt.plot(predicted_values, label='Predicted', color='red', linestyle='--')

    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # If we have many data points, rotate date labels
    if dates is not None and len(dates) > 20:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return plt

def plot_feature_importance(feature_importance_df, top_n=10):
    """
    Plot feature importance

    Args:
        feature_importance_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Number of top features to display
    """
    top_features = feature_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    return plt

def plot_error_by_forecast_step(step_errors, metric='MAE'):
    """
    Plot error by forecast step

    Args:
        step_errors: List of errors by step
        metric: Name of the metric (for labeling)
    """
    steps = range(1, len(step_errors) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, step_errors, marker='o')
    plt.title(f'{metric} by Forecast Step')
    plt.xlabel('Forecast Step')
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.xticks(steps)
    plt.tight_layout()
    return plt

def run_bitcoin_prediction(file_path, input_seq_length=30, forecast_horizon=30, epochs=100):
    """
    Run the complete Bitcoin prediction workflow

    Args:
        file_path: Path to the CSV data file
        input_seq_length: Number of days to use as input
        forecast_horizon: Number of days to forecast
        epochs: Number of training epochs

    Returns:
        Trained model and evaluation results
    """
    print(Fore.BLUE + f"\nRunning Bitcoin price prediction with LSTM model (input: {input_seq_length} days, forecast: {forecast_horizon} days)..." + Style.RESET_ALL)

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler_y, feature_names = load_and_preprocess_data(file_path)

    # 2. Feature importance analysis
    feature_importance = analyze_feature_importance(X_train, y_train, feature_names)
    print("Top 10 Important Features:")
    print(feature_importance.head(10))

    # 3. Create sequences for multi-step forecasting
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, input_seq_length, forecast_horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, input_seq_length, forecast_horizon)

    print(f"Training shape: X={X_train_seq.shape}, y={y_train_seq.shape}")
    print(f"Testing shape: X={X_test_seq.shape}, y={y_test_seq.shape}")

    # 4. Train the model
    model, history = train_model(
        X_train_seq,
        y_train_seq,
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        forecast_horizon=forecast_horizon,
        epochs=epochs
    )

    # 5. Evaluate model
    y_pred = model.predict(X_test_seq)

    # Calculate metrics
    evaluation = evaluate_forecast(y_test_seq, y_pred, forecast_horizon, scaler_y)

    print(f"MSE: {evaluation['mse']:.4f}")
    print(f"MAE: {evaluation['mae']:.4f}")
    print(f"RMSE: {evaluation['rmse']:.4f}")
    print(f"Average Direction Accuracy: {np.mean(evaluation['direction_accuracy']):.4f}")

    return model, evaluation, feature_importance, history
