# LSTM Model for Bitcoin Price Prediction

This module implements an advanced LSTM (Long Short-Term Memory) model for Bitcoin price prediction, based on the model4.ipynb implementation.

## Features

- **Bidirectional LSTM Architecture**: Uses bidirectional LSTM layers for better capture of time series patterns
- **Multi-step Forecasting**: Capable of forecasting Bitcoin prices for multiple days ahead
- **Feature Importance Analysis**: Analyzes and visualizes the importance of different features
- **Comprehensive Evaluation**: Provides multiple metrics including direction accuracy for trading decisions
- **Sequence Creation**: Creates time-based sequences for time series forecasting

## Files Structure

- `ml_logic/model.py`: Main implementation of the LSTM model
- `model_train.py`: Script to train the LSTM model
- `model_pred.py`: Script to make predictions using the trained LSTM model
- `ml_logic/registry.py`: Updated to handle both the basic and LSTM models

## Usage

### Training the model

```bash
python -m senti_ai.train_lstm --input_seq_length 30 --forecast_horizon 30 --epochs 100
```

Options:
- `--input_seq_length`: Number of days to use as input (default: 30)
- `--forecast_horizon`: Number of days to forecast (default: 30)
- `--epochs`: Maximum number of training epochs (default: 100)
- `--data_source`: Source of data ("bigquery" or "local", default: "bigquery")
- `--data_path`: Path to local CSV file (required if data_source is "local")

### Making predictions

```bash
python -m senti_ai.predict_lstm --input_seq_length 30 --forecast_horizon 30
```

Options:
- `--input_seq_length`: Number of days to use as input (default: 30)
- `--forecast_horizon`: Number of days to forecast (default: 30)
- `--data_source`: Source of data ("bigquery" or "local", default: "bigquery")
- `--data_path`: Path to local CSV file (required if data_source is "local")
- `--no_plot`: Disable plot generation

## Model Architecture

1. **Input Layer**: Takes sequences of length `input_seq_length` with multiple features
2. **First LSTM Layer**: Bidirectional LSTM with 128 units and dropout
3. **Second LSTM Layer**: Bidirectional LSTM with 64 units and dropout
4. **Output Layer**: Dense layer with `forecast_horizon` units for multi-step prediction

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Measures average squared difference between predicted and actual values
- **MAE (Mean Absolute Error)**: Measures average absolute difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, useful for interpretation in the original scale
- **Direction Accuracy**: Measures how often the model correctly predicts the direction of price movement (up or down)

## Integration with Existing System

The LSTM model is integrated with the existing MLflow-based model registry system. It can be loaded and saved using the same registry functions as the basic model, with the addition of a `model_type` parameter to specify which model to use.

## Preprocessing

The model uses robust preprocessing techniques:
- Outlier handling with IQR-based winsorization
- Feature scaling with RobustScaler
- Sequence creation for time series forecasting

## Example Workflow

1. Load and preprocess data from BigQuery or local CSV
2. Create sequences for training and testing
3. Train the LSTM model with appropriate callbacks
4. Evaluate model performance on test data
5. Save model and results to MLflow
6. Generate future predictions and visualizations

## Future Improvements

- Implement hyperparameter tuning for optimal model configuration
- Add more advanced features such as sentiment analysis from news and social media
- Incorporate economic indicators and market sentiment metrics
- Implement ensemble methods combining LSTM with other models
