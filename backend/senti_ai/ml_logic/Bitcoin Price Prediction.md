# Bitcoin Price Prediction System

This repository contains a complete machine learning system for predicting Bitcoin prices using time series forecasting techniques. The system employs both a basic model and an LSTM-based deep learning approach to forecast future Bitcoin prices.

## Table of Contents

- [Setup](#setup)
- [Using the Makefile](#using-the-makefile)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Model Deployment](#model-deployment)
- [Complete Pipelines](#complete-pipelines)
- [Testing](#testing)
- [Clean-up and Reset](#clean-up-and-reset)

## Setup

Before using the system, ensure you have all the required dependencies installed and your environment properly configured.

1. Install the package:
```bash
make reinstall_package
```

2. Set up MLflow for experiment tracking:
```bash
make setup_mlflow
```

## Using the Makefile

The project includes a comprehensive Makefile to simplify common operations. Use `make help` to see all available commands and their descriptions.

## Model Training

### Train the Basic Model
```bash
make train_basic
```

### Train the LSTM Model
With default parameters:
```bash
make train_lstm
```

With custom parameters:
```bash
make train_lstm_custom INPUT_SEQ_LENGTH=60 FORECAST_HORIZON=30 EPOCHS=150
```

Parameters:
- `INPUT_SEQ_LENGTH`: Number of days to use as input for prediction
- `FORECAST_HORIZON`: Number of days to forecast
- `EPOCHS`: Maximum number of training epochs

## Making Predictions

### Predict with Basic Model
```bash
make predict_basic
```

### Predict with LSTM Model
With default parameters:
```bash
make predict_lstm
```

With custom parameters:
```bash
make predict_lstm_custom INPUT_SEQ_LENGTH=60 FORECAST_HORIZON=30
```

## Model Deployment

### Deploy to Staging
```bash
make deploy_staging MODEL_TYPE=lstm
```
or
```bash
make deploy_staging MODEL_TYPE=basic
```

### Deploy to Production
```bash
make deploy_production MODEL_TYPE=lstm
```
or
```bash
make deploy_production MODEL_TYPE=basic
```

## Complete Pipelines

### Run Full LSTM Pipeline
This sets up MLflow, trains the LSTM model, and deploys it to staging:
```bash
make run_lstm_pipeline
```

### Run Full Basic Model Pipeline
This sets up MLflow, trains the basic model, and deploys it to staging:
```bash
make run_basic_pipeline
```

## Testing

### Test GCP Setup
```bash
make test_gcp_setup
```

### Test API Endpoints
```bash
make test_api_root
make test_api_predict
make test_api_on_docker
make test_api_on_prod
```

### Test LSTM Model
```bash
make test_lstm
```

## Clean-up and Reset

### Clean Generated Files
```bash
make clean
```

### Clean MLflow Artifacts
```bash
make clean_mlflow
```

### Reset Data Files
Reset local files:
```bash
make reset_local_files
```

Reset BigQuery files:
```bash
make reset_bq_files
```

Reset Google Cloud Storage files:
```bash
make reset_gcs_files
```

Reset all files at once:
```bash
make reset_all_files
```

## Starting Fresh

If you want to restart the entire setup and begin again:

1. Clean up everything:
```bash
make clean_mlflow
make clean
make reset_all_files
```

2. Set up MLflow again:
```bash
make setup_mlflow
```

3. Start your model training pipeline from scratch:
```bash
make run_lstm_pipeline
```
or
```bash
make run_basic_pipeline
```

## Environment Variables

Make sure to set the necessary environment variables either in your shell or in a `.env` file. Key variables include:

- `GCP_PROJECT_AHMET`: Google Cloud Project ID
- `BQ_DATASET`: BigQuery dataset name
- `MLFLOW_TRACKING_URI`: URI for MLflow tracking server
- `MLFLOW_MODEL_NAME`: Name for the registered model
- `LSTM_INPUT_SEQ_LENGTH`: Default length of input sequence
- `LSTM_FORECAST_HORIZON`: Default forecast horizon
- `LSTM_MAX_EPOCHS`: Default maximum training epochs
- `DATA_SOURCE`: Source of data ("bigquery" or "local")
- `DATA_PATH`: Path to local CSV file (if using local data)

## API

The system includes a FastAPI backend that can be run with:
```bash
make run_api
```

## Workflow

The entire workflow can be orchestrated with Prefect:
```bash
make run_workflow
```
