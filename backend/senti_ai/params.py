import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

##################  VARIABLES  ##################
# DATA_SIZE = os.environ.get("DATA_SIZE")
# CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# MODEL_TARGET = os.environ.get("MODEL_TARGET")
# GCP_PROJECT = os.environ.get("GCP_PROJECT")
# GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
# GCP_REGION = os.environ.get("GCP_REGION")
# BQ_DATASET = os.environ.get("BQ_DATASET")
# BQ_REGION = os.environ.get("BQ_REGION")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INSTANCE = os.environ.get("INSTANCE")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
# EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")

################## VALIDATIONS #################

# env_valid_options = dict(
#     DATA_SIZE=["1k", "200k", "all"],
#     MODEL_TARGET=["local", "gcs", "mlflow"],
# )

# def validate_env_value(env, valid_options):
#     env_value = os.environ[env]
#     if env_value not in valid_options:
#         raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


# for env, valid_options in env_valid_options.items():
#     validate_env_value(env, valid_options)

GCP_PROJECT_AHMET = os.environ.get("GCP_PROJECT_AHMET")
GCP_PROJECT_BILL = os.environ.get("GCP_PROJECT_BILL")

GCP_REGION = os.environ.get("GCP_REGION")

BQ_REGION = os.environ.get("BQ_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")

AHMET_USER_KEY = os.environ.get("AHMET_USER_KEY")
KSENIA_USER_KEY = os.environ.get("KSENIA_USER_KEY")
BILL_USER_KEY = os.environ.get("BILL_USER_KEY")
PHILIP_USER_KEY = os.environ.get("PHILIP_USER_KEY")
PUSHOVER_API_TOKEN = os.environ.get("PUSHOVER_API_TOKEN")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKIN_PASSWORD = os.environ.get("MLFLOW_TRACKIN_PASSWORD")

PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")

# LSTM model settings
MODEL_NAME = os.getenv("MODEL_NAME", "sentiai_model")
MODEL_EXPERIMENT = os.getenv("MODEL_EXPERIMENT", "sentiai_lstm_experiment")
MODEL_INPUT_SEQ_LENGTH = int(os.getenv("LSTM_INPUT_SEQ_LENGTH", 30))
MODEL_FORECAST_HORIZON = int(os.getenv("LSTM_FORECAST_HORIZON", 30))
MODEL_MAX_EPOCHS = int(os.getenv("LSTM_MAX_EPOCHS", 100))
MODEL_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", 32))
MODEL_LEARNING_RATE = float(os.getenv("LSTM_LEARNING_RATE", 0.001))
MODEL_EARLY_STOPPING_PATIENCE = int(os.getenv("LSTM_EARLY_STOPPING_PATIENCE", 15))

# Data settings
DATA_SOURCE = os.getenv("DATA_SOURCE", "bigquery")
DATA_PATH = os.getenv("DATA_PATH")

# Model settings
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "basic")

# Testing settings
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
