import mlflow
from senti_ai.params import *
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_model():
    model_uri = f"bitcoin_price_prediction_model_test.h5"
    model = mlflow.sklearn.load_model(model_uri)
    mlflow.tensorflow.log_model(model=model,
                        artifact_path="model",
                        registered_model_name=MLFLOW_MODEL_NAME,
                        )
    return model

def save_model():
    return None

def save_results():
    return None
