import mlflow
from senti_ai.params import *
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from tensorflow.keras.models import load_model as tf_load_model
import os
import functools
import time
from datetime import datetime

def mlflow_run(experiment_name=None):
    """
    Decorator to create an MLflow run for a function.

    Args:
        experiment_name: Name of the MLflow experiment
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set up MLflow
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            # Start MLflow run
            with mlflow.start_run() as run:
                # Get run ID for logging
                run_id = run.info.run_id
                print(f"MLflow run started with ID: {run_id}")

                # Add run_id to kwargs
                kwargs['run_id'] = run_id

                # Execute the function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Log execution time
                mlflow.log_metric("execution_time_seconds", execution_time)

                # Log timestamp
                mlflow.log_param("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                print(f"MLflow run {run_id} completed in {execution_time:.2f} seconds")
                return result
        return wrapper
    return decorator

def log_model_metrics(metrics_dict, model_type="basic"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the function
            result = func(*args, **kwargs)
            # Log metrics if they exist in the result
            if isinstance(result, tuple) and len(result) >= 1:
                metrics = result[0]
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)

            # mlflow.log_param("model_type", model_type)
            return result
        return wrapper
    return decorator

@mlflow_run(experiment_name=f"{MLFLOW_MODEL_NAME}_experiments")
def load_model(model_type="basic", run_id=None):
    """
    Load either the basic model or the LSTM model.

    Args:
        model_type: Type of model to load ("basic" or "lstm")
        run_id: MLflow run ID

    Returns:
        Loaded model
    """
    # Log model type
    mlflow.log_param("model_action", "load")
    mlflow.log_param("model_type", model_type)

    if model_type == "lstm":
        model_uri = f"best_bitcoin_model.h5"
        if os.path.exists(model_uri):
            model = tf_load_model(model_uri)
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name=f"{MLFLOW_MODEL_NAME}_lstm",
            )
            mlflow.log_param("model_loaded", True)
        else:
            print(f"❌ LSTM model file not found at {model_uri}")
            mlflow.log_param("model_loaded", False)
            model = None
    else:
        model_uri = f"bitcoin_price_prediction_model_test.h5"
        if os.path.exists(model_uri):
            model = mlflow.sklearn.load_model(model_uri)
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME,
            )
            mlflow.log_param("model_loaded", True)
        else:
            print(f"❌ Basic model file not found at {model_uri}")
            mlflow.log_param("model_loaded", False)
            model = None

    return model

@mlflow_run(experiment_name=f"{MLFLOW_MODEL_NAME}_experiments")
@log_model_metrics({"model_saved": 1.0})
def save_model(model, model_type="basic", run_id=None):
    """
    Save model to disk and register with MLflow.

    Args:
        model: Model to save
        model_type: Type of model ("basic" or "lstm")
        run_id: MLflow run ID
    """
    # Log model type
    mlflow.log_param("model_action", "save")
    mlflow.log_param("model_type", model_type)

    if model_type == "lstm":
        model_path = "best_bitcoin_model.h5"
        model.save(model_path)

        # Log model architecture summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        mlflow.log_text("\n".join(model_summary), "model_summary.txt")

        # Register model with MLflow
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=f"{MLFLOW_MODEL_NAME}_lstm",
        )
        print(f"✅ LSTM model saved to {model_path} and registered in MLflow")
    else:
        model_path = "bitcoin_price_prediction_model_test.h5"
        mlflow.tensorflow.save_model(model, model_path)

        # Register model with MLflow
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )
        print(f"✅ Basic model saved to {model_path} and registered in MLflow")

    # Log model path as artifact
    mlflow.log_param("model_path", model_path)

    return {"model_saved": 1.0}

@mlflow_run(experiment_name=f"{MLFLOW_MODEL_NAME}_experiments")
def save_results(metrics, model_version, model_type="basic", run_id=None):
    """
    Save model evaluation results to MLflow.

    Args:
        metrics: Dictionary of evaluation metrics
        model_version: Version of the model
        model_type: Type of model ("basic" or "lstm")
        run_id: MLflow run ID
    """
    # Log model information
    mlflow.log_param("model_action", "evaluate")
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("model_version", model_version)

    # Log metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)

    # If there are step metrics, log them as a file
    step_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list) and key.startswith("step_"):
            step_metrics[key] = value

    if step_metrics:
        import json
        # Write step metrics to a JSON file
        step_metrics_json = json.dumps(step_metrics)
        mlflow.log_text(step_metrics_json, "step_metrics.json")

    print(f"✅ Results saved to MLflow for {model_type} model v{model_version}")

    return metrics

def register_model_for_deployment(model_type="basic", stage="Staging"):
    """
    Register the model for deployment in MLflow.

    Args:
        model_type: Type of model ("basic" or "lstm")
        stage: Deployment stage ("Staging", "Production", "Archived")

    Returns:
        Model version info
    """
    @mlflow_run(experiment_name=f"{MLFLOW_MODEL_NAME}_deployments")
    def _register_model(run_id=None):
        # Log deployment information
        mlflow.log_param("deployment_stage", stage)
        mlflow.log_param("model_type", model_type)

        # Set model name based on type
        if model_type == "lstm":
            model_name = f"{MLFLOW_MODEL_NAME}_lstm"
        else:
            model_name = MLFLOW_MODEL_NAME

        # Get MLflow client
        client = MlflowClient()

        # Get latest model version
        latest_versions = client.get_latest_versions(model_name)

        if not latest_versions:
            print(f"❌ No versions found for model {model_name}")
            mlflow.log_param("deployment_success", False)
            return None

        # Get the latest version
        latest_version = latest_versions[0]
        version_num = latest_version.version

        # Update model stage
        client.transition_model_version_stage(
            name=model_name,
            version=version_num,
            stage=stage
        )

        # Log success
        mlflow.log_param("deployment_success", True)
        mlflow.log_param("deployed_version", version_num)

        print(f"✅ Model {model_name} version {version_num} transitioned to {stage}")
        return latest_version

    return _register_model()

def create_mlflow_experiment():
    """
    Create MLflow experiments if they don't exist.
    """
    client = MlflowClient()

    # Get environment variables with defaults
    mlflow_exp = os.environ.get("MLFLOW_EXPERIMENT", "sentiai_experiment")
    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", "sentiai_model")

    # Create experiments if they don't exist
    experiment_names = [
        mlflow_exp,  # From your .env
        f"{mlflow_model_name}_experiments",  # Used by your decorators
        f"{mlflow_model_name}_deployments",  # Used by your decorators
        os.environ.get("MLFLOW_EXPERIMENT_PROD", "sentiai_model_prod"),  # Default if not set
        os.environ.get("MLFLOW_EXPERIMENT_DEV", "sentiai_model_dev")   # Default if not set
    ]

    for exp_name in experiment_names:
        # Check if experiment exists
        experiment = client.get_experiment_by_name(exp_name)

        if experiment is None:
            # Create experiment
            try:
                experiment_id = mlflow.create_experiment(
                    exp_name,
                    artifact_location=f"./mlruns/{exp_name}"
                )
                print(f"Created experiment {exp_name} with ID {experiment_id}")
            except Exception as e:
                print(f"Error creating experiment {exp_name}: {e}")
        else:
            print(f"Experiment {exp_name} already exists with ID {experiment.experiment_id}")

    return True
