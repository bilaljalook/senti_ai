# restore_mlflow.py
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Set the tracking URI
mlflow.set_tracking_uri("https://mlflow.ckna.net:5000/")

# Create client
client = MlflowClient()

# Check MLflow version
import pkg_resources
mlflow_version = pkg_resources.get_distribution("mlflow").version
print(f"MLflow version: {mlflow_version}")

# List all experiments
print("Listing all experiments:")
try:
    # Method available in MLflow 2.1.1
    experiments = client.get_experiment_by_name("sentiai_model_experiments")
    if experiments:
        print(f"Found experiment: {experiments.name}, Lifecycle stage: {experiments.lifecycle_stage}")

        # Check if it's deleted
        if experiments.lifecycle_stage == "deleted":
            print("Experiment is in deleted state.")
            try:
                client.restore_experiment(experiments.experiment_id)
                print("Experiment restored successfully!")
            except Exception as e:
                print(f"Error restoring experiment: {e}")
    else:
        print("Experiment not found. Will try to create a new one.")
        try:
            new_id = mlflow.create_experiment("sentiai_model_experiments")
            print(f"Created new experiment with ID: {new_id}")
        except Exception as e:
            print(f"Failed to create experiment: {e}")
except Exception as e:
    print(f"Error accessing experiment: {e}")

# List all experiments again to confirm
print("\nListing all experiments after operation:")
try:
    all_experiments = mlflow.search_experiments()
    for exp in all_experiments:
        print(f"ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle stage: {exp.lifecycle_stage}")
except Exception as e:
    print(f"Error listing experiments: {e}")
