.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

run_preprocess:
	python -c 'from taxifare.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from taxifare.interface.main import train; train()'

run_pred:
	python -c 'from taxifare.interface.main import pred; pred()'

run_evaluate:
	python -c 'from taxifare.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

run_api:
	uvicorn taxifare.api.fast:app --reload

#################### BITCOIN MODEL ACTIONS ###################
# Setup MLflow experiments
setup_mlflow:
	python -c 'from senti_ai.ml_logic.registry import create_mlflow_experiment; create_mlflow_experiment()'

# Train the basic model
train_basic:
	python -m senti_ai.interface.main train

# Train the LSTM model with default parameters
train_lstm:
	python -m senti_ai.train_lstm

# Train the LSTM model with custom parameters
train_lstm_custom:
	python -m senti_ai.train_lstm --input_seq_length $(INPUT_SEQ_LENGTH) --forecast_horizon $(FORECAST_HORIZON) --epochs $(EPOCHS)

# Make predictions with the basic model
predict_basic:
	python -m senti_ai.interface.main predict

# Make predictions with the LSTM model
predict_lstm:
	python -m senti_ai.predict_lstm

# Make predictions with the LSTM model with custom parameters
predict_lstm_custom:
	python -m senti_ai.predict_lstm --input_seq_length $(INPUT_SEQ_LENGTH) --forecast_horizon $(FORECAST_HORIZON)

# Deploy a model to MLflow staging environment
deploy_staging:
	python -c 'from senti_ai.ml_logic.registry import register_model_for_deployment; register_model_for_deployment(model_type="$(MODEL_TYPE)", stage="Staging")'

# Deploy a model to MLflow production environment
deploy_production:
	python -c 'from senti_ai.ml_logic.registry import register_model_for_deployment; register_model_for_deployment(model_type="$(MODEL_TYPE)", stage="Production")'

# Run full LSTM pipeline: train, evaluate, and deploy to staging
run_lstm_pipeline: setup_mlflow train_lstm
	python -c 'from senti_ai.ml_logic.registry import register_model_for_deployment; register_model_for_deployment(model_type="lstm", stage="Staging")'

# Run full basic model pipeline: train, evaluate, and deploy to staging
run_basic_pipeline: setup_mlflow train_basic
	python -c 'from senti_ai.ml_logic.registry import register_model_for_deployment; register_model_for_deployment(model_type="basic", stage="Staging")'

##################### TESTS #####################
test_gcp_setup:
	@pytest \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_env \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_setup_key_path \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_project \
	tests/all/test_gcp_setup.py::TestGcpSetup::test_code_get_wagon_project

default:
	cat tests/api/test_output.txt

test_kitt:
	@echo "\n 🧪 computing and saving your progress at 'tests/api/test_output.txt'..."
	@pytest tests/api -c "./tests/pytest_kitt.ini" 2>&1 > tests/api/test_output.txt || true
	@echo "\n 🙏 Please: \n git add tests \n git commit -m 'checkpoint' \n ggpush"

test_api_root:
	pytest \
	tests/api/test_endpoints.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_is_dict --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_has_key --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_val_is_float --asyncio-mode=strict -W "ignore"

test_api_on_docker:
	pytest \
	tests/api/test_docker_endpoints.py --asyncio-mode=strict -W "ignore"

test_api_on_prod:
	pytest \
	tests/api/test_cloud_endpoints.py --asyncio-mode=strict -W "ignore"

test_notifications:
	pytest \
	tests/notifications/test_pushover_notification.py::test_send_pushover_notification -W "ignore"

# Test the LSTM model
test_lstm:
	pytest \
	tests/ml_logic/test_model.py -W "ignore"

################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/mlops
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
GS_DIR=gs://datascience-mlops/taxi-fare-ny

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${BQ_DATASET}
	-bq show ${BQ_DATASET}.processed_1k
	-bq show ${BQ_DATASET}.processed_200k
	-bq show ${BQ_DATASET}.processed_all
	-gsutil ls gs://${BUCKET_NAME}

reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/mlops/data/
	mkdir ~/.lewagon/mlops/data/raw
	mkdir ~/.lewagon/mlops/data/processed
	mkdir ~/.lewagon/mlops/training_outputs
	mkdir ~/.lewagon/mlops/training_outputs/metrics
	mkdir ~/.lewagon/mlops/training_outputs/models
	mkdir ~/.lewagon/mlops/training_outputs/params

reset_local_files_with_csv_solutions: reset_local_files
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_query_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/raw/query_2009-01-01_2015-01-01_all.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_1k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_1k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_200k.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_200k.csv
	-curl ${HTTPS_DIR}solutions/data_processed_fixture_2009-01-01_2015-01-01_all.csv > ${ML_DIR}/data/processed/processed_2009-01-01_2015-01-01_all.csv

reset_bq_files:
	-bq rm -f --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_1k
	-bq rm -f --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_200k
	-bq rm -f --project_id ${GCP_PROJECT} ${BQ_DATASET}.processed_all
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_1k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_200k
	-bq mk --sync --project_id ${GCP_PROJECT} --location=${BQ_REGION} ${BQ_DATASET}.processed_all

reset_gcs_files:
	-gsutil rm -r gs://${BUCKET_NAME}
	-gsutil mb -p ${GCP_PROJECT} -l ${GCP_REGION} gs://${BUCKET_NAME}

reset_all_files: reset_local_files reset_bq_files reset_gcs_files

##################### CLEANING #####################

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints

# Clean MLflow artifacts
clean_mlflow:
	@rm -fr mlruns
	@rm -f best_bitcoin_model.h5
	@rm -f bitcoin_price_prediction_model_test.h5

# Help target to show available commands
help:
	@echo "Available commands:"
	@echo "make setup_mlflow                 - Set up MLflow experiments"
	@echo "make train_basic                  - Train the basic model"
	@echo "make train_lstm                   - Train the LSTM model with default parameters"
	@echo "make train_lstm_custom            - Train the LSTM model with custom parameters (use with INPUT_SEQ_LENGTH, FORECAST_HORIZON, EPOCHS)"
	@echo "make predict_basic                - Make predictions with the basic model"
	@echo "make predict_lstm                 - Make predictions with the LSTM model"
	@echo "make predict_lstm_custom          - Make predictions with custom parameters"
	@echo "make deploy_staging MODEL_TYPE=lstm|basic   - Deploy a model to staging"
	@echo "make deploy_production MODEL_TYPE=lstm|basic - Deploy a model to production"
	@echo "make run_lstm_pipeline            - Run full LSTM pipeline"
	@echo "make run_basic_pipeline           - Run full basic model pipeline"
	@echo "make clean_mlflow                 - Clean MLflow artifacts"
