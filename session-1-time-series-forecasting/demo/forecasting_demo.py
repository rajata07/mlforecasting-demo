"""
=============================================================================
  DEMO: Time Series Forecasting with Azure ML AutoML
  Session 1 — Energy Demand Forecasting
=============================================================================

  This script demonstrates end-to-end time series forecasting using
  Azure Machine Learning's AutoML (SDK v2).

  Prerequisites:
    pip install azure-ai-ml azure-identity pandas

  Before running:
    1. az login
    2. Replace <placeholders> with your Azure resource details
=============================================================================
"""

# ── Step 0: Configuration ──────────────────────────────────────────────────
SUBSCRIPTION_ID = "<YOUR_SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<YOUR_RESOURCE_GROUP>"
WORKSPACE_NAME = "<YOUR_WORKSPACE_NAME>"
COMPUTE_NAME = "cpu-cluster"
EXPERIMENT_NAME = "energy-demand-forecasting-demo"

TARGET_COLUMN = "demand"
TIME_COLUMN = "timestamp"
FORECAST_HORIZON = 48  # 48 hours ahead

# ── Step 1: Connect to Azure ML Workspace ──────────────────────────────────
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

print("🔗 Connecting to Azure ML Workspace...")
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)
print(f"✅ Connected to: {ml_client.workspace_name}")


# ── Step 2: Create or Reference Compute ────────────────────────────────────
from azure.ai.ml.entities import AmlCompute

print(f"\n📦 Setting up compute: {COMPUTE_NAME}...")
try:
    compute = ml_client.compute.get(COMPUTE_NAME)
    print(f"✅ Compute '{COMPUTE_NAME}' already exists.")
except Exception:
    compute = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="STANDARD_DS3_V2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print(f"✅ Compute '{COMPUTE_NAME}' created.")


# ── Step 3: Prepare Training Data ──────────────────────────────────────────
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

print("\n📊 Configuring training data...")

# Option A: Use local MLTable folder
my_training_data_input = Input(
    type=AssetTypes.MLTABLE,
    path="./train_data",  # Folder must contain MLTable definition + CSV
)

# Option B: Use a registered dataset (uncomment if applicable)
# my_training_data_input = Input(
#     type=AssetTypes.MLTABLE,
#     path="azureml:energy-demand-dataset:1",
# )

print("✅ Training data configured.")


# ── Step 4: Configure AutoML Forecasting Job ──────────────────────────────
from azure.ai.ml import automl

print("\n⚙️ Configuring AutoML Forecasting job...")

forecasting_job = automl.forecasting(
    compute=COMPUTE_NAME,
    experiment_name=EXPERIMENT_NAME,
    training_data=my_training_data_input,
    target_column_name=TARGET_COLUMN,
    primary_metric="normalized_root_mean_squared_error",
    n_cross_validations=5,
    enable_model_explainability=True,
)

# Forecast-specific settings
forecasting_job.set_forecast_settings(
    time_column_name=TIME_COLUMN,
    forecast_horizon=FORECAST_HORIZON,
    country_or_region_for_holidays="US",
    # Uncomment for multi-series:
    # time_series_id_column_names=["region", "station_id"],
)

# Training limits
forecasting_job.set_limits(
    timeout_minutes=120,
    trial_timeout_minutes=30,
    max_concurrent_trials=4,
    max_trials=20,
    enable_early_termination=True,
)

# Model search (optional — restrict for faster demo)
forecasting_job.set_training(
    allowed_training_algorithms=[
        "LightGBM",
        "ExponentialSmoothing",
        "Prophet",
        "ARIMA",
    ],
    enable_dnn_training=False,  # Set True to include TCNForecaster
)

print("✅ AutoML job configured.")


# ── Step 5: Submit the Job ─────────────────────────────────────────────────
print("\n🚀 Submitting AutoML forecasting job...")
returned_job = ml_client.jobs.create_or_update(forecasting_job)

print(f"✅ Job created: {returned_job.name}")
print(f"📊 Studio URL: {returned_job.services['Studio'].endpoint}")
print("\n⏳ Streaming job output (Ctrl+C to stop and check Studio)...")

try:
    ml_client.jobs.stream(returned_job.name)
except KeyboardInterrupt:
    print("\n⏸ Stopped streaming. Check Azure ML Studio for progress.")


# ── Step 6: Retrieve Results ──────────────────────────────────────────────
print("\n📈 Retrieving job results...")
completed_job = ml_client.jobs.get(returned_job.name)
print(f"✅ Job status: {completed_job.status}")
print(f"📊 Best model: Check Studio UI → Experiment → {EXPERIMENT_NAME}")


# ── Step 7 (Optional): Deploy Best Model ─────────────────────────────────
# Uncomment the following to deploy the best model as a managed endpoint.
#
# from azure.ai.ml.entities import (
#     ManagedOnlineEndpoint,
#     ManagedOnlineDeployment,
#     Model,
# )
#
# model = Model(
#     path=f"azureml://jobs/{returned_job.name}/outputs/best_model",
#     name="energy-demand-forecast-model",
#     type=AssetTypes.MLFLOW_MODEL,
# )
# registered_model = ml_client.models.create_or_update(model)
#
# endpoint_name = "energy-forecast-ep"
# endpoint = ManagedOnlineEndpoint(name=endpoint_name)
# ml_client.online_endpoints.begin_create_or_update(endpoint).result()
#
# deployment = ManagedOnlineDeployment(
#     name="blue",
#     endpoint_name=endpoint_name,
#     model=registered_model,
#     instance_type="Standard_DS3_v2",
#     instance_count=1,
# )
# ml_client.online_deployments.begin_create_or_update(deployment).result()
# print(f"✅ Endpoint deployed: {endpoint_name}")

print("\n🎉 Demo complete!")
