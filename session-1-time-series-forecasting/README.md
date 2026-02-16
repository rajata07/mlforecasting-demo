# Session 1: Time Series Forecasting with Azure Machine Learning

> **Duration:** ~90 minutes  
> **Level:** Intermediate  
> **Audience:** Data Scientists, ML Engineers, Business Analysts  
> **Prerequisites:** Basic Python, familiarity with ML concepts, Azure subscription

---

## Table of Contents

1. [Session Overview](#session-overview)
2. [What is Time Series Forecasting?](#what-is-time-series-forecasting)
3. [Why Azure for Time Series Forecasting?](#why-azure-for-time-series-forecasting)
4. [Azure Services for Forecasting](#azure-services-for-forecasting)
5. [Core Concepts Deep Dive](#core-concepts-deep-dive)
6. [AutoML Forecasting in Azure ML](#automl-forecasting-in-azure-ml)
7. [Step-by-Step Demo](#step-by-step-demo)
8. [Best Practices](#best-practices)
9. [Real-World Use Cases](#real-world-use-cases)
10. [Q&A Talking Points](#qa-talking-points)
11. [Resources & References](#resources--references)

---

## Session Overview

This session demonstrates how Azure Machine Learning enables **automated time series forecasting** using AutoML. Attendees will learn foundational time series concepts, understand the Azure ML ecosystem for forecasting, and walk through a hands-on demo that trains, evaluates, and deploys a forecasting model — all with minimal code.

### Key Takeaways for the Audience

- Understand how time series forecasting works conceptually
- Know which Azure services support forecasting and when to use each
- See a live demo of AutoML forecasting (SDK v2 + Studio)
- Learn best practices for production-grade forecasting pipelines

---

## What is Time Series Forecasting?

### Definition

Time series forecasting is the process of using **historical data points collected over time** to predict future values. Unlike traditional regression, time series data has an inherent temporal ordering that must be preserved.

### Mathematical Foundation

A time series model predicts future values based on historical observations:

$$y_{t+1} = f(y_t, y_{t-1}, \ldots, y_{t-s})$$

Where:
- $y_t$ = the observed value at time $t$
- $f$ = a function that maps historical values to future predictions
- $s$ = the lookback window (how much history the model uses)

### Types of Forecasting Models

#### 1. Time Series Models (Univariate)
These use **only historical values** of the target variable:

| Model | Description | Best For |
|-------|-------------|----------|
| **ARIMA/ARIMAX** | Auto-Regressive Integrated Moving Average | Stationary data with clear autocorrelation |
| **Exponential Smoothing** | Weighted average with exponentially decreasing weights | Data with trend and/or seasonality |
| **Prophet** | Facebook's additive model for business time series | Strong seasonal effects + holidays |
| **Naive / Seasonal Naive** | Simple baseline — last value or same period last season | Benchmarking other models |

#### 2. Regression / Explanatory Models (Multivariate)
These use **external predictor variables** alongside time:

$$y = g(\text{price}, \text{day of week}, \text{holiday}, \text{weather}, \ldots)$$

| Model | Description | Best For |
|-------|-------------|----------|
| **LightGBM / XGBoost** | Gradient-boosted decision trees | Tabular data with many features |
| **Elastic Net / LASSO** | Regularized linear regression | High-dimensional sparse data |
| **Random Forest** | Ensemble of decision trees | Robust general-purpose predictions |
| **TCNForecaster** | Temporal Convolutional Network (deep learning) | Large datasets, complex patterns |

### Key Time Series Components

```
                 📈 Trend        — Long-term increase or decrease
                 🔄 Seasonality  — Repeating patterns (daily, weekly, yearly)
                 📊 Cyclicality  — Irregular long-term oscillations
                 🎲 Noise        — Random variation
```

Understanding these components is critical for choosing the right model and configuring forecasting parameters correctly.

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. Many classical models require stationarity. AutoML in Azure handles this automatically by applying **differencing transforms** when it detects non-stationary data.

---

## Why Azure for Time Series Forecasting?

| Capability | Benefit |
|-----------|---------|
| **AutoML** | Automatically tries dozens of models + hyperparameters |
| **Feature Engineering** | Auto-generates lag features, rolling windows, calendar features |
| **Scalability** | Distributed compute for training thousands of time series |
| **MLOps** | Built-in experiment tracking, model registry, deployment |
| **Enterprise-grade** | RBAC, VNet, managed identity, compliance certifications |
| **No-code + Pro-code** | Azure ML Studio UI or Python SDK v2 + CLI v2 |

---

## Azure Services for Forecasting

### Primary: Azure Machine Learning (AutoML Forecasting)

The **recommended service** for time series forecasting. Provides:

- Automated model selection from 15+ algorithms
- Automatic feature engineering (lags, rolling windows, holidays)
- Cross-validation with time-series-aware splits
- Multi-series (many models) and hierarchical time series support
- Deep learning integration (TCNForecaster)

### Alternative / Complementary Services

| Service | When to Use |
|---------|-------------|
| **Azure AI Services (Anomaly Detector)** | When you need anomaly detection in time series rather than forecasting |
| **Azure Databricks + AutoML** | When your data lakehouse is in Databricks and you want forecast within that ecosystem |
| **Azure Synapse Analytics** | When forecasting is part of a larger data engineering + analytics workflow |
| **Power BI Forecasting** | Quick visual forecasts for business dashboards (limited control) |
| **Azure Stream Analytics** | Real-time event processing with basic temporal analytics |

---

## Core Concepts Deep Dive

### 1. Forecast Horizon

The **number of time steps** into the future you want to predict.

```
Example: If your data is daily and you want to predict 2 weeks ahead:
    forecast_horizon = 14
```

Longer horizons generally produce less accurate predictions. Choose the horizon based on your **business decision-making cycle**.

### 2. Lag Features

AutoML automatically creates **lag features** — previous values of the target variable used as predictors:

```
Time    Target    Lag_1    Lag_2    Lag_7
t       100       95       90       80
t+1     105       100      95       85
t+2     110       105      100      90
```

These capture autocorrelation patterns and can dramatically improve model accuracy.

### 3. Rolling Window Aggregations

Statistical summaries computed over a sliding window of past values:

```
Rolling_Mean_7d    Rolling_Std_7d    Rolling_Min_7d    Rolling_Max_7d
92.5               5.2               80                100
95.0               4.8               85                105
97.0               4.5               90                110
```

### 4. Calendar / Holiday Features

AutoML generates features like:
- Day of week, month, quarter, year
- Is weekend / is holiday
- Country-specific holidays (configurable via `country_or_region_for_holidays`)

### 5. Time Series ID Columns

When you have **multiple time series** in a single dataset (e.g., sales for different stores and products), you must specify the columns that uniquely identify each series:

```python
forecasting_job.set_forecast_settings(
    time_series_id_column_names=['store', 'brand']
)
```

### 6. Cross-Validation for Time Series

Unlike standard k-fold CV, time series cross-validation uses **rolling origin** evaluation to respect temporal ordering:

```
Fold 1: Train [------|] Validate [--]
Fold 2: Train [--------|] Validate [--]
Fold 3: Train [----------|] Validate [--]
```

AutoML handles this automatically with configurable `n_cross_validations` and `cv_step_size`.

### 7. Model Grouping in AutoML

Azure ML AutoML uses different grouping strategies depending on the model type:

| Strategy | Models |
|----------|--------|
| **One series per model (1:1)** | Naive, Seasonal Naive, ARIMA, ARIMAX, Exponential Smoothing, Prophet |
| **All series share one model (N:1)** | LightGBM, XGBoost, Random Forest, Decision Tree, Elastic Net, TCNForecaster |

This mixed approach is applied **by default** — no configuration needed.

### 8. Evaluation Metrics

| Metric | Formula Intuition | Use When |
|--------|-------------------|----------|
| **NRMSE** (Normalized RMSE) | Penalizes large errors proportionally | Default — good general metric |
| **NMAE** (Normalized MAE) | Treats all errors equally | Robust to outliers |
| **R²** | Proportion of variance explained | Comparing against baseline |
| **MAPE** | Percentage error per data point | Business stakeholders need % accuracy |

---

## AutoML Forecasting in Azure ML

### How It Works (End-to-End Flow)

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Your Data   │───▶│  Data Preparation │───▶│ Feature         │
│  (CSV/MLTable│    │  - Handle missing │    │ Engineering     │
│   /Parquet)  │    │  - Detect freq    │    │ - Lags          │
└──────────────┘    │  - Stationarity   │    │ - Rolling window│
                    └──────────────────┘    │ - Calendar      │
                                            │ - Holidays      │
                                            └────────┬────────┘
                                                     │
                    ┌──────────────────┐              ▼
                    │  Model Sweeping  │◀────────────────────
                    │  15+ algorithms  │
                    │  Hyperparameter  │
                    │  tuning          │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐    ┌──────────────────┐
                    │  Best Model      │───▶│  Deploy / Batch  │
                    │  Selection       │    │  Inference       │
                    │  (+ Ensemble)    │    │                  │
                    └──────────────────┘    └──────────────────┘
```

### Supported Algorithms in AutoML Forecasting

AutoML sweeps over **all of these** unless you restrict the search space:

**Classical Time Series:**
- Naive, Seasonal Naive, Average, Seasonal Average
- Exponential Smoothing
- ARIMA, ARIMAX
- Prophet

**Machine Learning:**
- Linear SGD, LARS LASSO, Elastic Net
- K Nearest Neighbors
- Decision Tree, Random Forest, Extremely Randomized Trees
- Gradient Boosted Trees, LightGBM, XGBoost

**Deep Learning:**
- TCNForecaster (Temporal Convolutional Network)

### Scaling Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Single model** | One AutoML job, one series | Simple PoC or single-product forecast |
| **Many Models** | Train independent models per series partition | 100s–1000s of products/stores |
| **Hierarchical (HTS)** | Train + reconcile across a hierarchy | Forecast at SKU + store + region levels |

---

## Step-by-Step Demo

> **Scenario:** Forecast energy demand for the next 48 hours using the UCI Energy Dataset.

### Prerequisites

```bash
# Install Azure ML SDK v2
pip install azure-ai-ml azure-identity

# Login to Azure
az login
```

### Step 1: Connect to Azure ML Workspace

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Connect to your workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<YOUR_SUBSCRIPTION_ID>",
    resource_group_name="<YOUR_RESOURCE_GROUP>",
    workspace_name="<YOUR_WORKSPACE_NAME>"
)
print(f"Connected to workspace: {ml_client.workspace_name}")
```

### Step 2: Create or Reference Compute

```python
from azure.ai.ml.entities import AmlCompute

# Create compute cluster (if it doesn't exist)
compute_name = "cpu-cluster"
try:
    ml_client.compute.get(compute_name)
    print(f"Compute '{compute_name}' already exists.")
except Exception:
    compute = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_DS3_V2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print(f"Compute '{compute_name}' created.")
```

### Step 3: Prepare Training Data

Your data should be in **MLTable format** for AutoML. Here's how to prepare it:

```python
import pandas as pd

# Load sample energy consumption data
# Columns: timestamp, demand, temperature, humidity, holiday
df = pd.read_csv("energy_demand.csv", parse_dates=["timestamp"])
print(f"Dataset shape: {df.shape}")
print(df.head())
```

**Create MLTable definition file** (`train_data/MLTable`):

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable

paths:
  - file: ./train.csv

transformations:
  - read_delimited:
      delimiter: ","
      header: all_files_same_headers
      encoding: utf8
```

```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

# Define training data input
my_training_data_input = Input(
    type=AssetTypes.MLTABLE,
    path="./train_data"
)
```

### Step 4: Configure the AutoML Forecasting Job

```python
from azure.ai.ml import automl

# Set forecasting variables
target_column_name = "demand"
time_column_name = "timestamp"
forecast_horizon = 48  # Predict next 48 hours

# Create the forecasting job
forecasting_job = automl.forecasting(
    compute=compute_name,
    experiment_name="energy-demand-forecasting",
    training_data=my_training_data_input,
    target_column_name=target_column_name,
    primary_metric="normalized_root_mean_squared_error",
    n_cross_validations=5,
    enable_model_explainability=True,
)

# Set forecast-specific settings
forecasting_job.set_forecast_settings(
    time_column_name=time_column_name,
    forecast_horizon=forecast_horizon,
    country_or_region_for_holidays="US",  # Auto-generate US holiday features
    # time_series_id_column_names=['region'],  # Uncomment if multiple series
)

# Set training limits
forecasting_job.set_limits(
    timeout_minutes=120,
    trial_timeout_minutes=30,
    max_concurrent_trials=4,
    max_trials=20,
    enable_early_termination=True,
)
```

### Step 5: (Optional) Customize Model Search & Features

```python
# Restrict or expand model search space
forecasting_job.set_training(
    # Only try these models (faster demo):
    allowed_training_algorithms=[
        "LightGBM", "ExponentialSmoothing", "Prophet", "ARIMA"
    ],
    enable_dnn_training=False,  # Set True to include TCNForecaster
)

# Custom featurization (optional)
from azure.ai.ml.automl import ColumnTransformer

forecasting_job.set_featurization(
    mode="custom",
    transformer_params={
        "imputer": [
            ColumnTransformer(
                fields=["temperature"],
                parameters={"strategy": "median"}
            ),
            ColumnTransformer(
                fields=["humidity"],
                parameters={"strategy": "ffill"}
            ),
        ]
    }
)
```

### Step 6: Submit the Job

```python
# Submit the AutoML forecasting job
returned_job = ml_client.jobs.create_or_update(forecasting_job)

print(f"Created job: {returned_job.name}")
print(f"Studio URL: {returned_job.services['Studio'].endpoint}")
```

**At this point, switch to Azure ML Studio to show the audience:**
1. The experiment tracking dashboard
2. Models being trained in real-time
3. Leaderboard of model performance
4. Feature importance charts
5. Residual analysis plots

### Step 7: Retrieve the Best Model & Results

```python
# Wait for job completion (or monitor in Studio)
ml_client.jobs.stream(returned_job.name)

# Get the best model details
best_run = ml_client.jobs.get(returned_job.name)
print(f"Best model: {best_run}")
```

### Step 8: Build a Train → Inference → Evaluate Pipeline

```python
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

# Get components from Azure ML registry
inference_component = ml_client.components.get(
    name="automl_forecasting_inference",
    version="latest"
)

metrics_component = ml_client.components.get(
    name="compute_metrics",
    label="latest"
)

@pipeline(description="Forecasting Train + Evaluate Pipeline")
def forecasting_pipeline(
    train_data,
    test_data,
    target_column,
    time_column,
    horizon,
):
    # Training step
    train_node = automl.forecasting(
        training_data=train_data,
        target_column_name=target_column,
        primary_metric="normalized_root_mean_squared_error",
        n_cross_validations="auto",
        outputs={"best_model": Output(type=AssetTypes.MLFLOW_MODEL)},
    )
    train_node.set_forecast_settings(
        time_column_name=time_column,
        forecast_horizon=horizon,
    )

    # Inference step (rolling forecast on test data)
    inference_node = inference_component(
        test_data=test_data,
        model_path=train_node.outputs.best_model,
        target_column_name=target_column,
        forecast_mode="rolling",
        step=1,
    )

    # Metrics step
    metrics_node = metrics_component(
        task="tabular-forecasting",
        ground_truth=inference_node.outputs.inference_output_file,
        prediction=inference_node.outputs.inference_output_file,
        evaluation_config=inference_node.outputs.evaluation_config_output_file,
    )

    return {
        "metrics": metrics_node.outputs.evaluation_result,
        "forecasts": inference_node.outputs.inference_output_file,
    }

# Create and submit the pipeline
pipeline_job = forecasting_pipeline(
    train_data=Input(type=AssetTypes.MLTABLE, path="./train_data"),
    test_data=Input(type=AssetTypes.URI_FOLDER, path="./test_data"),
    target_column="demand",
    time_column="timestamp",
    horizon=48,
)
pipeline_job.settings.default_compute = compute_name

returned_pipeline = ml_client.jobs.create_or_update(
    pipeline_job,
    experiment_name="energy-forecast-pipeline"
)
print(f"Pipeline URL: {returned_pipeline.services['Studio'].endpoint}")
```

### Step 9: Deploy as a Managed Online Endpoint (Optional)

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)

# Register the best model
model = Model(
    path=f"azureml://jobs/{returned_job.name}/outputs/best_model",
    name="energy-demand-forecast-model",
    type=AssetTypes.MLFLOW_MODEL,
)
registered_model = ml_client.models.create_or_update(model)

# Create endpoint
endpoint_name = "energy-forecast-endpoint"
endpoint = ManagedOnlineEndpoint(name=endpoint_name)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=registered_model,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

print(f"Endpoint ready: {endpoint_name}")
```

### Step 10: Azure ML Studio Walkthrough (No-Code Alternative)

> Show this as an alternative path for non-technical stakeholders

1. Navigate to [Azure ML Studio](https://ml.azure.com)
2. Go to **Automated ML** → **New Automated ML Job**
3. Select your dataset
4. Choose **Forecasting** as the task type
5. Configure:
   - Time column
   - Forecast horizon  
   - Time series identifiers
   - Primary metric
6. Set compute and limits
7. Click **Submit** and monitor results
8. Explore the **Models** tab for leaderboard
9. Click **Explain model** for feature importance
10. One-click **Deploy** to a managed endpoint

---

## Best Practices

### Data Preparation
- **Ensure regular frequency** — fill gaps or use `frequency` + `target_aggregate_function` settings
- **Handle short series** — use `short_series_handling_config` (`pad`, `drop`, or `auto`)
- **Include external features** — weather, events, promotions significantly improve accuracy

### Configuration
- **Start with `n_cross_validations="auto"`** — AutoML picks the right number
- **Set `enable_early_termination=True`** — saves compute cost
- **Use `country_or_region_for_holidays`** — free accuracy boost from holiday features
- **Limit `max_trials` for demos** — 10-20 is enough to show results quickly

### Production
- **Use pipelines** — orchestrate train → inference → evaluate in a repeatable workflow
- **Schedule retraining** — time series models degrade as new patterns emerge
- **Monitor data drift** — Azure ML provides built-in data drift monitoring
- **Deploy with managed endpoints** — automatic scaling, blue/green deployments

---

## Real-World Use Cases

| Industry | Use Case | Forecast Target |
|----------|----------|-----------------|
| **Retail** | Demand planning | Product sales by store |
| **Energy** | Load forecasting | Electricity demand by region |
| **Finance** | Revenue projections | Monthly recurring revenue |
| **Supply Chain** | Inventory optimization | SKU demand at warehouse level |
| **Healthcare** | Patient volume | ER admissions per hospital |
| **Transportation** | Ride demand | Trips per zone per hour |
| **Manufacturing** | Predictive maintenance | Equipment failure probability |

---

## Q&A Talking Points

**Q: How does AutoML handle missing data in time series?**
> AutoML applies forward-fill (ffill) for target values and configurable imputation (median, constant, ffill) for features. You can customize this via `set_featurization()`.

**Q: Can I forecast thousands of products/stores simultaneously?**
> Yes! Use the **Many Models** pattern (partitioned training) or **Hierarchical Time Series (HTS)** for reconciled forecasts across aggregation levels.

**Q: How long does training take?**
> Depends on data size, number of models, and compute. For a typical dataset (<1M rows), expect 30-90 minutes on a 4-node cluster with `max_trials=20`.

**Q: Can I bring my own model?**
> Absolutely. AutoML is one option. You can also train custom models (e.g., PyTorch, TensorFlow) on Azure ML compute and use the same MLOps infrastructure.

**Q: What about real-time forecasting?**
> Deploy the trained model as a managed online endpoint and call it via REST API. For streaming scenarios, pair with Azure Event Hubs or Stream Analytics.

---

## Resources & References

- [Set up AutoML for Time Series Forecasting (SDK v2)](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-forecast?view=azureml-api-2)
- [Forecasting Methods in AutoML — Concepts](https://learn.microsoft.com/azure/machine-learning/concept-automl-forecasting-methods?view=azureml-api-2)
- [Lag Features for Time Series in AutoML](https://learn.microsoft.com/azure/machine-learning/concept-automl-forecasting-lags?view=azureml-api-2)
- [Model Sweeping & Selection in AutoML](https://learn.microsoft.com/azure/machine-learning/concept-automl-forecasting-sweeping?view=azureml-api-2)
- [Tutorial: Forecast Demand with AutoML (Studio)](https://learn.microsoft.com/azure/machine-learning/tutorial-automated-ml-forecast?view=azureml-api-2)
- [Many Models Pipeline Example (GitHub)](https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/pipelines/1k_demand_forecast_pipeline/)
- [Azure ML Python SDK v2 Reference](https://learn.microsoft.com/python/api/overview/azure/ai-ml-readme)

---

## Agenda Slide (Copy-Paste for your Presentation)

```
Session 1: Time Series Forecasting with Azure ML
─────────────────────────────────────────────────
1. 📌 Introduction & What is Time Series?        (10 min)
2. 📊 Core Concepts (Horizon, Lags, Seasonality)  (15 min)
3. ☁️  Azure Services Landscape                    (10 min)
4. 🤖 AutoML Forecasting Deep Dive                (15 min)
5. 💻 Live Demo: Energy Demand Forecasting         (25 min)
   - SDK v2 walkthrough
   - Studio UI walkthrough
   - Pipeline: Train → Infer → Evaluate
6. 🏭 Best Practices & Production Patterns         (10 min)
7. ❓ Q&A                                          (5 min)
─────────────────────────────────────────────────
```
