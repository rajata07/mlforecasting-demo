# Session 1 — LOW-CODE: Time Series Forecasting with Azure ML Studio

> **Duration:** ~60 minutes  
> **Level:** Beginner to Intermediate  
> **Audience:** Business Analysts, Citizen Data Scientists, Decision Makers, anyone with ZERO coding experience  
> **Prerequisites:** Azure subscription, web browser, a CSV dataset

---

## Table of Contents

1. [What Does "Low-Code" Mean Here?](#what-does-low-code-mean-here)
2. [Low-Code Options for Forecasting on Azure](#low-code-options-for-forecasting-on-azure)
3. [Option 1: AutoML in Azure ML Studio (No-Code)](#option-1-automl-in-azure-ml-studio-no-code)
4. [Option 2: Azure ML Designer (Drag-and-Drop)](#option-2-azure-ml-designer-drag-and-drop)
5. [Option 3: Power BI Forecasting (Visual)](#option-3-power-bi-forecasting-visual)
6. [Step-by-Step Demo: No-Code AutoML Forecasting](#step-by-step-demo-no-code-automl-forecasting)
7. [Step-by-Step Demo: Designer Pipeline](#step-by-step-demo-designer-pipeline)
8. [Low-Code vs. Pro-Code — When to Use What](#low-code-vs-pro-code--when-to-use-what)
9. [Resources & References](#resources--references)

---

## What Does "Low-Code" Mean Here?

Azure provides **three tiers** of experience for time series forecasting:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SPECTRUM OF APPROACHES                            │
│                                                                      │
│   NO-CODE              LOW-CODE               PRO-CODE               │
│   ────────             ────────               ────────               │
│   Azure ML Studio      Azure ML Designer      Python SDK v2          │
│   AutoML Wizard         (drag & drop)          CLI v2                 │
│                                                Jupyter Notebooks     │
│   Power BI              Custom Components                            │
│   Forecasting           (some YAML)                                  │
│                                                                      │
│   ◀─── Zero code ──────────────────────── Full code control ───▶    │
│   ◀─── Fastest start ──────────────────── Most flexible ───────▶    │
└──────────────────────────────────────────────────────────────────────┘
```

This README focuses on the **No-Code** and **Low-Code** approaches. The main [README.md](./README.md) covers the Pro-Code approach.

---

## Low-Code Options for Forecasting on Azure

| Option | Code Required | Best For | Limitations |
|--------|--------------|----------|-------------|
| **AutoML in Azure ML Studio** | **Zero** | Full forecasting workflow — data → train → deploy | Limited custom preprocessing |
| **Azure ML Designer** | **Minimal** (drag & drop + optional YAML) | Visual pipeline building with reusable components | Not all forecasting-specific features available |
| **Power BI Forecasting** | **Zero** | Quick visual trendline forecasts for dashboards | Very limited — only simple exponential smoothing |

---

## Option 1: AutoML in Azure ML Studio (No-Code)

### What Is It?

Azure ML Studio provides a **fully wizard-driven** interface at [ml.azure.com](https://ml.azure.com) for **Automated Machine Learning**. You upload data, select "Time Series Forecasting" as the task type, configure a few settings via dropdowns, and click **Submit**. Azure trains and evaluates dozens of models automatically — no code whatsoever.

### What You Get (Without Writing Any Code)

- Automatic model selection from 15+ algorithms (ARIMA, Prophet, LightGBM, etc.)
- Automatic feature engineering (lags, rolling windows, calendar features)
- Time-series-aware cross-validation
- Model leaderboard ranked by your chosen metric
- Feature importance / model explainability charts
- One-click deployment to a REST endpoint
- Downloadable model (MLflow format)

### Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Your CSV    │────▶│  Azure ML Studio      │────▶│  Trained Model   │
│  (upload)    │     │  AutoML Wizard        │     │  + REST Endpoint │
│              │     │                       │     │                  │
│              │     │  1. Upload data       │     │  Leaderboard     │
│              │     │  2. Select Forecasting│     │  Metrics         │
│              │     │  3. Configure         │     │  Explanations    │
│              │     │  4. Submit            │     │                  │
└──────────────┘     └──────────────────────┘     └──────────────────┘
                           ⬆ ALL via browser UI
```

---

## Option 2: Azure ML Designer (Drag-and-Drop)

### What Is It?

The **Designer** is a visual pipeline builder inside Azure ML Studio. You drag components (data sources, transforms, training steps, evaluation) onto a canvas and connect them with wires. Think of it like a flowchart that runs actual ML workloads.

### What You Get

- Visual drag-and-drop pipeline authoring
- Reusable components (pre-built + your own)
- Can include Python script components for custom logic
- Pipeline runs tracked and versioned automatically
- Can schedule pipelines for recurring forecasting

### When to Use Designer vs. AutoML Studio

| Criteria | AutoML Studio | Designer |
|----------|---------------|----------|
| **Ease** | Easier — wizard | Moderate — visual pipeline |
| **Customization** | Limited | More control over data flow |
| **Reusability** | One-off experiment | Reusable pipeline template |
| **Multi-step workflows** | Single train job | Train → Transform → Score → Evaluate |
| **Scheduling** | Manual resubmit | Can schedule recurring runs |

---

## Option 3: Power BI Forecasting (Visual)

### What Is It?

Power BI has a built-in **Analytics pane** that can add a forecast trendline to any line chart. It uses simple exponential smoothing. This is purely for visualization — not for building production ML models.

### How to Use It

1. Open a Power BI report with a time series line chart
2. Select the chart → **Analytics pane** (magnifying glass icon)
3. Expand **Forecast** → Toggle **On**
4. Configure:
   - **Forecast length** (e.g., 10 data points)
   - **Confidence interval** (default 95%)
   - **Seasonality** (Auto or manual)
5. The forecast line appears on the chart

### Limitations

- Single algorithm only (exponential smoothing)
- No model metrics / evaluation
- No multivariate support
- No deployment capability
- Purely visual — cannot export predictions as data

**Verdict:** Use Power BI forecasting only for **quick visual exploration** or executive dashboards. For anything production-grade, use AutoML in Azure ML Studio.

---

## Step-by-Step Demo: No-Code AutoML Forecasting

> **Scenario:** Forecast bike rental demand using no code.  
> **Dataset:** [bike-no.csv](https://github.com/Azure/azureml-examples/blob/v1-archive/v1/python-sdk/tutorials/automl-with-azureml/forecasting-bike-share/bike-no.csv) (public sample)  
> **Time:** ~25 minutes (including training wait time)

### Step 1: Sign in to Azure ML Studio

1. Go to [ml.azure.com](https://ml.azure.com)
2. Select your **subscription** and **workspace**
3. If you don't have a workspace, click **Create workspace** and follow prompts

### Step 2: Navigate to Automated ML

1. In the left navigation, click **Authoring** → **Automated ML**
2. Click **+ New Automated ML job**
3. Under **Training method**, select **Train automatically** (default)
4. Click **Start configuring job**

### Step 3: Basic Settings

| Field | Value |
|-------|-------|
| **Job name** | `bike-forecast-demo` (auto-generated is fine) |
| **Experiment name** | `bike-share-forecasting` |

Click **Next**.

### Step 4: Create and Upload Your Dataset

1. Under **Select task type**, choose **Time series forecasting**
2. Click **Create** to create a new data asset
3. Fill in:
   - **Name**: `bike-rental-data`
   - **Type**: Tabular
4. Click **Next**
5. **Data source**: Select **From local files**
6. Click **Next**
7. **Storage type**: Azure Blob Storage → select `workspaceblobstore` (default)
8. Click **Next**
9. **Upload**: Click **Upload files** → select `bike-no.csv`
10. Click **Next**
11. **Settings**: Verify the preview looks correct

    | Field | Value |
    |-------|-------|
    | File format | Delimited |
    | Delimiter | Comma |
    | Encoding | UTF-8 |
    | Column headers | All files have same headers |

12. Click **Next**
13. **Schema**: Deselect `casual` and `registered` columns (they're breakdowns of the target `cnt`)
14. Click **Next** → **Create**

### Step 5: Configure Forecasting Task Settings

1. **Target column**: Select `cnt` (the rental count to predict)
2. **Time column**: Select `date`
3. **Time series identifiers**: Leave blank (single series)
4. **Frequency**: Autodetect
5. **Forecast horizon**: Deselect Autodetect, enter `14` (predict 14 days ahead)
6. **Enable deep learning**: Leave unchecked

### Step 6: Additional Configuration Settings

Click **View additional configuration settings**:

| Setting | Value |
|---------|-------|
| **Primary metric** | Normalized root mean squared error |
| **Explain best model** | ✅ Enable |
| **Blocked models** | Extreme Random Trees (optional) |
| **Forecast target lags** | None |
| **Target rolling window size** | None |

Under **Limits**:

| Setting | Value |
|---------|-------|
| **Max concurrent trials** | 6 |
| **Metric score threshold** | None |
| **Experiment timeout (minutes)** | 120 |

Click **Save**.

### Step 7: Validation Settings

1. **Validation type**: k-fold cross-validation
2. **Number of cross validations**: 5

Click **Next**.

### Step 8: Configure Compute

1. **Compute type**: Compute cluster
2. Click **+ New** to create a compute cluster:

    | Field | Value |
    |-------|-------|
    | Virtual machine tier | Dedicated |
    | Virtual machine type | CPU |
    | Virtual machine size | Standard_DS12_V2 |
    | Compute name | `bike-compute` |
    | Min nodes | 1 |
    | Max nodes | 6 |
    | Idle seconds before scale down | 120 |

3. Click **Create** and wait ~2 minutes
4. Select the new compute from the dropdown

Click **Next**.

### Step 9: Review and Submit

1. Review all settings on the summary page
2. Click **Submit training job**
3. The job status shows as **Running**

> **Demo talking point:** "Notice we haven't written a single line of code. Everything was configured through the browser."

⏳ **Training takes 15-30 minutes.** During this time, show the audience:

### Step 10: Explore Results (While Training Runs)

#### Models Tab
- Navigate to **Models + child jobs**
- Watch models appear in the leaderboard as they complete
- Click any **Algorithm name** to see:
  - **Overview**: Algorithm details, duration, metric score
  - **Metrics**: Evaluation charts, residual plots, predicted vs. actual

#### Model Explanations
1. Select the best model → **Explain model**
2. Select your compute → **Create**
3. After ~2-5 minutes, go to **Explanations (preview)** tab
4. View **Aggregate feature importance** — shows which features drove predictions

> **Demo talking point:** "The model automatically figured out that 'temperature' and 'hour of day' are the most important features — without us telling it anything about the data."

### Step 11: Deploy the Best Model (One-Click)

1. Select the best model from the leaderboard
2. Click **Deploy** → **Deploy to web service**
3. Configure:

    | Field | Value |
    |-------|-------|
    | Deployment name | `bikeshare-deploy` |
    | Compute type | Azure Container Instance (ACI) |
    | Enable authentication | Disabled |

4. Click **Deploy**
5. Wait ~20 minutes for deployment

Once deployed, you have a **REST API endpoint** that accepts input data and returns forecasts — no code needed to create it.

### Step 12: Test the Endpoint

1. Go to **Endpoints** in the left navigation
2. Select your endpoint
3. Click **Test** tab
4. Paste sample JSON input
5. See the forecast response

---

## Step-by-Step Demo: Designer Pipeline

> **Scenario:** Build a visual forecasting pipeline using Designer's drag-and-drop interface.  
> **Time:** ~15 minutes to build, variable training time

### Step 1: Open the Designer

1. Go to [ml.azure.com](https://ml.azure.com) → **Designer** in left navigation
2. Click **+ New pipeline** (Custom components v2)
3. Name your pipeline: `forecast-pipeline`

### Step 2: Build Your Pipeline Visually

Drag these components from the left **Component** panel onto the canvas:

```
┌──────────────────┐
│  Import Data     │  ← Your dataset (registered or uploaded)
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Select Columns  │  ← Choose relevant features
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Clean Missing   │  ← Handle missing values
│  Data            │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Split Data      │  ← 70/30 train/test split
└────┬────────┬────┘
     │        │
┌────▼────┐ ┌─▼──────────┐
│Train    │ │             │
│Model    │ │  Score      │
│         │ │  Model      │
└────┬────┘ └──────┬──────┘
     │             │
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │  Evaluate   │
     │  Model      │
     └─────────────┘
```

### Step 3: Connect Components

- Wire each component's output port to the next component's input port
- For **Train Model**: connect the algorithm component to the left port, data to the right port
- For **Score Model**: connect trained model to left port, test data to right port

### Step 4: Configure and Submit

1. Click **Settings** → Select your compute target
2. Click **Submit** → Create or select an experiment
3. Monitor the pipeline run — each component shows green (✅) as it completes

### Step 5: View Results

- Click the **Evaluate Model** component → **Preview data** → See metrics
- Right-click **Score Model** → **Preview data** → See predictions

> **Demo talking point:** "The Designer gives you more control over the data flow while remaining entirely visual. It's great for teams that want a reusable, scheduled pipeline without writing code."

---

## Low-Code vs. Pro-Code — When to Use What

### Decision Matrix

```
                        ┌─────────────────────────────────────────┐
                        │     START HERE: What's your situation?   │
                        └─────────────────┬───────────────────────┘
                                          │
                          ┌───────────────▼────────────────┐
                          │  Do you need to write code?     │
                          └───────┬───────────────┬────────┘
                                  │               │
                              NO  │               │  YES / WANT TO
                                  │               │
                     ┌────────────▼──┐     ┌──────▼──────────────┐
                     │ Quick visual  │     │ Need full control   │
                     │ forecast      │     │ over pipeline,      │
                     │ for a         │     │ custom models,      │
                     │ dashboard?    │     │ or CI/CD?           │
                     └──┬────────┬──┘     └───────┬─────────────┘
                        │        │                │
                    YES │        │ NO             │
                        │        │                │
               ┌────────▼──┐  ┌──▼────────────┐  ┌▼──────────────┐
               │ Power BI  │  │ Azure ML      │  │ Python SDK v2 │
               │ Forecast  │  │ Studio AutoML │  │ + CLI v2      │
               │           │  │ (No-Code)     │  │ (Pro-Code)    │
               └───────────┘  └───────────────┘  └───────────────┘
```

### Detailed Comparison

| Criteria | Power BI Forecast | AutoML Studio (No-Code) | Designer (Low-Code) | SDK v2 (Pro-Code) |
|----------|-------------------|-------------------------|--------------------|--------------------|
| **Skill level** | Business user | Citizen data scientist | Intermediate | Data scientist / ML engineer |
| **Code required** | None | None | Minimal (optional) | Python / CLI |
| **Time to first result** | 2 minutes | 30-60 minutes | 1-2 hours | 2-4 hours |
| **Model selection** | 1 algorithm | 15+ algorithms | Your choice | 15+ or custom |
| **Feature engineering** | None | Automatic | Manual (components) | Automatic + custom |
| **Evaluation metrics** | None | Full suite | Full suite | Full suite + custom |
| **Deployment** | N/A (visual only) | One-click REST API | Pipeline output | Full MLOps |
| **Retraining** | Manual | Manual resubmit | Scheduled pipeline | CI/CD automated |
| **Multi-series** | No | Yes | With custom components | Yes (Many Models, HTS) |
| **Cost control** | Included in Power BI Pro | Pay for compute | Pay for compute | Pay for compute |
| **Best for** | Executive dashboards | PoC / MVP / quick wins | Repeatable workflows | Production systems |

### Recommendations

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| "Show me a quick trend for next quarter" | **Power BI Forecasting** | Instant visual, no setup |
| "Build a forecasting PoC in 1 day" | **AutoML Studio (No-Code)** | Zero code, automatic best model |
| "Business analyst needs to forecast monthly" | **AutoML Studio (No-Code)** | No programming skills needed |
| "Need a repeatable forecast pipeline" | **Designer** or **SDK v2** | Schedule + reuse |
| "Forecast 10,000 SKUs across 500 stores" | **SDK v2 (Pro-Code)** | Many Models / HTS pattern needed |
| "Integrate forecast into CI/CD" | **SDK v2 (Pro-Code)** | Full MLOps with GitHub Actions |
| "Custom preprocessing + exotic models" | **SDK v2 (Pro-Code)** | Full flexibility |

### The Golden Rule

> **Start with the simplest approach that meets your requirements.** Use AutoML Studio to validate the business case (PoC), then graduate to SDK v2 only when you need more control. Most forecasting projects can start — and even remain — on the no-code path.

---

## Resources & References

- [Tutorial: Forecast Demand with No-Code AutoML in Azure ML Studio](https://learn.microsoft.com/azure/machine-learning/tutorial-automated-ml-forecast?view=azureml-api-2)
- [Set up No-Code AutoML Training in Studio](https://learn.microsoft.com/azure/machine-learning/how-to-use-automated-ml-for-ml-models?view=azureml-api-2)
- [What is Azure ML Designer?](https://learn.microsoft.com/azure/machine-learning/concept-designer?view=azureml-api-2)
- [Tutorial: Train a No-Code Regression Model with Designer](https://learn.microsoft.com/azure/machine-learning/tutorial-designer-automobile-price-train-score?view=azureml-api-1)
- [Create Pipelines with Components in Studio Designer](https://learn.microsoft.com/azure/machine-learning/how-to-create-component-pipelines-ui?view=azureml-api-2)
- [Compare Microsoft ML Products & Technologies](https://learn.microsoft.com/azure/architecture/ai-ml/guide/data-science-and-machine-learning)

---

## Agenda Slide (Low-Code Session Variant)

```
Session 1 (Low-Code): Time Series Forecasting — No Code Required
──────────────────────────────────────────────────────────────────
1. 📌 What is Time Series Forecasting?              (10 min)
2. ☁️  Low-Code Options on Azure (3 tiers)            (10 min)
3. 💻 Live Demo: AutoML Studio — Zero Code            (25 min)
   - Upload CSV
   - Configure forecasting
   - Explore model leaderboard
   - Explain model
   - One-click deploy
4. 💻 Quick Demo: Designer Pipeline (drag & drop)     (10 min)
5. 📊 When to Use Low-Code vs. Pro-Code               (5 min)
──────────────────────────────────────────────────────────────────
```
