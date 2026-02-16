# Azure ML — Low-Code vs. Pro-Code Recommendations

> **Cross-Session Summary** — When to use which approach for Forecasting & Image Labelling

---

## Quick Decision Matrix

| Scenario | Approach | Session |
|----------|----------|---------|
| Quick sales forecast for a business meeting | **Low-Code** — AutoML Studio wizard | Session 1 |
| Automated weekly demand-forecasting pipeline | **Pro-Code** — SDK v2 + CLI | Session 1 |
| Simple trendline in a dashboard | **Low-Code** — Power BI visual | Session 1 |
| Multi-step, multi-series forecasting with custom features | **Pro-Code** — SDK v2 | Session 1 |
| Read text from receipts/images | **No-Code** — Azure AI Vision (pre-built) | Session 2 |
| Classify 5 product types, ~200 images | **No-Code** — Custom Vision Portal | Session 2 |
| Label 50K images with a 10-person team | **Low-Code** — Azure ML Data Labeling | Session 2 |
| Production object detection with CI/CD | **Pro-Code** — SDK v2 AutoML for Vision | Session 2 |
| Run model offline on mobile/edge device | **No-Code** — Custom Vision (export ONNX) | Session 2 |
| Semantic segmentation for satellite imagery | **Pro-Code** — SDK v2 | Session 2 |

---

## General Guidance

### Start Low-Code When:
- You're **prototyping** or validating whether ML can solve your problem
- The team has **no ML engineers** yet
- Dataset is **small** (< 10K samples for vision, < 100K rows for forecasting)
- You need results in **hours, not days**
- The task follows a **standard pattern** (classification, regression, forecasting)

### Graduate to Pro-Code When:
- You need **reproducible MLOps pipelines** (CI/CD, version control, automated retraining)
- Dataset is **large or growing continuously**
- You need **custom preprocessing**, feature engineering, or model architectures
- Multiple models feed into a **larger system**
- **Regulatory requirements** demand full audit trails and code review

### Stay No-Code When:
- **Pre-built models already solve the problem** (Azure AI Vision, Cognitive Services)
- The "model" is a simple **dashboard visual** (Power BI forecasting)
- You need a **one-time analysis**, not an ongoing system

---

## Low-Code Options Summary Across Both Sessions

| Low-Code Tool | Forecasting (Session 1) | Image Labelling (Session 2) |
|--------------|:-----------------------:|:---------------------------:|
| **Azure ML Studio — AutoML Wizard** | ✅ Full support | ⚠️ Vision tasks need SDK/CLI for training |
| **Azure ML Designer** | ✅ Drag-and-drop pipelines | ⚠️ Limited for vision |
| **Power BI** | ✅ Simple exponential smoothing | ❌ |
| **Custom Vision Portal** | ❌ | ✅ Classification + Detection (retiring 2028) |
| **Azure AI Vision Studio** | ❌ | ✅ Pre-built (no training) |
| **Azure ML Data Labeling** | ❌ | ✅ Enterprise annotation |

---

## The Bottom Line

> **Low-code is the best starting point for both tasks.** Use Azure ML Studio's AutoML wizard for forecasting PoCs and Custom Vision Portal (or Azure AI Vision) for image tasks. Once you validate the approach, graduate to Pro-Code (SDK v2) for production-grade MLOps.

---

## Files in This Repository

| File | Description |
|------|-------------|
| [session-1-time-series-forecasting/README.md](session-1-time-series-forecasting/README.md) | Pro-Code forecasting guide (~90 min) |
| [session-1-time-series-forecasting/README-LOW-CODE.md](session-1-time-series-forecasting/README-LOW-CODE.md) | Low-Code forecasting guide (~60 min) |
| [session-1-time-series-forecasting/demo/](session-1-time-series-forecasting/demo/) | Runnable Python demo + data generator |
| [session-2-image-labelling/README.md](session-2-image-labelling/README.md) | Pro-Code image labelling guide (~90 min) |
| [session-2-image-labelling/README-LOW-CODE.md](session-2-image-labelling/README-LOW-CODE.md) | Low-Code image labelling guide (~60 min) |
| [session-2-image-labelling/demo/](session-2-image-labelling/demo/) | Runnable Python demo (AutoML Vision + Custom Vision + AI Vision) |
