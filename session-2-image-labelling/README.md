# Session 2: Image Labelling & Computer Vision with Azure

> **Duration:** ~90 minutes  
> **Level:** Intermediate  
> **Audience:** Data Scientists, ML Engineers, Project Managers overseeing annotation teams  
> **Prerequisites:** Basic understanding of ML, Azure subscription

---

## Table of Contents

1. [Session Overview](#session-overview)
2. [What is Image Labelling?](#what-is-image-labelling)
3. [Why is Labelling Critical?](#why-is-labelling-critical)
4. [Azure Services for Image Labelling & Computer Vision](#azure-services-for-image-labelling--computer-vision)
5. [Core Concepts Deep Dive](#core-concepts-deep-dive)
6. [Azure ML Data Labeling — Deep Dive](#azure-ml-data-labeling--deep-dive)
7. [ML-Assisted Labelling](#ml-assisted-labelling)
8. [AutoML for Computer Vision](#automl-for-computer-vision)
9. [Azure AI Custom Vision](#azure-ai-custom-vision)
10. [Azure AI Vision (Image Analysis 4.0)](#azure-ai-vision-image-analysis-40)
11. [Step-by-Step Demo](#step-by-step-demo)
12. [Best Practices](#best-practices)
13. [Real-World Use Cases](#real-world-use-cases)
14. [Q&A Talking Points](#qa-talking-points)
15. [Resources & References](#resources--references)

---

## Session Overview

This session covers the **end-to-end image labelling and computer vision workflow** on Azure — from raw unlabelled images to trained, deployed models. Attendees will learn how Azure ML's Data Labeling tool works, how ML-assisted labelling accelerates annotation, and how labeled data feeds into AutoML for Vision or Azure AI Custom Vision to produce production-ready models.

### Key Takeaways for the Audience

- Understand the different types of image labelling tasks
- Know which Azure services support labelling and when to use each
- See a live demo of Azure ML Data Labeling (setup + annotation + export)
- Understand ML-assisted labelling (pre-labels + active learning)
- Walk through training a model from labeled data to deployment

---

## What is Image Labelling?

### Definition

Image labelling (also called **image annotation**) is the process of assigning **metadata** to images so that machine learning models can learn to recognize patterns. This metadata can be:

- **A class label** — "this image shows a cat"
- **Bounding boxes** — coordinates around objects in the image
- **Polygons** — precise outlines of object boundaries
- **Pixel-level masks** — every pixel classified into a category

### Types of Image Labelling Tasks

```
┌───────────────────────────────────────────────────────────────────┐
│                    IMAGE LABELLING TASKS                          │
├──────────────────┬──────────────────┬──────────────────┬──────────┤
│  Classification  │ Object Detection │ Instance         │ Semantic │
│  (Multi-class /  │ (Bounding Box)   │ Segmentation     │ Segment. │
│   Multi-label)   │                  │ (Polygon)        │ (Pixel)  │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│  "Is this a      │  "Where are the  │  "Outline each   │  "Color  │
│   dog or cat?"   │   cars?"         │   person exactly" │  every   │
│                  │                  │                  │  pixel"  │
│  Label: cat      │  Box: (x,y,w,h)  │  Polygon:        │  Mask:   │
│                  │  Label: car      │  [(x1,y1),...,    │  class   │
│                  │                  │   (xn,yn)]       │  per px  │
│                  │                  │  Label: person   │          │
├──────────────────┼──────────────────┼──────────────────┼──────────┤
│  Easiest         │ ← Complexity →   │                  │  Hardest │
│  Fastest         │ ← Time/Image →   │                  │  Slowest │
└──────────────────┴──────────────────┴──────────────────┴──────────┘
```

### Comparison of Task Types

| Task | Output | Complexity | Time per Image | Typical Use Case |
|------|--------|------------|----------------|------------------|
| **Multi-class Classification** | Single label per image | Low | 2-5 seconds | Product categorization |
| **Multi-label Classification** | Multiple labels per image | Low-Medium | 5-10 seconds | Scene tagging |
| **Object Detection (Bounding Box)** | Box coordinates + label per object | Medium | 15-60 seconds | Autonomous driving, retail |
| **Instance Segmentation (Polygon)** | Polygon vertices + label per object | High | 1-5 minutes | Medical imaging, agriculture |
| **Semantic Segmentation (Pixel mask)** | Class assigned to every pixel | Very High | 5-30 minutes | Satellite imagery, pathology |

---

## Why is Labelling Critical?

### The Data-Centric AI Paradigm

> "Data quality is more important than model architecture." — Andrew Ng

```
Model Performance = f(Data Quality, Data Quantity, Model Architecture)
                        ^^^^^^^^^^^^
                        MOST IMPACTFUL
```

**Key statistics:**
- Most ML projects spend **80% of time on data** (including labelling)
- Poor labels can degrade model accuracy by **20-40%**
- Inconsistent labels cause models to learn noise instead of patterns
- ML-assisted labelling can reduce annotation time by **50-70%**

### Common Labelling Challenges

| Challenge | Impact | Azure Solution |
|-----------|--------|----------------|
| **Slow manual labelling** | Time bottleneck | ML-assisted pre-labels |
| **Inconsistent labels** | Noisy training data | Quality control + consensus labelling |
| **Expensive annotators** | Budget constraints | Active learning prioritizes high-value images |
| **Label format wrangling** | Engineering overhead | Built-in JSONL/COCO export |
| **Scalability** | Can't label millions of images manually | Pre-labelling + Azure ML compute |

---

## Azure Services for Image Labelling & Computer Vision

### Overview of the Azure Ecosystem

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IMAGE WORKFLOW                              │
│                                                                     │
│   Raw Images  ──▶  LABELLING  ──▶  TRAINING  ──▶  DEPLOYMENT      │
│                                                                     │
│   ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌──────────┐  │
│   │ Azure Blob │  │ Azure ML   │  │ AutoML for   │  │ Managed  │  │
│   │ Storage    │  │ Data       │  │ Vision       │  │ Endpoint │  │
│   │            │  │ Labeling   │  │              │  │          │  │
│   └────────────┘  └────────────┘  │ OR           │  │ OR       │  │
│                                   │              │  │          │  │
│                   ┌────────────┐  │ Custom Vision│  │ REST API │  │
│                   │ 3rd Party  │  │ Service      │  │          │  │
│                   │ (Labelbox, │  │              │  │ OR       │  │
│                   │  Scale AI) │  │ OR           │  │          │  │
│                   └────────────┘  │ Custom Code  │  │ Edge     │  │
│                                   └──────────────┘  │ Deploy   │  │
│                                                     └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Service Comparison

| Service | Best For | Labelling | Training | Deployment |
|---------|----------|-----------|----------|------------|
| **Azure ML Data Labeling** | Enterprise annotation workflows | ✅ Full-featured | Via AutoML for Vision | Azure ML Endpoints |
| **Azure AI Custom Vision** | Quick prototypes, small datasets | ✅ Built-in simple UI | ✅ One-click | ✅ REST + Edge |
| **Azure AI Vision (4.0)** | Pre-built vision capabilities | ❌ No labelling needed | ❌ Pre-trained | ✅ REST API |
| **AutoML for Vision** | Custom models from labeled data | Import from Data Labeling | ✅ Full AutoML | Azure ML Endpoints |

---

## Core Concepts Deep Dive

### 1. Annotation Formats

#### JSONL (JSON Lines) — Azure ML Native Format

Each image gets one JSON line with labels:

```json
{"image_url": "azureml://stores/blob/paths/image001.jpg", "label": "cat"}
{"image_url": "azureml://stores/blob/paths/image002.jpg", "label": "dog"}
```

For object detection:

```json
{
  "image_url": "azureml://stores/blob/paths/image003.jpg",
  "label": [
    {
      "label": "car",
      "topX": 0.12, "topY": 0.34,
      "bottomX": 0.56, "bottomY": 0.78
    },
    {
      "label": "person",
      "topX": 0.60, "topY": 0.10,
      "bottomX": 0.80, "bottomY": 0.90
    }
  ]
}
```

> **Note:** Bounding box coordinates in Azure ML are **normalized** (0.0 to 1.0 relative to image dimensions).

#### COCO Format

The industry-standard format, also supported for export:

```json
{
  "images": [{"id": 1, "file_name": "image001.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 60]}],
  "categories": [{"id": 1, "name": "car"}]
}
```

### 2. MLTable for Computer Vision

To use labeled data with AutoML for Vision, it must be wrapped in an `MLTable`:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable

paths:
  - file: ./annotations.jsonl

transformations:
  - read_json_lines:
      encoding: utf8
      invalid_lines: error
      include_path_column: false
```

### 3. Label Quality Metrics

| Metric | Description |
|--------|-------------|
| **Inter-Annotator Agreement (IAA)** | How consistently multiple labelers label the same image |
| **Cohen's Kappa** | Statistical measure of IAA adjusted for chance agreement |
| **Intersection over Union (IoU)** | Overlap between predicted and ground-truth boxes/polygons |
| **Consensus Labels** | Multiple labelers review same image, majority vote wins |

### 4. Transfer Learning

Instead of training from scratch, Azure's vision models use **pre-trained backbones** (e.g., ResNet, YOLO, EfficientNet) and **fine-tune** them on your labeled data:

```
Pre-trained on ImageNet (14M images)
         │
         ▼
    ┌─────────────┐
    │  Freeze most │
    │  layers      │
    │              │
    │  Fine-tune   │
    │  last layers │
    │  on YOUR     │
    │  data        │
    └─────────────┘
         │
         ▼
  Your custom model
  (needs only 50-500 images!)
```

This is why Azure's Custom Vision and AutoML for Vision can produce good results with **relatively small datasets** (50+ images per class).

---

## Azure ML Data Labeling — Deep Dive

### Project Types Supported

1. **Image Classification Multi-class** — Single label per image
2. **Image Classification Multi-label** — One or more labels per image
3. **Object Identification (Bounding Box)** — Draw rectangles around objects
4. **Instance Segmentation (Polygon)** — Draw precise outlines around objects
5. **Semantic Segmentation (Pixel-level)** — Classify every pixel *(Preview)*

### Project Lifecycle

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   1. CREATE  │───▶│  2. SETUP    │───▶│  3. LABEL    │───▶│  4. EXPORT   │
│   Project    │    │  Labels,     │    │  Annotate    │    │  Dataset     │
│              │    │  Instructions│    │  images      │    │  (JSONL/COCO)│
│  Choose type │    │  Add data    │    │  Review      │    │  Use in      │
│  Name it     │    │  Assign team │    │  QA          │    │  AutoML      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                              │
                                        ┌─────▼─────┐
                                        │ ML-Assist │
                                        │ Pre-label │
                                        │ Cluster   │
                                        └───────────┘
```

### Setting Up a Project (Azure ML Studio Steps)

1. Go to [ml.azure.com](https://ml.azure.com) → **Data Labeling** → **Add project**
2. **Project name**: e.g., "product-classification-v1"
3. **Media type**: Image
4. **Labeling task type**: Choose one of:
   - Image Classification Multi-class
   - Image Classification Multi-label
   - Object Identification (Bounding Box)
   - Polygon (Instance Segmentation)
   - Semantic Segmentation (Preview)
5. **Add data**: Select a datastore pointing to your Azure Blob container
6. **Add labels**: Define your label taxonomy (e.g., "Car", "Truck", "Bicycle")
7. **Instructions**: Write clear guidelines for annotators
8. **Quality control** (optional): Enable consensus labelling
9. **ML-assisted labelling** (optional): Enable for auto-pre-labels
10. **Create** → Project initializes

### Labelling Tools in the Studio UI

| Tool | Shortcut | Purpose |
|------|----------|---------|
| **Tag selector** | 1-9 | Select a label from the taxonomy |
| **Rectangular box** | R | Draw bounding boxes |
| **Draw polygon** | P | Draw polygon outlines |
| **Lock/unlock regions** | L | Prevent accidental edits |
| **Add/remove polygon points** | U | Refine polygon shapes |
| **Template-based box** | T | Create same-size boxes quickly |
| **Delete all regions** | — | Clear all annotations from image |
| **Move region** | — | Reposition existing annotations |
| **Submit** | — | Save and move to next batch |

### Exporting Labels

Two export formats supported:

| Format | When to Use |
|--------|-------------|
| **Azure ML Dataset** | Feed directly into AutoML for Vision training |
| **COCO** | Use with external tools (Detectron2, YOLO, etc.) |

Export via Studio UI:
1. Go to **Data Labeling** → Your project → **Project details**
2. Click **Export**
3. Choose format
4. Download or reference as a dataset

---

## ML-Assisted Labelling

### How It Works

ML-assisted labelling uses your initial manual labels to train a model that **automatically pre-labels** remaining images. This creates a human-in-the-loop workflow:

```
┌───────────────┐     ┌────────────────┐     ┌──────────────────┐
│  1. Manually  │────▶│  2. Model      │────▶│  3. Pre-labeled  │
│  label ~200   │     │  trains on     │     │  images appear   │
│  images       │     │  your labels   │     │  with predictions│
└───────────────┘     └────────────────┘     └────────┬─────────┘
                                                       │
                      ┌────────────────┐               │
                      │  4. Labeler    │◀──────────────┘
                      │  reviews &     │
                      │  corrects      │
                      │  pre-labels    │
                      └────────┬───────┘
                               │
                               ▼
                      ┌────────────────┐
                      │  5. Model      │
                      │  re-trains     │──── Cycle repeats
                      │  (improved!)   │
                      └────────────────┘
```

### Two Mechanisms

#### 1. Clustering (Classification Tasks)
- Groups similar images together on the same labelling page
- Makes manual labelling faster (similar images can be batch-tagged)
- Uses embedding from a truncated neural network

#### 2. Pre-labelling (All Tasks)
- After enough manual labels, model predicts labels for remaining images
- Predictions with confidence above a threshold are shown as pre-labels
- Labelers **review and correct** rather than label from scratch
- Typically reduces annotation time by **50-70%**

### Confidence Threshold

AutoML evaluates the pre-labelling model on a test set of manually labeled items. It picks a **confidence threshold** that balances:
- Showing enough pre-labels to save time
- Keeping accuracy high enough that corrections are minimal

---

## AutoML for Computer Vision

After exporting labeled data from Azure ML Data Labeling, you can train models using **AutoML for Vision** (preview).

### Supported Tasks

| Task | AutoML Function | Supported Models |
|------|----------------|-----------------|
| **Image Classification (Multi-class)** | `automl.image_classification()` | ResNet, EfficientNet, ViT |
| **Image Classification (Multi-label)** | `automl.image_classification_multilabel()` | ResNet, EfficientNet, ViT |
| **Object Detection** | `automl.image_object_detection()` | YOLOv5, Faster R-CNN, RetinaNet |
| **Instance Segmentation** | `automl.image_instance_segmentation()` | Mask R-CNN |

### Code Example — Image Classification

```python
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.automl import ClassificationMultilabelPrimaryMetrics

# Create image classification job
image_classification_job = automl.image_classification(
    experiment_name="product-classification",
    compute="gpu-cluster",
    training_data=Input(
        type=AssetTypes.MLTABLE,
        path="./training-mltable-folder"
    ),
    validation_data=Input(
        type=AssetTypes.MLTABLE,
        path="./validation-mltable-folder"
    ),
    target_column_name="label",
    primary_metric="accuracy",
)

# Set limits
image_classification_job.set_limits(
    timeout_minutes=120,
    max_trials=10,
    max_concurrent_trials=2,
)

# Submit
returned_job = ml_client.jobs.create_or_update(image_classification_job)
print(f"Studio URL: {returned_job.services['Studio'].endpoint}")
```

### Code Example — Object Detection

```python
from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes

image_object_detection_job = automl.image_object_detection(
    experiment_name="defect-detection",
    compute="gpu-cluster",
    training_data=Input(
        type=AssetTypes.MLTABLE,
        path="./training-mltable-folder"
    ),
    validation_data=Input(
        type=AssetTypes.MLTABLE,
        path="./validation-mltable-folder"
    ),
    target_column_name="label",
    primary_metric="mean_average_precision",
)

returned_job = ml_client.jobs.create_or_update(image_object_detection_job)
```

---

## Azure AI Custom Vision

### What is Custom Vision?

A **simpler, more accessible** alternative to AutoML for Vision. It provides a web-based UI for uploading images, labelling them, and training models with **zero code**.

### When to Use Custom Vision vs. AutoML for Vision

| Criteria | Custom Vision | AutoML for Vision |
|----------|---------------|-------------------|
| **Ease of use** | Very easy — web UI | Requires SDK/code knowledge |
| **Dataset size** | 50-100K images | 10-500K+ images |
| **Customization** | Limited | Full control over hyperparameters |
| **Model types** | Classification + Object Detection | Classification + Detection + Segmentation |
| **Edge deployment** | ✅ Built-in export to ONNX, TensorFlow, CoreML | Via Azure ML managed endpoints |
| **Cost** | Pay per prediction + training | Pay for compute time |

### Code Example — Custom Vision (Python SDK)

```python
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
)
from msrest.authentication import ApiKeyCredentials
import os

# Authenticate
ENDPOINT = os.environ["CUSTOM_VISION_ENDPOINT"]
training_key = os.environ["CUSTOM_VISION_TRAINING_KEY"]
prediction_key = os.environ["CUSTOM_VISION_PREDICTION_KEY"]

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Create project
project = trainer.create_project("ProductClassification")

# Create tags (labels)
electronics_tag = trainer.create_tag(project.id, "Electronics")
clothing_tag = trainer.create_tag(project.id, "Clothing")
food_tag = trainer.create_tag(project.id, "Food")

# Upload and tag images
image_list = []
for img_file in os.listdir("./images/electronics"):
    with open(f"./images/electronics/{img_file}", "rb") as f:
        image_list.append(
            ImageFileCreateEntry(
                name=img_file,
                contents=f.read(),
                tag_ids=[electronics_tag.id]
            )
        )

# Batch upload
upload_result = trainer.create_images_from_files(
    project.id,
    ImageFileCreateBatch(images=image_list)
)
if not upload_result.is_batch_successful:
    print("Upload failed!")

# Train the model
import time
iteration = trainer.train_project(project.id)
while iteration.status != "Completed":
    iteration = trainer.get_iteration(project.id, iteration.id)
    print(f"Training status: {iteration.status}")
    time.sleep(10)

# Publish the model
publish_iteration_name = "classifyModel"
trainer.publish_iteration(
    project.id,
    iteration.id,
    publish_iteration_name,
    os.environ["CUSTOM_VISION_PREDICTION_RESOURCE_ID"]
)

# Test prediction
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open("./test_images/test_product.jpg", "rb") as image_contents:
    results = predictor.classify_image(
        project.id, publish_iteration_name, image_contents.read()
    )
    for prediction in results.predictions:
        print(f"  {prediction.tag_name}: {prediction.probability * 100:.2f}%")
```

---

## Azure AI Vision (Image Analysis 4.0)

### Pre-Built Capabilities (No Labelling Needed!)

For many common scenarios, you don't need custom labelling at all. Azure AI Vision 4.0 provides pre-trained models for:

| Feature | Description |
|---------|-------------|
| **Caption** | Generate natural language description of the image |
| **Dense Captions** | Detailed captions for all regions |
| **Tags** | Content tags (e.g., "outdoor", "building", "sky") |
| **Object Detection** | Detect 600+ object categories |
| **People Detection** | Detect people + positions |
| **Read (OCR)** | Extract text from images |
| **Smart Cropping** | Identify regions of interest |

### Code Example — Image Analysis

```python
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Setup client
client = ImageAnalysisClient(
    endpoint=os.environ["VISION_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["VISION_KEY"])
)

# Analyze an image
result = client.analyze_from_url(
    image_url="https://example.com/sample-image.jpg",
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.READ,
    ],
    gender_neutral_caption=True,
)

# Print results
print("Caption:", result.caption.text,
      f"(Confidence: {result.caption.confidence:.2f})")

print("\nTags:")
for tag in result.tags.list:
    print(f"  {tag.name}: {tag.confidence:.2f}")

print("\nObjects detected:")
for obj in result.objects.list:
    print(f"  {obj.tags[0].name} at {obj.bounding_box}")

print("\nText (OCR):")
if result.read is not None:
    for block in result.read.blocks:
        for line in block.lines:
            print(f"  {line.text}")
```

---

## Step-by-Step Demo

> **Scenario:** Build a product image classifier for an e-commerce company. Classify product images into categories: Electronics, Clothing, Food, Furniture.

### Part A: Azure ML Data Labeling (Studio UI — 20 min)

#### Step 1: Prepare Images in Azure Blob Storage

```bash
# Upload sample images to Azure Blob
az storage container create \
    --account-name <STORAGE_ACCOUNT> \
    --name product-images

az storage blob upload-batch \
    --account-name <STORAGE_ACCOUNT> \
    --destination product-images \
    --source ./sample_images/
```

#### Step 2: Create a Data Labeling Project

1. Navigate to [ml.azure.com](https://ml.azure.com) → your workspace
2. Click **Data Labeling** in the left navigation
3. Click **+ Add project**
4. Configure:
   - **Project name**: `product-classification-demo`
   - **Media type**: Image
   - **Labeling task type**: Image Classification Multi-class
5. Click **Next**

#### Step 3: Connect Data

1. Select **Add datastore** (or choose existing)
2. Point to your Azure Blob container with product images
3. Azure ML will discover all images in the container
4. Click **Next**

#### Step 4: Define Labels

Add the following labels:
- `Electronics`
- `Clothing`
- `Food`
- `Furniture`

Click **Next**

#### Step 5: Write Labelling Instructions

Write clear, unambiguous instructions for labellers:

```markdown
## Product Classification Instructions

**Task:** Assign each image to exactly ONE product category.

### Categories:

1. **Electronics** — Phones, laptops, tablets, headphones, cameras, TVs
2. **Clothing** — Shirts, pants, dresses, shoes, accessories
3. **Food** — Packaged food, fresh produce, beverages, snacks
4. **Furniture** — Tables, chairs, sofas, shelves, beds

### Guidelines:
- If a product could fit multiple categories, choose the PRIMARY category
- If the image is unclear or not a product, skip it
- Label based on the MAIN subject of the image, not background objects
```

#### Step 6: Enable ML-Assisted Labelling

1. Check **Enable ML-assisted labeling**
2. This will use your initial manual labels to train a model
3. After ~200 manual labels, pre-labels will start appearing

#### Step 7: Start Labelling

1. Click **Start labeling**
2. Images appear in batches
3. Select the correct category for each image
4. Use keyboard shortcuts (1-4) for speed
5. Click **Submit** after each batch

**Demo talking point:** Show the audience:
- How fast the UI is for classification
- Keyboard shortcuts for speed
- Multi-image selection for batch labelling
- Pre-labels appearing after ML model trains

#### Step 8: Monitor Progress

1. Go to **Dashboard** tab to see:
   - Completion percentage
   - Labeller productivity
   - Label distribution
   - ML-assist model training status

#### Step 9: Export Labeled Data

1. Click **Export** on the project overview page
2. Choose **Azure ML Dataset** format
3. The exported dataset is now available under **Datasets** → ready for training

### Part B: Training with AutoML for Vision (SDK — 15 min)

#### Step 1: Setup Environment

```bash
pip install azure-ai-ml azure-identity
```

#### Step 2: Connect to Workspace

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<YOUR_SUBSCRIPTION_ID>",
    resource_group_name="<YOUR_RESOURCE_GROUP>",
    workspace_name="<YOUR_WORKSPACE_NAME>"
)
```

#### Step 3: Create GPU Compute

```python
from azure.ai.ml.entities import AmlCompute

compute_name = "gpu-cluster"
try:
    ml_client.compute.get(compute_name)
except Exception:
    gpu_compute = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_NC6",  # GPU instance
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(gpu_compute).result()
```

#### Step 4: Reference the Labeled Data

```python
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

# Reference the exported dataset from Data Labeling
training_data = Input(
    type=AssetTypes.MLTABLE,
    path="azureml:product-classification-labeled:1"  # Exported dataset
)

# Or use local MLTable
training_data = Input(
    type=AssetTypes.MLTABLE,
    path="./training-mltable-folder"
)
```

#### Step 5: Configure and Submit AutoML for Vision

```python
from azure.ai.ml import automl

image_classification_job = automl.image_classification(
    experiment_name="product-image-classification",
    compute=compute_name,
    training_data=training_data,
    target_column_name="label",
    primary_metric="accuracy",
)

# Set training limits
image_classification_job.set_limits(
    timeout_minutes=60,
    max_trials=10,
    max_concurrent_trials=2,
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(image_classification_job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.services['Studio'].endpoint}")
```

#### Step 6: Monitor in Azure ML Studio

Show the audience:
1. **Runs** page with all trial details
2. **Model leaderboard** — best accuracy across models
3. **Metrics** — accuracy, precision, recall, confusion matrix
4. **Images** tab — predictions visualized on sample images

#### Step 7: Register and Deploy the Best Model

```python
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

# Register the best model
best_model = Model(
    path=f"azureml://jobs/{returned_job.name}/outputs/best_model",
    name="product-classifier",
    type=AssetTypes.MLFLOW_MODEL,
)
registered_model = ml_client.models.create_or_update(best_model)

# Deploy to managed endpoint
endpoint = ManagedOnlineEndpoint(name="product-classifier-endpoint")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint.name,
    model=registered_model,
    instance_type="Standard_NC6",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()

print(f"Endpoint ready: {endpoint.name}")
```

### Part C: Quick Custom Vision Demo (Studio UI — 10 min)

As an alternative for simpler workflows, show [customvision.ai](https://www.customvision.ai):

1. **Create project** → Classification → General domain
2. **Upload images** → Tag them in the browser
3. **Train** → Click the green "Train" button
4. **Test** → Upload a test image → See predictions
5. **Export** → Download as ONNX, TensorFlow, or CoreML for edge deployment

### Part D: Azure AI Vision — Pre-Built (5 min)

Show pre-built analysis capabilities with **no training required**:

```python
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    endpoint=os.environ["VISION_ENDPOINT"],
    credential=AzureKeyCredential(os.environ["VISION_KEY"])
)

# Analyze a product image
result = client.analyze_from_url(
    image_url="https://example.com/product.jpg",
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
    ],
)

print(f"Caption: {result.caption.text}")
print(f"Tags: {[t.name for t in result.tags.list]}")
print(f"Objects: {[o.tags[0].name for o in result.objects.list]}")
```

**Demo talking point:** "For many scenarios, you might not need custom labelling at all — Azure AI Vision provides pre-built capabilities that work out of the box."

---

## Best Practices

### Labelling Strategy

- **Start small** — label 50-100 images per class, train a model, evaluate, then decide if you need more
- **Write clear instructions** — ambiguous guidelines cause inconsistent labels
- **Enable ML-assisted labelling** — after ~200 manual labels, pre-labels dramatically speed things up
- **Use consensus labelling** — for critical projects, have 2-3 labellers per image
- **Balance your classes** — aim for similar numbers of images per category

### Data Quality

- **Diverse images** — include different angles, lighting, backgrounds
- **Representative of production** — training images should match real-world conditions
- **Consistent resolution** — ideally 256x256 to 1024x1024 pixels
- **Avoid label leakage** — don't include text or watermarks that reveal the class

### Training

- **Use GPU compute** for vision tasks — CPU is 10-50x slower
- **Start with AutoML defaults** — they're well-tuned for most scenarios
- **Check the confusion matrix** — it reveals which classes the model confuses
- **Use validation data** — hold out 10-20% of labeled data for evaluation

### Production

- **Monitor prediction confidence** — flag low-confidence predictions for human review
- **Retrain periodically** — as new product types or visual patterns emerge
- **A/B test** — deploy new models alongside existing ones
- **Edge deployment** — for latency-sensitive scenarios, export to ONNX

---

## Real-World Use Cases

| Industry | Use Case | Labelling Type | Azure Service |
|----------|----------|----------------|---------------|
| **Retail** | Product categorization | Multi-class classification | Custom Vision / AutoML |
| **Manufacturing** | Defect detection | Object detection (bbox) | AutoML for Vision |
| **Healthcare** | Medical image analysis | Instance segmentation | AutoML for Vision |
| **Agriculture** | Crop disease detection | Multi-class classification | Custom Vision |
| **Construction** | Safety equipment detection | Object detection (bbox) | AutoML for Vision |
| **Insurance** | Damage assessment | Multi-label classification | AutoML for Vision |
| **Autonomous Vehicles** | Road scene understanding | Semantic segmentation | AutoML for Vision |
| **Fashion** | Style tagging | Multi-label classification | Custom Vision |

---

## Q&A Talking Points

**Q: How many labeled images do I need?**
> For Custom Vision: minimum 50 per class. For AutoML for Vision: recommended 100+ per class. More data generally means better accuracy, but ML-assisted labelling can help you get there faster.

**Q: Can I import labels from other tools (Labelbox, CVAT, etc.)?**
> Yes! Convert your labels to JSONL format (Azure ML native) or COCO format. Azure provides [conversion scripts](https://github.com/Azure/azureml-examples/blob/main/v1/python-sdk/tutorials/automl-with-azureml/) for common formats.

**Q: What if my classes are imbalanced (e.g., 1000 images of "normal" vs. 50 of "defect")?**
> AutoML handles class imbalance automatically through techniques like oversampling and weighted loss. For severe imbalance, consider collecting more minority-class images or using data augmentation.

**Q: Can I deploy models to edge devices (IoT)?**
> Custom Vision can export directly to ONNX, TensorFlow Lite, CoreML, and Docker. For AutoML models, export to ONNX and deploy via Azure IoT Edge.

**Q: How does pricing work?**
> Azure ML Data Labeling: free (you pay for compute and storage). Custom Vision: per prediction + per training hour. AutoML for Vision: pay for GPU compute hours during training.

**Q: Can labellers work remotely?**
> Yes. Azure ML Data Labeling is fully web-based. Assign labellers via their Microsoft account or Azure AD identity. They only need browser access to ml.azure.com.

---

## Resources & References

- [Set up an Image Labeling Project in Azure ML](https://learn.microsoft.com/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2)
- [Labeling Images — The Annotator's Guide](https://learn.microsoft.com/azure/machine-learning/how-to-label-data?view=azureml-api-2)
- [Prepare Data for AutoML Computer Vision](https://learn.microsoft.com/azure/machine-learning/how-to-prepare-datasets-for-automl-images?view=azureml-api-2)
- [Train Computer Vision Models with AutoML](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2)
- [Azure AI Custom Vision — Quickstart](https://learn.microsoft.com/azure/ai-services/custom-vision-service/quickstarts/image-classification?pivots=programming-language-python)
- [Azure AI Vision — Image Analysis 4.0](https://learn.microsoft.com/azure/ai-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40?pivots=programming-language-python)
- [Export Labels from Data Labeling Projects](https://learn.microsoft.com/azure/machine-learning/how-to-manage-labeling-projects?view=azureml-api-2#export-the-labels)
- [ML-Assisted Data Labeling](https://learn.microsoft.com/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2#use-ml-assisted-data-labeling)

---

## Agenda Slide (Copy-Paste for your Presentation)

```
Session 2: Image Labelling & Computer Vision on Azure
──────────────────────────────────────────────────────
1. 📌 Introduction & Why Labelling Matters          (10 min)
2. 🏷️  Types of Image Labelling Tasks                (10 min)
3. ☁️  Azure Services Landscape                       (10 min)
4. 🔧 Azure ML Data Labeling — Deep Dive             (10 min)
5. 💻 Live Demo Part A: Data Labeling Project         (20 min)
   - Create project in Studio
   - Label images
   - ML-assisted pre-labels
   - Export labeled data
6. 💻 Live Demo Part B: Train with AutoML for Vision  (15 min)
   - AutoML image classification
   - Model evaluation in Studio
7. 🚀 Quick Demos: Custom Vision + AI Vision 4.0      (10 min)
8. ❓ Q&A                                             (5 min)
──────────────────────────────────────────────────────
```
