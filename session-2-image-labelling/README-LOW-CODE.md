# Session 2 — LOW-CODE: Image Labelling & Computer Vision — No Code Required

> **Duration:** ~60 minutes  
> **Level:** Beginner to Intermediate  
> **Audience:** Business Analysts, Project Managers, Citizen Data Scientists, anyone with ZERO coding experience  
> **Prerequisites:** Azure subscription, web browser, sample images (JPG/PNG)

---

## Table of Contents

1. [What Does "Low-Code" Mean Here?](#what-does-low-code-mean-here)
2. [Low-Code Options for Image Labelling on Azure](#low-code-options-for-image-labelling-on-azure)
3. [Option 1: Azure ML Data Labeling + AutoML Studio](#option-1-azure-ml-data-labeling--automl-studio)
4. [Option 2: Azure Custom Vision Portal](#option-2-azure-custom-vision-portal)
5. [Option 3: Azure AI Vision Studio](#option-3-azure-ai-vision-studio)
6. [Step-by-Step Demo: Custom Vision Portal (End-to-End)](#step-by-step-demo-custom-vision-portal-end-to-end)
7. [Step-by-Step Demo: Azure ML Data Labeling](#step-by-step-demo-azure-ml-data-labeling)
8. [Step-by-Step Demo: Azure AI Vision Studio](#step-by-step-demo-azure-ai-vision-studio)
9. [Low-Code vs. Pro-Code — When to Use What](#low-code-vs-pro-code--when-to-use-what)
10. [Resources & References](#resources--references)

---

## What Does "Low-Code" Mean Here?

Azure provides **multiple tiers** of experience for image labelling and computer vision:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SPECTRUM OF APPROACHES                            │
│                                                                      │
│   NO-CODE               LOW-CODE               PRO-CODE             │
│   ────────              ────────               ────────             │
│   Custom Vision Portal  Azure ML Data          Python SDK v2        │
│   (customvision.ai)     Labeling + AutoML      CLI v2               │
│                         Studio UI              Jupyter Notebooks    │
│   Azure AI Vision                                                    │
│   Studio (try.it)       AutoML for Vision      Custom PyTorch /     │
│                         (needs some config)    HuggingFace          │
│                                                                      │
│   ◀─── Zero code ──────────────────────── Full code control ───▶    │
│   ◀─── Fastest start ──────────────────── Most flexible ───────▶    │
└──────────────────────────────────────────────────────────────────────┘
```

This README focuses on **No-Code** and **Low-Code** approaches. See the main [README.md](./README.md) for the Pro-Code path.

---

## Low-Code Options for Image Labelling on Azure

| Option | Code Required | Labelling | Training | Deployment | Best For |
|--------|:------------:|:---------:|:--------:|:----------:|----------|
| **Custom Vision Portal** | **Zero** | ✅ Built-in | ✅ One-click | ✅ REST + Edge export | Quick prototypes, small datasets |
| **Azure ML Data Labeling + Studio** | **Zero** | ✅ Full-featured | ✅ AutoML via Studio | ✅ One-click | Enterprise workflows, large teams |
| **Azure AI Vision Studio** | **Zero** | ❌ Not needed | ❌ Pre-trained | ✅ REST API | Pre-built capabilities, no training needed |

---

## Option 1: Azure ML Data Labeling + AutoML Studio

### What Is It?

The **Azure ML Data Labeling** tool is a browser-based annotation workspace inside [ml.azure.com](https://ml.azure.com). You create a labeling project, upload images, assign labelers, annotate images using visual tools, and export labeled data — all without code.

After labeling, you can submit the exported dataset to **AutoML for Vision** directly from the Studio UI, train a model, and deploy it — still without code.

### Complete No-Code Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload      │────▶│  Label       │────▶│  AutoML      │────▶│  Deploy      │
│  Images      │     │  in Studio   │     │  Train       │     │  Endpoint    │
│  (Blob       │     │  (browser)   │     │  (Studio UI) │     │  (one-click) │
│   Storage)   │     │              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                     │  ML-assisted  │
                     │  pre-labels   │
                     │  speed things │
                     │  up!          │
                     └──────────────┘
                       ZERO CODE throughout
```

### What You Get

- Label images using visual tools (tags, bounding boxes, polygons)
- Assign multiple labelers with quality control
- ML-assisted pre-labels (model learns from your initial labels and suggests the rest)
- Clustering groups similar images for faster batch labeling
- Export to Azure ML Dataset or COCO format
- Train directly via AutoML in Studio
- One-click deployment to managed endpoint

---

## Option 2: Azure Custom Vision Portal

### What Is It?

[customvision.ai](https://www.customvision.ai) is a **completely standalone web portal** for building image classifiers and object detectors. It handles the entire lifecycle — upload, label, train, evaluate, deploy, and export — in a single browser-based experience.

> **Important Note:** Microsoft has announced planned retirement of Custom Vision by **9/25/2028**. It is fully supported until then. For new projects, consider starting with Azure ML AutoML for Vision. For quick prototypes, Custom Vision remains an excellent choice.

### Complete No-Code Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload      │────▶│  Tag         │────▶│  Train       │────▶│  Test &      │
│  Images      │     │  Images      │     │  (one click) │     │  Deploy      │
│  (drag/drop) │     │  (in browser)│     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                │  REST API   │
                                                                │  OR         │
                                                                │  Export to  │
                                                                │  ONNX /     │
                                                                │  TensorFlow │
                                                                │  CoreML /   │
                                                                │  Docker     │
                                                                └──────────────┘
```

### Key Advantages

- **Simplest possible experience** — literally drag images, click tags, click train
- **Works with 50+ images per class** — great for small datasets
- **Edge export** — download models as ONNX, TensorFlow Lite, CoreML, or Docker containers
- **Smart Labeler** — after initial training, suggests labels for new images
- **Two task types**: Image Classification (multi-class/multi-label) + Object Detection

---

## Option 3: Azure AI Vision Studio

### What Is It?

[Azure AI Vision Studio](https://portal.vision.cognitive.azure.com/) lets you **try pre-built computer vision models** directly in the browser. No labeling, no training — just upload an image and see results.

### What You Can Do (No Training Needed)

| Capability | What It Does |
|------------|-------------|
| **Caption** | Generates a natural language description of any image |
| **Dense Captions** | Describes every region of an image |
| **Tags** | Returns content tags (e.g., "sky", "building", "outdoor") |
| **Object Detection** | Detects 600+ pre-trained object categories |
| **People Detection** | Detects people and body positions |
| **OCR / Read** | Extracts text from images and documents |
| **Smart Crop** | Identifies the most interesting region of an image |
| **Face Detection** | Detects faces and attributes |

### When This Is Enough

For many real-world scenarios, you may **not need custom labelling or training at all**. Azure AI Vision's pre-trained models handle:

- General image categorization
- Text extraction from photos/documents
- People counting / detection
- Generic object detection (cars, animals, furniture, etc.)

> **Rule of thumb:** If your categories are common/general (animals, vehicles, food, text), try Azure AI Vision first. If your categories are domain-specific (specific product SKUs, rare defects, custom medical images), you need custom labelling + training.

---

## Step-by-Step Demo: Custom Vision Portal (End-to-End)

> **Scenario:** Classify product images into categories (Electronics, Food, Clothing) using zero code.  
> **Time:** ~20 minutes  
> **Dataset:** Any collection of product images (50+ per category recommended)

### Step 1: Create Azure Resources

1. Go to [portal.azure.com](https://portal.azure.com)
2. Search for **Custom Vision** → Click **Create**
3. Configure:

    | Field | Value |
    |-------|-------|
    | Create options | Both (Training + Prediction) |
    | Subscription | Your subscription |
    | Resource group | Create new or select existing |
    | Region | Your nearest region |
    | Name | `product-vision-demo` |
    | Training pricing tier | Free F0 (or Standard S0) |
    | Prediction pricing tier | Free F0 (or Standard S0) |

4. Click **Create** and wait for deployment

### Step 2: Create a New Project

1. Go to [customvision.ai](https://www.customvision.ai) → Sign in
2. Click **New Project**
3. Fill in:

    | Field | Value |
    |-------|-------|
    | Name | `ProductClassification` |
    | Description | Classify product images |
    | Resource | Select the resource you just created |
    | Project Type | Classification |
    | Classification Type | Multiclass (one label per image) |
    | Domain | General or Retail |

4. Click **Create project**

### Step 3: Upload and Tag Images

1. Click **Add images**
2. Browse and select your Electronics images (~50 photos)
3. In the **My Tags** field, type `Electronics` → press Enter
4. Click **Upload [N] files**
5. Wait for upload → Click **Done**
6. Repeat for `Clothing` and `Food` categories

> **Demo talking point:** "Notice how simple this is — we're just dragging images and typing category names. No code, no configuration files, no data formatting."

### Step 4: Train the Model

1. Click the green **Train** button (top right)
2. Choose training type:
   - **Quick Training** (~5 minutes) — good for demo
   - **Advanced Training** (1-24 hours) — better accuracy, specify budget
3. Click **Train**
4. Wait for training to complete

### Step 5: Evaluate Results

After training, the **Performance** tab shows:

| Metric | What It Means |
|--------|--------------|
| **Precision** | Of images the model labeled as X, what % were actually X? |
| **Recall** | Of all actual X images, what % did the model correctly find? |
| **AP (Average Precision)** | Combined precision/recall at various thresholds |

- Review per-tag metrics
- Click a tag to see which images were misclassified

> **Demo talking point:** "With just 50 images per class and zero code, we're already getting 85-95% accuracy thanks to transfer learning."

### Step 6: Quick Test

1. Click **Quick Test** (top right)
2. Upload a new image that wasn't in the training set
3. See the predictions with confidence scores

### Step 7: Publish and Use

1. Click **Publish** → Give it a name like `classifyModel`
2. Select your Prediction resource
3. Click **Publish**

The model is now available as a REST API endpoint. The **Prediction URL** is displayed — you can call it from any application.

### Step 8: Export for Edge (Optional)

1. Click **Export** in the Performance tab
2. Choose your format:
   - **ONNX** — cross-platform
   - **TensorFlow** — Android, Linux
   - **CoreML** — iOS, macOS
   - **Docker** (Linux or Windows container)
   - **VAIDK** — Vision AI Dev Kit
3. Click **Download**

> **Demo talking point:** "We can run this model offline on a phone or an IoT device — no cloud connection needed."

---

## Step-by-Step Demo: Azure ML Data Labeling

> **Scenario:** Set up an enterprise labeling project with ML-assisted annotations.  
> **Time:** ~15 minutes (setup) + labeling time

### Step 1: Upload Images to Blob Storage

1. Go to [portal.azure.com](https://portal.azure.com) → your Storage Account
2. Create a container: `product-images`
3. Upload your images via **Upload** button

### Step 2: Create a Data Labeling Project

1. Go to [ml.azure.com](https://ml.azure.com) → your workspace
2. Click **Data Labeling** → **+ Add project**
3. Configure:

    | Setting | Value |
    |---------|-------|
    | Project name | `product-labeling-demo` |
    | Media type | Image |
    | Labeling task type | Image Classification Multi-class |

4. Click **Next**

### Step 3: Add Data

1. Select or create a **Datastore** pointing to your Blob container
2. Azure ML discovers all images automatically
3. Click **Next**

### Step 4: Define Labels

Add your label taxonomy:
- `Electronics`
- `Clothing`
- `Food`
- `Furniture`

Click **Next**

### Step 5: Write Instructions

Provide clear labeling guidelines (see main README for example).

### Step 6: Enable ML-Assisted Labeling

1. Check ✅ **Enable ML-assisted labeling**
2. This trains a model on your initial labels and pre-labels the rest

### Step 7: Start Labeling

1. Click **Start labeling**
2. Images appear in batches of 4-9
3. Select images → click the correct label → **Submit**
4. Use keyboard shortcuts (1-4) for speed

### Step 8: Watch ML-Assist Kick In

After ~200 manual labels:
- **Clustering**: Similar images group together on the same page
- **Pre-labels**: Predicted labels appear — you just confirm or correct

> **Demo talking point:** "After labeling about 200 images manually, the system started suggesting labels automatically. We went from 5 seconds per image to 1 second per image."

### Step 9: Export

1. Click **Export** on the project overview
2. Choose **Azure ML Dataset**
3. Use the exported dataset for training via AutoML in Studio

---

## Step-by-Step Demo: Azure AI Vision Studio

> **Scenario:** Try pre-built image analysis — no training or labeling needed.  
> **Time:** ~5 minutes

### Step 1: Open Vision Studio

1. Go to [portal.vision.cognitive.azure.com](https://portal.vision.cognitive.azure.com/)
2. Sign in with your Azure account
3. Select your Azure AI Vision resource

### Step 2: Try Image Captioning

1. Click **Image analysis** → **Add captions to images**
2. Upload an image or select a sample
3. See the auto-generated caption and confidence score

### Step 3: Try Object Detection

1. Click **Detect common objects in images**
2. Upload an image
3. See bounding boxes drawn around detected objects

### Step 4: Try OCR

1. Click **Extract text from images**
2. Upload an image containing text
3. See all extracted text with bounding polygons

> **Demo talking point:** "For common use cases like reading text from receipts, detecting cars in parking lots, or captioning images — you don't need to label or train anything. These pre-built models work out of the box."

---

## Low-Code vs. Pro-Code — When to Use What

### Decision Tree

```
                    ┌──────────────────────────────────┐
                    │  Do you need CUSTOM categories    │
                    │  that Azure AI Vision doesn't     │
                    │  already recognize?               │
                    └──────────┬────────────┬───────────┘
                               │            │
                           NO  │            │  YES
                               │            │
                    ┌──────────▼──┐    ┌────▼─────────────────┐
                    │ Azure AI    │    │ How many images?      │
                    │ Vision      │    │ How complex?           │
                    │ (Pre-built) │    └──┬──────────────┬─────┘
                    │ Done!       │       │              │
                    └─────────────┘   < 10K images   > 10K images
                                     Simple task     Complex / Enterprise
                                         │              │
                              ┌──────────▼──┐    ┌──────▼──────────────┐
                              │ Custom      │    │ Azure ML Data       │
                              │ Vision      │    │ Labeling            │
                              │ Portal      │    │ + AutoML for Vision │
                              │ (No-Code)   │    │ (Low/Pro-Code)      │
                              └─────────────┘    └─────────────────────┘
```

### Detailed Comparison

| Criteria | Azure AI Vision (Pre-built) | Custom Vision Portal | Azure ML Data Labeling + AutoML | SDK v2 (Pro-Code) |
|----------|-----------------------------|---------------------|---------------------------------|--------------------|
| **Code required** | None | None | None (Studio) / Minimal | Full Python |
| **Labeling** | Not needed | Built-in (simple UI) | Full-featured + ML-assist | Import from any source |
| **Training** | Not needed | One-click | AutoML via Studio | Full control |
| **Custom categories** | ❌ Pre-defined only | ✅ Your own | ✅ Your own | ✅ Your own |
| **Dataset size** | N/A | 50-100K images | Unlimited | Unlimited |
| **Task types** | Detection, Caption, OCR | Classification, Detection | Classification, Detection, Segmentation | All + custom architectures |
| **Edge export** | ❌ | ✅ ONNX, TF, CoreML | Via model export | Via model export |
| **Team labeling** | N/A | ❌ Single user | ✅ Multi-labeler + QA | ✅ Multi-labeler + QA |
| **ML-assisted labeling** | N/A | Smart Labeler (basic) | ✅ Full (pre-labels + clustering) | ✅ Full |
| **Time to first result** | 2 minutes | 30 minutes | 1-2 hours | 4+ hours |
| **Best for** | Generic tasks, OCR | Quick PoC, small datasets | Enterprise, large teams | Production systems |

### Recommendations

| Scenario | Recommended Approach | Why |
|----------|---------------------|-----|
| "Read text from receipts/documents" | **Azure AI Vision** (pre-built) | OCR works out of the box |
| "Detect common objects (cars, people)" | **Azure AI Vision** (pre-built) | 600+ pre-trained categories |
| "Classify 5 product types, 200 images" | **Custom Vision Portal** | Fastest path, no code |
| "Detect manufacturing defects" | **Custom Vision** or **AutoML for Vision** | Need custom domain-specific training |
| "Label 50K medical images with 10 labelers" | **Azure ML Data Labeling + AutoML** | Enterprise labeling at scale |
| "Semantic segmentation for satellite imagery" | **SDK v2 (Pro-Code)** | Advanced task type needs code |
| "Run model on mobile phone offline" | **Custom Vision** (export to ONNX/CoreML) | Built-in edge export |
| "CI/CD for model retraining pipeline" | **SDK v2 (Pro-Code)** | Full MLOps integration |

### The Golden Rule

> **Don't label or train anything until you've checked whether Azure AI Vision's pre-built models already solve your problem.** If they don't, start with Custom Vision Portal for a quick PoC. Graduate to Azure ML Data Labeling + AutoML for Vision when you need enterprise-grade labeling workflows, larger datasets, or more model control.

---

## Important: Custom Vision Migration Notice

Microsoft has announced **planned retirement of Azure Custom Vision by September 25, 2028**. While fully supported until then, Microsoft recommends these migration paths:

| Current Approach | Recommended Migration Target |
|-----------------|------------------------------|
| Custom Vision Classification | **Azure ML AutoML for Vision** (image classification) |
| Custom Vision Object Detection | **Azure ML AutoML for Vision** (object detection) |
| Custom Vision Edge Export | Export AutoML model to ONNX |
| Simple image categorization | **Azure Content Understanding** (preview) or **Foundry Models** |

For new long-term projects, consider starting directly with Azure ML AutoML for Vision.

---

## Resources & References

- [Custom Vision Portal — Build a Classifier (Quickstart)](https://learn.microsoft.com/azure/ai-services/custom-vision-service/getting-started-build-a-classifier)
- [Custom Vision — What Is It?](https://learn.microsoft.com/azure/ai-services/custom-vision-service/overview)
- [Set up an Image Labeling Project in Azure ML](https://learn.microsoft.com/azure/machine-learning/how-to-create-image-labeling-projects?view=azureml-api-2)
- [Labeling Images — The Annotator's Guide](https://learn.microsoft.com/azure/machine-learning/how-to-label-data?view=azureml-api-2)
- [Azure AI Vision Studio](https://portal.vision.cognitive.azure.com/)
- [Azure AI Vision Image Analysis 4.0 Quickstart](https://learn.microsoft.com/azure/ai-services/computer-vision/quickstarts-sdk/image-analysis-client-library-40)
- [Custom Vision Migration Guide](https://aka.ms/custom-vision-migration)
- [AutoML for Computer Vision](https://learn.microsoft.com/azure/machine-learning/how-to-auto-train-image-models?view=azureml-api-2)
- [Azure Content Understanding (Preview)](https://learn.microsoft.com/azure/ai-services/content-understanding/overview)

---

## Agenda Slide (Low-Code Session Variant)

```
Session 2 (Low-Code): Image Labelling & Vision — No Code Required
──────────────────────────────────────────────────────────────────
1. 📌 Why Labelling Matters & Task Types              (10 min)
2. ☁️  Low-Code Options on Azure (3 tiers)              (5 min)
3. 💻 Live Demo: Custom Vision Portal (end-to-end)     (20 min)
   - Create project
   - Upload & tag images
   - Train (one click)
   - Quick test + evaluate
   - Export for edge
4. 💻 Demo: Azure ML Data Labeling + ML-Assist          (15 min)
5. 💻 Quick Demo: Azure AI Vision Studio (pre-built)     (5 min)
6. 📊 When to Use Low-Code vs. Pro-Code                  (5 min)
──────────────────────────────────────────────────────────────────
```
