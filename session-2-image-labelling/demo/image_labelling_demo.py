"""
=============================================================================
  DEMO: Image Labelling & Classification with Azure
  Session 2 — Product Image Classification
=============================================================================

  This script demonstrates:
    Part A: Training an image classifier with AutoML for Vision
    Part B: Using Azure AI Vision (pre-built) for image analysis
    Part C: Using Custom Vision SDK

  Prerequisites:
    pip install azure-ai-ml azure-identity
    pip install azure-ai-vision-imageanalysis       # For Part B
    pip install azure-cognitiveservices-vision-customvision msrest  # For Part C

  Before running:
    1. az login
    2. Replace <placeholders> with your Azure resource details
    3. Complete the Data Labeling steps in Azure ML Studio first (see README)
=============================================================================
"""

# ============================================================================
# PART A: AutoML for Vision — Image Classification
# ============================================================================

def demo_automl_vision():
    """Train an image classifier using AutoML for Vision (SDK v2)."""

    # ── Configuration ──
    SUBSCRIPTION_ID = "<YOUR_SUBSCRIPTION_ID>"
    RESOURCE_GROUP = "<YOUR_RESOURCE_GROUP>"
    WORKSPACE_NAME = "<YOUR_WORKSPACE_NAME>"
    COMPUTE_NAME = "gpu-cluster"
    EXPERIMENT_NAME = "product-image-classification-demo"

    # ── Step 1: Connect to Workspace ──
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

    # ── Step 2: Create GPU Compute ──
    from azure.ai.ml.entities import AmlCompute

    print(f"\n📦 Setting up GPU compute: {COMPUTE_NAME}...")
    try:
        ml_client.compute.get(COMPUTE_NAME)
        print(f"✅ Compute '{COMPUTE_NAME}' already exists.")
    except Exception:
        gpu_compute = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="STANDARD_NC6",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(gpu_compute).result()
        print(f"✅ Compute '{COMPUTE_NAME}' created.")

    # ── Step 3: Reference Labeled Data ──
    from azure.ai.ml import Input
    from azure.ai.ml.constants import AssetTypes

    print("\n📊 Configuring training data...")

    # Option A: From Data Labeling export (registered dataset)
    # training_data = Input(
    #     type=AssetTypes.MLTABLE,
    #     path="azureml:product-classification-labeled:1",
    # )

    # Option B: From local MLTable folder
    training_data = Input(
        type=AssetTypes.MLTABLE,
        path="./training-mltable-folder",
    )

    print("✅ Training data configured.")

    # ── Step 4: Configure AutoML for Vision ──
    from azure.ai.ml import automl

    print("\n⚙️ Configuring AutoML Image Classification job...")

    image_classification_job = automl.image_classification(
        experiment_name=EXPERIMENT_NAME,
        compute=COMPUTE_NAME,
        training_data=training_data,
        target_column_name="label",
        primary_metric="accuracy",
    )

    image_classification_job.set_limits(
        timeout_minutes=60,
        max_trials=10,
        max_concurrent_trials=2,
    )

    print("✅ AutoML job configured.")

    # ── Step 5: Submit ──
    print("\n🚀 Submitting AutoML Image Classification job...")
    returned_job = ml_client.jobs.create_or_update(image_classification_job)
    print(f"✅ Job created: {returned_job.name}")
    print(f"📊 Studio URL: {returned_job.services['Studio'].endpoint}")

    return returned_job


# ============================================================================
# PART B: Azure AI Vision 4.0 — Pre-Built Image Analysis
# ============================================================================

def demo_azure_ai_vision():
    """Analyze images using Azure AI Vision (no training required)."""

    import os
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential

    # ── Configuration ──
    endpoint = os.environ.get("VISION_ENDPOINT", "<YOUR_VISION_ENDPOINT>")
    key = os.environ.get("VISION_KEY", "<YOUR_VISION_KEY>")

    print("🔗 Connecting to Azure AI Vision...")
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    # ── Analyze an image ──
    sample_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"

    print(f"\n🖼️ Analyzing image: {sample_url}")
    result = client.analyze_from_url(
        image_url=sample_url,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.TAGS,
            VisualFeatures.OBJECTS,
            VisualFeatures.READ,
        ],
        gender_neutral_caption=True,
    )

    # ── Print results ──
    print("\n📋 Analysis Results:")
    print("─" * 50)

    if result.caption is not None:
        print(f"\n🏷️ Caption: '{result.caption.text}'")
        print(f"   Confidence: {result.caption.confidence:.4f}")

    if result.tags is not None:
        print(f"\n🏷️ Tags:")
        for tag in result.tags.list:
            print(f"   • {tag.name} ({tag.confidence:.2f})")

    if result.objects is not None:
        print(f"\n📦 Objects Detected:")
        for obj in result.objects.list:
            print(f"   • {obj.tags[0].name} at {obj.bounding_box}")

    if result.read is not None:
        print(f"\n📝 Text (OCR):")
        for block in result.read.blocks:
            for line in block.lines:
                print(f"   {line.text}")

    print("\n✅ Azure AI Vision analysis complete!")


# ============================================================================
# PART C: Azure Custom Vision — Full Workflow
# ============================================================================

def demo_custom_vision():
    """
    Create a Custom Vision project, upload images, train, and predict.
    
    Requires:
      pip install azure-cognitiveservices-vision-customvision msrest
    
    Environment variables:
      CUSTOM_VISION_ENDPOINT
      CUSTOM_VISION_TRAINING_KEY
      CUSTOM_VISION_PREDICTION_KEY
      CUSTOM_VISION_PREDICTION_RESOURCE_ID
    """

    import os
    import time
    import uuid
    from azure.cognitiveservices.vision.customvision.training import (
        CustomVisionTrainingClient,
    )
    from azure.cognitiveservices.vision.customvision.prediction import (
        CustomVisionPredictionClient,
    )
    from azure.cognitiveservices.vision.customvision.training.models import (
        ImageFileCreateBatch,
        ImageFileCreateEntry,
    )
    from msrest.authentication import ApiKeyCredentials

    # ── Configuration ──
    ENDPOINT = os.environ.get(
        "CUSTOM_VISION_ENDPOINT", "<YOUR_CUSTOM_VISION_ENDPOINT>"
    )
    training_key = os.environ.get(
        "CUSTOM_VISION_TRAINING_KEY", "<YOUR_TRAINING_KEY>"
    )
    prediction_key = os.environ.get(
        "CUSTOM_VISION_PREDICTION_KEY", "<YOUR_PREDICTION_KEY>"
    )
    prediction_resource_id = os.environ.get(
        "CUSTOM_VISION_PREDICTION_RESOURCE_ID", "<YOUR_PREDICTION_RESOURCE_ID>"
    )
    publish_iteration_name = "classifyModel"

    # ── Step 1: Create training client ──
    print("🔗 Connecting to Custom Vision...")
    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

    # ── Step 2: Create project ──
    project_name = f"product-demo-{uuid.uuid4().hex[:8]}"
    print(f"\n📁 Creating project: {project_name}")
    project = trainer.create_project(project_name)

    # ── Step 3: Create tags ──
    print("\n🏷️ Creating tags...")
    tags = {}
    for tag_name in ["Electronics", "Clothing", "Food", "Furniture"]:
        tags[tag_name] = trainer.create_tag(project.id, tag_name)
        print(f"   ✅ Tag: {tag_name}")

    # ── Step 4: Upload images ──
    print("\n📸 Uploading images...")
    base_image_dir = "./sample_images"

    for category, tag in tags.items():
        category_dir = os.path.join(base_image_dir, category)
        if not os.path.exists(category_dir):
            print(f"   ⚠️ Directory not found: {category_dir} — skipping")
            continue

        image_list = []
        for img_file in os.listdir(category_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                with open(os.path.join(category_dir, img_file), "rb") as f:
                    image_list.append(
                        ImageFileCreateEntry(
                            name=img_file,
                            contents=f.read(),
                            tag_ids=[tag.id],
                        )
                    )

        if image_list:
            # Upload in batches of 64
            for i in range(0, len(image_list), 64):
                batch = image_list[i : i + 64]
                upload_result = trainer.create_images_from_files(
                    project.id, ImageFileCreateBatch(images=batch)
                )
                if not upload_result.is_batch_successful:
                    print(f"   ❌ Batch upload failed for {category}")
                else:
                    print(f"   ✅ Uploaded {len(batch)} images for {category}")

    # ── Step 5: Train ──
    print("\n🏋️ Training model...")
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print(f"   Status: {iteration.status}")
        time.sleep(10)
    print("   ✅ Training complete!")

    # ── Step 6: Publish ──
    print("\n📢 Publishing model...")
    trainer.publish_iteration(
        project.id,
        iteration.id,
        publish_iteration_name,
        prediction_resource_id,
    )
    print(f"   ✅ Published as: {publish_iteration_name}")

    # ── Step 7: Test prediction ──
    print("\n🔮 Testing prediction...")
    prediction_credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": prediction_key}
    )
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    test_image = os.path.join(base_image_dir, "test", "test_image.jpg")
    if os.path.exists(test_image):
        with open(test_image, "rb") as image_contents:
            results = predictor.classify_image(
                project.id, publish_iteration_name, image_contents.read()
            )
            print("\n📋 Predictions:")
            for prediction in results.predictions:
                print(
                    f"   {prediction.tag_name}: "
                    f"{prediction.probability * 100:.2f}%"
                )
    else:
        print(f"   ⚠️ Test image not found: {test_image}")

    print("\n✅ Custom Vision demo complete!")


# ============================================================================
# MAIN — Run selected demo parts
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Image Labelling & Computer Vision Demo")
    print("  Session 2")
    print("=" * 60)

    if len(sys.argv) > 1:
        part = sys.argv[1].lower()
    else:
        print("\nUsage: python image_labelling_demo.py [part]")
        print("  part A  — AutoML for Vision (image classification)")
        print("  part B  — Azure AI Vision (pre-built analysis)")
        print("  part C  — Custom Vision (full workflow)")
        print("  all     — Run all parts")
        sys.exit(0)

    if part in ("a", "automl", "all"):
        print("\n" + "=" * 60)
        print("  PART A: AutoML for Vision")
        print("=" * 60)
        demo_automl_vision()

    if part in ("b", "vision", "all"):
        print("\n" + "=" * 60)
        print("  PART B: Azure AI Vision 4.0")
        print("=" * 60)
        demo_azure_ai_vision()

    if part in ("c", "custom", "all"):
        print("\n" + "=" * 60)
        print("  PART C: Azure Custom Vision")
        print("=" * 60)
        demo_custom_vision()

    print("\n🎉 Demo session complete!")
