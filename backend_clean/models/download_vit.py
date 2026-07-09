"""
Run this ONCE to download the ViT deepfake detection model.
It saves to models/vit_finetuned/ so your existing load code works as-is.

Usage:
    python download_vit.py
"""

import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Matches the path in your existing code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")

# Best publicly available ViT fine-tuned for deepfake detection on HuggingFace
# Trained on FaceForensics++ — works for both image and video frame detection
MODEL_ID = "dima806/deepfake_vs_real_image_detection"

print(f"Downloading model: {MODEL_ID}")
print(f"Saving to: {MODEL_PATH}")
print("This may take a few minutes (~330MB)...\n")

os.makedirs(MODEL_PATH, exist_ok=True)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

processor.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)

print("\nDone. Model saved locally.")
print(f"Your existing vit_model.py will now load from: {MODEL_PATH}")

# Quick sanity check
print("\nRunning sanity check...")
from PIL import Image
import torch
import numpy as np

dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=dummy, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

# Print the label mapping so you know which index is FAKE
print(f"Model labels: {model.config.id2label}")
print(f"Dummy image probs: {probs.tolist()}")
print("\nNote: Check the label mapping above.")
print("In your predict_vit(), probs[1] assumes index 1 = FAKE.")
print("Verify this matches id2label before using in production.")