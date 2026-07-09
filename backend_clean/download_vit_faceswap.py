"""
Run this ONCE to download the second ViT model (face swap specialist).
Saves to models/vit_faceswap/ — your vit_model.py loads it from there automatically.

Usage:
    python download_vit_faceswap.py
"""

import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_faceswap")

# Trained on FaceForensics++ — specifically strong on face swaps
MODEL_ID = "Wvolf/ViT_Deepfake_Detection"

print(f"Downloading model: {MODEL_ID}")
print(f"Saving to: {MODEL_PATH}")
print("This may take a few minutes...\n")

os.makedirs(MODEL_PATH, exist_ok=True)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

processor.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)

print("\nDone. Model saved locally.")

# Sanity check
print("\nRunning sanity check...")
dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=dummy, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

print(f"Model labels: {model.config.id2label}")
print(f"Dummy image probs: {probs.tolist()}")
print("\nVerify which index maps to FAKE — should match what vit_model.py expects.")