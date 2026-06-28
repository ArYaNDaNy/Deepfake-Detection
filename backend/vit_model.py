# vit_model.py
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")

# Convert to forward slashes for HuggingFace compatibility
MODEL_PATH = MODEL_PATH.replace("\\", "/")

print(f"Loading model from: {MODEL_PATH}")

# Load processor + model locally
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded successfully on {device}")

# ---------- SINGLE IMAGE ----------
def predict_vit(pil_image: Image.Image) -> float:
    """Return fake probability (0–1)."""
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    return float(probs[1])  # index 1 = fake

# ---------- BATCH INFERENCE ----------
def predict_vit_batch(pil_images: list) -> list:
    """Batch prediction for speed."""
    if len(pil_images) == 0:
        return []

    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

    return probs.cpu().numpy().tolist()
