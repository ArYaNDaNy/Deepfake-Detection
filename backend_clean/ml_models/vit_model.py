import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model 1: dima806 — strong on AI-generated faces
MODEL_1_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")
# Model 2: Wvolf — strong on face swaps
MODEL_2_PATH = os.path.join(BASE_DIR, "models", "vit_faceswap")

# Hardware — MPS > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"🔥 Hardware acceleration target: {device}")

processor1 = None
model1     = None
fake_label_idx1 = 1

processor2 = None
model2     = None
fake_label_idx2 = 1


def _resolve_fake_index(model) -> int:
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        for idx, label in model.config.id2label.items():
            if any(w in str(label).lower() for w in ["fake", "synthesized", "spoof", "deepfake"]):
                return int(idx)
    return 1


# --- Load Model 1 ---
try:
    processor1 = AutoImageProcessor.from_pretrained(MODEL_1_PATH, local_files_only=True)
    model1 = AutoModelForImageClassification.from_pretrained(MODEL_1_PATH, local_files_only=True)
    model1.to(device)
    model1.eval()
    fake_label_idx1 = _resolve_fake_index(model1)
    print(f"✅ Model 1 (dima806) loaded. FAKE index: {fake_label_idx1}")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load Model 1. Error: {e}")

# --- Load Model 2 ---
try:
    processor2 = AutoImageProcessor.from_pretrained(MODEL_2_PATH, local_files_only=True)
    model2 = AutoModelForImageClassification.from_pretrained(MODEL_2_PATH, local_files_only=True)
    model2.to(device)
    model2.eval()
    fake_label_idx2 = _resolve_fake_index(model2)
    print(f"✅ Model 2 (Wvolf) loaded. FAKE index: {fake_label_idx2}")
except Exception as e:
    print(f"WARNING: Failed to load Model 2. Falling back to single ViT. Error: {e}")


def calibrate_probability(logits: torch.Tensor, temperature: float = 1.4) -> torch.Tensor:
    """Temperature scaling — softens overconfident predictions."""
    return F.softmax(logits / temperature, dim=-1)


def predict_vit(pil_image: Image.Image) -> float:
    """Single image — averaged across both models."""
    scores = []
    if model1 is not None and processor1 is not None:
        inputs = processor1(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = calibrate_probability(model1(**inputs).logits, temperature=1.4)[0]
        scores.append(float(probs[fake_label_idx1].cpu().item()))

    if model2 is not None and processor2 is not None:
        inputs = processor2(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = calibrate_probability(model2(**inputs).logits, temperature=1.4)[0]
        scores.append(float(probs[fake_label_idx2].cpu().item()))

    return float(sum(scores) / len(scores)) if scores else 0.5


def predict_vit_batch(pil_images: list) -> list:
    """Batch inference — averaged across both models per image."""
    if not pil_images:
        return []

    scores_all = [[] for _ in range(len(pil_images))]

    if model1 is not None and processor1 is not None:
        inputs = processor1(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = calibrate_probability(model1(**inputs).logits, temperature=1.4)
        for i, p in enumerate(probs[:, fake_label_idx1].cpu().numpy().tolist()):
            scores_all[i].append(p)

    if model2 is not None and processor2 is not None:
        inputs = processor2(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = calibrate_probability(model2(**inputs).logits, temperature=1.4)
        for i, p in enumerate(probs[:, fake_label_idx2].cpu().numpy().tolist()):
            scores_all[i].append(p)

    return [float(sum(s) / len(s)) if s else 0.5 for s in scores_all]