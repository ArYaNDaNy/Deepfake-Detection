"""
Video deepfake detection using yermandy/deepfake-detection.
CLIP ViT-L/14 fine-tuned on FaceForensics++.
Output: [Real, Fake] logits — apply softmax, take index 1 as fake prob.
"""

import os
import torch
import numpy as np
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "video_clip", "model.torchscript")
PROC_PATH  = os.path.join(BASE_DIR, "models", "video_clip", "clip_processor")
print(f"Looking for model at: {MODEL_PATH}")
print(f"Exists: {os.path.exists(MODEL_PATH)}")
# MPS > CPU for M3
device = torch.device("cpu")
print(f"🔥 Video model device: {device}")

video_model = None
processor   = None

try:
    video_model = torch.jit.load(MODEL_PATH, map_location=device)
    video_model.eval()
    print("✅ Video model (CLIP/FF++) loaded")
except Exception as e:
    print(f"WARNING: Video model not loaded: {e}")

try:
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(PROC_PATH)
    print("✅ CLIP processor loaded")
except Exception as e:
    print(f"WARNING: CLIP processor not loaded: {e}")


def predict_video_frames(face_arrays: list, batch_size: int = 8) -> float:
    """
    Runs CLIP-based deepfake detector on face crops.
    Output shape [B, 2] — softmax, take index 1 as fake probability.
    Returns weighted avg+max fake probability.
    """
    if video_model is None or processor is None or len(face_arrays) == 0:
        return 0.5

    try:
        pil_faces = [Image.fromarray(f) for f in face_arrays]
        all_scores = []

        for i in range(0, len(pil_faces), batch_size):
            batch = pil_faces[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                logits = video_model(pixel_values.float())          # [B, 2]
                probs  = torch.softmax(logits.float(), dim=-1)      # [B, 2]
                fake_probs = probs[:, 0].cpu().numpy()      # index - 0= Fake

            all_scores.extend(fake_probs.tolist())

        if not all_scores:
            return 0.5

        avg_score = float(np.mean(all_scores))
        max_score = float(np.max(all_scores))

        # Weighted: avg for consistency, max to catch single fake face
        final = (0.65 * avg_score) + (0.35 * max_score)
        return float(np.clip(final, 0.0, 1.0))

    except Exception as e:
        print(f"Error in predict_video_frames: {e}")
        return 0.5