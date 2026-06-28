# merged_main.py
import io
import os
import base64
import tempfile
import numpy as np
from PIL import Image
import imageio.v3 as iio
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# local modules
from vit_model import predict_vit, predict_vit_batch
from video_utils import extract_faces_from_video
from heuristics import texture_score, color_anomaly, boundary_artifact, eye_glint_score

# audio expert
try:
    from models.audio import audio_expert
except Exception:
    audio_expert = None

app = FastAPI(title="Deepfake Detection Backend (Merged)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def frame_to_base64(frame_rgb, max_side=640, quality=80):
    """Convert RGB numpy frame -> base64 data URL (resized if large)."""
    pil = Image.fromarray(frame_rgb.astype("uint8"))
    w, h = pil.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        pil = pil.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def clamp01(x):
    return float(min(max(x, 0.0), 1.0))

def compute_heuristics(face_rgb):
    """Return heuristics dict (for logging)."""
    try:
        tex = texture_score(face_rgb)
        col = color_anomaly(face_rgb)
        bound = boundary_artifact(face_rgb)
        eye = eye_glint_score(face_rgb)
        return {
            "texture": clamp01(tex),
            "color": clamp01(col),
            "boundary": clamp01(bound),
            "eye_glint": clamp01(eye)
        }
    except Exception:
        return {}

# ---------------- Endpoints ----------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), sample_rate: int = 1, max_faces: int = 120):
    """
    Merged analyze endpoint.
    """
    filename = (file.filename or "").lower()
    content = await file.read()

    # ============ IMAGE PROCESSING ============
    if filename.endswith((".jpg", ".jpeg", ".png")):
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        prob = predict_vit(pil_img)  # 0..1 fake probability (ViT)
        label = "FAKE" if prob >= 0.5 else "REAL"

        # ✅ CORRECTED LOGIC
        if label == "REAL" or label == "Authentic":
            authentic_prob = round((1 - prob) * 100, 2)  # Real %
            fake_prob = round(prob * 100, 2)             # Fake %
            probability = authentic_prob                 # Show Real %
        else:
            fake_prob = round(prob * 100, 2)             # Fake %
            authentic_prob = round((1 - prob) * 100, 2)  # Real %
            probability = fake_prob                      # Show Fake %

        response = {
            "label": label,
            "confidence": round(prob * 100, 2),
            "authentic_probability": authentic_prob,
            "fake_probability": fake_prob,
            "probability": probability,
            "suspicious_frames": [],
            "file_type": "image",
            "filename": file.filename
        }

        print("\n" + "="*50)
        print("IMAGE ANALYSIS RESULT:")
        print(json.dumps(response, indent=2))
        print("="*50 + "\n")

        return response

    # ============ VIDEO PROCESSING ============
    elif filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(content)
        tmp.close()
        
        try:
            faces = extract_faces_from_video(tmp.name, max_frames=400)
        except Exception as e:
            faces = []
            print("Warning: extract_faces_from_video failed:", e)

        # Fallback: sample video frames raw
        if not faces:
            frames = []
            try:
                reader = iio.imiter(tmp.name)
                for idx, fr in enumerate(reader):
                    if idx % max(1, sample_rate) == 0:
                        frames.append(fr)
                    if idx >= 120:
                        break
            except Exception:
                frames = []
            faces = frames

        if not faces:
            try:
                os.unlink(tmp.name)
            except:
                pass
            raise HTTPException(status_code=400, detail="No valid frames/faces extracted from video.")

        # Limit and sample faces
        max_eval = min(len(faces), max(1, max_faces))
        sampled = faces[::max(1, max(1, len(faces)//max_eval))][:max_eval]

        pil_list = []
        meta = []
        for idx, face_rgb in enumerate(sampled):
            try:
                pil = Image.fromarray(face_rgb).convert("RGB")
            except Exception:
                continue
            pil_resized = pil.resize((224, 224))
            pil_list.append(pil_resized)
            meta.append((idx, face_rgb))

        if not pil_list:
            try:
                os.unlink(tmp.name)
            except:
                pass
            raise HTTPException(status_code=400, detail="No processable frames for ViT.")

        # Batch prediction
        vit_probs = predict_vit_batch(pil_list)

        frame_details = []
        final_vit_scores = []
        suspicious_frames = []

        for (frame_idx, face_rgb), vit_p in zip(meta, vit_probs):
            vit_score = float(vit_p)
            final_vit_scores.append(vit_score * 100)

            heur = compute_heuristics(face_rgb)

            frame_details.append({
                "frame_index": int(frame_idx),
                "vit_prob": float(round(vit_score, 4)),
                "heuristics": heur
            })

            if vit_score >= 0.7:
                suspicious_frames.append(frame_to_base64(face_rgb, max_side=320, quality=70))

        # Calculate average confidence (fake probability)
        avg_conf = float(np.mean(final_vit_scores))
        label = "FAKE" if avg_conf >= 50 else "REAL"
        
        # ✅ CORRECTED - Same as image logic
        if label == "REAL" or label == "Authentic":
            authentic_prob = round(100 - avg_conf, 2)
            fake_prob = round(avg_conf, 2)
            probability = authentic_prob
        else:
            fake_prob = round(avg_conf, 2)
            authentic_prob = round(100 - avg_conf, 2)
            probability = fake_prob

        response = {
            "label": label,
            "confidence": round(avg_conf, 2),
            "authentic_probability": authentic_prob,
            "fake_probability": fake_prob,
            "probability": probability,
            "suspicious_frames": suspicious_frames,
            "file_type": "video",
            "filename": file.filename,
            "total_frames_analyzed": len(final_vit_scores),
            "suspicious_frame_count": len(suspicious_frames),
            "frame_details": frame_details
        }

        response_for_print = response.copy()
        response_for_print["suspicious_frames"] = f"[{len(suspicious_frames)} base64 images]"
        print("\n" + "="*50)
        print("VIDEO ANALYSIS RESULT:")
        print(json.dumps(response_for_print, indent=2))
        print("="*50 + "\n")

        try:
            os.unlink(tmp.name)
        except:
            pass

        return response

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    content = await file.read()

    if not filename.endswith((".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    if audio_expert is None:
        raise HTTPException(status_code=500, detail="Audio expert missing on server.")

    try:
        result = audio_expert.predict(content)
    except Exception as e:
        print("Audio prediction failed:", e)
        raise HTTPException(status_code=500, detail="Audio analysis failed")

    return {
        "file_type": "audio",
        "label": result.get("verdict"),
        "confidence": round(result.get("fake_probability", 0) * 100, 2),
        "features": result.get("features_summary", {}),
        "filename": file.filename
    }
