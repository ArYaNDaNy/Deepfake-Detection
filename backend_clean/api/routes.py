import os
import shutil
import subprocess
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import io

from utils.video_utils import extract_faces_from_video, extract_face_crop
from ml_models.vit_model import predict_vit, predict_vit_batch
from ml_models.audio_expert import predict_audio
from ml_models.heuristics import (
    texture_score, color_anomaly, boundary_artifact,
    eye_glint_score, noise_analysis, frequency_analysis,
    calculate_pixel_glitch_score, calculate_video_glitch_score
)
from ml_models.fusion_model import fuse_image_scores, fuse_video_scores

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/analyze")
async def auto_route_analysis(file: UploadFile = File(...)):
    """
    Polymorphic dynamic orchestration gateway. 
    Intercepts generic requests and seamlessly distributes them to the underlying modal experts.
    """
    filename_lower = file.filename.lower()
    
    # 1. Image Modality Routing
    if filename_lower.endswith(('.png', '.jpg', '.jpeg')):
        return await analyze_image(file)
        
    # 2. Audio Modality Routing
    elif filename_lower.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        return await analyze_audio(file)
        
    # 3. Video Modality Routing
    elif filename_lower.endswith(('.mp4', '.mov', '.avi')):
        return await analyze_video(file)
        
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported media format. Please upload an authorized Image, Audio, or Video asset."
        )


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Must be an image file.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_image)

        face_crop = extract_face_crop(img_array)
        heuristic_input = face_crop if face_crop is not None else img_array

        vit_prob = predict_vit(pil_image)

        heuristics_dict = {
            "texture":   texture_score(heuristic_input),
            "color":     color_anomaly(heuristic_input),
            "boundary":  boundary_artifact(heuristic_input),
            "eye_glint": eye_glint_score(heuristic_input),
            "noise":     noise_analysis(heuristic_input),
            "frequency": frequency_analysis(heuristic_input),
        }

        final_score = fuse_image_scores(vit_prob, heuristics_dict)

        # Matched directly to your frontend transformer mapping parameters!
        return {
            "filename": file.filename,
            "file_type": "image",
            "label": "FAKE" if final_score > 0.50 else "REAL",
            "confidence": round(final_score * 100, 2),
            "probability": final_score,
            "suspicious_frames": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="Must be an audio file.")

    audio_path     = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    converted_path = os.path.join(TEMP_DIR, f"temp_{file.filename}.wav")

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file.filename.lower().endswith('.wav'):
            subprocess.run([
                "ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1",
                converted_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            analysis_path = converted_path
        else:
            analysis_path = audio_path

        fake_probability = predict_audio(analysis_path)

        return {
            "filename": file.filename,
            "file_type": "audio",
            "label": "FAKE" if fake_probability > 0.50 else "REAL",
            "confidence": round(fake_probability * 100, 2),
            "probability": fake_probability,
            "suspicious_frames": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(converted_path):
            os.remove(converted_path)


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Must be a video file.")

    video_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    audio_path = os.path.join(TEMP_DIR, f"temp_{file.filename}.wav")

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        has_audio = False
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
                audio_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            has_audio = True
        except:
            pass

        face_arrays = extract_faces_from_video(video_path, 40)
        if not face_arrays:
            return {
                "filename": file.filename,
                "file_type": "video",
                "label": "REAL",
                "confidence": 0.0,
                "probability": 0.0,
                "suspicious_frames": []
            }

        pil_faces = [Image.fromarray(f) for f in face_arrays]
        vit_probs = predict_vit_batch(pil_faces)
        avg_vit_prob = float(np.median(vit_probs))

        pixel_glitch = calculate_pixel_glitch_score(face_arrays)
        vit_glitch = calculate_video_glitch_score(vit_probs)

        avg_heuristics = {
            "texture":   float(np.mean([texture_score(f) for f in face_arrays])),
            "color":     float(np.mean([color_anomaly(f) for f in face_arrays])),
            "boundary":  float(np.mean([boundary_artifact(f) for f in face_arrays])),
            "eye_glint": float(np.mean([eye_glint_score(f) for f in face_arrays])),
            "noise":     float(np.mean([noise_analysis(f) for f in face_arrays])),
            "frequency": float(np.mean([frequency_analysis(f) for f in face_arrays])),
        }

        video_final_score = fuse_video_scores(
            avg_vit_prob, avg_heuristics, pixel_glitch, vit_glitch
        )

        if has_audio:
            audio_score = predict_audio(audio_path)
            overall_score = (0.70 * video_final_score) + (0.30 * audio_score)
        else:
            overall_score = video_final_score

        # Identify indices of suspicious frames exceeding threshold boundaries
        suspicious_frames = [idx for idx, prob in enumerate(vit_probs) if prob > 0.65]

        return {
            "filename": file.filename,
            "file_type": "video",
            "label": "FAKE" if overall_score > 0.50 else "REAL",
            "confidence": round(overall_score * 100, 2),
            "probability": overall_score,
            "suspicious_frames": suspicious_frames
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        if has_audio and os.path.exists(audio_path):
            os.remove(audio_path)