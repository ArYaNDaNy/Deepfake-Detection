# VisionGuard 🛡️

A multimodal deepfake detection system that analyzes **images**, **videos**, and **audio** to determine whether media is real or synthetically generated. Built with a dual Vision Transformer ensemble for image/video analysis and a calibrated SVM classifier for audio detection.

---

## Demo

| Modality | Test Dataset | Accuracy |
|---|---|---|
| Image | 140k Real & Fake Faces (StyleGAN) | 98.8% |
| Audio | FoR-norm (Fake-or-Real Dataset) | 76.2% |
| Video | DFDC Face Crops | 68.0% |

---

## Architecture

```
Deepfake-Detection/
├── frontend/          # React + Vite + Tailwind UI
└── backend_clean/     # FastAPI inference server
    ├── ml_models/
    │   ├── vit_model.py        # Dual ViT ensemble (image + video frames)
    │   ├── audio_expert.py     # SVM classifier on 77 acoustic features
    │   ├── heuristics.py       # OpenCV-based artifact detection
    │   └── fusion_model.py     # Score fusion logic
    ├── utils/
    │   └── video_utils.py      # MediaPipe face extraction
    ├── models/                 # Downloaded model weights (not in repo)
    │   ├── vit_finetuned/      # dima806/deepfake_vs_real_image_detection
    │   ├── vit_faceswap/       # Wvolf/ViT_Deepfake_Detection
    │   └── audio_logistic_model.pkl
    ├── scripts/                # This file is not pushed due to large dataset being present
    │   └── train_audio.py      # Audio SVM training script
    ├── test/                   # This file is not pushed due to large dataset being present
    │   ├── test_images.py
    │   ├── test_videos.py
    │   └── test_audio.py
    └── api.py                  # FastAPI routes
```

### Detection Pipeline

**Image**
```
Input Image → MediaPipe Face Crop → Dual ViT (dima806 + Wvolf) 
→ Temperature Calibration (T=1.4) → OpenCV Heuristics 
(texture, boundary, noise, frequency) → Conflict-Aware Fusion → Score
```

**Video**
```
Input Video → MediaPipe Face Extraction (40 frames) → Dual ViT Batch 
→ Median Aggregation → Pixel Glitch Score → Heuristics → Fusion → Score
FFmpeg Audio Track → Audio Pipeline → 70/30 Weighted Final Score
```

**Audio**
```
Input Audio → Librosa Feature Extraction (77 features: MFCC + Delta + 
Delta² + Spectral Contrast + Pitch + Shimmer + Jitter + Mel Stats) 
→ StandardScaler → SVM (RBF, C=50) → Fake Probability
```

### Models Used

| Model | Source | Used For |
|---|---|---|
| `dima806/deepfake_vs_real_image_detection` | HuggingFace | Primary ViT — AI-generated faces |
| `Wvolf/ViT_Deepfake_Detection` | HuggingFace | Secondary ViT — face swaps |
| SVM RBF (C=50) | Trained locally | Audio deepfake detection |
| MediaPipe Face Detection | Google | Face crop extraction |

---

## Getting Started

### Prerequisites

- Python 3.10
- Node.js or [Bun](https://bun.sh)
- ffmpeg (`brew install ffmpeg` on Mac)

---

### Backend Setup

```bash
cd backend_clean

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Download Models

```bash
# Download primary ViT model
python download_vit.py            # else python3 download_vit.py

# Download secondary ViT model (face swap specialist)
python download_vit_faceswap.py   # else python3 download_vit_faceswap.py
```

#### Train Audio Model(This part is not pushed as dataset included 10k files)

The audio SVM must be trained before running the server.

1. Download the [FoR-norm dataset](https://bil.eecs.yorku.ca/datasets/) and place it at:
```
backend_clean/datasets/audio_real/    ← for-norm/training/real/
backend_clean/datasets/audio_fake/    ← for-norm/training/fake/
```

2. Train:
```bash
cd scripts
python3 train_audio.py
```

#### Start the Server

```bash
cd backend_clean
uvicorn main:app --reload --port 8000
```

API will be available at `http://localhost:8000`

---

### Frontend Setup

```bash
cd frontend

# Install dependencies
bun install       # or: npm install

# Start dev server
bun run dev       # or: npm run dev
```

Frontend will be available at `http://localhost:5173`

---

## API Reference

### `POST /analyze-image`
Detects deepfakes in a still image.

**Request:** `multipart/form-data` — `file` (.jpg, .png, .jpeg)

**Response:**
```json
{
  "status": "success",
  "modality": "image",
  "fake_probability": 0.74,
  "breakdown": {
    "vit_confidence": 0.81,
    "face_detected": true,
    "heuristics": {
      "texture": 0.92,
      "boundary": 0.43,
      "noise": 0.12,
      "frequency": 0.21
    }
  }
}
```

---

### `POST /analyze-video`
Detects deepfakes in a video file.

**Request:** `multipart/form-data` — `file` (.mp4, .mov, .avi)

**Response:**
```json
{
  "status": "success",
  "modality": "video",
  "fake_probability": 0.68,
  "breakdown": {
    "video_score": 0.71,
    "audio_score": 0.58,
    "vit_median": 0.63,
    "pixel_glitch": 0.82,
    "heuristics_average": {}
  }
}
```

---

### `POST /analyze-audio`
Detects synthetic/cloned audio.

**Request:** `multipart/form-data` — `file` (.wav, .mp3, .flac, .ogg, .m4a)

**Response:**
```json
{
  "status": "success",
  "modality": "audio",
  "fake_probability": 0.92,
  "breakdown": {
    "audio_score": 0.92,
    "model": "svm_rbf_77features"
  }
}
```

---

### Score Interpretation

| Score | Label | Meaning |
|---|---|---|
| 0.00 – 0.35 | ✅ Real | Likely authentic |
| 0.35 – 0.60 | ⚠️ Suspicious | Inconclusive — manual review recommended |
| 0.60 – 1.00 | 🚨 Fake | Likely synthetic or manipulated |

---

## Testing(This is also not included in repo as it has a lot of files 1GB+)

Place test samples in the following structure:
```
samples/
├── image/
│   ├── real/
│   └── fake/
├── video/
│   ├── real/
│   └── fake/
└── audio/
    ├── real/
    └── fake/
```

Run tests (server must be running):
```bash
cd test
python3 test_images.py --samples_dir ../samples/image
python3 test_videos.py --samples_dir ../samples/video
python3 test_audio.py  --samples_dir ../samples/audio
```

---

## Known Limitations

**Image**
- Performs well on GAN and diffusion-generated faces
- Reduced accuracy on face swaps extracted from compressed video
- Top-tier generative models (Gemini, Midjourney v6) can bypass detection

**Audio**
- Trained on FoR-norm — may underperform on newer voice cloning systems not covered by the dataset
- Sensitive to audio quality and recording conditions

**Video**
- Treats frames independently — no temporal sequence model
- Struggles on motion-stabilized datasets (e.g. DFDC) where frame-to-frame differences are minimal
- Processing time: ~15–20 seconds per video on Apple M3

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Tailwind CSS |
| Backend | FastAPI, Uvicorn |
| Vision Models | PyTorch, HuggingFace Transformers |
| Audio | Librosa, Scikit-Learn |
| Face Detection | MediaPipe |
| Video Processing | OpenCV, FFmpeg |

---

## Team

Built as a project for nexevolve hackathon somaiya college vidyavihar where we held 5th place
