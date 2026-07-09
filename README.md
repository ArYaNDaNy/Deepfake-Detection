# VisionGuard 🛡️

A multimodal deepfake detection system that analyzes **images**, **videos**, and **audio** to determine whether media is real or synthetically generated. Built with a dual Vision Transformer ensemble for image/video analysis and a calibrated SVM classifier for audio detection.

---

## 📊 Demo & Accuracy

| Modality | Test Dataset | Accuracy |
| --- | --- | --- |
| **Image** | 140k Real & Fake Faces (StyleGAN) | **98.8%** |
| **Audio** | FoR-norm (Fake-or-Real Dataset) | **76.2%** |
| **Video** | DFDC Face Crops | **68.0%** |

---

## 🏗️ Architecture

```text
VisionGuard/
├── frontend/          # React + Vite + Tailwind UI
└── backend_clean/     # FastAPI inference server
    ├── ml_models/
    │   ├── vit_model.py        # Dual ViT ensemble (image + video frames)
    │   ├── audio_expert.py     # SVM classifier on 77 acoustic features
    │   ├── heuristics.py       # OpenCV-based artifact detection (texture, noise, etc.)
    │   └── fusion_model.py     # Score fusion logic (conflict-aware)
    ├── utils/
    │   └── video_utils.py      # MediaPipe face extraction
    ├── models/                 # Downloaded model weights (not tracked in repo)
    │   ├── vit_finetuned/      # dima806/deepfake_vs_real_image_detection
    │   ├── vit_faceswap/       # Wvolf/ViT_Deepfake_Detection
    │   ├── download_vit.py
    │   ├── download_vit_faceswap.py
    │   └── audio_logistic_model.pkl
    ├── scripts/
    │   └── train_audio.py      # Audio SVM training script
    ├── test/
    │   ├── test_images.py
    │   ├── test_videos.py
    │   └── test_audio.py
    └── main.py                 # FastAPI routes and server entry point

```

---

## ⚙️ Detection Pipeline

### Image Pipeline

```text
Input Image
    ↓
MediaPipe Face Detection (Cropping)
    ↓
Dual ViT Ensemble (dima806 + Wvolf)
    ↓
Temperature Calibration (T=1.4)
    ↓
OpenCV Heuristics (Texture, Boundary, Noise, Frequency)
    ↓
Conflict-Aware Fusion
    ↓
Final Fake Probability

```

### Video Pipeline

```text
Input Video
    ↓
Extract Faces via MediaPipe (40 frames)
    ↓
Dual ViT Batch Frame Analysis
    ↓
Median Aggregation & Pixel Glitch Score
    ↓
Heuristics Analysis
    ↓
FFmpeg Audio Track Extraction
    ↓
Audio Pipeline (see below)
    ↓
70/30 Weighted Image + Audio Final Score

```

### Audio Pipeline

```text
Input Audio
    ↓
Librosa Feature Extraction 
(77 Features: MFCC + Delta + Delta² + Spectral Contrast + Pitch + Shimmer + Jitter + Mel Stats)
    ↓
StandardScaler
    ↓
SVM (RBF, C=50)
    ↓
Fake Probability

```

---

## 🤖 Models Used

| Model | Source | Purpose |
| --- | --- | --- |
| **dima806/deepfake_vs_real_image_detection** | HuggingFace | Primary ViT — AI-generated faces |
| **Wvolf/ViT_Deepfake_Detection** | HuggingFace | Secondary ViT — Face swap specialist |
| **SVM (RBF, C=50)** | Trained locally | Audio deepfake detection |
| **MediaPipe Face Detection** | Google | Face crop extraction |

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.10.x** (Recommended. Python 3.12 also works but may show harmless deprecation warnings)
* **Node.js** or **Bun**
* **FFmpeg** installed and available in your system PATH (e.g., `brew install ffmpeg` on Mac)
* **Git**

---

### Backend Setup

```bash
cd backend_clean

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

Verify the core installations:

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import mediapipe as mp; print(mp.__version__)"

```

#### Download Models

Fetch the required Vision Transformer models from HuggingFace:

```bash
python models/download_vit.py
python models/download_vit_faceswap.py

```

#### Train Audio Model

The audio SVM must be trained before running the server. Download the **FoR-norm dataset** and place the folders as follows:

```text
backend_clean/datasets/audio_real/    ← (for-norm/training/real/)
backend_clean/datasets/audio_fake/    ← (for-norm/training/fake/)

```

Run the training script:

```bash
cd scripts
python train_audio.py

```

#### Start Backend Server

```bash
cd ../  # Ensure you are back in backend_clean/
python -m uvicorn main:app --reload --port 8000

```

*The backend API will be available at `http://localhost:8000*`

---

### Frontend Setup

```bash
cd frontend

# Install dependencies
bun install
# or
npm install

# Start development server
bun run dev
# or
npm run dev

```

*The frontend UI will be available at `http://localhost:5173*`

---

## 🔌 API Reference

### `POST /analyze-image`

Detects deepfakes in a still image.

* **Accepts:** `multipart/form-data` — file (`.jpg`, `.jpeg`, `.png`)
* **Response:**

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

### `POST /analyze-video`

Detects deepfakes in a video file utilizing both frame analysis and audio extraction.

* **Accepts:** `multipart/form-data` — file (`.mp4`, `.mov`, `.avi`)
* **Response:**

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

### `POST /analyze-audio`

Detects synthetic/cloned audio.

* **Accepts:** `multipart/form-data` — file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`)
* **Response:**

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

## 📈 Score Interpretation

| Score | Label | Meaning |
| --- | --- | --- |
| **0.00 – 0.35** | ✅ Real | Likely authentic |
| **0.35 – 0.60** | ⚠️ Suspicious | Inconclusive — manual review recommended |
| **0.60 – 1.00** | 🚨 Fake | Likely synthetic or manipulated |

---

## 🧪 Testing

Place test samples in the following directory structure:

```text
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

With the server running, execute the test scripts:

```bash
cd backend_clean/test
python test_images.py --samples_dir ../../samples/image
python test_videos.py --samples_dir ../../samples/video
python test_audio.py  --samples_dir ../../samples/audio

```

---

## 🛠️ Common Setup Issues & Fixes

**1. `torch._C` Error**
(`ModuleNotFoundError: No module named 'torch._C'`)

* **Fix:** Uninstall and cleanly reinstall PyTorch.
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio

```



**2. MediaPipe `solutions` Missing**
(`AttributeError: module 'mediapipe' has no attribute 'solutions'`)

* **Fix:** Downgrade/Pin the MediaPipe version.
```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.21

```



**3. Missing ViT Models** (`FileNotFoundError`)

* **Fix:** You skipped the model download step. Run `python download_vit.py` and `python download_vit_faceswap.py`.

**4. Missing Audio Model** (`audio_logistic_model.pkl not found`)

* **Fix:** The SVM needs to be trained locally. Run `python train_audio.py` in the `scripts` folder.

**5. `pkg_resources` Missing**

* **Fix:** Downgrade setuptools.
```bash
pip install "setuptools<81"

```



**6. `pkg_resources` Deprecation Warning**

* **Note:** Generated internally by **librosa**. Safe to ignore.

**7. MediaPipe / Protobuf / SWIG Deprecation Warnings**

* **Note:** Examples include `SwigPyObject`, `MessageMapContainer`, or `ScalarMapContainer`. These are safe to ignore on Python 3.12.

**8. Audio Deprecation Warnings**

* **Note:** Warnings mentioning `aifc`, `audioop`, or `sunau` come from **audioread**. Safe to ignore.

**9. FFmpeg Not Found**

* **Fix:** Install FFmpeg at the OS level and ensure it is added to your PATH. Verify with `ffmpeg -version`.

**10. Scikit-Learn Version Warning** (`InconsistentVersionWarning`)

* **Fix:** Either install the exact matching version (`pip install scikit-learn==1.4.1.post1`) or retrain the audio model locally.

**11. Wrong Python Environment**

* **Fix:** Always ensure you are using the virtual environment. Verify by running `where python` (Windows) or `which python` (Mac/Linux). It should point to `backend_clean/venv/`.

---

## ⚠️ Known Limitations

**Image Detection:**

* Performs exceptionally well on GAN and diffusion-generated faces.
* Reduced accuracy on face-swaps extracted from highly compressed video files.
* Top-tier commercial generative models (e.g., Midjourney v6, Gemini) may occasionally bypass detection.

**Audio Detection:**

* Currently trained exclusively on the FoR-norm dataset; may underperform on newer, out-of-distribution voice cloning systems.
* Sensitive to audio recording quality and heavy background noise.

**Video Detection:**

* Treats frames independently (uses spatial models rather than a temporal sequence model).
* Struggles on heavily motion-stabilized datasets (like DFDC) where frame-to-frame differences are minimal.
* Processing time is relatively heavy: ~15–20 seconds per video on an Apple M3 chip (longer on older hardware).

---

## 💻 Tech Stack

| Layer | Technology |
| --- | --- |
| **Frontend** | React, Vite, Tailwind CSS |
| **Backend** | FastAPI, Uvicorn |
| **Vision Models** | PyTorch, HuggingFace Transformers |
| **Audio Processing** | Librosa, Scikit-Learn |
| **Face Detection** | MediaPipe |
| **Video Processing** | OpenCV, FFmpeg |

---

## 🏆 Development & Team Notes

* Large datasets and model weights are intentionally excluded from the Git repository to save space and bandwidth. Always run the download scripts.
* Avoid running uvicorn with `python -X dev` unless you are actively debugging, as it enables excessive dependency deprecation warnings.
* **Origin:** This project was initially built during the **NexEvolve Hackathon** at K. J. Somaiya College of Engineering (securing 5th place) and was subsequently expanded as a final year BE (Artificial Intelligence) project at **Don Bosco Institute of Technology, Mumbai**.
