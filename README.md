# VisionGuard 🛡️

A multimodal deepfake detection system that analyzes **images**,
**videos**, and **audio** to determine whether media is real or
synthetically generated. Built with a dual Vision Transformer ensemble
for image/video analysis and a calibrated SVM classifier for audio
detection.

------------------------------------------------------------------------

## Demo

  Modality   Test Dataset                           Accuracy
  ---------- ----------------------------------- -----------
  Image      140k Real & Fake Faces (StyleGAN)     **98.8%**
  Audio      FoR-norm (Fake-or-Real Dataset)       **76.2%**
  Video      DFDC Face Crops                       **68.0%**

------------------------------------------------------------------------

# Architecture

``` text
Deepfake-Detection/
├── frontend/          # React + Vite + Tailwind UI
└── backend_clean/     # FastAPI inference server
    ├── ml_models/
    │   ├── vit_model.py
    │   ├── audio_expert.py
    │   ├── heuristics.py
    │   └── fusion_model.py
    ├── utils/
    │   └── video_utils.py
    ├── models/
    │   ├── vit_finetuned/
    │   ├── vit_faceswap/
    │   └── audio_logistic_model.pkl
    ├── scripts/
    ├── test/
    └── main.py
```

## Detection Pipeline

### Image

``` text
Input Image
    ↓
MediaPipe Face Detection
    ↓
Dual ViT (dima806 + Wvolf)
    ↓
Temperature Calibration
    ↓
OpenCV Heuristics
    ↓
Conflict-Aware Fusion
    ↓
Final Fake Probability
```

### Video

``` text
Input Video
    ↓
Extract Faces (MediaPipe)
    ↓
Dual ViT Frame Analysis
    ↓
Median Aggregation
    ↓
Pixel Glitch + Heuristics
    ↓
FFmpeg Audio Extraction
    ↓
Audio Analysis
    ↓
Weighted Fusion
```

### Audio

``` text
Input Audio
    ↓
Librosa Feature Extraction (77 Features)
    ↓
StandardScaler
    ↓
SVM (RBF)
    ↓
Fake Probability
```

------------------------------------------------------------------------

# Models Used

  Model                                      Purpose
  ------------------------------------------ ------------------------
  dima806/deepfake_vs_real_image_detection   Primary image detector
  Wvolf/ViT_Deepfake_Detection               Face-swap detector
  SVM (RBF)                                  Audio detector
  MediaPipe Face Detection                   Face extraction

------------------------------------------------------------------------

# Getting Started

## Prerequisites

-   Python **3.10.x (Recommended)**
-   Python 3.12 also works (may show harmless deprecation warnings)
-   Node.js (or Bun)
-   FFmpeg installed and available in PATH
-   Git

------------------------------------------------------------------------

# Backend Setup

``` bash
cd backend_clean

python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Verify installation:

``` bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import mediapipe as mp; print(mp.__version__)"
```

## Download Models

``` bash
python download_vit.py
python download_vit_faceswap.py
```

## Train Audio Model

Download the FoR-norm dataset and place it as:

``` text
backend_clean/datasets/audio_real/
backend_clean/datasets/audio_fake/
```

Train:

``` bash
cd scripts
python train_audio.py
```

## Start Backend

``` bash
python -m uvicorn main:app --reload --port 8000
```

Backend:

    http://localhost:8000

------------------------------------------------------------------------

# Frontend Setup

``` bash
cd frontend

bun install
# or
npm install

bun run dev
# or
npm run dev
```

Frontend:

    http://localhost:5173

------------------------------------------------------------------------

# API

## POST /analyze-image

Accepts:

-   jpg
-   jpeg
-   png

Returns fake probability and heuristic breakdown.

## POST /analyze-video

Accepts:

-   mp4
-   mov
-   avi

Returns image + audio fusion score.

## POST /analyze-audio

Accepts:

-   wav
-   mp3
-   flac
-   ogg
-   m4a

Returns SVM fake probability.

------------------------------------------------------------------------

# Score Interpretation

  Score        Meaning
  ------------ ------------
  0.00--0.35   Real
  0.35--0.60   Suspicious
  0.60--1.00   Fake

------------------------------------------------------------------------

# Testing

Place samples:

``` text
samples/
    image/
    video/
    audio/
```

Run:

``` bash
python test_images.py
python test_videos.py
python test_audio.py
```

------------------------------------------------------------------------

# Common Setup Issues & Fixes

## 1. torch.\_C Error

    ModuleNotFoundError: No module named 'torch._C'

Fix:

``` bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```

------------------------------------------------------------------------

## 2. MediaPipe `solutions` Missing

    AttributeError: module 'mediapipe' has no attribute 'solutions'

Fix:

``` bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.21
```

------------------------------------------------------------------------

## 3. Missing ViT Models

    FileNotFoundError

Run:

``` bash
python download_vit.py
python download_vit_faceswap.py
```

------------------------------------------------------------------------

## 4. Missing Audio Model

    audio_logistic_model.pkl

Train locally:

``` bash
cd scripts
python train_audio.py
```

------------------------------------------------------------------------

## 5. pkg_resources Missing

    ModuleNotFoundError: No module named 'pkg_resources'

Fix:

``` bash
pip install "setuptools<81"
```

Verify:

``` bash
python -c "import pkg_resources"
```

------------------------------------------------------------------------

## 6. pkg_resources Deprecated Warning

Generated by **librosa** internally.

Safe to ignore.

------------------------------------------------------------------------

## 7. MediaPipe / protobuf / SWIG Deprecation Warnings

Examples:

-   SwigPyObject
-   MessageMapContainer
-   ScalarMapContainer

Safe to ignore on Python 3.12.

------------------------------------------------------------------------

## 8. Audio Deprecation Warnings

Warnings mentioning:

-   aifc
-   audioop
-   sunau

come from **audioread**.

Safe to ignore.

------------------------------------------------------------------------

## 9. FFmpeg Not Found

Install FFmpeg and add it to PATH.

Verify:

``` bash
ffmpeg -version
```

------------------------------------------------------------------------

## 10. Scikit-Learn Version Warning

    InconsistentVersionWarning

Install matching version:

``` bash
pip install scikit-learn==1.4.1.post1
```

or retrain the audio model.

------------------------------------------------------------------------

## 11. Verify Virtual Environment

``` bash
where python
python -c "import sys; print(sys.executable)"
```

Always ensure Python points to:

    backend_clean/venv/

------------------------------------------------------------------------

# Known Limitations

## Image

-   Performs best on GAN and diffusion generated faces.
-   Reduced accuracy on compressed face-swaps.
-   Latest commercial generators may bypass detection.

## Audio

-   Trained on FoR-norm only.
-   Sensitive to recording quality.

## Video

-   No temporal transformer.
-   Frame-based inference.
-   Processing takes \~15--20 seconds depending on hardware.

------------------------------------------------------------------------

# Tech Stack

  Layer            Technology
  ---------------- -----------------------
  Frontend         React, Vite, Tailwind
  Backend          FastAPI
  Vision           PyTorch, Hugging Face
  Audio            Librosa, Scikit-Learn
  Face Detection   MediaPipe
  Video            OpenCV, FFmpeg

------------------------------------------------------------------------

# Development Notes

-   Large datasets are intentionally excluded from Git.
-   Model weights are downloaded separately.
-   Audio model must be trained locally.
-   Always use a virtual environment.
-   Start the server using:

``` bash
python -m uvicorn main:app --reload --port 8000
```

Avoid running with `python -X dev` unless debugging, as it enables
additional deprecation warnings.

------------------------------------------------------------------------

# Team

Built during the **NexEvolve Hackathon** at **K. J. Somaiya College of
Engineering**, where the project secured **5th place**.
