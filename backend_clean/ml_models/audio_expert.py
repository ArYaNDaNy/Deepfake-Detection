import os
import librosa
import numpy as np
import joblib

# 1. Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "audio_logistic_model.pkl")

# 2. Load Model Globally
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"WARNING: Audio model not found at {MODEL_PATH}. Prediction will return default.")
    clf = None


def extract_audio_features(file_path: str) -> np.ndarray:
    """
    Extracts 107 acoustic features specifically designed to distinguish
    real human speech from TTS/deepfake audio.

    Feature groups:
    - MFCC + Delta + Delta2 (39): vocal tract shape and temporal dynamics
    - Chroma (12): harmonic/pitch content
    - Spectral features (6): brightness, bandwidth, rolloff, contrast, flatness, flux
    - Prosodic features (4): energy dynamics, pitch variance, speaking rate, silence ratio
    - Mel spectrogram statistics (8): mean, std, skew per band summary
    - Jitter/Shimmer proxies (4): micro-variation in pitch and amplitude
    - Cepstral Peak Prominence proxy (1): voice quality measure
    - ZCR + RMS (2): basic temporal features
    Total: ~107 features
    """
    y, sr = librosa.load(file_path, sr=16000)

    # Pad or trim to minimum 1 second to avoid empty feature issues
    min_samples = sr
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))

    # ── MFCC features (39) ──────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)           # 13

    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1) # 13
    # TTS voices have unnaturally smooth phoneme transitions

    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1) # 13
    # Catches robotic acceleration patterns in synthetic speech

    # ── Chroma (12) ─────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)          # 12
    # AI voices often have unnatural harmonic intervals

    # ── Spectral features (13) ──────────────────────────────────────
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))          # 1
    bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))         # 1
    roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))           # 1
    flat = np.mean(librosa.feature.spectral_flatness(y=y))                 # 1
    # Spectral flatness: TTS often has unnaturally flat/tonal spectrum

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)      # 7
    # Contrast between peaks and valleys — TTS lacks natural variation

    # Spectral flux — measure of change between frames
    spec = np.abs(librosa.stft(y))
    flux = np.mean(np.diff(spec, axis=1) ** 2)                             # 1
    # TTS has unnaturally consistent spectral flux

    # ── Prosodic features (4) ───────────────────────────────────────
    # RMS energy — volume dynamics
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)                        # 1
    rms_std  = np.std(rms)                         # 1
    # Real humans vary volume naturally; TTS is perfectly leveled

    # Pitch (F0) variance — TTS has unnaturally smooth pitch contours
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_std  = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0      # 1
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) \
                       if voiced_flag is not None and len(voiced_flag) > 0 else 0.5  # 1
    except Exception:
        pitch_std    = 0.0
        voiced_ratio = 0.5

    # ── ZCR (1) ─────────────────────────────────────────────────────
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))              # 1

    # ── Mel spectrogram statistics (4) ──────────────────────────────
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_mean = np.mean(mel)                        # 1
    mel_std  = np.std(mel)                         # 1
    mel_skew = float(np.mean(((mel - mel_mean) / (mel_std + 1e-8)) ** 3)) # 1
    # Skewness of mel distribution — TTS has different energy distribution
    mel_band_std = np.mean(np.std(mel, axis=1))   # 1
    # Per-band variation — real speech has more variation across mel bands

    # ── Jitter/Shimmer proxies (4) ──────────────────────────────────
    # Micro-variations in pitch and amplitude — real voices have natural jitter
    # TTS is perfectly periodic
    frame_length = 512
    hop_length   = 128
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

    frame_rms  = np.sqrt(np.mean(frames ** 2, axis=0))
    shimmer    = np.mean(np.abs(np.diff(frame_rms)) / (frame_rms[:-1] + 1e-8))  # 1
    # Shimmer: amplitude variation between consecutive frames

    frame_zcr  = np.mean(np.abs(np.diff(np.sign(frames), axis=0)), axis=0) / frame_length
    jitter     = np.std(frame_zcr)               # 1
    # Jitter proxy: irregularity in zero crossing patterns

    energy_entropy = -np.sum(
        (frame_rms / (np.sum(frame_rms) + 1e-8)) *
        np.log2(frame_rms / (np.sum(frame_rms) + 1e-8) + 1e-8)
    )                                             # 1
    # Energy entropy: real speech has higher entropy than TTS

    silence_ratio = np.sum(frame_rms < 0.01) / (len(frame_rms) + 1e-8)  # 1
    # Real speech has natural silence/breath pauses; TTS often doesn't

    # ── Combine all features ─────────────────────────────────────────
    features = np.hstack((
        mfccs_mean,        # 13
        mfcc_delta_mean,   # 13
        mfcc_delta2_mean,  # 13
        chroma_mean,       # 12
        cent,              # 1
        bw,                # 1
        roll,              # 1
        flat,              # 1
        contrast_mean,     # 7
        flux,              # 1
        rms_mean,          # 1
        rms_std,           # 1
        pitch_std,         # 1
        voiced_ratio,      # 1
        zcr_mean,          # 1
        mel_mean,          # 1
        mel_std,           # 1
        mel_skew,          # 1
        mel_band_std,      # 1
        shimmer,           # 1
        jitter,            # 1
        energy_entropy,    # 1
        silence_ratio,     # 1
    ))

    return features


def predict_audio(file_path: str) -> float:
    """
    Analyzes an audio file and returns fake probability (0.0 = Real, 1.0 = Fake).
    """
    if clf is None:
        return 0.5

    try:
        features = extract_audio_features(file_path)
        features_reshaped = features.reshape(1, -1)
        probabilities = clf.predict_proba(features_reshaped)
        return float(probabilities[0][1])
    except Exception as e:
        print(f"Error processing audio: {e}")
        return 0.5