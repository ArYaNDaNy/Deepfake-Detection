# models/audio.py
import io
import numpy as np
import librosa
import soundfile as sf   # pip install soundfile if not present
from sklearn.linear_model import LogisticRegression

class AudioDeepfakeDetector:
    def __init__(self):
        # Will be learned during calibration
        self.clf = None
        self.feature_names = [
            "mfcc_variance",
            "mfcc_delta_variance",
            "mfcc_delta_delta_variance",
            "spectral_entropy",
            "mel_roughness",
            "pitch_jitter",
            "zcr_variance",
            "spectral_centroid_variance",
        ]

    def preprocess_bytes(self, audio_bytes: bytes, target_sr=16000, max_seconds=10.0):
        """Load audio bytes -> mono numpy array sampled at target_sr, trimmed, truncated to max_seconds."""
        try:
            vf = io.BytesIO(audio_bytes)

            # Prefer librosa.load (which uses soundfile under the hood if available).
            # But sometimes direct soundfile.read works better with BytesIO; try both.
            try:
                audio_array, sr = librosa.load(vf, sr=target_sr, mono=True, duration=max_seconds)
            except Exception:
                # fallback: use soundfile and then resample if needed
                vf.seek(0)
                audio_array, sr = sf.read(vf, dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)  # to mono
                if sr != target_sr:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

                # truncate to max_seconds
                max_len = int(target_sr * max_seconds)
                if len(audio_array) > max_len:
                    audio_array = audio_array[:max_len]

            # trim leading/trailing silence
            try:
                audio_array, _ = librosa.effects.trim(audio_array, top_db=40)
            except Exception:
                pass

            # require minimum length (0.5s)
            if len(audio_array) < int(0.5 * target_sr):
                return None
            return audio_array.astype("float32")

        except Exception as e:
            print(f"[audio.preprocess_bytes] Error loading audio: {e}")
            return None

    def extract_features(self, audio_array):
        """Extract feature dict from audio_array sampled at 16000 Hz."""
        sr = 16000

        # 1. MFCCs and dynamics
        mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)

        mfcc_var = float(np.std(mfcc))
        mfcc_delta_var = float(np.std(mfcc_delta))
        mfcc_delta_delta_var = float(np.std(mfcc_delta_delta))

        # 2. Spectral entropy
        S = np.abs(librosa.stft(audio_array, n_fft=1024, hop_length=512))
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
        entropy = -np.sum(S_norm * np.log2(S_norm + 1e-12), axis=0)
        spectral_entropy = float(np.mean(entropy))

        # 3. Mel roughness
        mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_fft=1024, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_roughness = float(np.mean(np.std(mel_spec_db, axis=1)))

        # 4. Pitch jitter (yin)
        try:
            f0 = librosa.yin(audio_array, fmin=50, fmax=400, sr=sr)
            f0_clean = f0[f0 > 0]
            if len(f0_clean) > 1:
                pitch_jitter = float(np.std(np.diff(f0_clean)) / (np.mean(f0_clean) + 1e-12))
            else:
                pitch_jitter = 0.0
        except Exception:
            pitch_jitter = 0.0

        # 5. ZCR variance
        zcr = librosa.feature.zero_crossing_rate(audio_array)
        zcr_var = float(np.var(zcr))

        # 6. Spectral centroid variance
        spec_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
        spec_centroid_var = float(np.var(spec_centroid))

        return {
            "mfcc_variance": mfcc_var,
            "mfcc_delta_variance": mfcc_delta_var,
            "mfcc_delta_delta_variance": mfcc_delta_delta_var,
            "spectral_entropy": spectral_entropy,
            "mel_roughness": mel_roughness,
            "pitch_jitter": pitch_jitter,
            "zcr_variance": zcr_var,
            "spectral_centroid_variance": spec_centroid_var,
        }

    def _features_to_vector(self, feat_dict):
        """Convert feature dict to ordered numpy vector."""
        return np.array([feat_dict[name] for name in self.feature_names], dtype=np.float32)

    def predict(self, audio_bytes: bytes):
        """Predict using trained logistic regression classifier."""
        # safe check
        if not hasattr(self, "clf") or self.clf is None:
            return {"error": "Model not calibrated yet", "verdict": "ERROR"}

        audio_array = self.preprocess_bytes(audio_bytes)
        if audio_array is None:
            return {"error": "Could not process audio or audio too short", "verdict": "ERROR"}

        feats = self.extract_features(audio_array)
        x = self._features_to_vector(feats).reshape(1, -1)

        try:
            prob_fake = float(self.clf.predict_proba(x)[0][1])  # class 1 = fake
        except Exception as e:
            print(f"[audio.predict] predict_proba failed: {e}")
            return {"error": "Model prediction failure", "verdict": "ERROR"}

        verdict = "Fake" if prob_fake >= 0.5 else "Real"
        return {
            "verdict": verdict,
            "fake_probability": prob_fake,
            "features_summary": {k: round(v, 3) for k, v in feats.items()}
        }

    def calibrate_threshold(self, samples: list):
        """
        Train a logistic regression classifier on your labeled samples.
        samples: [{"audio_bytes": ..., "label": "real"/"fake", "filename": "..."}]
        """
        X = []
        y = []

        print(f"\n🎯 Extracting features from {len(samples)} samples...\n")
        for i, sample in enumerate(samples):
            audio_array = self.preprocess_bytes(sample["audio_bytes"])
            if audio_array is None:
                print(f"  skipped sample {i} ({sample.get('filename', '??')}) - could not load")
                continue

            feats = self.extract_features(audio_array)
            vec = self._features_to_vector(feats)
            label = 1 if str(sample["label"]).lower() == "fake" else 0

            X.append(vec)
            y.append(label)

            print(f"  [{i+1:2d}] {sample.get('filename', '?'):35s} → label={sample['label']}")

        if len(X) < 4 or len(set(y)) < 2:
            print("❌ Not enough diverse data to train classifier")
            return

        X = np.stack(X, axis=0)
        y = np.array(y, dtype=np.int64)

        print("\n🧠 Training logistic regression classifier...")
        self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)

        train_probs = self.clf.predict_proba(X)[:, 1]
        preds = (train_probs >= 0.5).astype(int)
        acc = float((preds == y).mean())

        print(f"✅ Training complete. In-sample accuracy: {acc:.3f}")
        print(f"   Class distribution: real={int((y==0).sum())}, fake={int((y==1).sum())}\n")


# Single exported instance for convenience
audio_expert = AudioDeepfakeDetector()
