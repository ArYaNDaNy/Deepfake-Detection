import cv2
import numpy as np


def texture_score(img_rgb: np.ndarray) -> float:
    """
    Checks for smoothed, low-texture surfaces (common GAN/diffusion artifact).
    Uses CLAHE to normalize lighting before running Laplacian.
    Returns 0.0 (Fake/Smooth) to 1.0 (Real/Textured).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_gray = clahe.apply(gray)
    var = cv2.Laplacian(norm_gray, cv2.CV_64F).var()
    return float(min(var / 100.0, 1.0))


def color_anomaly(img_rgb: np.ndarray) -> float:
    """
    YCrCb skin tone check.
    NOTE: Excluded from fusion — unreliable on fair skin.
    Returns 0.0 (Normal) to 1.0 (Anomalous).
    """
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (cr >= 120) & (cr <= 185) & (cb >= 70) & (cb <= 135)
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    skin_pixel_ratio = np.sum(skin_mask) / total_pixels
    anomaly_score = max(0.0, 1.0 - (skin_pixel_ratio / 0.25))
    return float(anomaly_score)


def boundary_artifact(img_rgb: np.ndarray) -> float:
    """
    Hunts for blending seams at face crop boundaries.
    Returns 0.0 (Clean) to 1.0 (Unnatural seam).
    """
    h, w, _ = img_rgb.shape
    if h < 20 or w < 20:
        return 0.0
    mask = np.zeros((h, w), dtype=np.float32)
    by = int(h * 0.15)
    bx = int(w * 0.15)
    mask[:by, :] = 1.0
    mask[-by:, :] = 1.0
    mask[:, :bx] = 1.0
    mask[:, -bx:] = 1.0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    boundary_edges = edges * mask
    edge_density = boundary_edges.mean() / 255.0
    return float(min(edge_density / 0.10, 1.0))


def eye_glint_score(img_rgb: np.ndarray) -> float:
    """
    Measures bright localized reflection ratio.
    Returns 0.0 (Matte/Fake) to 1.0 (Natural glint/Real).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_val = gray.mean() + 1e-5
    max_val = gray.max()
    ratio = max_val / mean_val
    return float(min(ratio / 2.5, 1.0))


def noise_analysis(img_rgb: np.ndarray) -> float:
    """
    Measures sensor noise levels.
    Real camera photos have natural grain; AI images are unnaturally clean.
    Returns 0.0 (Unnaturally clean = Fake) to 1.0 (Natural noise = Real).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_residual = gray - blurred
    noise_std = np.std(noise_residual)
    score = np.clip((noise_std - 1.5) / 2.5, 0.0, 1.0)
    return float(score)


def frequency_analysis(img_rgb: np.ndarray) -> float:
    """
    Analyzes FFT frequency spectrum.
    Real images have natural high-frequency content; AI images are low-freq biased.
    Returns 0.0 (Unnatural = Fake) to 1.0 (Natural = Real).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    low_freq = magnitude[cy-10:cy+10, cx-10:cx+10].mean()
    high_freq_mask = np.ones_like(magnitude, dtype=bool)
    high_freq_mask[cy-30:cy+30, cx-30:cx+30] = False
    high_freq = magnitude[high_freq_mask].mean()
    ratio = high_freq / (low_freq + 1e-5)
    return float(np.clip((ratio - 0.35) / 0.30, 0.0, 1.0))


def calculate_pixel_glitch_score(face_arrays: list) -> float:
    """
    Measures temporal inconsistency using pixel-level frame differences.
    Does NOT depend on ViT scores — reliable on any video quality.

    Real videos: consistent face appearance across frames → low std of diffs
    Fake videos: abrupt face region changes between frames → high std of diffs

    Returns 0.0 (Consistent/Real) to 1.0 (Inconsistent/Fake).
    """
    if len(face_arrays) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(face_arrays)):
        prev = face_arrays[i-1].astype(np.float32)
        curr = face_arrays[i].astype(np.float32)
        # Resize to same shape for comparison
        h = min(prev.shape[0], curr.shape[0])
        w = min(prev.shape[1], curr.shape[1])
        prev = cv2.resize(prev, (w, h))
        curr = cv2.resize(curr, (w, h))
        diff = np.mean(np.abs(curr - prev)) / 255.0
        diffs.append(diff)

    std_diff = np.std(diffs)
    # High std = inconsistent frame-to-frame changes = deepfake artifact
    glitch = np.clip(std_diff / 0.05, 0.0, 1.0)
    return float(glitch)


def calculate_video_glitch_score(vit_scores: list) -> float:
    """
    Legacy glitch score based on ViT variance.
    Kept for backward compatibility but no longer used in fusion.
    """
    if len(vit_scores) < 2:
        return 0.0
    std_dev = np.std(vit_scores)
    return float(min(std_dev / 0.35, 1.0))