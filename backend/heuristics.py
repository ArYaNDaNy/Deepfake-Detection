# heuristics.py
import cv2
import numpy as np
from PIL import Image

def texture_score(img):
    """Skin texture consistency (fake = overly smooth)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(var / 500.0, 1.0))

def color_anomaly(img):
    """Color channel deviation."""
    r, g, b = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    diff = (abs(r-g) + abs(r-b) + abs(g-b)) / 255
    return float(min(diff, 1.0))

def boundary_artifact(face):
    """Face boundary sharpness anomalies."""
    edges = cv2.Canny(face, 100, 200)
    return float(edges.mean() / 255)

def eye_glint_score(face):
    """Reflection highlight realism."""
    gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    return float(gray.max() / 255)