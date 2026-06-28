# fusion_model.py
import numpy as np
from vit_model import predict_vit
from heuristics import texture_score, color_anomaly, boundary_artifact, eye_glint_score


def run_fusion(pil_img, is_video=False):
    """
    Fusion of ViT + Heuristics.
    For video frames, weights are automatically reduced.
    """

    img = np.array(pil_img)
    pil_img = pil_img.resize((224, 224))
    vit_prob = predict_vit(pil_img)

    tex = texture_score(img)
    col = color_anomaly(img)
    bound = boundary_artifact(img)
    glint = eye_glint_score(img)

    if is_video:
        final_score = (
            0.95 * vit_prob +
            0.02 * (1 - tex) +
            0.01 * col +
            0.01 * bound +
            0.01 * (1 - glint)
        )
    else:
        # Image = clean → heuristics balanced
        final_score = (
            0.55 * vit_prob +
            0.15 * (1 - tex) +
            0.10 * col +
            0.10 * bound +
            0.10 * (1 - glint)
        )

    final_score = float(np.clip(final_score, 0, 1))

    return final_score, {
        "vit": vit_prob,
        "texture": tex,
        "color": col,
        "boundary": bound,
        "eye_glint": glint
    }