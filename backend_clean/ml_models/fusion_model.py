import numpy as np


def fuse_image_scores(vit_prob: float, heuristics: dict) -> float:
    """
    Blends dual ViT probability with physical heuristics for image detection.

    Weights:
    - ViT (0.55): Strong on AI-gen faces
    - Noise (0.18): Best signal for AI-generated images
    - Boundary (0.12): Catches blending seams
    - Frequency (0.10): AI upsampling artifacts
    - Texture (0.03): Weak on compressed images
    - Eye Glint (0.02): Mostly noise
    - Color (0.00): Excluded — broken for fair skin
    """
    tex_fake_score   = 1.0 - heuristics.get("texture", 1.0)
    bound_fake_score = heuristics.get("boundary", 0.0)
    glint_fake_score = heuristics.get("eye_glint", 0.0)
    noise_fake_score = 1.0 - heuristics.get("noise", 1.0)
    freq_fake_score  = 1.0 - heuristics.get("frequency", 1.0)

    heuristics_sum = (
        0.03 * tex_fake_score   +
        0.12 * bound_fake_score +
        0.02 * glint_fake_score +
        0.18 * noise_fake_score +
        0.10 * freq_fake_score
    )

    # Conflict: ViT confident Real but heuristics catching something
    heuristics_disagree = (vit_prob < 0.25 and heuristics_sum > 0.10) or \
                          (vit_prob > 0.75 and heuristics_sum < 0.10)

    if heuristics_disagree:
        final_score = (0.55 * vit_prob) + (0.45 * heuristics_sum)
    elif vit_prob < 0.25 or vit_prob > 0.75:
        final_score = (0.70 * vit_prob) + (0.30 * heuristics_sum)
    else:
        final_score = (0.50 * vit_prob) + (0.50 * heuristics_sum)

    # Suspicion floor — only when heuristics also agree something is off
    if vit_prob < 0.05 and heuristics_sum > 0.08:
        final_score = max(final_score, 0.20)

    return float(np.clip(final_score, 0.0, 1.0))


def fuse_video_scores(avg_vit_prob: float, avg_heuristics: dict,
                      pixel_glitch: float, vit_glitch: float) -> float:
    """
    Blends temporal video scores using pixel-level glitch as primary signal.

    Weights:
    - Pixel Glitch (0.60): Frame-to-frame pixel diff — reliable on any video quality
    - Boundary (0.15): Face swap blending seams
    - Noise (0.10): Compression/generation artifacts
    - Frequency (0.08): AI upsampling patterns
    - ViT (0.05): Almost ignored — too unreliable on compressed video frames
    - VIT Glitch (0.02): Legacy signal, minimal weight
    - Color (0.00): Excluded
    - Eye Glint (0.00): Excluded — too noisy in video
    """
    bound_fake_score = avg_heuristics.get("boundary", 0.0)
    noise_fake_score = 1.0 - avg_heuristics.get("noise", 1.0)
    freq_fake_score  = 1.0 - avg_heuristics.get("frequency", 1.0)

    final_score = (
        0.60 * pixel_glitch    +  # primary — pixel-level temporal inconsistency
        0.15 * bound_fake_score +  # blending seam detection
        0.10 * noise_fake_score +  # compression artifacts
        0.08 * freq_fake_score  +  # frequency pattern
        0.05 * avg_vit_prob     +  # minimal ViT contribution
        0.02 * vit_glitch          # legacy ViT variance signal
    )

    return float(np.clip(final_score, 0.0, 1.0))