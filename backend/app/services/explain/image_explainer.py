"""Grad-CAM explanation for the ConvNeXt-Tiny image model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.pipelines.image.preprocessing import IMAGE_TRANSFORM
from app.services.explain import core

DISPLAY_MAX_SIDE = 512


@dataclass
class ImageExplanation:
    overlay_jpeg: bytes          # original photo with heatmap blended in
    original_jpeg: bytes         # the photo as shown (same size as the overlay)
    cam: np.ndarray              # (7, 7) normalised Grad-CAM map at the model's last feature map
    fake_probability: float      # ConvNeXt-Tiny alone (softmax), recomputed in this pass
    fake_logit: float
    evidence_strength: float     # 0..1 = fake_probability / 0.5, capped; heatmap is faded by this (see core)
    focus_fraction: float        # share of the image whose (strength-scaled) evidence is >= 0.5
    peak_cell: tuple[int, int]   # (row, col) of the strongest cell in the 7x7 map


def gradcam_for_tensor(model: torch.nn.Module, x: torch.Tensor):
    """x: (1, 3, 224, 224) preprocessed exactly like ImagePipeline. Returns (cam (h, w) float32 np,
    logits (2,), ). Index 0 is FAKE, index 1 is REAL (see ImagePipeline.predict)."""
    with torch.enable_grad():
        feats = model.features(x)                       # (1, 768, 7, 7) - last stage
        logits = model.classifier(model.avgpool(feats))  # (1, 2)
        cam = core.cam_from_features(feats, logits[:, 0].sum())
    return cam[0].cpu().numpy().astype(np.float32), logits[0].detach().cpu()


def explain_image(path: str, model: torch.nn.Module, device: torch.device) -> ImageExplanation:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        x = IMAGE_TRANSFORM(rgb).unsqueeze(0).to(device)
        shown = core.fit_max_side(rgb, DISPLAY_MAX_SIDE)

    cam, logits = gradcam_for_tensor(model, x)
    probs = torch.softmax(logits, dim=0)
    strength = core.evidence_strength(float(probs[0]))
    scaled = cam * strength
    heat = core.overlay(shown, scaled)

    return ImageExplanation(
        overlay_jpeg=core.encode(heat),
        original_jpeg=core.encode(shown),
        cam=cam,
        fake_probability=float(probs[0]),
        evidence_strength=strength,
        fake_logit=float(logits[0]),
        focus_fraction=float((core.upsample_cam(scaled, (56, 56)) >= 0.5).mean()),
        peak_cell=tuple(int(v) for v in np.unravel_index(int(cam.argmax()), cam.shape)),
    )
