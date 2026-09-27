"""Shared helpers for TruthLens explanations (Grad-CAM + image encoding).

Everything here is derived from the real model: Grad-CAM (Selvaraju et al. 2017) weights the last
convolutional feature maps by the gradient of the FAKE logit, so the map shows which spatial regions
pushed the model's own FAKE score up. It is an attribution of the model's behaviour, not proof that a
region was manipulated.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

# Piecewise-linear "jet-like" colormap control points (position, R, G, B) - no matplotlib dependency.
_CMAP = np.array(
    [
        [0.00, 0, 0, 143],
        [0.25, 0, 128, 255],
        [0.50, 0, 255, 128],
        [0.75, 255, 255, 0],
        [1.00, 255, 0, 0],
    ],
    dtype=np.float32,
)


def colormap(values: np.ndarray) -> np.ndarray:
    """values in [0, 1] (any shape) -> uint8 RGB array with a trailing channel axis of 3."""
    v = np.clip(values, 0.0, 1.0)
    out = np.empty(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, _CMAP[:, 0], _CMAP[:, c + 1])
    return out.astype(np.uint8)


def cam_from_features(feats: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Grad-CAM. feats: (N, C, h, w) last-stage feature maps that `score` was computed from (still attached
    to the autograd graph); score: scalar (sum over the batch is fine, samples are independent).
    Returns (N, h, w), each map normalised to [0, 1] (all zeros if the model saw no positive evidence).
    Uses autograd.grad on `feats` only, so no parameter .grad is touched (safe next to the live model)."""
    (grads,) = torch.autograd.grad(score, feats)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1)).detach()
    peak = cam.amax(dim=(1, 2), keepdim=True)
    return torch.where(peak > 0, cam / peak.clamp_min(1e-12), torch.zeros_like(cam))


def evidence_strength(fake_probability: float) -> float:
    """How much of the heatmap to show, in [0, 1], from the model's own FAKE probability.

    Grad-CAM is normalised per image, so on an image the model is sure is REAL the map would still show a
    bright "peak" that is just noise. Scaling by the model's FAKE probability (full strength from 0.5 up)
    makes such maps fade to (almost) nothing instead of implying evidence that is not there."""
    return float(min(1.0, max(0.0, fake_probability / 0.5)))


def upsample_cam(cam_hw: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Bicubic-upsample a small (h, w) float map to (H, W), clipped to [0, 1]."""
    img = Image.fromarray(cam_hw.astype(np.float32), mode="F")
    up = img.resize(size_wh, Image.Resampling.BICUBIC)
    return np.clip(np.asarray(up, dtype=np.float32), 0.0, 1.0)


def overlay(base_rgb: Image.Image, cam_hw: np.ndarray, max_alpha: float = 0.65) -> Image.Image:
    """Blend the heatmap over the picture; cold (zero) regions stay transparent so the photo remains legible."""
    w, h = base_rgb.size
    heat = upsample_cam(cam_hw, (w, h))
    color = colormap(heat).astype(np.float32)
    alpha = (0.15 + (max_alpha - 0.15) * heat)[..., None] * (heat[..., None] > 0.03)
    base = np.asarray(base_rgb.convert("RGB"), dtype=np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(blended.astype(np.uint8))


def fit_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def encode(img: Image.Image, fmt: str = "JPEG", quality: int = 82) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt.upper())
    return buf.getvalue()


def data_uri(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
