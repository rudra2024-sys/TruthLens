"""Face-swap check for still images: runs the deployed Video Model v1 face-crop classifier on a single photo.

Why this exists: the whole-image ConvNeXt + CLIP ensemble is near chance on face swaps (AUC 0.54-0.61 on 356 held-out
video frames), because a swap is a real photo with one face replaced. Video Model v1 round 4 looks only at the
Haar-detected face crop and scores AUC 0.95 on the same frames (CLAUDE.md section 28).

Applies only when a frontal face is actually detected and is at least MIN_FACE_PX wide - the centre-square fallback
the video pipeline uses when no face is found is NOT a face check, so it returns None instead. Same checkpoint,
cascade, crop padding and preprocessing as the video path (no new weights); one frame instead of sixteen.
"""

from __future__ import annotations

from dataclasses import dataclass

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from app.services.video.model_v1 import common, optimized

MIN_FACE_PX = 64
MTCNN_MIN_PROB = 0.9
FACE_PADDING = common.FACE_PADDING_FRACTION

_mtcnn = None


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "image_face_swap" / "face_swap_photo_r2_blend50.pt"


def checkpoint_path() -> str | None:
    """Weights for the image face-swap check. Default: the photo model round 2 file if it is present (CLAUDE.md section 28),
    else the video round-4 weights as before. IMAGE_FACE_SWAP_CHECKPOINT overrides; set it to an empty string to force the
    video round-4 weights (rollback without deleting the file). The video module never reads any of this."""
    v = os.getenv("IMAGE_FACE_SWAP_CHECKPOINT")
    if v is not None:
        return v or None
    return str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None


def calibration_key() -> str:
    """A separate checkpoint has different logits, so it needs its own Platt entry (image_face_swap_r2)."""
    return "image_face_swap_r2" if checkpoint_path() else "image_face_swap"


def _get_mtcnn():
    """MTCNN (weights ship inside the facenet-pytorch package, already a backend dependency). Lazy: only built if Haar
    finds nothing, so the common path pays nothing."""
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=40, post_process=False)
    return _mtcnn


def _mtcnn_crop(frame_bgr: np.ndarray):
    """Largest MTCNN face (>= MIN_FACE_PX wide, prob >= 0.9) with the same 20% padding as the Haar path, as an RGB crop
    plus the box width; None if there is none. Measured as a Haar *fallback* only (CLAUDE.md section 28): used alone,
    its crops score ~5 points worse because the model was trained on Haar-style crops."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = _get_mtcnn().detect(Image.fromarray(rgb))
    if boxes is None:
        return None
    cands = [b for b, p in zip(boxes, probs) if p is not None and p >= MTCNN_MIN_PROB and (b[2] - b[0]) >= MIN_FACE_PX]
    if not cands:
        return None
    x0, y0, x1, y1 = max(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    h, w = rgb.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    a, b = max(0, int(x0 - bw * FACE_PADDING)), max(0, int(y0 - bh * FACE_PADDING))
    c, d = min(w, int(x1 + bw * FACE_PADDING)), min(h, int(y1 + bh * FACE_PADDING))
    crop = rgb[b:d, a:c]
    return (crop, int(bw)) if crop.size else None


def _score_crop(crop_rgb: np.ndarray) -> float:
    resized = Image.fromarray(crop_rgb).resize((common.IMAGE_SIZE, common.IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(resized, dtype=np.uint8).astype(np.float32) / 255.0 - common.IMAGENET_MEAN) / common.IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy()[None]).contiguous().float()
    model, device, _ = optimized._get_model_and_device(checkpoint_path())
    with torch.inference_mode():
        return float(model(tensor.to(device)).squeeze(-1).item())


@dataclass
class FaceSwapScore:
    logit: float          # raw model logit for FAKE (sigmoid(logit) is the uncalibrated P(FAKE))
    face_px: int          # width of the detected face box in the original image


def _load_bgr(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def score_bgr(frame_bgr: np.ndarray) -> FaceSwapScore | None:
    """None when no face (>= MIN_FACE_PX wide) is found. Haar first (what the model was trained on); MTCNN only when
    Haar finds no usable face."""
    cascade = common.get_face_cascade()
    _, info = common.crop_face_or_center_with_info(frame_bgr, cascade)
    bbox = info["face_bbox"]
    if bbox is not None and bbox[2] >= MIN_FACE_PX:
        proc, _ = common.preprocess_frame_with_info(frame_bgr, cascade)
        model, device, _ = optimized._get_model_and_device(checkpoint_path())
        tensor = torch.from_numpy(proc[None]).contiguous().float().to(device)
        with torch.inference_mode():
            logit = float(model(tensor).squeeze(-1).item())
        return FaceSwapScore(logit=logit, face_px=int(bbox[2]))
    found = _mtcnn_crop(frame_bgr)
    if found is None:
        return None
    crop, face_px = found
    return FaceSwapScore(logit=_score_crop(crop), face_px=face_px)


def score_image(path: str) -> FaceSwapScore | None:
    return score_bgr(_load_bgr(path))
