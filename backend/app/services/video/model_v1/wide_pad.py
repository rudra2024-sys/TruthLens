"""Opt-in wider face-crop padding for Video Model v1 -- NOT part of the frozen Celeb-DF-v2 parity code in
common.py (kept separate so common.py/optimized.py stay exactly what they always were).

Added 2026-10-03 as the third of three video-parsing experiments requested in the same session as sections
19/20 (rotation fallback, cascade consensus). Face-swap boundary artifacts often sit right at the jaw/hairline
-- near the edge of or just outside the model's trained 20% padding (common.FACE_PADDING_FRACTION) -- so a
wider crop might give the model more of the manipulated region to look at. Unlike sections 19/20 this isn't a
response to a diagnosed failure on a specific clip; it's a hypothesis to test. See CLAUDE.md section 22 for
the measured answer.

Same crop geometry as common.crop_face_or_center_with_info (center-square fallback, empty-crop safety net)
with one number changed: the padding fraction around a detected face.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from . import common

WIDE_PADDING_FRACTION = 0.35  # vs common.FACE_PADDING_FRACTION's 0.20


def crop_face_or_center_wide(
    frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier, padding_fraction: float = WIDE_PADDING_FRACTION
) -> np.ndarray:
    """Identical to common.crop_face_or_center_with_info's crop logic, except the padding fraction is a
    parameter instead of the hardcoded common.FACE_PADDING_FRACTION. The face-finding step itself (cascade,
    scaleFactor, minNeighbors, minSize) and the center-square/empty-crop fallbacks are untouched."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_rgb.shape[:2]

    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop = frame_rgb[top : top + side, left : left + side]
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_w = int(fw * padding_fraction)
        pad_h = int(fh * padding_fraction)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(w, x + fw + pad_w)
        y1 = min(h, y + fh + pad_h)
        crop = frame_rgb[y0:y1, x0:x1]

    if crop.size == 0:
        crop = frame_rgb
    return crop


def preprocess_frame_wide(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> np.ndarray:
    """Same resize/normalise as common.preprocess_frame, built on crop_face_or_center_wide instead of
    common.crop_face_or_center."""
    crop_rgb = crop_face_or_center_wide(frame_bgr, cascade)
    resized = Image.fromarray(crop_rgb).resize(
        (common.IMAGE_SIZE, common.IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    normalized = resized_rgb.astype(np.float32) / 255.0
    normalized = (normalized - common.IMAGENET_MEAN) / common.IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).copy()
