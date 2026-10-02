"""Opt-in cross-cascade consensus check for Video Model v1's face crop -- NOT part of the frozen Celeb-DF-v2
parity code in common.py (kept separate so common.py/optimized.py stay exactly what they always were).

Added 2026-10-02 after diagnosing `fake video 3.mp4` (CLAUDE.md section 11's "useful finding"): the default
Haar cascade (`haarcascade_frontalface_default.xml`) sometimes locks onto a wall poster or a wrist instead of
a face (frames 2, 4, 8 of that clip, confirmed visually with a saved boxed-frame image) -- and on exactly those
frames, a second, independently-trained cascade (`haarcascade_frontalface_alt2.xml`) finds nothing at all. On
the frames where the default cascade finds the actual face (13, 14), alt2 independently finds a near-identical
box. This module requires that agreement before trusting a detection: a default-cascade face is only used if
at least one alt2 detection overlaps it by IoU >= MIN_IOU; otherwise the frame falls back to the existing
center-square crop, exactly as if no face had been found at all.

Unlike orientation.py's rotation fallback (which only ever touches a clip where EVERY sampled frame already
fails to find a face, so it cannot regress a working clip), this is not provably safe by construction: a frame
where the default cascade alone correctly finds a face that alt2 happens to miss would be downgraded from a
good crop to a center crop. Whether the poster/wrist fix outweighs that cost is an empirical question -- see
CLAUDE.md section 20 for the measured answer, not an assumption baked into the code.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from . import common

MIN_IOU = 0.3

_alt2_cascade_cache: cv2.CascadeClassifier | None = None


def get_alt2_cascade() -> cv2.CascadeClassifier:
    global _alt2_cascade_cache
    if _alt2_cascade_cache is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        _alt2_cascade_cache = cv2.CascadeClassifier(path)
    return _alt2_cascade_cache


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def corroborated_face(gray: np.ndarray, cascade: cv2.CascadeClassifier, alt2: cv2.CascadeClassifier):
    """The largest `cascade` detection corroborated by an overlapping `alt2` detection (IoU >= MIN_IOU), or
    None if `cascade` found nothing, `alt2` found nothing, or none of `cascade`'s candidates overlap any of
    `alt2`'s."""
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    faces2 = alt2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces2) == 0:
        return None
    candidates = [f for f in faces if any(_iou(tuple(f), tuple(g)) >= MIN_IOU for g in faces2)]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f[2] * f[3])


def crop_face_or_center_consensus(
    frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier, alt2: cv2.CascadeClassifier
) -> np.ndarray:
    """Same crop geometry as common.crop_face_or_center_with_info (20% padding, center-square fallback,
    empty-crop safety net) but only trusts a face detection corroborated by alt2 (see corroborated_face)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_rgb.shape[:2]

    face = corroborated_face(gray, cascade, alt2)

    if face is None:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop = frame_rgb[top : top + side, left : left + side]
    else:
        x, y, fw, fh = (int(v) for v in face)
        pad_w = int(fw * common.FACE_PADDING_FRACTION)
        pad_h = int(fh * common.FACE_PADDING_FRACTION)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(w, x + fw + pad_w)
        y1 = min(h, y + fh + pad_h)
        crop = frame_rgb[y0:y1, x0:x1]

    if crop.size == 0:
        crop = frame_rgb
    return crop


def preprocess_frame_consensus(
    frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier, alt2: cv2.CascadeClassifier
) -> np.ndarray:
    """Same resize/normalise as common.preprocess_frame, built on crop_face_or_center_consensus instead of
    common.crop_face_or_center."""
    crop_rgb = crop_face_or_center_consensus(frame_bgr, cascade, alt2)
    resized = Image.fromarray(crop_rgb).resize(
        (common.IMAGE_SIZE, common.IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    normalized = resized_rgb.astype(np.float32) / 255.0
    normalized = (normalized - common.IMAGENET_MEAN) / common.IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).copy()
