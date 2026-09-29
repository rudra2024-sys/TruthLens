"""Frozen, shared semantics for TruthLens Video Model v1.

This module is a line-by-line port of the recovered original Celeb-DF-v2
evaluator (the script that produced the ROC-AUC 0.695383 / PR-AUC 0.927163
/ 6,529-video result), NOT a from-scratch reimplementation. Everything
here MUST be identical between baseline.py (naive reference) and
optimized.py (fast path) -- frame position selection, sequential
frame-matching semantics, face detection + 20% padding + center-square
fallback (with the original's empty-crop safety net), PIL BILINEAR
resize, and ImageNet normalization. Only decode mechanism (grab()-skip vs
full read()), model-loading lifecycle, and batching are allowed to differ
-- see the module docstrings in baseline.py / optimized.py.

Corrected in this revision after a direct line-by-line diff against the
recovered original evaluator found two prediction-changing bugs and one
robustness gap in the previous version:
  - frame indices were `round()`ed instead of `np.linspace(...).astype
    (int)`-truncated, causing ~6-7 of 16 selected frames to differ from
    the original on typical videos
  - resize used cv2.INTER_LINEAR instead of the original's
    PIL.Image.Resampling.BILINEAR (different algorithm, different pixel
    values)
  - the original's `if crop.size == 0: crop = frame_rgb` safety fallback
    was missing
  - short-video handling (frame_count < NUM_FRAMES) invented a
    repeat/clamp fallback the original does not have; the original's
    sequential frame-matching loop under-selects on the resulting
    duplicate positions and drops such a video as unscorable
    (`only_N_frames` error) -- see ShortVideoError below
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

NUM_FRAMES = 16
IMAGE_SIZE = 224
THRESHOLD = 0.525
FACE_PADDING_FRACTION = 0.20
LABEL_CONVENTION = {0: "REAL", 1: "FAKE"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "checkpoints"
    / "video"
    / "epoch_11_model_only.pt"
)


class ShortVideoError(ValueError):
    """Raised when the original evaluator's sequential frame-matching
    loop cannot find all NUM_FRAMES positions. This is not a bug: the
    original evaluator has no fallback for videos shorter than
    NUM_FRAMES -- np.linspace produces duplicate integer positions for
    them, and once its `current` counter advances past a repeated
    position value it can never match it again, so such a video comes up
    short and gets dropped from scoring (its `only_N_frames` error path)
    rather than scored with a repeated/clamped frame. This exactly
    reproduces that behavior instead of inventing a fallback. Expected to
    never trigger on any video with frame_count >= NUM_FRAMES (all 6,529
    Celeb-DF-v2 videos in the original evaluation satisfied that -- it
    reported 0 inference errors)."""


def get_device() -> torch.device:
    requested = os.getenv("VIDEO_MODEL_V1_DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VIDEO_MODEL_V1_DEVICE=cuda was requested, but CUDA is unavailable."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    return Path(
        checkpoint_path
        or os.getenv("VIDEO_MODEL_V1_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path | None = None) -> dict:
    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video model v1 checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


_face_cascade_cache: cv2.CascadeClassifier | None = None


def get_face_cascade() -> cv2.CascadeClassifier:
    """Loads the Haar frontal-face cascade once per process and reuses it."""
    global _face_cascade_cache
    if _face_cascade_cache is None:
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        _face_cascade_cache = cascade
    return _face_cascade_cache


def evenly_spaced_frame_indices(frame_count: int, num_frames: int = NUM_FRAMES) -> list[int]:
    """Exactly matches the original evaluator:
    np.linspace(0, frame_count - 1, num_frames).astype(int) -- truncating,
    not rounding. For frame_count < num_frames this deliberately produces
    duplicate positions (see ShortVideoError); that is the original's
    actual behavior, not a bug to be smoothed over.

    Precondition: frame_count >= 1. The original only ever computes
    positions after checking total_frames > 0 (its "invalid_frame_count"
    error path handles <= 0 separately) -- see decode_selected_frames.
    """
    return np.linspace(0, frame_count - 1, num_frames).astype(int).tolist()


def decode_selected_frames(
    video_path: str, use_grab_skip: bool, on_frame=None
) -> list[np.ndarray]:
    """Port of the original evaluator's read_video_fast(): computes
    NUM_FRAMES evenly spaced positions from the container's reported
    frame count, then walks frames in order with a `current` counter,
    capturing a frame exactly when `current` reaches the next target
    position.

    use_grab_skip=False replicates the original literally (cap.read()
    every frame, discarding the ones that aren't wanted) -- this is what
    baseline.py uses. use_grab_skip=True substitutes cv2.VideoCapture.
    grab() for the discarded reads, which decodes/color-converts nothing
    for skipped frames -- a decode-only optimization proven pixel-
    identical to read()-and-discard for the frames actually kept; this is
    what optimized.py uses. The selection semantics, including the
    original's under-selection on short videos with duplicate positions,
    are identical either way because both paths share this one function.

    Raises ValueError if the container can't be opened or reports
    frame_count <= 0 (the original's "invalid_frame_count" error path).
    Raises ShortVideoError if fewer than NUM_FRAMES positions are matched
    (the original's "only_N_frames" error path).

    on_frame (optional): called as on_frame(kept, total) after each of the
    NUM_FRAMES frames is captured, so callers can report progress. It never
    influences frame selection; leaving it None is exactly the original
    behaviour. An exception raised by the callback propagates (used for
    cancellation) after the capture is released.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"video_open_failed: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"invalid_frame_count: {video_path}")

        positions = evenly_spaced_frame_indices(frame_count)

        frames: list[np.ndarray] = []
        current = 0
        next_position_index = 0
        while next_position_index < len(positions):
            target = positions[next_position_index]

            if use_grab_skip and current != target:
                if not cap.grab():
                    break
                current += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break
            if current == target:
                frames.append(frame)
                next_position_index += 1
                if on_frame is not None:
                    on_frame(len(frames), len(positions))
            current += 1

        if len(frames) != len(positions):
            raise ShortVideoError(
                f"only_{len(frames)}_frames (expected {len(positions)}) for "
                f"{video_path}"
            )
        return frames
    finally:
        cap.release()


def crop_face_or_center_with_info(
    frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier
) -> tuple[np.ndarray, dict]:
    """Largest Haar-detected frontal face, padded 20% on each side and
    clamped to frame bounds; center-square crop when no face is found;
    falls back to the full frame if the computed crop is empty -- all
    matching the original evaluator exactly. Detection runs on grayscale
    derived from the BGR frame (as the original does); the returned crop
    is RGB (the original crops from its already-BGR2RGB-converted frame,
    not from the raw BGR frame).

    Returns (crop_rgb, info) where info = {"face_bbox": (x,y,w,h) | None,
    "crop_bounds": (x0,y0,x1,y1)} -- diagnostic only, for tests/reports.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_rgb.shape[:2]

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        bounds = (left, top, left + side, top + side)
        crop = frame_rgb[top : top + side, left : left + side]
        face_bbox = None
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

        pad_w = int(fw * FACE_PADDING_FRACTION)
        pad_h = int(fh * FACE_PADDING_FRACTION)

        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(w, x + fw + pad_w)
        y1 = min(h, y + fh + pad_h)

        bounds = (x0, y0, x1, y1)
        crop = frame_rgb[y0:y1, x0:x1]
        face_bbox = (int(x), int(y), int(fw), int(fh))

    if crop.size == 0:
        crop = frame_rgb

    return crop, {"face_bbox": face_bbox, "crop_bounds": bounds}


def crop_face_or_center(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> np.ndarray:
    """Largest Haar-detected frontal face, padded 20% on each side and
    clamped to frame bounds; center-square crop when no face is found.
    Returns an RGB crop -- see crop_face_or_center_with_info."""
    crop, _info = crop_face_or_center_with_info(frame_bgr, cascade)
    return crop


def preprocess_frame(frame_bgr: np.ndarray, cascade: cv2.CascadeClassifier) -> np.ndarray:
    """BGR frame -> (3, 224, 224) float32, ImageNet-normalized.

    Matches the original exactly: crop from the RGB-converted frame, then
    PIL BILINEAR resize (not cv2.resize/INTER_LINEAR -- a different
    algorithm with different pixel values), then /255.0 and normalize.
    """
    crop_rgb = crop_face_or_center(frame_bgr, cascade)
    resized = Image.fromarray(crop_rgb).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    resized_rgb = np.asarray(resized, dtype=np.uint8)
    normalized = resized_rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).copy()
