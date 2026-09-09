"""Optimized inference implementation for TruthLens Video Model v1.

Same evaluated protocol as baseline.py -- frame selection, sequential
frame-matching semantics, face crop + 20% padding + center-square/empty-
crop fallback, PIL BILINEAR resize, normalization, mean-logit pooling,
sigmoid, 0.525 threshold, FP16 CUDA autocast -- all shared via common.py
(a line-by-line port of the recovered original Celeb-DF-v2 evaluator) so
the two implementations cannot silently diverge on model semantics.

What differs from baseline.py (implementation only):
  - the model and Haar cascade are loaded once per process and cached,
    not rebuilt on every call
  - only the 16 needed frames are decoded: common.decode_selected_frames
    is called with use_grab_skip=True, which uses cv2.VideoCapture.grab()
    to skip frames we don't need instead of read()+discard, avoiding the
    full decode baseline.py does -- proven pixel-identical to
    read()-and-discard for the frames actually kept
  - all 16 frames are stacked into one batch and run through the model
    in a single forward pass instead of 16
  - torch.inference_mode() instead of no_grad(), and the batch tensor is
    built once via np.stack instead of allocated/concatenated per frame
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import common

_MODEL_CACHE: dict[str, tuple[nn.Module, torch.device]] = {}


def _get_model_and_device(checkpoint_path: str | None) -> tuple[nn.Module, torch.device, bool]:
    """Returns (model, device, was_loaded_this_call)."""
    key = checkpoint_path or "__default__"
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], False

    model = common.build_model()
    common.load_checkpoint(model, checkpoint_path)
    model.eval()
    device = common.get_device()
    model.to(device)
    _MODEL_CACHE[key] = (model, device)
    return model, device, True


def clear_model_cache() -> None:
    """Test/benchmark hook to force a cold reload."""
    _MODEL_CACHE.clear()


@dataclass
class VideoModelV1Result:
    logit: float
    probability: float
    verdict: str
    frame_logits: list[float]
    timings: dict[str, float] = field(default_factory=dict)


def _decode_selected_frames(video_path: str) -> list[np.ndarray]:
    return common.decode_selected_frames(video_path, use_grab_skip=True)


def predict(video_path: str, checkpoint_path: str | None = None) -> VideoModelV1Result:
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    model, device, was_loaded = _get_model_and_device(checkpoint_path)
    cascade = common.get_face_cascade()
    load_s = time.perf_counter() - t0 if was_loaded else 0.0

    t0 = time.perf_counter()
    frames = _decode_selected_frames(video_path)
    batch = np.stack(
        [common.preprocess_frame(f, cascade) for f in frames], axis=0
    )
    tensor = torch.from_numpy(batch).contiguous().float().to(device)
    preprocess_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ):
            logits = model(tensor).squeeze(-1)  # (16,)
        mean_logit_t = logits.mean()
        probability_t = torch.sigmoid(mean_logit_t)
    frame_logits = [float(v) for v in logits.detach().cpu().tolist()]
    mean_logit = float(mean_logit_t.item())
    probability = float(probability_t.item())
    inference_s = time.perf_counter() - t0

    verdict = "FAKE" if probability >= common.THRESHOLD else "REAL"
    total_s = time.perf_counter() - t_total0

    return VideoModelV1Result(
        logit=mean_logit,
        probability=probability,
        verdict=verdict,
        frame_logits=frame_logits,
        timings={
            "model_load_s": load_s,
            "preprocess_s": preprocess_s,
            "inference_s": inference_s,
            "total_s": total_s,
        },
    )
