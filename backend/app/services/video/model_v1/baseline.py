"""Naive, unoptimized reference implementation of TruthLens Video Model v1
inference -- a faithful, un-batched port of the recovered original
Celeb-DF-v2 evaluator's single-video path (see common.py for the shared,
line-by-line-matched semantics).

Intentionally inefficient in ways optimized.py fixes:
  - decodes every frame in the video (common.decode_selected_frames with
    use_grab_skip=False), never skipping via grab()
  - rebuilds the model and the Haar cascade from scratch on every call
    (no caching across requests)
  - runs one forward pass per frame (16 separate model() calls) instead
    of a single batched pass

Frame selection, sequential frame-matching semantics (including the
original's under-selection on short videos -- see common.ShortVideoError),
face detection + 20% padding + center-square/empty-crop fallback, PIL
BILINEAR resize, and normalization all come from common.py, so this
cannot silently drift from optimized.py on model semantics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import common


@dataclass
class VideoModelV1Result:
    logit: float
    probability: float
    verdict: str  # "REAL" or "FAKE"
    frame_logits: list[float]
    timings: dict[str, float] = field(default_factory=dict)


def predict(video_path: str, checkpoint_path: str | None = None) -> VideoModelV1Result:
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    model = common.build_model()
    common.load_checkpoint(model, checkpoint_path)
    model.eval()
    device = common.get_device()
    model.to(device)
    cascade = common.get_face_cascade()
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    frames = common.decode_selected_frames(video_path, use_grab_skip=False)
    preprocessed = [common.preprocess_frame(f, cascade) for f in frames]
    preprocess_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    frame_logits: list[float] = []
    with torch.no_grad():
        for arr in preprocessed:
            tensor = torch.from_numpy(arr).unsqueeze(0).float().to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                logit = model(tensor).item()
            frame_logits.append(logit)
    inference_s = time.perf_counter() - t0

    mean_logit = float(np.mean(frame_logits))
    probability = float(torch.sigmoid(torch.tensor(mean_logit)).item())
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
