"""Opt-in variant of `optimized.predict()` that uses the wider face-crop padding in `wide_pad.py` instead of
common.py's trained 20% padding. Not the production default -- see `backend.py`'s `VideoModelV1WidePadBackend`
(`VIDEO_MODEL_BACKEND=model_v1_wide_pad`) and CLAUDE.md section 22.

Reuses the exact same model, checkpoint, decode and scoring code as `optimized.predict()`. The ONLY difference
is the padding fraction used when cropping around a detected face (35% instead of 20%); frames where no face
is found are unaffected (same center-square fallback either way).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import common, optimized, wide_pad


@dataclass
class WidePadResult:
    logit: float
    probability: float
    verdict: str
    frame_logits: list[float]
    timings: dict[str, float] = field(default_factory=dict)


def predict(video_path: str, checkpoint_path: str | None = None) -> WidePadResult:
    t_total0 = time.perf_counter()

    model, device, _ = optimized._get_model_and_device(checkpoint_path)
    cascade = common.get_face_cascade()

    frames = common.decode_selected_frames(video_path, use_grab_skip=True)
    processed = [wide_pad.preprocess_frame_wide(f, cascade) for f in frames]
    batch = np.stack(processed, axis=0)
    tensor = torch.from_numpy(batch).contiguous().float().to(device)

    with torch.inference_mode():
        logits = model(tensor).squeeze(-1)  # (16,)
        mean_logit_t = logits.mean()
        probability_t = torch.sigmoid(mean_logit_t)
    frame_logits = [float(v) for v in logits.detach().cpu().tolist()]
    mean_logit = float(mean_logit_t.item())
    probability = float(probability_t.item())
    verdict = "FAKE" if probability >= common.THRESHOLD else "REAL"

    return WidePadResult(
        logit=mean_logit,
        probability=probability,
        verdict=verdict,
        frame_logits=frame_logits,
        timings={"total_s": time.perf_counter() - t_total0},
    )
