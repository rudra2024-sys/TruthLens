"""Opt-in variant of `optimized.predict()` that adds the conservative orientation fallback in `orientation.py`
before the existing, unmodified preprocessing pipeline. Not the production default -- see `backend.py`'s
`VideoModelV1RotationAwareBackend` (`VIDEO_MODEL_BACKEND=model_v1_rotation_aware`) and CLAUDE.md section 19.

Reuses the exact same model, checkpoint, Haar cascade, decode, crop and scoring code as `optimized.predict()`.
The ONLY difference is that every sampled frame may be rotated (by one of 0/90/180/270 degrees, chosen once
for the whole clip by `orientation.detect_clip_rotation`) before being handed to the unmodified
`common.preprocess_frame()`. When the native orientation already finds a face in at least one frame,
`detect_clip_rotation()` returns 0 and every frame is preprocessed exactly as `optimized.predict()` would --
confirmed bit-identical in `tests/test_video_rotation_fallback.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import common, optimized, orientation


@dataclass
class RotationAwareResult:
    logit: float
    probability: float
    verdict: str
    frame_logits: list[float]
    rotation_degrees: int  # 0 unless every sampled frame failed to find a face at the native orientation
    timings: dict[str, float] = field(default_factory=dict)


def predict(video_path: str, checkpoint_path: str | None = None) -> RotationAwareResult:
    t_total0 = time.perf_counter()

    model, device, _ = optimized._get_model_and_device(checkpoint_path)
    cascade = common.get_face_cascade()

    frames = common.decode_selected_frames(video_path, use_grab_skip=True)
    rotation_degrees = orientation.detect_clip_rotation(frames, cascade)
    rotated_frames = [orientation.rotate_frame(f, rotation_degrees) for f in frames]

    processed = [common.preprocess_frame(f, cascade) for f in rotated_frames]
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

    return RotationAwareResult(
        logit=mean_logit,
        probability=probability,
        verdict=verdict,
        frame_logits=frame_logits,
        rotation_degrees=rotation_degrees,
        timings={"total_s": time.perf_counter() - t_total0},
    )
