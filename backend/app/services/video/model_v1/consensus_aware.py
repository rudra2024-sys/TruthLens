"""Opt-in variant of `optimized.predict()` that uses the cross-cascade consensus crop in `consensus.py`
instead of the single-cascade crop in `common.py`. Not the production default -- see `backend.py`'s
`VideoModelV1ConsensusBackend` (`VIDEO_MODEL_BACKEND=model_v1_consensus`) and CLAUDE.md section 20.

Reuses the exact same model, checkpoint, decode and scoring code as `optimized.predict()`. The ONLY difference
is that each frame's face crop requires alt2 corroboration (see `consensus.corroborated_face`) instead of
trusting the default cascade alone; an uncorroborated detection falls back to the same center-square crop
`common.py` already uses when no face is found at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import common, consensus, optimized


@dataclass
class ConsensusAwareResult:
    logit: float
    probability: float
    verdict: str
    frame_logits: list[float]
    timings: dict[str, float] = field(default_factory=dict)


def predict(video_path: str, checkpoint_path: str | None = None) -> ConsensusAwareResult:
    t_total0 = time.perf_counter()

    model, device, _ = optimized._get_model_and_device(checkpoint_path)
    cascade = common.get_face_cascade()
    alt2 = consensus.get_alt2_cascade()

    frames = common.decode_selected_frames(video_path, use_grab_skip=True)
    processed = [consensus.preprocess_frame_consensus(f, cascade, alt2) for f in frames]
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

    return ConsensusAwareResult(
        logit=mean_logit,
        probability=probability,
        verdict=verdict,
        frame_logits=frame_logits,
        timings={"total_s": time.perf_counter() - t_total0},
    )
