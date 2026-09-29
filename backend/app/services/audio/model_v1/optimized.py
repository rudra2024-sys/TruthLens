"""Inference for TruthLens Audio Model v1.

Decoding is reused as-is from the existing, already-production-proven PyAV
path (app/services/models/audio_model.py::decode_audio_mono_16k) -- no new
decode logic. Everything after decoding follows this model's own trained
preprocessing (see common.py): RMS-normalize the *whole* clip once, then
window -- training itself windowed a pre-normalized clip at a random offset
each epoch (its augmentation), so slicing several deterministic offsets at
inference time is faithful to that, not a deviation from it.

The model is loaded once per process and cached, not reloaded per request
(mirrors app/services/video/model_v1/optimized.py's _MODEL_CACHE pattern).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.services.models.audio_model import decode_audio_mono_16k

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
    """Test hook to force a cold reload."""
    _MODEL_CACHE.clear()


@dataclass
class AudioModelV1Result:
    deepfake_probability_mean: float
    deepfake_probability_std: float
    windows_analyzed: int
    duration_s: float
    timings: dict[str, float] = field(default_factory=dict)


def _tile_pad(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length]
    reps = int(np.ceil(length / x.shape[0]))
    return np.tile(x, reps)[:length]


def _build_windows(samples: np.ndarray) -> np.ndarray:
    n = samples.shape[0]
    if n <= 0:
        raise ValueError("Empty audio signal.")

    if n <= common.WINDOW_SAMPLES:
        return _tile_pad(samples, common.WINDOW_SAMPLES)[None, :]

    n_windows = min(common.MAX_WINDOWS, int(np.ceil(n / common.WINDOW_SAMPLES)))
    starts = np.linspace(0, max(n - common.WINDOW_SAMPLES, 0), n_windows).astype(int)
    windows = np.stack([samples[s:s + common.WINDOW_SAMPLES] for s in starts])
    if windows.shape[1] < common.WINDOW_SAMPLES:
        windows = np.stack([_tile_pad(w, common.WINDOW_SAMPLES) for w in windows])
    return windows


def predict(path: str, checkpoint_path: str | None = None) -> AudioModelV1Result:
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    model, device, was_loaded = _get_model_and_device(checkpoint_path)
    load_s = time.perf_counter() - t0 if was_loaded else 0.0

    t0 = time.perf_counter()
    samples = decode_audio_mono_16k(path)
    duration_s = float(samples.shape[0] / common.TARGET_SR)
    samples = common.rms_normalize(samples)  # whole clip, once -- matches training
    windows = _build_windows(samples)
    tensor = torch.from_numpy(windows).contiguous().float().to(device)
    preprocess_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)  # (N, 2)
        probs = F.softmax(logits, dim=-1)[:, 1]  # deepfake-class probability per window
    per_window = probs.detach().cpu().numpy()
    inference_s = time.perf_counter() - t0

    total_s = time.perf_counter() - t_total0

    return AudioModelV1Result(
        deepfake_probability_mean=float(per_window.mean()),
        deepfake_probability_std=float(per_window.std()),
        windows_analyzed=int(windows.shape[0]),
        duration_s=duration_s,
        timings={
            "model_load_s": load_s,
            "preprocess_s": preprocess_s,
            "inference_s": inference_s,
            "total_s": total_s,
        },
    )
