"""Per-window explanation for Audio Model v1 (wav2vec2-base, mean-pooled multi-window deepfake classifier).

Uses the production preprocessing (`audio/model_v1/optimized.py`) and the cached production model, so the
windows scored here are exactly the windows the verdict was computed from (see `audio/backend.py`'s
AudioModelV1Backend: the verdict is `_band_verdict` of these windows' mean). Adds:
  * each window's own deepfake probability, with its position in the clip,
  * a gradient-based saliency curve over time for the single most suspicious window, rendered against that
    window's own spectrogram.

wav2vec2's frontend is a stack of 1D convolutions over the raw waveform, not the single late 2D conv feature
map that Grad-CAM (Selvaraju et al. 2017) targets, so there is no direct Grad-CAM analogue here. The
substitute used is a plain input-gradient saliency map (Simonyan et al. 2013): the gradient of the model's
own deepfake probability with respect to each input sample, showing which moments in time the score is most
sensitive to. Like Grad-CAM elsewhere in this codebase, this is an attribution of the model's own behaviour,
not proof that a time segment was manipulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.services.explain import core
from app.services.audio.model_v1 import common, optimized
from app.services.models.audio_model import decode_audio_mono_16k

SPEC_SIZE = (640, 220)  # (width, height) of the rendered spectrogram image, display only
N_FFT = 512
HOP = 160  # 10ms at this model's 16kHz target rate
SALIENCY_BINS = 80  # time resolution of the saliency curve, independent of the STFT framing above


@dataclass
class WindowExplanation:
    order: int
    start_s: float
    end_s: float
    probability: float  # this window's own deepfake probability


@dataclass
class AudioExplanation:
    windows: list[WindowExplanation]
    mean_probability: float  # same quantity the verdict thresholds (see AudioModelV1Backend)
    threshold: float
    windows_above_threshold: int
    duration_s: float
    saliency_window_order: int  # which window (by `order` above) the spectrogram/saliency below are from
    spectrogram_jpeg: bytes
    saliency_jpeg: bytes | None


def _window_starts(n_samples: int) -> np.ndarray:
    """Mirrors `optimized._build_windows`'s start-index formula exactly. Kept as a separate computation rather
    than changing that production function to also return starts - same reasoning as video_explainer.py
    computing frame positions (`common.evenly_spaced_frame_indices`) alongside, not inside,
    `common.decode_selected_frames`: production inference code stays untouched, explanation code mirrors it."""
    if n_samples <= common.WINDOW_SAMPLES:
        return np.array([0])
    n_windows = min(common.MAX_WINDOWS, int(np.ceil(n_samples / common.WINDOW_SAMPLES)))
    return np.linspace(0, max(n_samples - common.WINDOW_SAMPLES, 0), n_windows).astype(int)


def _render_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """Log-magnitude STFT via numpy's FFT only (no extra dependency, same spirit as watermark.py's PyWavelets
    reimplementation avoiding a conflicting dep). Returns (freq_bins, time_frames) normalised to [0, 1]."""
    window = np.hanning(N_FFT).astype(np.float32)
    n = waveform.shape[0]
    n_frames = max(1, 1 + max(0, n - N_FFT) // HOP)
    frames = np.zeros((n_frames, N_FFT), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP
        chunk = waveform[start : start + N_FFT]
        frames[i, : chunk.shape[0]] = chunk
    frames *= window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    log_mag = np.log1p(mag).T  # (freq_bins, time_frames)
    log_mag -= log_mag.min()
    peak = log_mag.max()
    if peak > 0:
        log_mag /= peak
    return log_mag


def _spectrogram_image(waveform: np.ndarray) -> Image.Image:
    spec = _render_spectrogram(waveform)
    gray = (np.flipud(spec) * 255).astype(np.uint8)  # flip so low frequencies sit at the bottom, as usual
    img = Image.fromarray(gray, mode="L").convert("RGB")
    return img.resize(SPEC_SIZE, Image.Resampling.BICUBIC)


def _saliency_curve(model: torch.nn.Module, device: torch.device, win: np.ndarray) -> np.ndarray:
    """Gradient of the deepfake probability w.r.t. the raw waveform, pooled into SALIENCY_BINS time bins and
    normalised to [0, 1] (per-window max-normalisation, the same convention Grad-CAM uses per-image elsewhere
    in this codebase - see core.cam_from_features)."""
    x = torch.from_numpy(win).contiguous().float().to(device).unsqueeze(0)
    x.requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=-1)[:, 1].sum()
        (grad,) = torch.autograd.grad(prob, x)
    sal = np.abs(grad.detach().cpu().numpy()[0])
    bins = np.array_split(sal, SALIENCY_BINS)
    curve = np.array([b.mean() for b in bins], dtype=np.float32)
    peak = curve.max()
    return curve / peak if peak > 0 else curve


def explain_audio(path: str, checkpoint_path: str | None = None) -> AudioExplanation:
    model, device, _ = optimized._get_model_and_device(checkpoint_path)

    samples = decode_audio_mono_16k(path)
    duration_s = float(samples.shape[0] / common.TARGET_SR)
    normalized = common.rms_normalize(samples)  # whole clip, once - matches training, same as optimized.predict
    windows = optimized._build_windows(normalized)
    starts = _window_starts(normalized.shape[0])

    tensor = torch.from_numpy(windows).contiguous().float().to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    mean_probability = float(probs.mean())
    out_windows = [
        WindowExplanation(
            order=i,
            start_s=float(starts[i]) / common.TARGET_SR,
            end_s=float(starts[i] + common.WINDOW_SAMPLES) / common.TARGET_SR,
            probability=float(probs[i]),
        )
        for i in range(len(probs))
    ]

    from app.core.config import settings

    top = int(np.argmax(probs))
    spec_img = _spectrogram_image(windows[top])
    curve = _saliency_curve(model, device, windows[top])
    cam = np.tile(curve[None, :], (8, 1))  # uniform across frequency bins - only time varies
    strength = core.evidence_strength(float(probs[top]))
    heat = core.overlay(spec_img, cam * strength) if strength > 0 else None

    return AudioExplanation(
        windows=out_windows,
        mean_probability=mean_probability,
        threshold=float(settings.FAKE_THRESHOLD),
        windows_above_threshold=int(sum(1 for p in probs if p >= settings.FAKE_THRESHOLD)),
        duration_s=duration_s,
        saliency_window_order=top,
        spectrogram_jpeg=core.encode(spec_img),
        saliency_jpeg=core.encode(heat) if heat is not None else None,
    )
