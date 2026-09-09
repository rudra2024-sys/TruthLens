"""Audio spoof/deepfake-voice detection using AASIST, pretrained on ASVspoof2019-LA.

AASIST classifies short (~4s) raw-waveform windows as bonafide vs spoof. For
longer clips we slide non-overlapping windows across the whole file and
average, so the verdict reflects the entire recording, not just its start.
"""
from __future__ import annotations

import numpy as np

from app.services.models.sessions import aasist_predict_logits

TARGET_SR = 16000
WINDOW_SAMPLES = 64600  # ~4.04s at 16kHz, AASIST's trained input length
MAX_WINDOWS = 12  # cap work on very long files

# Which output column of the AASIST logits is "spoof" vs "bonafide".
#
# THIS IS AN UNVALIDATED ASSUMPTION, not a confirmed fact. It comes from a
# code-comment convention only ("[spoof, bonafide]" in sessions.py), and has
# never been checked against real labeled bonafide/spoof audio: no such data
# exists anywhere in this repository or (per an exhaustive prior whole-drive
# search) on the development machine. For a 2-class softmax,
# probs[:, 0] == 1 - probs[:, 1] always, so probability math alone cannot
# distinguish the two possible orientations -- only real labeled data can.
#
# Do NOT flip this without empirical evidence (running known bonafide and
# known spoof audio through score_audio_path and confirming the direction).
# Flipping it blind would just swap which unverified assumption is active.
SPOOF_CLASS_INDEX = 0


def decode_audio_mono_16k(path: str) -> np.ndarray:
    """Decode any container PyAV/ffmpeg supports (wav/mp3/flac/m4a/...) to mono float32 @16kHz."""
    import av

    container = av.open(path)
    if not container.streams.audio:
        raise ValueError("No audio stream found in file.")
    stream = container.streams.audio[0]

    resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)

    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()  # shape (1, n_samples) for mono flt
            chunks.append(arr.reshape(-1))
    container.close()

    if not chunks:
        raise ValueError("Could not decode any audio samples from file.")
    return np.concatenate(chunks).astype(np.float32)


def _tile_pad(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length]
    reps = int(np.ceil(length / x.shape[0]))
    return np.tile(x, reps)[:length]


def build_windows(samples: np.ndarray) -> np.ndarray:
    n = samples.shape[0]
    if n <= 0:
        raise ValueError("Empty audio signal.")

    if n <= WINDOW_SAMPLES:
        return _tile_pad(samples, WINDOW_SAMPLES)[None, :]

    n_windows = min(MAX_WINDOWS, int(np.ceil(n / WINDOW_SAMPLES)))
    starts = np.linspace(0, max(n - WINDOW_SAMPLES, 0), n_windows).astype(int)
    windows = np.stack([samples[s:s + WINDOW_SAMPLES] for s in starts])
    # last window may be short if n < WINDOW_SAMPLES * n_windows edge case
    if windows.shape[1] < WINDOW_SAMPLES:
        windows = np.stack([_tile_pad(w, WINDOW_SAMPLES) for w in windows])
    return windows


def score_audio_path(path: str) -> dict:
    samples = decode_audio_mono_16k(path)
    duration_s = float(samples.shape[0] / TARGET_SR)

    windows = build_windows(samples)
    logits = aasist_predict_logits(windows)  # (N, 2) -> [spoof, bonafide] (unvalidated, see SPOOF_CLASS_INDEX)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    spoof_probs = probs[:, SPOOF_CLASS_INDEX]

    return {
        "spoof_probability": float(spoof_probs.mean()),
        "spoof_probability_std": float(spoof_probs.std()),
        "windows_analyzed": int(windows.shape[0]),
        "duration_s": duration_s,
    }
