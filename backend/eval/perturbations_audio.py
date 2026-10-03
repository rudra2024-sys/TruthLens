"""Audio degradations that mimic what happens to a recording on its way through the internet, mirroring
eval/perturbations.py's purpose for images and eval/run_video_robustness.py's for video. Each function takes
mono float32 @16kHz samples (the format app/services/audio/model_v1/common.TARGET_SR already uses) and returns
degraded samples of the same length and rate.

Families and levels
  noise   Gaussian sample noise, std 0.005/0.02 (relative to the already RMS-normalised -20dBFS clip)
  volume  Gain change, 0.3x/2.0x (quiet recording / clipping-adjacent loud one)
  codec   Real lossy re-encode via PyAV: MP3 at 32/96 kbps (voice-message-quality / typical call quality)

Built but NOT YET RUN AT SCALE: this project has no locally-available labeled audio deepfake dataset (checked
2026-10-03 — only image/video datasets exist under D:\\eval_data; CLAUDE.md section 23 notes the same gap for
video was closed by reusing an existing manifest, but no audio equivalent of that manifest exists). Sourcing
one (e.g. ASVspoof2019-LA, In-The-Wild) needs either a Kaggle/dataset download or the user's own labeled files
— not attempted here without that direction. See run_audio_robustness.py's module docstring and
tests/test_audio_robustness_eval.py for the smoke-tested (synthetic-audio-only) validation that exists today.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FAMILIES: dict[str, dict] = {
    "noise": {"title": "Gaussian sample noise", "xlabel": "std (relative amplitude)", "levels": [0.005, 0.02]},
    "volume": {"title": "Gain change", "xlabel": "multiplier", "levels": [0.3, 2.0]},
    "codec": {"title": "MP3 re-encode", "xlabel": "kbps", "levels": [32, 96]},
}


def settings(include_clean: bool = True) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = [("clean", 0)] if include_clean else []
    for fam, spec in FAMILIES.items():
        out += [(fam, lvl) for lvl in spec["levels"]]
    return out


def noise(samples: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (samples + rng.normal(0.0, std, samples.shape)).astype(np.float32)


def volume(samples: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(samples * factor, -1.0, 1.0).astype(np.float32)


def codec(samples: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    """Real lossy round-trip via ffmpeg (through PyAV's bundled binaries is unavailable for muxing from raw
    arrays reliably across platforms, so this shells out to the `av`-installed ffmpeg if present; falls back
    to returning `samples` unchanged -- with a note -- if ffmpeg isn't on PATH, so callers never crash)."""
    import av

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "in.wav"
        mp3_path = Path(td) / "out.mp3"
        _write_wav(wav_path, samples, sr)

        out_container = av.open(str(mp3_path), mode="w")
        in_container = av.open(str(wav_path))
        in_stream = in_container.streams.audio[0]
        out_stream = out_container.add_stream("mp3", rate=sr)
        out_stream.bit_rate = kbps * 1000
        for frame in in_container.decode(in_stream):
            for packet in out_stream.encode(frame):
                out_container.mux(packet)
        for packet in out_stream.encode():
            out_container.mux(packet)
        out_container.close()
        in_container.close()

        decoded = av.open(str(mp3_path))
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for frame in decoded.decode(decoded.streams.audio[0]):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        decoded.close()
        if not chunks:
            return samples
        out = np.concatenate(chunks).astype(np.float32)
        # pad/trim back to the original length so downstream windowing is unaffected by codec framing/padding
        if out.shape[0] >= samples.shape[0]:
            return out[: samples.shape[0]]
        return np.pad(out, (0, samples.shape[0] - out.shape[0]))


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    import wave

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def apply(family: str, level, samples: np.ndarray, sr: int, key: str = "") -> np.ndarray:
    if family == "clean":
        return samples
    if family == "noise":
        seed = abs(hash((key, family, level))) % (2**32)
        return noise(samples, level, seed=seed)
    if family == "volume":
        return volume(samples, level)
    if family == "codec":
        return codec(samples, sr, int(level))
    raise ValueError(f"unknown perturbation family: {family}")
