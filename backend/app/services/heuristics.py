"""Shared content-heuristic helpers — scores derived from file bytes, not filenames."""
from __future__ import annotations

import math
import os
import struct
from collections import Counter


def clamp01(v: float) -> float:
    return round(min(1.0, max(0.0, float(v))), 4)


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent  # 0..8 for bytes


def normalized_entropy(data: bytes) -> float:
    return clamp01(shannon_entropy(data) / 8.0)


def sample_windows(path: str, window: int = 8192, count: int = 8) -> list[bytes]:
    size = os.path.getsize(path)
    if size <= 0:
        return []
    if size <= window:
        with open(path, "rb") as f:
            return [f.read()]
    chunks = []
    with open(path, "rb") as f:
        for i in range(count):
            pos = int((i / max(count - 1, 1)) * (size - window))
            f.seek(pos)
            chunks.append(f.read(window))
    return chunks


def packing_density_score(path: str) -> float:
    """High entropy packing often correlates with compressed / synthetic streams."""
    chunks = sample_windows(path, window=4096, count=6)
    if not chunks:
        return 0.5
    ents = [normalized_entropy(c) for c in chunks]
    mean = sum(ents) / len(ents)
    # Variance: very flat high entropy → push score up slightly
    var = sum((e - mean) ** 2 for e in ents) / len(ents)
    return clamp01(0.65 * mean + 0.35 * min(1.0, var * 8))


def analyze_image(path: str) -> tuple[float, float]:
    """Return (noise_residual_score, fft_score) in 0..1 from image pixels."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((256, 256))
        arr = np.asarray(im, dtype=np.float32) / 255.0

    # Noise residual via simple high-pass (pixel - local mean)
    # Approximate 3x3 box blur without scipy
    pad = np.pad(arr, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    residual = arr - blur
    noise_mag = float(np.mean(np.abs(residual)))
    # Natural photos tend to have moderate residual; overly smooth → suspicious
    # Map: very smooth (GAN-like) → higher fake score
    noise_score = clamp01(1.0 - min(1.0, noise_mag * 12.0))

    gray = np.mean(arr, axis=2)
    # FFT high-frequency energy ratio
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = np.sqrt(cy ** 2 + cx ** 2) + 1e-8
    high = mag[r > 0.35 * r_max].sum()
    total = mag.sum() + 1e-8
    high_ratio = float(high / total)
    # Unusual HF concentration or extreme lack both raise score via distance from typical mid-band
    fft_score = clamp01(abs(high_ratio - 0.18) * 4.5)

    # Histogram peaking (posterization / limited palette)
    hist, _ = np.histogram(gray.ravel(), bins=64, range=(0, 1), density=True)
    hist_peak = float(hist.max())
    noise_score = clamp01(0.7 * noise_score + 0.3 * min(1.0, hist_peak / 8.0))

    return noise_score, fft_score


def analyze_audio(path: str) -> tuple[float, float]:
    """Return (spectral_entropy_score, amplitude_score). Prefer WAV PCM; else byte heuristics."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".wav", ".wave"}:
        try:
            return _analyze_wav(path)
        except Exception:
            pass
    # Compressed / unknown: byte entropy proxies
    chunks = sample_windows(path, window=8192, count=10)
    if not chunks:
        return 0.5, 0.5
    ents = [normalized_entropy(c) for c in chunks]
    spectral = clamp01(sum(ents) / len(ents))
    # Amplitude proxy: variation across windows
    spreads = []
    for c in chunks:
        if len(c) < 2:
            continue
        vals = list(c)
        mean = sum(vals) / len(vals)
        spreads.append(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) / 128.0)
    amp = clamp01(sum(spreads) / max(len(spreads), 1))
    # Very uniform high entropy compressed streams → elevate risk slightly
    return clamp01(0.55 * spectral + 0.2), clamp01(1.0 - amp * 0.85)


def _analyze_wav(path: str) -> tuple[float, float]:
    import wave

    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        # Cap frames for speed
        take = min(nframes, 16000 * 5)
        raw = wf.readframes(take)

    if sw == 2:
        samples = struct.unpack("<" + "h" * (len(raw) // 2), raw[: (len(raw) // 2) * 2])
        if nch > 1:
            samples = samples[::nch]
        # Normalize
        arr = [s / 32768.0 for s in samples]
    else:
        # Fallback to bytes
        return normalized_entropy(raw), packing_density_score(path)

    if not arr:
        return 0.5, 0.5

    # Simple spectral proxy: frame energy variance + zero-crossing rate
    frame = 512
    energies = []
    zcrs = []
    for i in range(0, len(arr) - frame, frame):
        chunk = arr[i : i + frame]
        energies.append(sum(x * x for x in chunk) / frame)
        zc = sum(1 for a, b in zip(chunk, chunk[1:]) if (a >= 0) != (b >= 0))
        zcrs.append(zc / frame)

    e_mean = sum(energies) / max(len(energies), 1)
    e_var = sum((e - e_mean) ** 2 for e in energies) / max(len(energies), 1)
    z_mean = sum(zcrs) / max(len(zcrs), 1)

    # Synthetic TTS often has unusually steady energy / ZCR
    spectral = clamp01(1.0 - min(1.0, math.sqrt(e_var) * 8.0))
    amplitude = clamp01(abs(z_mean - 0.12) * 5.0)
    return spectral, amplitude


def analyze_video(path: str) -> tuple[float, float, int]:
    """Return (structure_score, consistency_score, windows_sampled)."""
    chunks = sample_windows(path, window=16384, count=12)
    n = max(len(chunks), 1)
    if not chunks:
        return 0.5, 0.5, 0

    ents = [normalized_entropy(c) for c in chunks]
    mean = sum(ents) / len(ents)
    var = sum((e - mean) ** 2 for e in ents) / len(ents)
    structure = clamp01(0.5 * mean + 0.5 * min(1.0, abs(mean - 0.92) * 6))
    consistency = clamp01(min(1.0, var * 25) if var > 0.002 else 0.35 + mean * 0.4)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    # Extremely tiny "videos" or oddly packed containers nudge risk
    if size_mb < 0.05:
        structure = clamp01(structure + 0.15)

    return structure, consistency, n
