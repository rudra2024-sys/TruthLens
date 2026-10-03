"""Tests for the audio robustness eval harness (CLAUDE.md section 24). This harness has NOT been run against
real labeled audio -- no labeled audio deepfake dataset exists locally (see eval/perturbations_audio.py's
module docstring). These tests confirm the MECHANISM works (shapes, no crashes, no degenerate output), not
that the deployed model is or isn't robust -- that question is still open.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from eval import perturbations_audio as PA
from tests.conftest import BACKEND

CKPT = Path(os.getenv("AUDIO_MODEL_V1_CHECKPOINT") or BACKEND / "checkpoints" / "audio" / "wav2vec2_v9.pt")


def _tone(seconds=2.0, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.shape[0])).astype(np.float32)


def test_settings_includes_clean_and_every_family_level():
    s = PA.settings()
    assert s[0] == ("clean", 0)
    families = {fam for fam, _ in s}
    assert families == {"clean", "noise", "volume", "codec"}


def test_noise_and_volume_preserve_length_and_dtype():
    samples = _tone()
    for family, level in [("noise", 0.02), ("volume", 0.3), ("volume", 2.0)]:
        out = PA.apply(family, level, samples, 16000, key="x")
        assert out.shape == samples.shape and out.dtype == np.float32


def test_volume_scales_amplitude_in_the_right_direction():
    samples = _tone()
    quiet = PA.apply("volume", 0.3, samples, 16000)
    loud = PA.apply("volume", 2.0, samples, 16000)
    assert np.abs(quiet).mean() < np.abs(samples).mean() < np.abs(loud).mean()


def test_noise_is_deterministic_per_key():
    samples = _tone()
    a = PA.apply("noise", 0.02, samples, 16000, key="same.wav")
    b = PA.apply("noise", 0.02, samples, 16000, key="same.wav")
    c = PA.apply("noise", 0.02, samples, 16000, key="different.wav")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_codec_round_trip_preserves_length_and_stays_non_degenerate():
    """Real MP3 re-encode via PyAV -- confirms ffmpeg/libmp3lame actually works in this environment, not just
    that the function returns something."""
    samples = _tone(seconds=1.0)
    out = PA.codec(samples, 16000, 32)
    assert out.shape == samples.shape
    assert np.abs(out).mean() > 1e-4          # not silence
    assert not np.array_equal(out, samples)   # actually lossy, not a no-op


def test_clean_is_a_true_noop():
    samples = _tone()
    assert np.array_equal(PA.apply("clean", 0, samples, 16000), samples)


def test_harness_runs_end_to_end_on_synthetic_audio_without_crashing(tmp_path):
    """Smoke test only -- synthetic sine-tone audio looks like neither genuine nor spoof speech to the model,
    so the resulting scores say nothing about real robustness. This just confirms the plumbing (decode,
    windowing, model forward pass under every perturbation setting) doesn't crash and produces sane,
    non-NaN, non-degenerate probabilities."""
    if not CKPT.is_file():
        pytest.skip("audio checkpoint not present")
    import csv
    import subprocess
    import sys

    import wave

    manifest = tmp_path / "manifest.csv"
    wav_path = tmp_path / "a.wav"
    samples = _tone(seconds=3.0)
    pcm16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16.tobytes())
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["path", "label"])
        wtr.writerow([str(wav_path), 0])

    out = tmp_path / "pred.csv"
    env = dict(os.environ, AUDIO_MODEL_V1_DEVICE="cpu")
    result = subprocess.run(
        [sys.executable, "-m", "eval.run_audio_robustness", "--manifest", str(manifest), "--out", str(out)],
        cwd=str(BACKEND), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "(decode failures: 0)" in result.stdout

    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert len(rows) == len(PA.settings())
    probs = [float(r["audio_fake"]) for r in rows]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert len(set(probs)) > 1   # not every setting collapsed to the identical score
