"""Short-term-energy voice-activity-detection for the monitoring session's audio signal (item
7, added 2026-10-09) -- "talking/speech detected," deliberately not "another voice detected":
classic energy-based VAD can tell you speech-level audio activity happened, it cannot tell you
whose voice it was (that would need a trained speaker-verification model, out of scope here).
During an exam, sustained talking at all is itself worth flagging regardless of whose voice it
is, so this is still a genuinely useful signal honestly described, not an overclaim.

No new dependency -- reuses app.services.models.audio_model.decode_audio_mono_16k (PyAV, already
a requirement) for container decoding, same as the AASIST audio-deepfake detector does.
"""

from __future__ import annotations

import numpy as np

from app.services.models.audio_model import decode_audio_mono_16k

FRAME_MS = 25
FRAME_SAMPLES = int(16000 * FRAME_MS / 1000)

# How many times the clip's own noise floor a frame's RMS energy must exceed to count as
# speech. A placeholder multiplier -- classic adaptive-threshold VAD, not validated against
# labeled speech/silence recordings from this exact microphone/room setup.
NOISE_FLOOR_MULTIPLIER = 3.0


def speech_ratio(audio_path: str) -> float:
    """Fraction of ~25ms frames classified as speech, 0.0-1.0. Adaptive per-clip threshold
    (NOISE_FLOOR_MULTIPLIER x the clip's own 10th-percentile frame energy) rather than a fixed
    absolute energy cutoff, since raw energy depends heavily on microphone gain/distance, which
    varies per device -- an adaptive threshold is at least self-consistent within one clip even
    without labeled data to calibrate an absolute one."""
    samples = decode_audio_mono_16k(audio_path)
    if samples.size < FRAME_SAMPLES:
        return 0.0

    n_frames = samples.size // FRAME_SAMPLES
    framed = samples[: n_frames * FRAME_SAMPLES].reshape(n_frames, FRAME_SAMPLES)
    energy = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1))

    noise_floor = max(float(np.percentile(energy, 10)), 1e-6)
    threshold = noise_floor * NOISE_FLOOR_MULTIPLIER
    speech_frames = int(np.sum(energy > threshold))
    return speech_frames / n_frames
