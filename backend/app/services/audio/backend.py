"""Swappable audio-model backend, mirroring services/video/backend.py's pattern.

Audio Model v1 (wav2vec2-base, fine-tuned for voice-cloning/deepfake detection)
is now the active production backend -- see AudioModelV1Backend below. It was
trained and validated across 5/6 held-out test sets including a 59-system
unseen-voice-conversion stress test (98.3%) and fresh unseen-speaker genuine
audio (10/10) -- see CLAUDE.md and app/services/audio/model_v1/common.py.
AASIST (2019-vintage ASVspoof, the prior production model) is kept registered
and selectable via AUDIO_MODEL_BACKEND=aasist as a fallback, but is no longer
the default.

This file is the seam: run_audio_detection() calls get_audio_backend() instead
of calling a scorer directly, and every backend returns the DetectionResult
shape from detection_interface.py. A future Audio Model v2 can be added as one
more class here and switched on via the AUDIO_MODEL_BACKEND env var, WITHOUT
touching detect.py, the DB-writing code in audio/detector.py, or the frontend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import settings
from app.services.detection_errors import UnprocessableMediaError
from app.services.detection_interface import DetectionResult
from app.services.models.audio_model import score_audio_path
from app.services.audio.model_v1 import common as model_v1_common
from app.services.audio.model_v1 import optimized as model_v1_optimized


@dataclass(frozen=True)
class AudioModelMetadata:
    name: str
    architecture: str
    checkpoint_path: str | None
    validated: bool  # has this backend been empirically checked against labeled audio data?


@runtime_checkable
class AudioBackend(Protocol):
    metadata: AudioModelMetadata

    def score(self, path: str) -> DetectionResult: ...


def _band_verdict(score: float, threshold: float, margin: float = 0.2) -> str:
    if score >= threshold + margin:
        return "FAKE"
    if score <= threshold - margin:
        return "REAL"
    return "UNCERTAIN"


class AasistBackend:
    """Prior production default (2019-vintage ASVspoof2019-LA, binary,
    unvalidated bonafide/spoof label ordering -- see the SPOOF_CLASS_INDEX
    comment in app/services/models/audio_model.py). Kept as an explicit
    fallback, selectable via AUDIO_MODEL_BACKEND=aasist, for instant
    rollback -- behavior is unchanged from the pre-swap run_audio_detection."""

    metadata = AudioModelMetadata(
        name="AASIST",
        architecture="AASIST (spectro-temporal graph attention), ASVspoof2019-LA",
        checkpoint_path=None,  # bundled ONNX weight, not an env-configurable checkpoint
        validated=False,  # never checked against labeled bonafide/spoof data, see audio_model.py
    )

    def score(self, path: str) -> DetectionResult:
        try:
            result = score_audio_path(path)
        except ValueError as e:
            raise UnprocessableMediaError(
                "The uploaded file could not be read as audio. "
                "It may be corrupt or in an unsupported format."
            ) from e

        spoof_probability = float(result.get("spoof_probability", 0.0))
        verdict = _band_verdict(spoof_probability, settings.FAKE_THRESHOLD)

        return DetectionResult(
            verdict=verdict,
            confidence=spoof_probability,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (audio/detector.py) fills in the real elapsed time
            raw_scores={
                "mean": spoof_probability,
                "std": float(result.get("spoof_probability_std", 0.0)),
                "windows": result.get("windows_analyzed"),
                "duration_s": result.get("duration_s"),
            },
        )


class AudioModelV1Backend:
    """Production backend: wav2vec2-base fine-tuned for voice-cloning/deepfake
    detection -- see app/services/audio/model_v1/. Delegates all preprocessing
    and inference to model_v1.optimized.predict(); this class only adapts its
    result to the DetectionResult shape run_audio_detection() expects. The
    model is loaded once per process and cached by optimized.py, not reloaded
    per request.

    Verdict uses the same shared FAKE_THRESHOLD + band-margin convention as
    every other detector (image/video/the prior AASIST path) -- confidence
    calibration for this specific model is a separate, not-yet-done follow-up,
    not something this integration assumes.
    """

    metadata = AudioModelMetadata(
        name="TruthLens Audio Model v1 (wav2vec2-base)",
        architecture="wav2vec2-base backbone, mean-pooled, linear head",
        checkpoint_path=str(model_v1_common.resolve_checkpoint_path()),
        validated=True,
    )

    def score(self, path: str) -> DetectionResult:
        try:
            result = model_v1_optimized.predict(path)
        except ValueError as e:
            # Covers decode_audio_mono_16k's ValueErrors (no audio stream,
            # nothing decodable) -- "this file can't be analyzed", not a
            # server failure.
            raise UnprocessableMediaError(
                "The uploaded file could not be read as audio. "
                "It may be corrupt or in an unsupported format."
            ) from e

        verdict = _band_verdict(result.deepfake_probability_mean, settings.FAKE_THRESHOLD)

        return DetectionResult(
            verdict=verdict,
            confidence=result.deepfake_probability_mean,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (audio/detector.py) fills in the real elapsed time
            raw_scores={
                "mean": result.deepfake_probability_mean,
                "std": result.deepfake_probability_std,
                "windows": result.windows_analyzed,
                "duration_s": result.duration_s,
            },
        )


_BACKENDS: dict[str, type[AudioBackend]] = {
    "aasist": AasistBackend,
    "model_v1": AudioModelV1Backend,
    # Register Audio Model v2 here once it exists, e.g.:
    # "model_v2": AudioModelV2Backend,
}


def get_audio_backend() -> AudioBackend:
    name = os.getenv("AUDIO_MODEL_BACKEND", "model_v1")
    backend_cls = _BACKENDS.get(name, AudioModelV1Backend)
    return backend_cls()
