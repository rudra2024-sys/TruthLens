"""Swappable video-model backend (Review-2 prep, Priority 6).

Video Model v1 (EfficientNet-B0, epoch 11) is now the active production
backend -- see VideoModelV1Backend below. It was verified line-by-line
against the recovered original Celeb-DF-v2 evaluator (frame sampling,
Haar face detection, 20% padding, center-square/empty-crop fallback, PIL
BILINEAR resize, ImageNet normalization, mean-logit pooling, sigmoid,
threshold 0.525) in app/services/video/model_v1/, and its optimized
inference path was benchmarked against a naive baseline with 100%
prediction agreement. The old byte-entropy heuristic
(services/heuristics.py::analyze_video) is kept registered and selectable
via VIDEO_MODEL_BACKEND=heuristic as a fallback, but is no longer the
default.

This file is the seam: run_video_detection() calls get_video_backend()
instead of calling a scorer directly, and every backend returns the
DetectionResult shape from detection_interface.py. A future Video Model
v2 can be added as one more class here and switched on via the
VIDEO_MODEL_BACKEND env var, WITHOUT touching detect.py, the DB-writing
code in video/detector.py, or the frontend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.config import settings
from app.services.detection_errors import UnprocessableMediaError
from app.services.detection_interface import DetectionResult
from app.services.heuristics import analyze_video, clamp01
from app.services.video.model_v1 import common as model_v1_common
from app.services.video.model_v1 import optimized as model_v1_optimized


@dataclass(frozen=True)
class VideoModelMetadata:
    name: str
    architecture: str
    checkpoint_path: str | None
    validated: bool  # has this backend been empirically checked against labeled video data?


@runtime_checkable
class VideoBackend(Protocol):
    metadata: VideoModelMetadata

    def score(self, path: str) -> DetectionResult: ...


def _band_verdict(score: float, threshold: float, margin: float = 0.2) -> str:
    if score >= threshold + margin:
        return "FAKE"
    if score <= threshold - margin:
        return "REAL"
    return "UNCERTAIN"


class HeuristicVideoBackend:
    """Current production default. Explicitly NOT a trained model -- see
    CLAUDE.md Sec 3 and the Sep 2026 audit. Behavior is unchanged from the
    pre-existing run_video_detection: same 0.75/0.25 weighting, same
    threshold band, same model_used label."""

    metadata = VideoModelMetadata(
        name="Video forensic heuristics v1",
        architecture="byte-entropy heuristic (no neural network)",
        checkpoint_path=None,
        validated=False,
    )

    def score(self, path: str) -> DetectionResult:
        structure, consistency, windows = analyze_video(path)
        confidence = clamp01(0.75 * structure + 0.25 * consistency)
        verdict = _band_verdict(confidence, settings.FAKE_THRESHOLD)
        return DetectionResult(
            verdict=verdict,
            confidence=confidence,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (video/detector.py) fills in the real elapsed time
            raw_scores={
                "structure": structure,
                "consistency": consistency,
                "windows": windows,
            },
        )


class VideoModelV1Backend:
    """Production backend: EfficientNet-B0 (epoch 11), verified against
    the recovered original Celeb-DF-v2 evaluator -- see
    app/services/video/model_v1/. Delegates all preprocessing and
    inference to model_v1.optimized.predict(); this class only adapts its
    result to the DetectionResult shape run_video_detection() expects, it
    does not reimplement any decoding/model logic itself. The model is
    loaded once per process and cached by optimized.py, not reloaded per
    request.

    verdict/confidence are passed through exactly as the verified
    protocol computes them (probability >= 0.525 -> FAKE, else REAL) --
    no added margin/UNCERTAIN banding, unlike HeuristicVideoBackend's
    _band_verdict, since that would be new decision logic the evaluation
    never covered.
    """

    metadata = VideoModelMetadata(
        name="TruthLens Video Model v1 (EfficientNet-B0, epoch 11)",
        architecture="EfficientNet-B0, 16-frame mean-logit pooling",
        checkpoint_path=str(model_v1_common.resolve_checkpoint_path()),
        validated=True,
    )

    def score(self, path: str) -> DetectionResult:
        try:
            result = model_v1_optimized.predict(path)
        except ValueError as e:
            # Covers model_v1_common.ShortVideoError (a ValueError
            # subclass) and the plain ValueErrors decode_selected_frames
            # raises for an unopenable file or an unusable frame count --
            # all "this file can't be analyzed", not a server failure.
            raise UnprocessableMediaError(
                "This video could not be analyzed -- it may be too short, "
                "corrupted, or in an unsupported format."
            ) from e

        return DetectionResult(
            verdict=result.verdict,
            confidence=result.probability,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (video/detector.py) fills in the real elapsed time
            raw_scores={
                "structure": result.probability,
                "windows": len(result.frame_logits),
            },
        )


_BACKENDS: dict[str, type[VideoBackend]] = {
    "heuristic": HeuristicVideoBackend,
    "model_v1": VideoModelV1Backend,
    # Register Video Model v2 here once it exists, e.g.:
    # "model_v2": VideoModelV2Backend,
}


def get_video_backend() -> VideoBackend:
    name = os.getenv("VIDEO_MODEL_BACKEND", "model_v1")
    backend_cls = _BACKENDS.get(name, VideoModelV1Backend)
    return backend_cls()
