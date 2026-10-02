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
from app.services.video import cnn_gru_v5
from app.services.video.model_v1 import common as model_v1_common
from app.services.video.model_v1 import optimized as model_v1_optimized
from app.services.video.model_v1 import rotation_aware as model_v1_rotation_aware


@dataclass(frozen=True)
class VideoModelMetadata:
    name: str
    architecture: str
    checkpoint_path: str | None
    validated: bool  # has this backend been empirically checked against labeled video data?


@runtime_checkable
class VideoBackend(Protocol):
    metadata: VideoModelMetadata

    def score(self, path: str, progress=None) -> DetectionResult: ...


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

    def score(self, path: str, progress=None) -> DetectionResult:
        # progress is accepted for interface parity; the byte-entropy heuristic is one quick pass, nothing to report.
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

    confidence is passed through exactly as the verified protocol
    computes it (the raw sigmoid probability). The verdict is banded
    +-0.2 around the evaluated 0.525 threshold via _band_verdict, the
    same margin already used by HeuristicVideoBackend and by the image/
    audio detectors -- added post-Review-2-eve per explicit user request,
    since real-world (non-Celeb-DF-v2) footage regularly lands close to
    0.525 and a forced binary call there reads as false confidence. The
    0.525 threshold value itself is unchanged; only the surrounding
    decision band is new. This does NOT fix a confidently-wrong
    prediction far from the threshold -- it only stops genuinely
    borderline scores from being forced to a hard REAL/FAKE.
    """

    metadata = VideoModelMetadata(
        name="TruthLens Video Model v1 (EfficientNet-B0, epoch 11)",
        architecture="EfficientNet-B0, 16-frame mean-logit pooling",
        checkpoint_path=str(model_v1_common.resolve_checkpoint_path()),
        validated=True,
    )

    def score(self, path: str, progress=None) -> DetectionResult:
        try:
            result = model_v1_optimized.predict(path, progress=progress)
        except ValueError as e:
            # Covers model_v1_common.ShortVideoError (a ValueError
            # subclass) and the plain ValueErrors decode_selected_frames
            # raises for an unopenable file or an unusable frame count --
            # all "this file can't be analyzed", not a server failure.
            raise UnprocessableMediaError(
                "This video could not be analyzed -- it may be too short, "
                "corrupted, or in an unsupported format."
            ) from e

        verdict = _band_verdict(result.probability, model_v1_common.THRESHOLD)

        return DetectionResult(
            verdict=verdict,
            confidence=result.probability,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (video/detector.py) fills in the real elapsed time
            raw_scores={
                "structure": result.probability,
                "windows": len(result.frame_logits),
            },
        )


class CnnGruV5Backend:
    """EXPERIMENTAL, opt-in only -- NOT the production default. CNN+BiGRU temporal head
    trained Sep-Oct 2026 (see eval/video_head_experiments/, CLAUDE.md sections 3/17/18).
    Reuses the same frozen EfficientNet-B0 feature extractor as VideoModelV1Backend, with
    a trained BiGRU head on top -- see app/services/video/cnn_gru_v5.py.

    metadata.validated=False on purpose: the checkpoint's self-reported real-world AUC
    (0.76 on the 10 local ground-truth videos) is NOT an independent measurement -- its
    own checkpoint_selection process used those same 10 videos to pick the epoch. Re-run
    against the live wiring here (eval/video_head_experiments/eval_cnn_rnn.py, pointed at
    this checkpoint) confirms the underlying scores are degenerate: 9 of 10 videos land
    within ~0.06 of 0.0 regardless of true label, and the AUC is propped up almost
    entirely by one fake video scoring 0.99. Select via VIDEO_MODEL_BACKEND=cnn_gru_v5 to
    keep evaluating it against new real-world footage -- do not make it the default
    without a larger, honestly held-out real-world AUC measurement (see CLAUDE.md).
    """

    metadata = VideoModelMetadata(
        name="TruthLens Video CNN+BiGRU v5 (experimental)",
        architecture="EfficientNet-B0 features (frozen) + BiGRU temporal head",
        checkpoint_path=str(cnn_gru_v5.resolve_checkpoint_path()),
        validated=False,
    )

    def score(self, path: str, progress=None) -> DetectionResult:
        # progress is accepted for interface parity; this backend has no per-frame callback yet.
        try:
            result = cnn_gru_v5.predict(path)
        except ValueError as e:
            raise UnprocessableMediaError(
                "This video could not be analyzed -- it may be too short, "
                "corrupted, or in an unsupported format."
            ) from e

        verdict = _band_verdict(result.probability, model_v1_common.THRESHOLD)

        return DetectionResult(
            verdict=verdict,
            confidence=result.probability,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (video/detector.py) fills in the real elapsed time
            raw_scores={
                "structure": result.probability,
                "windows": result.num_frames,
            },
        )


class VideoModelV1RotationAwareBackend:
    """EXPERIMENTAL, opt-in only -- NOT the production default. Identical to VideoModelV1Backend (same
    checkpoint, same architecture, same crop math) except frames are rotated 90/180/270 degrees before the
    face crop when the native orientation finds no face in any of the 16 sampled frames -- see
    app/services/video/model_v1/{orientation,rotation_aware}.py and CLAUDE.md section 19.

    metadata.validated=False until measured against more than the 10 local ground-truth videos: it is known to
    turn a 0-face-found clip into a correctly-cropped one (confirmed by hand on the Akool/Magic Hour blind spot
    in CLAUDE.md section 3), but whether that crop change actually flips the model's verdict toward correct --
    as opposed to just being a better-looking crop that still scores wrong -- needs the broader eval harness
    (backend/eval/), not just this one clip, before this could be considered for the default.
    """

    metadata = VideoModelMetadata(
        name="TruthLens Video Model v1 (EfficientNet-B0, epoch 11) + orientation fallback",
        architecture="EfficientNet-B0, 16-frame mean-logit pooling, Haar rotation fallback",
        checkpoint_path=str(model_v1_common.resolve_checkpoint_path()),
        validated=False,
    )

    def score(self, path: str, progress=None) -> DetectionResult:
        # progress is accepted for interface parity; this backend has no per-frame callback yet.
        try:
            result = model_v1_rotation_aware.predict(path)
        except ValueError as e:
            raise UnprocessableMediaError(
                "This video could not be analyzed -- it may be too short, "
                "corrupted, or in an unsupported format."
            ) from e

        verdict = _band_verdict(result.probability, model_v1_common.THRESHOLD)

        return DetectionResult(
            verdict=verdict,
            confidence=result.probability,
            model_used=self.metadata.name,
            processing_time_ms=0.0,  # caller (video/detector.py) fills in the real elapsed time
            raw_scores={
                "structure": result.probability,
                "windows": len(result.frame_logits),
                "rotation_degrees": result.rotation_degrees,
            },
        )


_BACKENDS: dict[str, type[VideoBackend]] = {
    "heuristic": HeuristicVideoBackend,
    "model_v1": VideoModelV1Backend,
    "cnn_gru_v5": CnnGruV5Backend,
    "model_v1_rotation_aware": VideoModelV1RotationAwareBackend,
    # Register Video Model v2 here once it exists, e.g.:
    # "model_v2": VideoModelV2Backend,
}


def get_video_backend() -> VideoBackend:
    name = os.getenv("VIDEO_MODEL_BACKEND", "model_v1")
    backend_cls = _BACKENDS.get(name, VideoModelV1Backend)
    return backend_cls()
