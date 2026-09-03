"""
Future detector interface — NOT YET WIRED IN.

This module exists to sketch a common shape that image/video/audio detectors
and a future fusion/ensemble layer could converge on. It is intentionally
*not* imported by main.py, detect.py, or any of the three live detectors
(services/image/detector.py, services/video/detector.py,
services/audio/detector.py) — none of their behavior changes as a result of
this file existing.

Why not wire it in now: each live detector currently returns a different,
bespoke shape (see their individual DB writes in detect.py's three branches),
and detect.py's dispatch is simple if/elif routing rather than a registry.
Making the real detectors conform to DetectionResult below, and turning
detect.py's dispatch into something that uses this interface, touches
working, currently-correct code across four files simultaneously. That is a
refactor with real regression risk, not an additive change, so it has been
deliberately deferred rather than done under this pass. See CLAUDE.md Phase 5
notes for the same conclusion.

Suggested follow-up (separate, explicitly-approved task):
1. Wrap (not rewrite) each existing detector so it returns a `DetectionResult`
   alongside whatever it already writes to the DB.
2. Only then consider a `FusionDetector` that takes several `DetectionResult`s
   (e.g. image + a future second image model) and combines them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class DetectionResult:
    """Common shape a detector's raw output could be normalized to."""

    verdict: str  # "REAL" | "FAKE" | "UNCERTAIN" — the vocabulary the DB/API already use
    confidence: float  # 0.0-1.0
    model_used: str
    processing_time_ms: float
    raw_scores: dict = field(default_factory=dict)  # detector-specific extra fields


@runtime_checkable
class ImageDetector(Protocol):
    def detect(self, file_path: str) -> DetectionResult: ...


@runtime_checkable
class AudioDetector(Protocol):
    def detect(self, file_path: str) -> DetectionResult: ...


@runtime_checkable
class VideoDetector(Protocol):
    def detect(self, file_path: str) -> DetectionResult: ...


@runtime_checkable
class FusionDetector(Protocol):
    """Future ensemble layer: combine multiple DetectionResults for the same
    file (e.g. two independent image models) into one final verdict."""

    def fuse(self, results: list[DetectionResult]) -> DetectionResult: ...
