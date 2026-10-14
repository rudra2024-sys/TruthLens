"""Quality gate for government-ID-card photos (added 2026-10-12). Deliberately much narrower
than enrollment_quality.py's selfie gate -- an ID card's face photo is a small inset, not a
frame-filling selfie, so ENROLLMENT_MIN_FACE_AREA_FRACTION / ENROLLMENT_EDGE_MARGIN_PX don't
transfer and are skipped outright here rather than replaced with an untuned guess (no real
Aadhaar/PAN sample photos were available to calibrate against this session -- revisit with real
samples). Keeps only: exactly one face found, and the same ENROLLMENT_MIN_BLUR_VARIANCE blur bar
as the selfie gate (a focus property, not a framing one -- reusing enrollment_quality.py's
_blur_variance directly, same cross-module reuse convention detector.py already uses for this
file's sibling checks).
"""

from __future__ import annotations

from app.core.config import settings
from app.services.identity.enrollment_quality import _blur_variance


def assess_government_id_quality(image_path: str, detection: dict) -> list[str]:
    """detection: the dict returned by IdentityPipeline.detect_faces() (count/landmarks/box/
    image_size). Returns a list of human-readable issue strings -- empty means the photo passes."""
    issues: list[str] = []

    count = detection["count"]
    box = detection["box"]

    if count > 1:
        return ["More than one face was detected in this photo. Use a clear photo of just the ID card."]

    if _blur_variance(image_path, box) < settings.ENROLLMENT_MIN_BLUR_VARIANCE:
        issues.append("This photo looks too blurry. Use a sharper, well-lit photo of the ID card.")

    return issues
