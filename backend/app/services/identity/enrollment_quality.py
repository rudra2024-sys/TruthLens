"""Enrollment photo quality gate (added 2026-10-11). A bad reference photo silently degrades
every future identity-match check against it with no warning to the user -- this catches the
obvious cases before a reference is ever stored, using signal already computed by
IdentityPipeline.detect_faces() (no second MTCNN pass) plus a standard, dependency-free blur
check via OpenCV (already a dependency -- see head_pose.py's own note on this).

All four thresholds below are placeholders based on general photo-quality heuristics, not
empirically tuned against a labeled dataset of real enrollment photos -- same honesty tier as
IDENTITY_MATCH_THRESHOLD and the monitoring-signal thresholds in config.py. Deliberately checked
independently (not short-circuited) so a caller gets every issue at once, not one at a time
across repeated failed uploads.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings


def _blur_variance(image_path: str, box: list[float] | None) -> float:
    """Variance of the Laplacian over the face crop (or the whole image if no box) -- the
    standard, well-established no-training-required blur-detection heuristic: a sharp image has
    strong edges in many directions (high variance), a blurry one doesn't."""
    with Image.open(image_path) as image:
        image = image.convert("L")  # grayscale
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 > x1 and y2 > y1:
                image = image.crop((x1, y1, x2, y2))
        gray = np.array(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_enrollment_quality(image_path: str, detection: dict) -> list[str]:
    """detection: the dict returned by IdentityPipeline.detect_faces() (count/landmarks/box/
    image_size). Returns a list of human-readable issue strings -- empty means the photo passes.
    Multiple issues can be returned together so a resubmission doesn't have to fail repeatedly on
    one problem at a time."""
    issues: list[str] = []

    count = detection["count"]
    box = detection["box"]
    image_width, image_height = detection["image_size"]

    if count > 1:
        issues.append(
            "More than one face was detected in this photo. Use a photo with only you in it."
        )
        # Can't meaningfully check framing/blur against a single face when there isn't one --
        # the caller still gets a clear, actionable reason to resubmit.
        return issues

    if count == 1 and box is not None:
        x1, y1, x2, y2 = box
        face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        image_area = image_width * image_height
        if image_area > 0 and face_area / image_area < settings.ENROLLMENT_MIN_FACE_AREA_FRACTION:
            issues.append(
                "Your face is too small in this photo. Move closer or crop tighter."
            )

        margin = settings.ENROLLMENT_EDGE_MARGIN_PX
        if (
            x1 <= margin or y1 <= margin
            or x2 >= image_width - margin or y2 >= image_height - margin
        ):
            issues.append(
                "Your face is cut off at the edge of this photo. Center it with margin on all sides."
            )

    # Blur check runs whenever this isn't the multi-face case, cropped to the face box when one
    # was found above detect_faces()'s stricter 0.95 confidence bar, or the whole image otherwise
    # -- deliberately NOT nested inside the count==1 branch above (see enroll_reference's call
    # site for why: a blurry photo can fail that stricter bar, i.e. count==0/box is None, while
    # still producing a usable embedding via the caller's separate, more lenient _embed() call,
    # and this check must still catch it).
    if _blur_variance(image_path, box) < settings.ENROLLMENT_MIN_BLUR_VARIANCE:
        issues.append(
            "This photo looks too blurry. Use a sharper, well-focused photo."
        )

    return issues
