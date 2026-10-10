"""Coarse head-pose (yaw/pitch) estimation from MTCNN's 5-point landmarks (item 10, added
2026-10-09). A geometric estimate, not a trained model -- no new dependency, reuses OpenCV
(already a dependency via opencv-python-headless) and the landmarks IdentityPipeline.detect_faces
already computes as part of its existing MTCNN forward pass.

Validation status, stated plainly: OpenCV's default solvePnP method (SOLVEPNP_ITERATIVE) needs
at least 6 points and MTCNN only gives 5 -- confirmed directly (it raises). SOLVEPNP_EPNP
supports 5 and was confirmed to run correctly and move in the geometrically expected direction
on hand-constructed synthetic landmark shifts. It was NOT validated against real photos at known
head angles (no labeled head-pose dataset exists locally) -- same "measure, don't assume"
honesty tier as IDENTITY_MATCH_THRESHOLD in config.py. The self-calibrating baseline design in
monitoring.py (compare each session to its own first good reading, not a universal angle cutoff)
exists specifically because of this validation gap, not just as a nicety.
"""

from __future__ import annotations

import numpy as np
import cv2

# Generic 3D face-model points (mm, arbitrary unit -- only relative scale matters for solvePnP),
# order matching MTCNN's own 5-point landmark order: left_eye, right_eye, nose_tip,
# left_mouth_corner, right_mouth_corner.
_MODEL_POINTS = np.array([
    (-30.0, 30.0, -30.0),
    (30.0, 30.0, -30.0),
    (0.0, 0.0, 0.0),
    (-25.0, -30.0, -20.0),
    (25.0, -30.0, -20.0),
], dtype=np.float64)


def estimate_yaw_pitch(landmarks: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
    """landmarks: (5,2) array in MTCNN's order. image_size: (width, height) of the source image,
    used to build a plausible camera matrix (focal length approximated from image width, a
    standard simplification when the real camera's intrinsics aren't known). Returns (yaw_deg,
    pitch_deg)."""
    width, height = image_size
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    image_points = np.asarray(landmarks, dtype=np.float64)
    ok, rvec, _tvec = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not ok:
        raise RuntimeError("solvePnP failed to converge on these landmarks")

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return float(yaw), float(pitch)


def mouth_width(landmarks: np.ndarray) -> float:
    """Euclidean distance between the two mouth-corner landmarks (indices 3, 4) -- a weak,
    width-only proxy for mouth activity. See this module's and monitoring.py's docstrings for
    why this is informational only (item 9), never auto-flagged."""
    left_mouth, right_mouth = landmarks[3], landmarks[4]
    return float(np.linalg.norm(np.asarray(left_mouth) - np.asarray(right_mouth)))
