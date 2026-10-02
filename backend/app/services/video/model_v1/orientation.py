"""Opt-in orientation-fallback preprocessing for Video Model v1 -- NOT part of the frozen Celeb-DF-v2 parity
code in common.py (see that module's docstring; this module is deliberately kept separate so common.py and
optimized.py stay byte-for-byte what they always were).

Added 2026-10-02 after diagnosing the known Akool/Magic Hour blind-spot clips (CLAUDE.md section 3): one of
the two local ground-truth clips (`fake video 4.mp4`, 480x848) finds a Haar face in 0 of its 16 sampled frames
at the stored orientation, and 16 of 16 after rotating each frame 90 degrees clockwise -- confirmed by hand,
including a visual check that the rotated frames show upright faces (and an upright "Magic Hour" watermark).
The container carries no rotation side-data PyAV or OpenCV can read (`CAP_PROP_ORIENTATION_META` is 0, no
DISPLAYMATRIX side packet), so there is no metadata flag to apply automatically -- this app's export pipeline
appears to just write the frame content sideways. Haar's frontal-face cascade is not rotation-invariant, so
this is a real, fixable detection failure, not a generator-specific blind spot like the ones in CLAUDE.md
section 3's deeper discussion.

This is a *preprocessing* change, not a model change: same checkpoint, same architecture, same crop math
(`common.crop_face_or_center`, untouched) -- the only difference is which rotation of the frame that crop math
runs on.
"""

from __future__ import annotations

import cv2
import numpy as np

_ROTATIONS: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame_bgr: np.ndarray, degrees: int) -> np.ndarray:
    code = _ROTATIONS[degrees]
    return frame_bgr if code is None else cv2.rotate(frame_bgr, code)


def _has_any_face(frames: list[np.ndarray], degrees: int, cascade: cv2.CascadeClassifier) -> bool:
    for f in frames:
        gray = cv2.cvtColor(rotate_frame(f, degrees), cv2.COLOR_BGR2GRAY)
        if len(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))):
            return True
    return False


def _count_faces(frames: list[np.ndarray], degrees: int, cascade: cv2.CascadeClassifier) -> int:
    count = 0
    for f in frames:
        gray = cv2.cvtColor(rotate_frame(f, degrees), cv2.COLOR_BGR2GRAY)
        if len(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))):
            count += 1
    return count


def detect_clip_rotation(frames: list[np.ndarray], cascade: cv2.CascadeClassifier) -> int:
    """Picks the single rotation (0/90/180/270 degrees) applied to every frame in the clip.

    Deliberately conservative: returns 0 (does nothing) the moment the native orientation finds a face in ANY
    sampled frame, so this can only ever change behaviour on a clip where every single one of the 16 sampled
    frames failed to find a face normally -- exactly the observed failure mode, and the only case where
    changing the crop carries no regression risk relative to today's production behaviour (a clip that already
    finds a face is left bit-for-bit alone). Among 90/180/270, picks whichever finds a face in the most frames;
    ties favour the lower degree value.
    """
    if _has_any_face(frames, 0, cascade):
        return 0
    best_degrees, best_hits = 0, 0
    for degrees in (90, 180, 270):
        hits = _count_faces(frames, degrees, cascade)
        if hits > best_hits:
            best_degrees, best_hits = degrees, hits
    return best_degrees
