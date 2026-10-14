"""Motion-based liveness heuristic (added 2026-10-11, item 9).

NOT blink detection. MTCNN's 5-point landmarks (left eye, right eye, nose, left mouth corner,
right mouth corner -- see pipelines/identity/inference.py) have no eyelid points, so a true
blink signal isn't computable from them; there is no training infrastructure in this environment
to build a trained anti-spoof model either (same constraint as every other ML item in this
codebase). What IS buildable from data already being recorded: frame-to-frame landmark motion
across consecutive checks in the same session. A printed photo held in front of the camera shows
near-zero motion between checks; a live face shows natural micro-movement even when someone is
trying to hold still. Real, imperfect signal, same honesty tier as head_pose.py/audio_vad.py: a
genuinely still real face can occasionally trigger it, and a photo subtly joggled by whoever is
holding it can evade it.
"""

from __future__ import annotations

import math


def _box_diagonal(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def _max_displacement(landmarks_a, landmarks_b) -> float:
    return max(
        math.hypot(ax - bx, ay - by)
        for (ax, ay), (bx, by) in zip(landmarks_a, landmarks_b)
    )


def assess_motion_liveness(history: list[tuple[list, list]], epsilon_fraction: float) -> bool:
    """history: consecutive single-face checks' (landmarks, box) pairs, oldest first, same
    session. Returns True ("looks static, possibly a held-up photo") only when every
    consecutive pair's max per-landmark displacement -- normalized by that pair's face-box
    diagonal, so it's scale-invariant to how close the person is to the camera -- stays below
    epsilon_fraction. Needs at least 2 entries to compare anything; fewer always returns False
    (not enough history to judge, not a claim of liveness either)."""
    if len(history) < 2:
        return False

    for (landmarks_a, box_a), (landmarks_b, box_b) in zip(history, history[1:]):
        diagonal = (_box_diagonal(box_a) + _box_diagonal(box_b)) / 2
        if diagonal <= 0:
            return False
        normalized = _max_displacement(landmarks_a, landmarks_b) / diagonal
        if normalized >= epsilon_fraction:
            return False
    return True
