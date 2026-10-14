"""Unit tests for the motion-based liveness heuristic (added 2026-10-11, item 9) -- pure
function, synthetic landmark/box data, no pipeline or DB needed.
"""

from app.services.identity.liveness import assess_motion_liveness

_BOX = [200.0, 150.0, 400.0, 400.0]  # 200x250 box -> diagonal ~320.2
_LANDMARKS = [
    [250.0, 200.0], [350.0, 200.0], [300.0, 250.0], [260.0, 320.0], [340.0, 320.0],
]
_EPS = 0.004  # same as config.py's default


def _shift(landmarks, dx, dy):
    return [[x + dx, y + dy] for x, y in landmarks]


def test_identical_landmarks_across_checks_are_flagged_static():
    history = [(_LANDMARKS, _BOX) for _ in range(3)]
    assert assess_motion_liveness(history, _EPS) is True


def test_naturally_varying_landmarks_are_not_flagged():
    # ~3px shifts on a ~320px diagonal box -> ~0.01 normalized displacement, well above epsilon.
    history = [
        (_LANDMARKS, _BOX),
        (_shift(_LANDMARKS, 3.0, -2.0), _BOX),
        (_shift(_LANDMARKS, -1.0, 4.0), _BOX),
    ]
    assert assess_motion_liveness(history, _EPS) is False


def test_one_large_movement_among_otherwise_static_frames_is_not_flagged():
    history = [
        (_LANDMARKS, _BOX),
        (_LANDMARKS, _BOX),
        (_shift(_LANDMARKS, 50.0, 0.0), _BOX),
    ]
    assert assess_motion_liveness(history, _EPS) is False


def test_fewer_than_two_entries_is_never_flagged():
    assert assess_motion_liveness([], _EPS) is False
    assert assess_motion_liveness([(_LANDMARKS, _BOX)], _EPS) is False


def test_zero_area_box_is_not_flagged():
    degenerate_box = [100.0, 100.0, 100.0, 100.0]
    history = [(_LANDMARKS, degenerate_box), (_LANDMARKS, degenerate_box)]
    assert assess_motion_liveness(history, _EPS) is False
