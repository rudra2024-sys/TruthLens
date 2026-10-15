"""Tests for the opt-in orientation-fallback video backend (CLAUDE.md section 19).

This backend is opt-in only (VIDEO_MODEL_BACKEND=model_v1_rotation_aware), never the default -- these tests
guard that wiring and the conservative "only touch clips where native orientation finds nothing" behaviour,
not a general claim about accuracy. The checkpoint-dependent tests skip themselves when the checkpoint or the
local ground-truth videos are absent, matching the pattern in test_models.py / test_video_cnn_gru_v5.py.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.video.model_v1 import common as model_v1_common
from app.services.video.model_v1 import orientation
from tests.conftest import BACKEND, TEST_DATA

# Mirrors resolve_checkpoint_path()'s own fallback so this always agrees with whatever checkpoint
# the backend-under-test picks up implicitly (VideoModelV1RotationAwareBackend.score() calls
# predict() with no explicit checkpoint) -- a hardcoded path here would silently diverge from the
# production default whenever that default changes (see CLAUDE.md section 27).
CKPT = model_v1_common.resolve_checkpoint_path()


# ---------------------------------------------------------------- orientation.py, no model/checkpoint needed

def _frontal_face_image_bgr():
    """A face photo from the repo's own test data, as a BGR array -- not a synthetic pattern, since Haar needs
    real face-like structure to fire. image4.png is pinned because it reliably detects exactly one face at the
    native orientation (confirmed by hand), which keeps these tests deterministic."""
    path = TEST_DATA / "chatgpt_everyday" / "image4.png"
    if not path.is_file():
        pytest.skip("sample face image not present")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip("sample face image could not be decoded")
    return img


def test_rotate_frame_0_degrees_is_a_noop():
    frame = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    assert np.array_equal(orientation.rotate_frame(frame, 0), frame)


def test_rotate_frame_round_trip():
    frame = np.random.default_rng(0).integers(0, 255, (10, 16, 3), dtype=np.uint8)
    back = orientation.rotate_frame(orientation.rotate_frame(frame, 90), 270)
    assert np.array_equal(back, frame)


def test_detect_clip_rotation_is_0_when_native_orientation_already_finds_a_face():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = _frontal_face_image_bgr()
    assert orientation.detect_clip_rotation([face] * 3, cascade) == 0


def test_detect_clip_rotation_finds_the_rotation_that_recovers_the_face():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face = _frontal_face_image_bgr()
    # Rotate the upright face 90 degrees clockwise so it is no longer upright; the fallback should find that
    # rotating it 270 degrees (counter-clockwise) recovers an upright, detectable face.
    sideways = orientation.rotate_frame(face, 90)
    assert orientation.detect_clip_rotation([sideways] * 3, cascade) == 270


def test_detect_clip_rotation_defaults_to_0_when_no_rotation_finds_a_face():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert orientation.detect_clip_rotation([blank] * 3, cascade) == 0


# ---------------------------------------------------------------- backend.py wiring

def test_default_backend_is_unaffected_by_registering_rotation_aware(monkeypatch):
    monkeypatch.delenv("VIDEO_MODEL_BACKEND", raising=False)
    from app.services.video.backend import VideoModelV1Backend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1Backend)


def test_rotation_aware_backend_is_selectable_via_env_var(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL_BACKEND", "model_v1_rotation_aware")
    from app.services.video.backend import VideoModelV1RotationAwareBackend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1RotationAwareBackend)


def test_rotation_aware_backend_metadata_marks_itself_unvalidated():
    """Guards against silently flipping this to the default before a broader eval -- see CLAUDE.md section 19."""
    from app.services.video.backend import VideoModelV1RotationAwareBackend
    assert VideoModelV1RotationAwareBackend.metadata.validated is False


# ---------------------------------------------------------------- real checkpoint + real sample videos

@pytest.mark.parametrize("path", [r"C:\fake photos\fake video 1.mp4", r"C:\real photos\real video 1.mp4"])
def test_rotation_aware_is_bit_identical_to_v1_when_native_orientation_finds_a_face(path):
    """The whole point of the conservative guard: a clip that already works today must be untouched."""
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized, rotation_aware

    plain = optimized.predict(path, str(CKPT))
    rotated = rotation_aware.predict(path, str(CKPT))

    assert rotated.rotation_degrees == 0
    assert rotated.probability == pytest.approx(plain.probability, abs=1e-6)
    assert rotated.frame_logits == pytest.approx(plain.frame_logits, abs=1e-6)


def test_rotation_aware_recovers_the_known_blind_spot_clip():
    """Regression pin for the diagnosed case (CLAUDE.md section 19): `fake video 4.mp4` finds 0/16 faces at
    its native orientation and is pushed toward REAL by the resulting center-crop; the rotation fallback should
    both find faces and move the raw probability toward (not necessarily across any particular band) FAKE.

    This is a historical fact about the epoch-11 checkpoint specifically (measured 2026-10-02, CLAUDE.md
    section 19) -- not a general claim that holds under any checkpoint. Round 4 (the new production default,
    section 27) was retrained on different data and, measured here, moves the OTHER way on this clip (plain
    0.300 -> rotated 0.132): the rotation fix's benefit on this file does not carry over to round 4's weights.
    Section 27 itself flags that sections 19-22's numbers are unverified for round 4. Pinned to epoch_11
    explicitly (not CKPT/the live default) so this keeps testing the original, still-true epoch-11 finding
    rather than silently asserting something false about round 4.
    """
    epoch_11_ckpt = BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt"
    path = r"C:\fake photos\fake video 4.mp4"
    if not epoch_11_ckpt.is_file() or not os.path.isfile(path):
        pytest.skip("epoch-11 checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized, rotation_aware

    plain = optimized.predict(path, str(epoch_11_ckpt))
    rotated = rotation_aware.predict(path, str(epoch_11_ckpt))

    assert rotated.rotation_degrees != 0
    assert rotated.probability > plain.probability


def test_rotation_aware_backend_matches_direct_module_call():
    path = r"C:\fake photos\fake video 4.mp4"
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import rotation_aware
    from app.services.video.backend import VideoModelV1RotationAwareBackend

    direct = rotation_aware.predict(path, str(CKPT))
    result = VideoModelV1RotationAwareBackend().score(path)

    assert result.confidence == pytest.approx(direct.probability)
    assert result.raw_scores["rotation_degrees"] == direct.rotation_degrees
    assert result.verdict in {"REAL", "FAKE", "UNCERTAIN"}
