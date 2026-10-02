"""Tests for the opt-in wide-face-padding video backend (CLAUDE.md section 22).

This backend is opt-in only (VIDEO_MODEL_BACKEND=model_v1_wide_pad), never the default -- it's a hypothesis
test (more context around the detected face might help), not a response to a diagnosed failure, so these
tests check the mechanism and wiring; CLAUDE.md section 22 has the measured accuracy verdict.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from app.services.video.model_v1 import wide_pad
from tests.conftest import BACKEND

CKPT = Path(os.getenv("VIDEO_MODEL_V1_CHECKPOINT") or BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt")


def test_wide_padding_produces_a_larger_crop_than_default_padding():
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    class _FixedFace:
        def detectMultiScale(self, *a, **k):
            return np.array([[100, 100, 50, 50]], dtype=np.int32)  # face at (100,100), 50x50

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    narrow = wide_pad.crop_face_or_center_wide(frame, _FixedFace(), padding_fraction=0.20)
    wide = wide_pad.crop_face_or_center_wide(frame, _FixedFace(), padding_fraction=0.35)
    assert wide.shape[0] > narrow.shape[0] and wide.shape[1] > narrow.shape[1]


def test_wide_padding_still_falls_back_to_center_square_when_no_face_found():
    import cv2

    class _NeverFinds:
        def detectMultiScale(self, *a, **k):
            return np.empty((0, 4), dtype=np.int32)

    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = wide_pad.crop_face_or_center_wide(frame, _NeverFinds())
    assert crop.shape[:2] == (200, 200)  # center-square of a 200x300 frame


def test_default_backend_is_unaffected_by_registering_wide_pad(monkeypatch):
    monkeypatch.delenv("VIDEO_MODEL_BACKEND", raising=False)
    from app.services.video.backend import VideoModelV1Backend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1Backend)


def test_wide_pad_backend_is_selectable_via_env_var(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL_BACKEND", "model_v1_wide_pad")
    from app.services.video.backend import VideoModelV1WidePadBackend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1WidePadBackend)


def test_wide_pad_backend_metadata_marks_itself_unvalidated():
    from app.services.video.backend import VideoModelV1WidePadBackend
    assert VideoModelV1WidePadBackend.metadata.validated is False


def test_wide_pad_changes_the_score_on_a_clip_with_a_found_face():
    """Sanity check that the wider crop actually feeds different pixels to the model (not a no-op)."""
    path = r"C:\fake photos\fake video 1.mp4"
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized, wide_pad_aware

    plain = optimized.predict(path, str(CKPT))
    wide = wide_pad_aware.predict(path, str(CKPT))
    assert wide.frame_logits != plain.frame_logits


def test_wide_pad_backend_matches_direct_module_call():
    path = r"C:\fake photos\fake video 1.mp4"
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import wide_pad_aware
    from app.services.video.backend import VideoModelV1WidePadBackend

    direct = wide_pad_aware.predict(path, str(CKPT))
    result = VideoModelV1WidePadBackend().score(path)

    assert result.confidence == pytest.approx(direct.probability)
    assert result.verdict in {"REAL", "FAKE", "UNCERTAIN"}
