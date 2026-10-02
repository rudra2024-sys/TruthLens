"""Tests for the opt-in cross-cascade consensus video backend (CLAUDE.md section 20).

This backend is opt-in only (VIDEO_MODEL_BACKEND=model_v1_consensus), never the default -- unlike the
orientation fallback (section 19), it is NOT provably regression-free, so these tests check the mechanism
(agreement required, disagreement rejected) rather than asserting it always helps. The checkpoint-dependent
tests skip themselves when the checkpoint or the local ground-truth videos are absent, matching the pattern in
test_models.py / test_video_cnn_gru_v5.py / test_video_rotation_fallback.py.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.video.model_v1 import consensus
from tests.conftest import BACKEND, TEST_DATA

CKPT = Path(os.getenv("VIDEO_MODEL_V1_CHECKPOINT") or BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt")


# ---------------------------------------------------------------- consensus.py, no model/checkpoint needed

def test_iou_identical_boxes_is_1():
    box = (10, 10, 50, 50)
    assert consensus._iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_0():
    assert consensus._iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_corroborated_face_returns_none_when_default_finds_nothing():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt2 = consensus.get_alt2_cascade()
    blank = np.zeros((200, 200), dtype=np.uint8)
    assert consensus.corroborated_face(blank, cascade, alt2) is None


def test_corroborated_face_rejects_an_uncorroborated_detection():
    """A spurious default-only detection (alt2 silent) must be rejected, not trusted."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    class _NeverFinds:
        def detectMultiScale(self, *a, **k):
            return np.empty((0, 4), dtype=np.int32)

    img_path = TEST_DATA / "chatgpt_everyday" / "image4.png"
    if not img_path.is_file():
        pytest.skip("sample face image not present")
    gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    assert consensus.corroborated_face(gray, cascade, _NeverFinds()) is None


def test_corroborated_face_accepts_a_corroborated_detection():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt2 = consensus.get_alt2_cascade()
    # image2.png is pinned here (confirmed by hand, distinct from image4.png used in
    # test_video_rotation_fallback.py, which default finds but alt2 does not corroborate) as a case where
    # both cascades independently agree on the same face region.
    img_path = TEST_DATA / "chatgpt_everyday" / "image2.png"
    if not img_path.is_file():
        pytest.skip("sample face image not present")
    gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
    face = consensus.corroborated_face(gray, cascade, alt2)
    assert face is not None


def test_crop_falls_back_to_center_square_when_uncorroborated():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    class _FindsSomething:
        def detectMultiScale(self, *a, **k):
            return np.array([[10, 10, 50, 50]], dtype=np.int32)

    class _NeverFinds:
        def detectMultiScale(self, *a, **k):
            return np.empty((0, 4), dtype=np.int32)

    frame = np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    crop = consensus.crop_face_or_center_consensus(frame, _FindsSomething(), _NeverFinds())
    # center-square crop of a 200x300 frame is 200x200
    assert crop.shape[:2] == (200, 200)


# ---------------------------------------------------------------- backend.py wiring

def test_default_backend_is_unaffected_by_registering_consensus(monkeypatch):
    monkeypatch.delenv("VIDEO_MODEL_BACKEND", raising=False)
    from app.services.video.backend import VideoModelV1Backend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1Backend)


def test_consensus_backend_is_selectable_via_env_var(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL_BACKEND", "model_v1_consensus")
    from app.services.video.backend import VideoModelV1ConsensusBackend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1ConsensusBackend)


def test_consensus_backend_metadata_marks_itself_unvalidated():
    from app.services.video.backend import VideoModelV1ConsensusBackend
    assert VideoModelV1ConsensusBackend.metadata.validated is False


# ---------------------------------------------------------------- real checkpoint + real sample videos

def test_consensus_rejects_the_poster_lock_on_on_the_diagnosed_clip():
    """Regression pin for the diagnosed case (CLAUDE.md section 20): on `fake video 3.mp4`, the default
    cascade's lone detections on frames 2/4/8 are NOT corroborated by alt2 (confirmed by hand), so the
    consensus-aware prediction must differ from the plain prediction on this clip."""
    path = r"C:\fake photos\fake video 3.mp4"
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized, consensus_aware

    plain = optimized.predict(path, str(CKPT))
    corroborated = consensus_aware.predict(path, str(CKPT))

    assert corroborated.frame_logits != plain.frame_logits


def test_consensus_backend_matches_direct_module_call():
    path = r"C:\fake photos\fake video 3.mp4"
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import consensus_aware
    from app.services.video.backend import VideoModelV1ConsensusBackend

    direct = consensus_aware.predict(path, str(CKPT))
    result = VideoModelV1ConsensusBackend().score(path)

    assert result.confidence == pytest.approx(direct.probability)
    assert result.verdict in {"REAL", "FAKE", "UNCERTAIN"}
