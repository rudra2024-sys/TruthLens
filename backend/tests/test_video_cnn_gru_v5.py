"""Tests for the experimental CNN+BiGRU v5 video backend (CLAUDE.md sections 3/17/18).

This backend is opt-in only (VIDEO_MODEL_BACKEND=cnn_gru_v5), never the default -- these
tests guard that wiring, not the model's real-world accuracy (which is known-weak, see
CLAUDE.md). The checkpoint-dependent tests skip themselves when the checkpoint or the
local ground-truth videos are absent, matching the pattern in test_models.py.
"""
import os
from pathlib import Path

import pytest

from tests.conftest import BACKEND

CKPT = Path(os.getenv("VIDEO_MODEL_CNN_GRU_V5_CHECKPOINT") or BACKEND / "checkpoints" / "video" / "cnn_gru_v5_head.pt")


def test_default_backend_is_unaffected_by_registering_cnn_gru_v5(monkeypatch):
    monkeypatch.delenv("VIDEO_MODEL_BACKEND", raising=False)
    from app.services.video.backend import VideoModelV1Backend, get_video_backend
    assert isinstance(get_video_backend(), VideoModelV1Backend)


def test_cnn_gru_v5_is_selectable_via_env_var(monkeypatch):
    monkeypatch.setenv("VIDEO_MODEL_BACKEND", "cnn_gru_v5")
    from app.services.video.backend import CnnGruV5Backend, get_video_backend
    assert isinstance(get_video_backend(), CnnGruV5Backend)


def test_cnn_gru_v5_metadata_marks_itself_unvalidated():
    """Guards against silently flipping this to the default -- it must keep signalling
    that it hasn't cleared the bar VideoModelV1Backend has (see CLAUDE.md)."""
    from app.services.video.backend import CnnGruV5Backend
    assert CnnGruV5Backend.metadata.validated is False


@pytest.mark.parametrize("path", [r"C:\fake photos\fake video 3.mp4", r"C:\real photos\real video 1.mp4"])
def test_cnn_gru_v5_backend_matches_direct_module_call(path):
    """The backend.py wiring must not change what cnn_gru_v5.predict() computes."""
    if not CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("cnn_gru_v5 checkpoint or sample video not present")
    from app.services.video import cnn_gru_v5
    from app.services.video.backend import CnnGruV5Backend

    direct = cnn_gru_v5.predict(path)
    result = CnnGruV5Backend().score(path)

    assert result.confidence == pytest.approx(direct.probability)
    assert 0.0 <= result.confidence <= 1.0
    assert result.verdict in {"REAL", "FAKE", "UNCERTAIN"}
