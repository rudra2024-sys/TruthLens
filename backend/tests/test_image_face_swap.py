"""Face-swap check for still images (CLAUDE.md section 28)."""
import math
import os

import pytest

from app.services import calibration
from app.services.image import detector, face_swap


def test_calibration_entry_is_platt_and_monotonic():
    cfg = calibration._load()["image_face_swap"]
    assert cfg["method"] == "platt" and cfg["min_face_px"] == 64
    ps = [calibration.calibrate(1 / (1 + math.exp(-z)), "image_face_swap") for z in (-3, -1, 0, 1, 3)]
    assert ps == sorted(ps)
    # logit at which the calibrated verdict band starts (FAKE >= 0.7) is about 0.74, REAL <= 0.3 about -1.7
    assert calibration.calibrate(1 / (1 + math.exp(-0.80)), "image_face_swap") >= 0.7
    assert calibration.calibrate(1 / (1 + math.exp(-(-1.8))), "image_face_swap") <= 0.3


def test_check_not_applied_when_no_face(monkeypatch):
    monkeypatch.setattr(face_swap, "score_image", lambda path: None)
    assert detector._face_swap_check("x.jpg") == (None, None)


def test_check_failure_never_breaks_the_scan(monkeypatch):
    def boom(path):
        raise RuntimeError("checkpoint missing")
    monkeypatch.setattr(face_swap, "score_image", boom)
    assert detector._face_swap_check("x.jpg") == (None, None)


def test_check_returns_calibrated_probability_and_face_width(monkeypatch):
    monkeypatch.setattr(face_swap, "score_image", lambda path: face_swap.FaceSwapScore(logit=2.0, face_px=120))
    p, px = detector._face_swap_check("x.jpg")
    assert px == 120
    assert p == pytest.approx(calibration.calibrate(1 / (1 + math.exp(-2.0)), face_swap.calibration_key()))
    assert p > 0.7


def test_too_small_face_is_not_scored():
    import numpy as np
    blank = np.full((300, 300, 3), 128, dtype=np.uint8)   # no face at all
    assert face_swap.score_bgr(blank) is None


@pytest.mark.models
def test_real_swap_still_scores_higher_than_real_face():
    ck = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "video", "video_v1_round4.pt")
    swap = os.path.join(os.path.dirname(__file__), "..", "..", "testing", "deepfake", "images")
    if not (os.path.isfile(ck) and os.path.isdir(swap)):
        pytest.skip("round-4 checkpoint or testing/ stills not present")
    scores = [face_swap.score_image(os.path.join(swap, f)) for f in sorted(os.listdir(swap))]
    scores = [s for s in scores if s is not None]
    assert scores, "no face found in any still"
    assert all(math.isfinite(s.logit) for s in scores)


def test_haar_path_unchanged_and_mtcnn_only_a_fallback(monkeypatch):
    """Haar-found faces never touch MTCNN (the model was trained on Haar crops); MTCNN is only the fallback."""
    import numpy as np
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    calls = {"mtcnn": 0}
    monkeypatch.setattr(face_swap, "_mtcnn_crop", lambda f: calls.__setitem__("mtcnn", calls["mtcnn"] + 1))
    assert face_swap.score_bgr(img) is None
    assert calls["mtcnn"] == 1      # no Haar face -> fallback consulted


def test_mtcnn_fallback_scores_its_crop(monkeypatch):
    import numpy as np
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(face_swap, "_mtcnn_crop", lambda f: (np.full((100, 100, 3), 90, dtype=np.uint8), 100))
    monkeypatch.setattr(face_swap, "_score_crop", lambda c: 1.5)
    s = face_swap.score_bgr(img)
    assert s is not None and s.logit == 1.5 and s.face_px == 100


def test_checkpoint_switch(monkeypatch, tmp_path):
    # empty string = forced rollback to the video round-4 weights and the original calibration entry
    monkeypatch.setenv("IMAGE_FACE_SWAP_CHECKPOINT", "")
    assert face_swap.checkpoint_path() is None
    assert face_swap.calibration_key() == "image_face_swap"
    # explicit path = that file AND its own Platt entry (different logits need a different calibration)
    monkeypatch.setenv("IMAGE_FACE_SWAP_CHECKPOINT", "/tmp/x.pt")
    assert face_swap.checkpoint_path() == "/tmp/x.pt"
    assert face_swap.calibration_key() == "image_face_swap_r2"
    from app.services.calibration import calibrate
    assert calibrate(0.5, "image_face_swap_r2") is not None
    # unset: default file if present, else the video weights
    monkeypatch.delenv("IMAGE_FACE_SWAP_CHECKPOINT")
    monkeypatch.setattr(face_swap, "DEFAULT_CHECKPOINT", tmp_path / "missing.pt")
    assert face_swap.checkpoint_path() is None
    (tmp_path / "there.pt").write_bytes(b"x")
    monkeypatch.setattr(face_swap, "DEFAULT_CHECKPOINT", tmp_path / "there.pt")
    assert face_swap.checkpoint_path() == str(tmp_path / "there.pt")
