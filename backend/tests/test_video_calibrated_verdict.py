"""Video Model v1 verdict on the Platt-calibrated scale + low-resolution guard (CLAUDE.md section 26, 2026-10-10).

The deployed model's raw scores sit in ~0.2-0.7, so the old +-0.2 band around the raw 0.525 left 98% of videos
UNCERTAIN. The verdict now bands the calibrated P(FAKE) around 0.5, and videos under the configured minimum short
side are always UNCERTAIN. Model output itself is stubbed; the real backend/detector/report code runs."""
import json
import math

import pytest
from pypdf import PdfReader

from app.services import calibration
from app.services.video import backend as video_backend
from app.services.video.model_v1 import optimized
from app.services.video.model_v1.optimized import VideoModelV1Result
from tests.conftest import video_bytes

A, B = 1.8237, -0.0946         # the fitted values in app/core/calibration.json (round 4, live 2026-10-14)


@pytest.fixture()
def platt_cfg(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"video": {"method": "platt", "a": A, "b": B, "min_short_side_px": 360}}))
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    calibration.clear_cache()
    yield
    calibration.clear_cache()


def raw_for(calibrated):
    """Raw model probability that the Platt map turns into `calibrated`."""
    z = (math.log(calibrated / (1 - calibrated)) - B) / A
    return 1 / (1 + math.exp(-z))


def stub_model(monkeypatch, raw_p, short_side=1080, faces=16):
    z = math.log(raw_p / (1 - raw_p))
    monkeypatch.setattr(optimized, "predict", lambda path, progress=None: VideoModelV1Result(
        logit=z, probability=raw_p, verdict="?", frame_logits=[z] * 16, frames_with_face=faces))
    monkeypatch.setattr(video_backend, "_short_side_px", lambda path: short_side)


def test_platt_calibration_matches_the_formula(platt_cfg):
    for p in (0.3, 0.5, 0.525, 0.6):
        z = math.log(p / (1 - p))
        assert calibration.calibrate(p, "video") == pytest.approx(1 / (1 + math.exp(-(A * z + B))))
    assert calibration.min_short_side_px("video") == 360


def test_raw_threshold_is_where_calibration_crosses_one_half(platt_cfg):
    t = calibration.raw_threshold("video", 0.525)
    assert calibration.calibrate(t, "video") == pytest.approx(0.5, abs=1e-9)


def test_temperature_config_still_works_for_image(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 2.0}}))
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    calibration.clear_cache()
    z = math.log(0.8 / 0.2)
    assert calibration.calibrate(0.8, "image") == pytest.approx(1 / (1 + math.exp(-z / 2.0)))
    assert calibration.raw_threshold("image", 0.5) == pytest.approx(0.5)
    calibration.clear_cache()


@pytest.mark.parametrize("calibrated, expected", [
    (0.95, "FAKE"), (0.75, "FAKE"), (0.6, "UNCERTAIN"), (0.4, "UNCERTAIN"), (0.25, "REAL"), (0.05, "REAL"),
])
def test_verdict_bands_the_calibrated_score(platt_cfg, monkeypatch, calibrated, expected):
    stub_model(monkeypatch, raw_for(calibrated))
    r = video_backend.VideoModelV1Backend().score("x.mp4")
    assert r.verdict == expected
    assert r.raw_scores["calibrated"] == pytest.approx(calibrated, abs=1e-6)
    assert r.confidence == pytest.approx(raw_for(calibrated))        # raw P(FAKE) still returned (eval relies on it)


def test_scores_the_old_band_called_uncertain_now_get_verdicts(platt_cfg, monkeypatch):
    """The point of the change: raw 0.72 / 0.33 sit inside the old raw band (0.325-0.725, UNCERTAIN); on the round-4
    calibrated scale they are 0.84 / 0.20, so FAKE / REAL. (Round 4's raw scores are more spread out than epoch 11's,
    so its Platt slope is gentler and e.g. raw 0.60 -> 0.66 correctly stays UNCERTAIN.)"""
    stub_model(monkeypatch, 0.72)
    assert video_backend.VideoModelV1Backend().score("x.mp4").verdict == "FAKE"
    stub_model(monkeypatch, 0.33)
    assert video_backend.VideoModelV1Backend().score("x.mp4").verdict == "REAL"
    stub_model(monkeypatch, 0.60)
    assert video_backend.VideoModelV1Backend().score("x.mp4").verdict == "UNCERTAIN"


@pytest.mark.parametrize("side, expected", [(224, "UNCERTAIN"), (359, "UNCERTAIN"), (360, "FAKE"), (1080, "FAKE")])
def test_low_resolution_videos_are_always_uncertain(platt_cfg, monkeypatch, side, expected):
    stub_model(monkeypatch, raw_for(0.99), short_side=side)
    r = video_backend.VideoModelV1Backend().score("x.mp4")
    assert r.verdict == expected
    assert r.raw_scores["low_resolution"] is (side < 360)
    assert r.raw_scores["short_side_px"] == side


def test_without_calibration_the_old_raw_band_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "_CONFIG_PATH", tmp_path / "missing.json")
    calibration.clear_cache()
    stub_model(monkeypatch, 0.60)
    r = video_backend.VideoModelV1Backend().score("x.mp4")
    assert r.verdict == "UNCERTAIN"                                  # 0.60 inside the old 0.325-0.725 band
    assert r.raw_scores["calibrated"] is None
    calibration.clear_cache()


def test_short_side_is_read_from_a_real_video(tmp_path):
    import cv2
    import numpy as np
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for _ in range(5):
        w.write(np.zeros((240, 320, 3), np.uint8))
    w.release()
    assert video_backend._short_side_px(path) == 240
    assert video_backend._short_side_px(str(tmp_path / "missing.mp4")) is None


def _scan(client, auth, upload, monkeypatch, raw_p, side):
    stub_model(monkeypatch, raw_p, short_side=side)
    monkeypatch.setattr("app.services.video.detector.get_video_backend", lambda: video_backend.VideoModelV1Backend())
    uid = upload(auth, video_bytes(), "clip.mp4", "video/mp4")
    body = client.post(f"/api/v1/detect/{uid}", headers=auth).json()
    return uid, body


def test_api_stores_confidence_on_the_verdict_scale(client, auth, upload, platt_cfg, monkeypatch):
    uid, body = _scan(client, auth, upload, monkeypatch, raw_for(0.9), 720)
    assert body["verdict"] == "FAKE"
    assert body["confidence_score"] == pytest.approx(0.9, abs=1e-4)          # gauge agrees with the verdict
    va = body["video_analysis"]
    assert va["xception_score"] == pytest.approx(raw_for(0.9))               # raw model output preserved
    assert va["calibrated_fake_probability"] == pytest.approx(0.9, abs=1e-4)
    assert va["short_side_px"] == 720 and va["low_resolution"] is False


def test_api_and_pdf_explain_a_withheld_low_resolution_verdict(client, auth, upload, platt_cfg, monkeypatch):
    raw = raw_for(0.95)
    uid, body = _scan(client, auth, upload, monkeypatch, raw, 224)
    assert body["verdict"] == "UNCERTAIN"
    assert body["video_analysis"]["low_resolution"] is True
    # the calibrated scale isn't valid this small: no calibrated value, confidence is the raw near-50% score
    assert body["video_analysis"]["calibrated_fake_probability"] is None
    assert body["calibrated_confidence"] is None
    assert body["confidence_score"] == pytest.approx(max(raw, 1 - raw))
    r = client.get(f"/api/v1/report/{uid}", headers=auth)
    assert r.status_code == 200
    import io
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(r.content)).pages)
    assert "Verdict withheld - low resolution" in text
    assert "224 px" in text
    assert "Calibrated FAKE Probability (used for the verdict)" not in text
