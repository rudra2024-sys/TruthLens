"""Post-hoc temperature-scaling calibration (app/services/calibration.py, added 2026-10-03) - see
CLAUDE.md section 25 for the measured ECE improvement (image 0.158->0.056, video 0.254->0.059) this is based
on. Additive only: a separate `calibrated_confidence` field, never replaces `confidence_score`."""
import json

import pytest

from app.services import calibration


@pytest.fixture(autouse=True)
def _clear_cache():
    calibration.clear_cache()
    yield
    calibration.clear_cache()


def test_calibrate_returns_none_for_a_media_type_with_no_fitted_temperature(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 2.0}}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    assert calibration.calibrate(0.9, "audio") is None


def test_calibrate_returns_none_when_config_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "_CONFIG_PATH", tmp_path / "missing.json")
    assert calibration.calibrate(0.9, "image") is None


def test_temperature_greater_than_1_pulls_extreme_scores_toward_half(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 4.0}}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    out = calibration.calibrate(0.99, "image")
    assert 0.5 < out < 0.99   # softened, not left at the extreme, not flipped past 0.5


def test_temperature_of_1_is_a_near_identity(tmp_path, monkeypatch):
    import math
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 1.0}}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    out = calibration.calibrate(0.7, "image")
    assert math.isclose(out, 0.7, abs_tol=1e-6)


def test_calibrate_is_monotonic_in_the_input_probability(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 3.0}}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    low = calibration.calibrate(0.3, "image")
    mid = calibration.calibrate(0.5, "image")
    high = calibration.calibrate(0.8, "image")
    assert low < mid < high


def test_calibrate_handles_extreme_inputs_without_crashing(tmp_path, monkeypatch):
    cfg = tmp_path / "calibration.json"
    cfg.write_text(json.dumps({"image": {"temperature": 2.0}}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_CONFIG_PATH", cfg)
    assert 0.0 <= calibration.calibrate(0.0, "image") <= 1.0
    assert 0.0 <= calibration.calibrate(1.0, "image") <= 1.0


def test_real_calibration_config_loads_if_present():
    """If eval/fit_calibration.py has been run, the real config should be loadable and sane."""
    calibration.clear_cache()
    out = calibration.calibrate(0.9, "image")
    if out is None:
        pytest.skip("app/core/calibration.json not present (eval/fit_calibration.py not run)")
    assert 0.0 <= out <= 1.0
