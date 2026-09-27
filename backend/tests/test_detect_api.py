"""Detection routes with stubbed detectors: dispatch, error mapping, idempotency, explain/provenance plumbing."""

import io

import pytest

from app.services.detection_errors import UnprocessableMediaError
from tests.conftest import TEST_DATA, jpeg_bytes, png_bytes


def _detect(client, headers, uid):
    return client.post(f"/api/v1/detect/{uid}", headers=headers)


# ---------------------------------------------------------------- dispatch by media type

@pytest.mark.parametrize("data,name,ctype,kind,analysis_key", [
    (png_bytes(), "a.png", "image/png", "image", "image_analysis"),
    (b"\x00" * 64, "a.mp4", "video/mp4", "video", "video_analysis"),
    (b"\x00" * 64, "a.wav", "audio/wav", "audio", "audio_analysis"),
])
def test_detection_dispatches_by_media_type(client, auth, upload, stub_detectors, data, name, ctype, kind, analysis_key):
    uid = upload(auth, data, name, ctype)
    r = _detect(client, auth, uid)
    assert r.status_code == 201
    body = r.json()
    assert body["model_used"] == f"stub-{kind}"
    assert body[analysis_key] is not None
    assert body["verdict"] == "FAKE" and body["upload_id"] == uid


def test_result_endpoint_returns_the_stored_result(client, scan, auth):
    uid = scan(auth)
    r = client.get(f"/api/v1/detect/{uid}/result", headers=auth)
    assert r.status_code == 200 and r.json()["upload_id"] == uid


def test_result_before_detection_is_404(client, auth, upload):
    uid = upload(auth, png_bytes())
    assert client.get(f"/api/v1/detect/{uid}/result", headers=auth).status_code == 404


def test_detection_is_idempotent(client, scan, auth):
    """Regression: a repeated POST used to store a second result row, after which /result, /report, /explain and
    /provenance answered 500 ("Multiple rows were found") for that upload forever."""
    uid = scan(auth)
    first = client.get(f"/api/v1/detect/{uid}/result", headers=auth).json()
    again = _detect(client, auth, uid)
    assert again.status_code == 201 and again.json()["result_id"] == first["result_id"]
    for path in (f"/api/v1/detect/{uid}/result", f"/api/v1/detect/{uid}/provenance", f"/api/v1/report/{uid}"):
        assert client.get(path, headers=auth).status_code == 200, path
    assert client.get("/api/v1/stats", headers=auth).json()["total_scans"] == 1


# ---------------------------------------------------------------- error mapping

def test_unprocessable_media_maps_to_422_with_the_safe_message(client, auth, upload, monkeypatch):
    async def boom(upload, db):
        raise UnprocessableMediaError("This file could not be read as an image.")
    monkeypatch.setattr("app.api.routes.detect.run_image_detection", boom)
    r = _detect(client, auth, upload(auth, png_bytes()))
    assert r.status_code == 422 and r.json()["detail"] == "This file could not be read as an image."


def test_unexpected_failure_maps_to_generic_500_without_leaking_internals(client, auth, upload, monkeypatch):
    async def boom(upload, db):
        raise RuntimeError("secret path C:/internal/model.pth exploded")
    monkeypatch.setattr("app.api.routes.detect.run_image_detection", boom)
    r = _detect(client, auth, upload(auth, png_bytes()))
    assert r.status_code == 500
    assert "secret" not in r.text and "model.pth" not in r.text


def test_missing_file_on_disk_is_a_clean_500(client, auth, upload, monkeypatch):
    async def gone(upload, db):
        raise FileNotFoundError("/srv/uploads/x.png")
    monkeypatch.setattr("app.api.routes.detect.run_image_detection", gone)
    r = _detect(client, auth, upload(auth, png_bytes()))
    assert r.status_code == 500 and "/srv/uploads" not in r.text and "upload it again" in r.json()["detail"]


def test_failed_detection_leaves_no_result_behind(client, auth, upload, monkeypatch):
    async def boom(upload, db):
        raise UnprocessableMediaError("bad file")
    monkeypatch.setattr("app.api.routes.detect.run_image_detection", boom)
    uid = upload(auth, png_bytes())
    assert _detect(client, auth, uid).status_code == 422
    assert client.get(f"/api/v1/detect/{uid}/result", headers=auth).status_code == 404


# ---------------------------------------------------------------- provenance endpoint (real files, no model needed)

def test_provenance_flags_ai_credentials_that_contradict_a_real_verdict(client, auth, upload, stub_detectors):
    sample = TEST_DATA / "chatgpt_everyday" / "image1.png"
    if not sample.is_file():
        pytest.skip("C2PA sample image not present")
    stub_detectors("image", verdict="REAL", confidence=0.02)
    uid = upload(auth, sample.read_bytes(), "gpt.png", "image/png")
    assert _detect(client, auth, uid).json()["verdict"] == "REAL"
    r = client.get(f"/api/v1/detect/{uid}/provenance", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] and body["level"] == "declared_ai"
    assert body["conflict_note"] and "REAL" in body["conflict_note"]
    assert body["c2pa"]["present"] and body["c2pa"]["content_intact"] is True
    assert body["caveat"]


def test_provenance_for_a_plain_file_says_nothing_and_never_invents_evidence(client, scan, auth):
    uid = scan(auth, jpeg_bytes(), "plain.jpg", "image/jpeg")
    body = client.get(f"/api/v1/detect/{uid}/provenance", headers=auth).json()
    assert body["level"] == "none" and body["conflict_note"] is None
    assert body["c2pa"]["present"] is False
    assert all(s["kind"] != "ai" for s in body["signals"])
    assert body["ela"]["applicable"] is True and body["ela"]["heatmap"].startswith("data:image/jpeg;base64,")


def test_provenance_not_available_for_audio(client, scan, auth):
    uid = scan(auth, b"\x00" * 64, "a.wav", "audio/wav")
    body = client.get(f"/api/v1/detect/{uid}/provenance", headers=auth).json()
    assert body["available"] is False and body["reason"]


# ---------------------------------------------------------------- explain endpoint plumbing (explainer stubbed)

def test_explain_endpoint_returns_data_uris_and_never_500s_when_unavailable(client, scan, auth, monkeypatch):
    from app.services.explain.image_explainer import ImageExplanation
    from app.services.explain.service import Explanation
    import numpy as np

    uid = scan(auth)
    img = ImageExplanation(overlay_jpeg=jpeg_bytes(), original_jpeg=jpeg_bytes(), cam=np.zeros((7, 7), np.float32),
                           fake_probability=0.9, fake_logit=2.2, evidence_strength=1.0, focus_fraction=0.1, peak_cell=(1, 2))
    monkeypatch.setattr("app.api.routes.detect.build_explanation",
                        lambda u, r: Explanation(available=True, media_type="image", method="stub", note="n", image=img,
                                                 verdict_driver="CLIP"))
    ok = client.get(f"/api/v1/detect/{uid}/explain", headers=auth).json()
    assert ok["available"] and ok["heatmap"].startswith("data:image/jpeg;base64,") and ok["verdict_driver"] == "CLIP"

    monkeypatch.setattr("app.api.routes.detect.build_explanation",
                        lambda u, r: Explanation(available=False, media_type="image", reason="model missing"))
    off = client.get(f"/api/v1/detect/{uid}/explain", headers=auth)
    assert off.status_code == 200 and off.json()["available"] is False and off.json()["heatmap"] is None


def test_explain_service_degrades_instead_of_raising_when_the_model_cannot_load(client, scan, auth, monkeypatch):
    """Whatever goes wrong inside an explanation (missing checkpoint, corrupt file...) must surface as available=False."""
    def broken():
        raise FileNotFoundError("no checkpoint")
    monkeypatch.setattr("app.services.image.detector.get_image_pipeline", broken)
    uid = scan(auth)
    r = client.get(f"/api/v1/detect/{uid}/explain", headers=auth)
    assert r.status_code == 200 and r.json()["available"] is False and r.json()["reason"]
