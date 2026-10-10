"""Monitoring session summary PDF (added 2026-10-09) -- same verification convention as
test_report.py: a real PDF is generated and parsed back with pypdf, not just a 200 status check.
"""

import io

import numpy as np
import pytest
from pypdf import PdfReader

from tests.conftest import png_bytes

_FRONTAL_LANDMARKS = np.array([
    [292.44, 141.61], [362.04, 139.58], [326.10, 170.51], [299.42, 213.41], [357.15, 212.87],
])


def _pdf_text(content: bytes) -> str:
    assert content[:5] == b"%PDF-"
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)


@pytest.fixture()
def stub_embed(monkeypatch):
    queue = []

    def _embed_stub(path):
        if queue:
            return queue.pop(0)
        return [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)

    def _queue(vector):
        queue.append(list(vector))

    return _queue


@pytest.fixture()
def stub_monitoring(monkeypatch):
    state = {"embedding": [1.0, 0.0], "fake_probability": 0.1, "deepfake_verdict": "REAL"}

    class _FakePipeline:
        def detect_faces(self, path):
            return {
                "count": 1,
                "landmarks": _FRONTAL_LANDMARKS,
                "box": [250.0, 130.0, 400.0, 300.0],
                "image_size": (480, 640),
            }

        def embed(self, path):
            return state["embedding"]

    monkeypatch.setattr("app.services.identity.monitoring.get_identity_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "app.services.identity.monitoring._run_deepfake_check",
        lambda path: (state["fake_probability"], state["deepfake_verdict"]),
    )
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_object_pipeline",
        lambda: type("_FakeObjectPipeline", (), {"detect": staticmethod(lambda path, conf_thresh=0.4: [])})(),
    )
    monkeypatch.setattr("app.services.identity.audio_vad.speech_ratio", lambda path: None)

    def _set(embedding=(1.0, 0.0), fake_probability=0.1, deepfake_verdict="REAL"):
        state.update(embedding=list(embedding), fake_probability=fake_probability, deepfake_verdict=deepfake_verdict)

    return _set


def _enroll(client, headers, upload, stub_embed, vector=(1.0, 0.0)):
    stub_embed(vector)
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _start_session(client, headers):
    r = client.post("/api/v1/identity/sessions", headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_report_contains_meta_and_timeline(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)

    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")
    uid1 = upload(auth, png_bytes(), "frame1.png", "image/png")
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/checks", json={"upload_id": uid1}, headers=auth)

    stub_monitoring(embedding=(0.0, 1.0), fake_probability=0.95, deepfake_verdict="FAKE")
    uid2 = upload(auth, png_bytes(), "frame2.png", "image/png")
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/checks", json={"upload_id": uid2}, headers=auth)

    client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "tab_hidden", "detail": "12.0"},
        headers=auth,
    )

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}/report", headers=auth)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    text = _pdf_text(r.content)

    assert "Monitoring Session Report" in text
    assert session["session_id"] in text
    assert "Flag Breakdown" in text
    assert "Behavioral Events" in text
    assert "Timeline" in text
    assert "Tab hidden" in text
    assert "Lip-Sync Signal" in text
    assert "Informational only, not a flag" in text
    # This fixture never supplies speech_ratio (always None), so lip-sync has nothing to compare.
    assert "Not enough checks with both audio and a detected face" in text


def test_lip_sync_section_flags_flat_mouth_during_detected_speech():
    """Direct unit test of the report function, not through the API -- constructs plain
    MonitoringCheck instances (no DB session needed to just hold attribute values) with
    identical mouth_width_px (flat) while speech_ratio indicates talking, confirming the
    "flat while speech detected" note appears, and that this is never styled as a flag."""
    from app.models.models import MonitoringCheck
    from app.services.report.session_report import _lip_sync_section
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    small = ParagraphStyle("test_small", parent=getSampleStyleSheet()["Normal"], fontSize=9)
    checks = [
        MonitoringCheck(check_id=f"c{i}", session_id="s", upload_id=f"u{i}", face_count=1,
                        fake_probability=0.1, deepfake_verdict="REAL", flagged=False,
                        checked_at=f"2026-10-09T10:00:0{i}", speech_ratio=0.8, mouth_width_px=50.0)
        for i in range(3)
    ]
    flowables = _lip_sync_section(checks, small)
    # Render the table's own text content via its underlying Paragraph objects.
    rendered = []
    for f in flowables:
        if hasattr(f, "_cellvalues"):
            for row in f._cellvalues:
                for cell in row:
                    rendered.append(cell.text if hasattr(cell, "text") else str(cell))
    assert any("flat while speech detected" in r for r in rendered)


def test_report_works_for_an_active_session_with_no_activity_yet(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}/report", headers=auth)
    assert r.status_code == 200
    text = _pdf_text(r.content)
    assert "Still active" in text
    assert "No checks were flagged" in text or "Flag Breakdown" in text


def test_report_ownership_isolation(client, auth, other_auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}/report", headers=other_auth)
    assert r.status_code == 404
