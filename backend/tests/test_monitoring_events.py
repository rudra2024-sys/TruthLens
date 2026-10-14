"""MonitoringEvent tests (added 2026-10-09) -- behavioral events (tab/focus loss, clipboard
paste, devtools heuristic, camera interruption) posted alongside the camera-based
MonitoringCheck within a session. No camera frame or model involved, so no stubbing needed --
just ownership, lifecycle, and persistence.
"""

import pytest

from tests.conftest import png_bytes


@pytest.fixture()
def stub_embed(monkeypatch):
    queue = []

    def _embed_stub(path):
        if queue:
            return queue.pop(0)
        return [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)
    monkeypatch.setattr(
        "app.services.identity.detector._detect_faces",
        lambda path: {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)},
    )
    monkeypatch.setattr("app.services.identity.detector.assess_enrollment_quality", lambda path, detection: [])

    def _queue(vector):
        queue.append(list(vector))

    return _queue


def _enroll(client, headers, upload, stub_embed, vector=(1.0, 0.0)):
    stub_embed(vector)
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _start_session(client, headers):
    r = client.post("/api/v1/identity/sessions", headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def test_post_event_persists_and_is_returned(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "tab_hidden", "detail": "7.3"},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["event_type"] == "tab_hidden"
    assert body["detail"] == "7.3"


@pytest.mark.parametrize("event_type", ["live_no_face", "live_looking_away", "live_multiple_faces"])
def test_live_attention_event_types_are_accepted(client, auth, upload, stub_embed, event_type):
    """Real-time client-side attention signals (added 2026-10-12, see LiveFaceMesh.jsx) go
    through the same event endpoint as tab_hidden etc. -- just three more accepted event_type
    values."""
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": event_type},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    assert r.json()["event_type"] == event_type


def test_rejects_unknown_event_type(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "not_a_real_type"},
        headers=auth,
    )
    assert r.status_code == 422


def test_event_after_session_ended_400s(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "clipboard_paste"},
        headers=auth,
    )
    assert r.status_code == 400


def test_event_ownership_isolation(client, auth, other_auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "clipboard_paste"},
        headers=other_auth,
    )
    assert r.status_code == 404


def test_get_session_includes_events_alongside_checks(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "devtools_suspected"},
        headers=auth,
    )
    client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/events",
        json={"event_type": "camera_interrupted", "detail": "track ended"},
        headers=auth,
    )

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 2
    assert body["checks"] == []
    types = {e["event_type"] for e in body["events"]}
    assert types == {"devtools_suspected", "camera_interrupted"}
