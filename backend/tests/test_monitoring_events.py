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

    def _queue(vector):
        queue.append(list(vector))

    return _queue


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
    assert body["session_id"] == session["session_id"]


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
