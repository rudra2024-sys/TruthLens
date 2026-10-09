"""Continuous-monitoring-session tests (added 2026-10-09) -- periodic identity/deepfake/
presence checks within a MonitoringSession, each of which can flag independently.

Stubs at the same primitive level test_identity.py's stub_embed does (the ML boundary: the
identity pipeline's embed(), and the deepfake ensemble call), while letting the REAL
_compute_signals banding/reason-building logic run -- so the integration (not just each piece
in isolation) is actually exercised. Never constructs real models in CI.

Covers "multiple faces" again too -- the first attempt (Haar cascade) was dropped after failing
on real input, then redone 2026-10-09 via IdentityPipeline.count_faces() (MTCNN detect + IoU
dedup), validated directly against real images (see monitoring.py's module docstring and
pipelines/identity/inference.py::count_faces's docstring for the full story).
"""

import pytest

from app.pipelines.identity.inference import NoFaceDetectedError
from tests.conftest import png_bytes, video_bytes


@pytest.fixture()
def stub_embed(monkeypatch):
    """Same technique as test_identity.py's fixture of the same name -- queues canned
    embeddings for app.services.identity.detector._embed, used here only to enroll a reference
    with a known vector."""
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
    """Controls face count, presence (via embed()/NoFaceDetectedError, the real production
    mechanism), and the deepfake signal independently, while leaving app.services.identity.
    monitoring._compute_signals's own banding/reason logic real."""
    state = {
        "face_count": 1,
        "embedding": [1.0, 0.0],
        "no_face": False,
        "fake_probability": 0.1,
        "deepfake_verdict": "REAL",
    }

    class _FakePipeline:
        def count_faces(self, path):
            return state["face_count"]

        def embed(self, path):
            if state["no_face"]:
                raise NoFaceDetectedError("stub: no face")
            return state["embedding"]

    monkeypatch.setattr("app.services.identity.monitoring.get_identity_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "app.services.identity.monitoring._run_deepfake_check",
        lambda path: (state["fake_probability"], state["deepfake_verdict"]),
    )

    def _set(face_count=1, embedding=(1.0, 0.0), no_face=False, fake_probability=0.1, deepfake_verdict="REAL"):
        state.update(
            face_count=face_count,
            embedding=list(embedding) if embedding is not None else None,
            no_face=no_face,
            fake_probability=fake_probability,
            deepfake_verdict=deepfake_verdict,
        )

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


def _submit_check(client, headers, upload, session_id):
    uid = upload(headers, png_bytes(), "frame.png", "image/png")
    return client.post(f"/api/v1/identity/sessions/{session_id}/checks", json={"upload_id": uid}, headers=headers)


def test_session_requires_enrollment(client, auth):
    r = client.post("/api/v1/identity/sessions", headers=auth)
    assert r.status_code == 404


def test_unflagged_check_when_everything_normal(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["flagged"] is False
    assert body["flag_reasons"] == []
    assert body["identity_verdict"] == "MATCH"
    assert body["deepfake_verdict"] == "REAL"
    assert body["face_count"] == 1


def test_check_flags_identity_mismatch_even_when_deepfake_is_clean(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(0.0, 1.0), fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["flagged"] is True
    assert "identity_mismatch" in body["flag_reasons"]
    assert "deepfake_signal" not in body["flag_reasons"]
    assert body["identity_verdict"] == "NO_MATCH"


def test_check_flags_deepfake_signal_even_when_identity_matches(client, auth, upload, stub_embed, stub_monitoring):
    """The literal "even if they have matched with the identity proof" case."""
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.95, deepfake_verdict="FAKE")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["identity_verdict"] == "MATCH"
    assert body["flagged"] is True
    assert "deepfake_signal" in body["flag_reasons"]
    assert "identity_mismatch" not in body["flag_reasons"]


def test_check_flags_no_face_and_skips_identity_subcheck(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(no_face=True, fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_count"] == 0
    assert body["identity_verdict"] is None
    assert body["similarity_score"] is None
    assert body["flagged"] is True
    assert body["flag_reasons"] == ["no_face"]


def test_check_flags_multiple_faces_and_skips_identity_subcheck(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(face_count=2, fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_count"] == 2
    assert body["identity_verdict"] is None
    assert body["similarity_score"] is None
    assert body["flagged"] is True
    assert body["flag_reasons"] == ["multiple_faces"]


def test_end_session_is_idempotent(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r1 = client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)
    assert r1.status_code == 200
    assert r1.json()["ended_at"] is not None

    r2 = client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)
    assert r2.status_code == 200
    assert r2.json()["ended_at"] == r1.json()["ended_at"]


def test_check_after_session_ended_400s(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 400


def test_get_session_returns_timeline(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")
    _submit_check(client, auth, upload, session["session_id"])
    stub_monitoring(no_face=True, fake_probability=0.05, deepfake_verdict="REAL")
    _submit_check(client, auth, upload, session["session_id"])

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == session["session_id"]
    assert len(body["checks"]) == 2


def test_ownership_isolation(client, auth, other_auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    assert client.get(f"/api/v1/identity/sessions/{session['session_id']}", headers=other_auth).status_code == 404
    assert client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=other_auth).status_code == 404

    uid = upload(other_auth, png_bytes(), "frame.png", "image/png")
    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks",
        json={"upload_id": uid},
        headers=other_auth,
    )
    assert r.status_code == 404
