"""Background monitoring-check jobs (added 2026-10-11, item 6) -- non-blocking, ownership,
concurrency limit, and that the job's result matches the synchronous endpoint's shape.
"""

import time

import numpy as np
import pytest

from app.pipelines.identity.inference import NoFaceDetectedError
from tests.conftest import png_bytes

_FRONTAL_LANDMARKS = np.array([
    [292.44, 141.61], [362.04, 139.58], [326.10, 170.51], [299.42, 213.41], [357.15, 212.87],
])


@pytest.fixture()
def stub_embed(monkeypatch):
    queue = []

    def _embed_stub(path):
        return queue.pop(0) if queue else [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)
    monkeypatch.setattr(
        "app.services.identity.detector._detect_faces",
        lambda path: {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)},
    )
    monkeypatch.setattr("app.services.identity.detector.assess_enrollment_quality", lambda path, detection: [])
    return lambda vector: queue.append(list(vector))


@pytest.fixture()
def stub_monitoring(monkeypatch):
    state = {"face_count": 1, "embedding": [1.0, 0.0], "no_face": False,
              "fake_probability": 0.1, "deepfake_verdict": "REAL"}

    class _FakePipeline:
        def detect_faces(self, path):
            return {
                "count": state["face_count"],
                "landmarks": _FRONTAL_LANDMARKS if state["face_count"] == 1 else None,
                "box": [250.0, 130.0, 400.0, 300.0] if state["face_count"] == 1 else None,
                "image_size": (480, 640),
            }

        def embed(self, path):
            if state["no_face"]:
                raise NoFaceDetectedError("stub: no face")
            return state["embedding"]

    monkeypatch.setattr("app.services.identity.monitoring.get_identity_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "app.services.identity.monitoring._run_deepfake_check",
        lambda path: (state["fake_probability"], state["deepfake_verdict"]),
    )
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_object_pipeline",
        lambda: type("_O", (), {"detect": staticmethod(lambda path, conf_thresh=0.4: [])})(),
    )
    monkeypatch.setattr("app.services.identity.audio_vad.speech_ratio", lambda path: None)
    return state


def _enroll(client, headers, upload, stub_embed):
    stub_embed((1.0, 0.0))
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _start_session(client, headers):
    r = client.post("/api/v1/identity/sessions", headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def start_job(client, headers, session_id, upload_id):
    r = client.post(
        f"/api/v1/identity/sessions/{session_id}/checks/jobs", json={"upload_id": upload_id}, headers=headers
    )
    assert r.status_code == 202, r.text
    return r.json()


def poll(client, headers, session_id, job_id, until=("done", "failed"), timeout=10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/v1/identity/sessions/{session_id}/checks/jobs/{job_id}", headers=headers).json()
        if last["state"] in until:
            return last
        time.sleep(0.02)
    raise AssertionError(f"job stuck: last={last}")


def test_check_job_completes_with_a_result_matching_the_sync_shape(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    uid = upload(auth, png_bytes(), "frame.png", "image/png")

    job = start_job(client, auth, session["session_id"], uid)
    assert job["state"] in ("queued", "running")
    done = poll(client, auth, session["session_id"], job["job_id"])
    assert done["state"] == "done", done
    result = done["result"]
    assert result["flagged"] is False
    assert result["identity_verdict"] == "MATCH"
    assert result["deepfake_verdict"] == "REAL"
    assert result["session_id"] == session["session_id"]


def test_check_job_rejects_an_already_ended_session(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)
    uid = upload(auth, png_bytes(), "frame.png", "image/png")

    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks/jobs", json={"upload_id": uid}, headers=auth
    )
    assert r.status_code == 400


def test_check_job_is_private_to_its_owner(client, auth, other_auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    uid = upload(auth, png_bytes(), "frame.png", "image/png")
    job = start_job(client, auth, session["session_id"], uid)
    poll(client, auth, session["session_id"], job["job_id"])

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}/checks/jobs/{job['job_id']}", headers=other_auth)
    assert r.status_code == 404

    # Can't even start a job against someone else's session.
    uid2 = upload(other_auth, png_bytes(), "frame2.png", "image/png")
    r2 = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks/jobs", json={"upload_id": uid2}, headers=other_auth
    )
    assert r2.status_code == 404


def test_a_running_check_job_does_not_block_other_requests(client, auth, upload, stub_embed, stub_monitoring, monkeypatch):
    """The real run_check, with a deliberately slow _compute_signals (run via run_in_threadpool,
    same as production) -- while it works, unrelated requests must still be served."""
    real_compute = __import__("app.services.identity.monitoring", fromlist=["_compute_signals"])._compute_signals

    def slow_compute(path, reference, audio_path):
        time.sleep(0.6)
        return real_compute(path, reference, audio_path)

    monkeypatch.setattr("app.services.identity.monitoring._compute_signals", slow_compute)

    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    uid = upload(auth, png_bytes(), "frame.png", "image/png")
    job = start_job(client, auth, session["session_id"], uid)

    t0 = time.time()
    latencies = []
    while time.time() - t0 < 0.4:
        s = time.time()
        assert client.get("/health").status_code == 200
        latencies.append(time.time() - s)
        time.sleep(0.03)
    assert max(latencies) < 0.3, f"server was blocked by the check (slowest /health took {max(latencies):.2f}s)"
    assert poll(client, auth, session["session_id"], job["job_id"])["state"] == "done"


def test_concurrency_limit_queues_extra_jobs(upload, auth, client, stub_embed, stub_monitoring, monkeypatch):
    """A manager with concurrency=1: a second job submitted while the first is still running
    stays queued until the first finishes, confirmed via a gate the test controls directly
    (same gate-based technique as test_jobs.py's slow_video_stub)."""
    import threading

    from app.services.monitoring_jobs import CheckJobManager

    gate = threading.Event()
    real_compute = __import__("app.services.identity.monitoring", fromlist=["_compute_signals"])._compute_signals

    def gated_compute(path, reference, audio_path):
        gate.wait(timeout=5)
        return real_compute(path, reference, audio_path)

    monkeypatch.setattr("app.services.identity.monitoring._compute_signals", gated_compute)
    import app.api.routes.identity as identity_routes
    monkeypatch.setattr(identity_routes, "check_job_manager", CheckJobManager(concurrency=1))

    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    uid1 = upload(auth, png_bytes(), "f1.png", "image/png")
    uid2 = upload(auth, png_bytes(), "f2.png", "image/png")

    job1 = start_job(client, auth, session["session_id"], uid1)
    job2 = start_job(client, auth, session["session_id"], uid2)

    time.sleep(0.1)  # let the scheduler get job1 into "running" and job2 stuck behind the semaphore
    state1 = client.get(f"/api/v1/identity/sessions/{session['session_id']}/checks/jobs/{job1['job_id']}", headers=auth).json()
    state2 = client.get(f"/api/v1/identity/sessions/{session['session_id']}/checks/jobs/{job2['job_id']}", headers=auth).json()
    assert state1["state"] == "running"
    assert state2["state"] == "queued"

    gate.set()
    assert poll(client, auth, session["session_id"], job1["job_id"])["state"] == "done"
    assert poll(client, auth, session["session_id"], job2["job_id"])["state"] == "done"
