"""Background detection jobs: progress, failure, cancel, queueing, idempotency, ownership, non-blocking."""

import threading
import time

import pytest
from fastapi.concurrency import run_in_threadpool

from app.services.detection_errors import UnprocessableMediaError
from app.services.jobs import JobManager, manager
from tests.conftest import make_stub, png_bytes, video_bytes

VIDEO = (video_bytes(), "clip.mp4", "video/mp4")


def slow_video_stub(steps=5, delay=0.03, gate=None, error=None, ran=None):
    """A video detector that reports progress from a WORKER THREAD, like the real one does.
    `ran` (a list) receives one entry per step that actually executed."""
    async def stub(upload, db, progress=None):
        def work():
            for i in range(steps):
                while gate is not None and not gate.is_set():
                    time.sleep(0.01)
                progress((i + 1) / steps, f"Reading frames ({i + 1}/{steps})")   # may raise JobCancelled
                if ran is not None:
                    ran.append(i)
                time.sleep(delay)
            if error:
                raise error
        await run_in_threadpool(work)
        return await make_stub("video")(upload, db)
    return stub


@pytest.fixture()
def use_stub(monkeypatch):
    def _use(**kw):
        monkeypatch.setattr("app.services.jobs.run_video_detection", slow_video_stub(**kw))
    return _use


def start(client, headers, uid):
    r = client.post(f"/api/v1/detect/{uid}/jobs", headers=headers)
    assert r.status_code == 202, r.text
    return r.json()


def poll(client, headers, job_id, until=("done", "failed", "cancelled"), timeout=15.0):
    seen = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        seen.append(j)
        if j["state"] in until:
            return j, seen
        time.sleep(0.02)
    raise AssertionError(f"job stuck: last={seen[-1] if seen else None}")


def video_upload(upload, headers):
    data, name, ctype = VIDEO
    return upload(headers, data, name, ctype)


# ------------------------------------------------------------------------- happy path

def test_job_runs_in_the_background_and_reports_monotonic_progress(client, auth, upload, use_stub):
    use_stub(steps=8, delay=0.05)
    uid = video_upload(upload, auth)
    job = start(client, auth, uid)
    assert job["state"] in ("queued", "running") and job["upload_id"] == uid and job["result_id"] is None

    final, seen = poll(client, auth, job["job_id"])
    assert final["state"] == "done" and final["progress"] == 1.0 and final["result_id"]
    progress = [j["progress"] for j in seen]
    assert progress == sorted(progress), "progress must never go backwards"
    assert any(0 < p < 1 for p in progress), "expected to observe intermediate progress"
    assert any("Reading frames" in j["stage"] for j in seen)

    result = client.get(f"/api/v1/detect/{uid}/result", headers=auth).json()
    assert result["result_id"] == final["result_id"] and result["video_analysis"] is not None


def test_job_for_an_image_upload_also_works(client, auth, upload, monkeypatch):
    monkeypatch.setattr("app.services.jobs.run_image_detection", make_stub("image"))
    uid = upload(auth, png_bytes())
    final, _ = poll(client, auth, start(client, auth, uid)["job_id"])
    assert final["state"] == "done" and final["media_type"] == "image"


# ------------------------------------------------------------------------- failures

def test_unprocessable_video_fails_with_the_safe_message_and_stores_nothing(client, auth, upload, use_stub):
    use_stub(steps=2, error=UnprocessableMediaError("This video could not be analyzed -- it may be too short."))
    uid = video_upload(upload, auth)
    final, _ = poll(client, auth, start(client, auth, uid)["job_id"])
    assert final["state"] == "failed" and "too short" in final["error"] and final["result_id"] is None
    assert client.get(f"/api/v1/detect/{uid}/result", headers=auth).status_code == 404


def test_unexpected_error_fails_generically_without_leaking_internals(client, auth, upload, use_stub):
    use_stub(steps=2, error=RuntimeError("secret C:/internal/model.pth exploded"))
    final, _ = poll(client, auth, start(client, auth, video_upload(upload, auth))["job_id"])
    assert final["state"] == "failed"
    assert "secret" not in str(final) and "model.pth" not in str(final)


# ------------------------------------------------------------------------- cancel

def test_cancelling_a_running_job_stops_it_and_stores_nothing(client, auth, upload, use_stub):
    gate, ran = threading.Event(), []
    use_stub(steps=50, delay=0.02, gate=gate, ran=ran)
    uid = video_upload(upload, auth)
    job = start(client, auth, uid)
    poll(client, auth, job["job_id"], until=("running",))
    assert client.delete(f"/api/v1/jobs/{job['job_id']}", headers=auth).status_code == 200
    gate.set()
    final, _ = poll(client, auth, job["job_id"])
    assert final["state"] == "cancelled" and final["result_id"] is None
    assert len(ran) < 5, f"the scan kept running after cancel ({len(ran)} of 50 steps executed): CPU wasted"
    assert client.get(f"/api/v1/detect/{uid}/result", headers=auth).status_code == 404
    # ... and the upload can be scanned again afterwards
    use_stub(steps=2)
    again, _ = poll(client, auth, start(client, auth, uid)["job_id"])
    assert again["state"] == "done"


def test_a_queued_job_reports_its_position_and_cancels_instantly(client, auth, upload, use_stub, monkeypatch):
    monkeypatch.setattr(manager, "concurrency", 1)
    gate = threading.Event()
    use_stub(steps=3, delay=0.01, gate=gate)
    first = start(client, auth, video_upload(upload, auth))
    poll(client, auth, first["job_id"], until=("running",))
    second = start(client, auth, video_upload(upload, auth))
    j2 = client.get(f"/api/v1/jobs/{second['job_id']}", headers=auth).json()
    assert j2["state"] == "queued" and j2["queue_position"] == 1

    cancelled = client.delete(f"/api/v1/jobs/{second['job_id']}", headers=auth).json()
    assert cancelled["state"] == "cancelled" and cancelled["queue_position"] is None
    gate.set()
    assert poll(client, auth, first["job_id"])[0]["state"] == "done"      # the running job is unaffected


# ------------------------------------------------------------------------- idempotency

def test_starting_twice_returns_the_same_job_and_never_duplicates_the_result(client, auth, upload, use_stub):
    gate = threading.Event()
    use_stub(steps=3, delay=0.01, gate=gate)
    uid = video_upload(upload, auth)
    a, b = start(client, auth, uid), start(client, auth, uid)
    assert a["job_id"] == b["job_id"]
    gate.set()
    done, _ = poll(client, auth, a["job_id"])
    again = start(client, auth, uid)                                       # already scanned
    assert again["state"] == "done" and again["result_id"] == done["result_id"]
    assert client.post(f"/api/v1/detect/{uid}", headers=auth).json()["result_id"] == done["result_id"]
    assert client.get("/api/v1/stats", headers=auth).json()["total_scans"] == 1


# ------------------------------------------------------------------------- access control

def test_jobs_are_private_to_their_owner(client, auth, other_auth, upload, use_stub):
    use_stub(steps=2)
    uid = video_upload(upload, auth)
    job = start(client, auth, uid)
    poll(client, auth, job["job_id"])
    assert client.get(f"/api/v1/jobs/{job['job_id']}", headers=other_auth).status_code == 404
    assert client.delete(f"/api/v1/jobs/{job['job_id']}", headers=other_auth).status_code == 404
    assert client.post(f"/api/v1/detect/{uid}/jobs", headers=other_auth).status_code == 404
    unknown = client.get("/api/v1/jobs/does-not-exist", headers=auth)
    assert unknown.status_code == 404 and unknown.json() == client.get(f"/api/v1/jobs/{job['job_id']}", headers=other_auth).json()
    for method, path in (("get", "/api/v1/jobs/x"), ("delete", "/api/v1/jobs/x"), ("post", "/api/v1/detect/x/jobs")):
        assert getattr(client, method)(path).status_code in (401, 403)


# ------------------------------------------------------------------------- the server stays responsive

def test_a_running_scan_does_not_block_other_requests(client, auth, upload, monkeypatch):
    """The real run_video_detection with a slow backend: while it works, unrelated requests must still be served."""
    from app.services.detection_interface import DetectionResult

    class SlowBackend:
        def score(self, path, progress=None):
            for i in range(10):
                if progress:
                    progress((i + 1) / 10, f"Reading frames ({i + 1}/10)")
                time.sleep(0.15)                                            # 1.5 s of blocking work
            return DetectionResult(verdict="REAL", confidence=0.1, model_used="slow", processing_time_ms=0.0,
                                   raw_scores={"structure": 0.1, "windows": 16})

    monkeypatch.setattr("app.services.video.detector.get_video_backend", lambda: SlowBackend())
    uid = video_upload(upload, auth)
    job = start(client, auth, uid)
    poll(client, auth, job["job_id"], until=("running",))
    t0 = time.time()
    latencies = []
    while time.time() - t0 < 1.0:
        s = time.time()
        assert client.get("/health").status_code == 200
        latencies.append(time.time() - s)
        time.sleep(0.05)
    assert max(latencies) < 0.5, f"server was blocked by the scan (slowest /health took {max(latencies):.2f}s)"
    assert poll(client, auth, job["job_id"])[0]["state"] == "done"


# ------------------------------------------------------------------------- manager unit tests

def test_finished_jobs_are_purged_and_active_ones_never_are():
    import asyncio

    from app.services.jobs import Job

    m = JobManager(keep_seconds=0)
    old = Job("old", "u1", "user", "video", state="done", finished_at=time.time() - 5)
    live = Job("live", "u2", "user", "video", state="running")
    m._jobs = {"old": old, "live": live}
    asyncio.run(_noop_submit(m))
    assert "old" not in m._jobs and "live" in m._jobs


async def _noop_submit(m):
    m._purge()


def test_queue_position_counts_only_older_queued_jobs():
    from app.services.jobs import Job

    m = JobManager()
    t = time.time()
    a = Job("a", "u1", "x", "video", state="running", created_at=t)
    b = Job("b", "u2", "x", "video", state="queued", created_at=t + 1)
    c = Job("c", "u3", "x", "video", state="queued", created_at=t + 2)
    m._jobs = {"a": a, "b": b, "c": c}
    assert (m.queue_position(a), m.queue_position(b), m.queue_position(c)) == (None, 1, 2)


def test_progress_never_decreases_and_stays_below_one_until_the_job_is_done():
    """Reports can arrive out of order or overshoot; the value shown to the user must not jump back or hit 100% early."""
    from app.services.jobs import Job

    m = JobManager()
    job = Job("j", "u", "x", "video")
    cb = m._callback(job)
    cb(0.6, "a")
    cb(0.2, "b")                                   # a late/out-of-order report
    assert job.progress == 0.6 and job.stage == "b"
    cb(1.0, "c")                                   # the scoring finished but the result is not stored yet
    assert job.progress == 0.99
    cb(-3, "d")
    assert job.progress == 0.99
    m._finish(job, "done", "Done", result_id="r")
    assert job.progress == 1.0 and job.state == "done"


def test_callback_raises_once_cancel_is_requested():
    from app.services.jobs import Job, JobCancelled

    m = JobManager()
    job = Job("j", "u", "x", "video")
    cb = m._callback(job)
    cb(0.1, "ok")
    job.cancel_requested = True
    with pytest.raises(JobCancelled):
        cb(0.2, "should abort")
