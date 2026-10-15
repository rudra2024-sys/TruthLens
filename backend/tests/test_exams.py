"""Proctored MCQ exam tests (added 2026-10-12) -- exam/question CRUD, the attempt lifecycle, and
server-side flag-threshold auto-termination via both the synchronous and background-job check
paths. See app/api/routes/exams.py, app/services/exams/{attempts,enforcement}.py.
"""

import asyncio
import time
import uuid

import numpy as np
import pytest
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import User
from app.pipelines.identity.inference import NoFaceDetectedError
from tests.conftest import PASSWORD, _signup, png_bytes

_FRONTAL_LANDMARKS = np.array([
    [292.44, 141.61], [362.04, 139.58], [326.10, 170.51], [299.42, 213.41], [357.15, 212.87],
])


def _signup_org(client, name="examorg"):
    email = f"{name}-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/signup", json={
        "name": name, "email": email, "password": PASSWORD, "account_type": "organization",
    })
    assert r.status_code == 201, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}, email


def _join_code_for(client, org_headers):
    return client.get("/api/v1/admin/join-code", headers=org_headers).json()["join_code"]


def _join(client, headers, join_code):
    r = client.post("/api/v1/identity/organization", json={"join_code": join_code}, headers=headers)
    assert r.status_code == 200, r.text


def _verify_id(client, headers, upload, stub_embed, vector=(1.0, 0.0)):
    stub_embed(vector)
    uid = upload(headers, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _ready_student(client, upload, stub_embed, join_code, name="student"):
    """Signs up, ID-verifies, and joins the given org -- a student who can start an exam."""
    headers, email = _signup(client, name)
    _verify_id(client, headers, upload, stub_embed)
    _join(client, headers, join_code)
    return headers, email


def _create_exam(client, org_headers, pass_threshold=1, max_flags=4, title="Quiz"):
    r = client.post(
        "/api/v1/exams",
        json={"title": title, "pass_threshold": pass_threshold, "max_flags": max_flags},
        headers=org_headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def _add_question(client, org_headers, exam_id, text="2+2?", options=("3", "4", "5"), correct=1):
    r = client.post(
        f"/api/v1/exams/{exam_id}/questions",
        json={"text": text, "options": list(options), "correct_option_index": correct},
        headers=org_headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


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
    monkeypatch.setattr("app.services.identity.detector.assess_government_id_quality", lambda path, detection: [])
    return lambda vector: queue.append(list(vector))


@pytest.fixture()
def stub_monitoring(monkeypatch):
    """Same configurable-state fixture as test_monitoring.py -- controls whether a check comes
    back flagged, via the real production mechanism (no_face -> NoFaceDetectedError)."""
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


# ---- exam/question CRUD -------------------------------------------------------------------

def test_create_exam_requires_organization_role(client):
    headers, _email = _signup(client, "notanorg")
    r = client.post("/api/v1/exams", json={"title": "X", "pass_threshold": 1}, headers=headers)
    assert r.status_code == 403


def test_create_exam_and_add_questions_admin_view_includes_answer_key(client):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    _add_question(client, org_headers, exam["exam_id"])

    r = client.get(f"/api/v1/exams/{exam['exam_id']}/questions/admin", headers=org_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["correct_option_index"] == 1


def test_student_question_view_never_includes_correct_option_index(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    _add_question(client, org_headers, exam["exam_id"])
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)

    r = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student)
    assert r.status_code == 201, r.text

    q = client.get(f"/api/v1/exams/{exam['exam_id']}/questions", headers=student)
    assert q.status_code == 200, q.text
    assert len(q.json()) == 1
    assert "correct_option_index" not in q.json()[0]


# ---- per-attempt identity confidence check (POST /{exam_id}/attempts/verify-identity) ---------

def test_verify_identity_match_lets_attempt_start(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    headers, _email2 = _signup(client, "freshverifystudent")
    _join(client, headers, join_code)

    stub_embed((1.0, 0.0))  # id photo embedding
    stub_embed((1.0, 0.0))  # live selfie embedding -- identical, similarity 1.0
    id_uid = upload(headers, png_bytes(), "id.png", "image/png")
    live_uid = upload(headers, png_bytes(), "live.png", "image/png")

    r = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": True},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["passed"] is True
    assert body["verdict"] == "MATCH"

    start = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=headers)
    assert start.status_code == 201, start.text


def test_verify_identity_mismatch_blocks_attempt_start(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    headers, _email2 = _signup(client, "mismatchstudent")
    _join(client, headers, join_code)

    stub_embed((1.0, 0.0))
    stub_embed((0.0, 1.0))  # orthogonal -- similarity 0.0, well below threshold
    id_uid = upload(headers, png_bytes(), "id.png", "image/png")
    live_uid = upload(headers, png_bytes(), "live.png", "image/png")

    r = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": True},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["passed"] is False
    assert r.json()["verdict"] == "NO_MATCH"

    start = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=headers)
    assert start.status_code == 400


def test_verify_identity_requires_consent(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    headers, _email2 = _signup(client, "noconsentverify")
    _join(client, headers, join_code)

    stub_embed((1.0, 0.0))
    id_uid = upload(headers, png_bytes(), "id.png", "image/png")
    live_uid = upload(headers, png_bytes(), "live.png", "image/png")

    r = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": False},
        headers=headers,
    )
    assert r.status_code == 400


def test_verify_identity_requires_joined_organization(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    headers, _email2 = _signup(client, "unjoinedverify")
    # Deliberately not joined to this exam's org.

    stub_embed((1.0, 0.0))
    id_uid = upload(headers, png_bytes(), "id.png", "image/png")
    live_uid = upload(headers, png_bytes(), "live.png", "image/png")

    r = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": True},
        headers=headers,
    )
    assert r.status_code == 400


def test_verify_identity_consumes_both_uploads(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    headers, _email2 = _signup(client, "consumeuploads")
    _join(client, headers, join_code)

    stub_embed((1.0, 0.0))
    stub_embed((1.0, 0.0))
    id_uid = upload(headers, png_bytes(), "id.png", "image/png")
    live_uid = upload(headers, png_bytes(), "live.png", "image/png")

    first = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": True},
        headers=headers,
    )
    assert first.status_code == 200, first.text

    # Re-using either upload id a second time must fail -- both were deleted after use.
    second = client.post(
        f"/api/v1/exams/{exam['exam_id']}/attempts/verify-identity",
        json={"id_upload_id": id_uid, "live_upload_id": live_uid, "consent": True},
        headers=headers,
    )
    assert second.status_code == 404


# ---- starting an attempt -------------------------------------------------------------------

def test_start_attempt_without_joining_exams_organization_is_rejected(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    headers, _email2 = _signup(client, "unjoinedstudent")
    _verify_id(client, headers, upload, stub_embed)
    # Deliberately not joined to this exam's org.
    r = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=headers)
    assert r.status_code == 400


def test_start_attempt_requires_government_id_verified_reference(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)

    headers, _email2 = _signup(client, "plainselfiestudent")
    # Plain selfie enrollment, NOT government-ID verification.
    stub_embed((1.0, 0.0))
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    _join(client, headers, join_code)

    r = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=headers)
    assert r.status_code == 400


def test_start_attempt_creates_linked_monitoring_session_with_matching_org_snapshot(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)

    r = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={"details": {"roll_no": "7"}}, headers=student)
    assert r.status_code == 201, r.text
    attempt = r.json()
    assert attempt["status"] == "in_progress"
    assert attempt["session_id"]

    # The org sees this session via the normal admin session list -- confirms the org_id snapshot matches.
    sessions = client.get("/api/v1/admin/sessions", headers=org_headers).json()
    assert any(s["session_id"] == attempt["session_id"] for s in sessions)


def test_cannot_start_second_concurrent_attempt(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)

    r1 = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student)
    assert r1.status_code == 201, r1.text
    r2 = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student)
    assert r2.status_code == 400


def test_retake_allowed_after_terminal_attempt(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=0)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)

    first = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()
    client.post(f"/api/v1/exams/attempts/{first['attempt_id']}/finish", headers=student)

    second = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student)
    assert second.status_code == 201, second.text
    assert second.json()["attempt_id"] != first["attempt_id"]


# ---- answering + finishing ------------------------------------------------------------------

def test_submit_answer_upserts_not_duplicates(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers)
    question = _add_question(client, org_headers, exam["exam_id"], correct=1)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    r1 = client.post(
        f"/api/v1/exams/attempts/{attempt['attempt_id']}/answers",
        json={"question_id": question["question_id"], "selected_option_index": 0}, headers=student,
    )
    assert r1.status_code == 201, r1.text
    r2 = client.post(
        f"/api/v1/exams/attempts/{attempt['attempt_id']}/answers",
        json={"question_id": question["question_id"], "selected_option_index": 1}, headers=student,
    )
    assert r2.status_code == 201, r2.text
    assert r1.json()["answer_id"] == r2.json()["answer_id"]
    assert r2.json()["selected_option_index"] == 1


def test_finish_attempt_computes_score_and_pass_fail(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=1)
    q1 = _add_question(client, org_headers, exam["exam_id"], text="q1", correct=1)
    q2 = _add_question(client, org_headers, exam["exam_id"], text="q2", correct=0)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    client.post(
        f"/api/v1/exams/attempts/{attempt['attempt_id']}/answers",
        json={"question_id": q1["question_id"], "selected_option_index": 1}, headers=student,
    )
    client.post(
        f"/api/v1/exams/attempts/{attempt['attempt_id']}/answers",
        json={"question_id": q2["question_id"], "selected_option_index": 2}, headers=student,  # wrong
    )

    r = client.post(f"/api/v1/exams/attempts/{attempt['attempt_id']}/finish", headers=student)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["score"] == 1
    assert body["status"] == "passed"
    assert body["ended_at"]


def test_finish_attempt_is_idempotent(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=0)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    first = client.post(f"/api/v1/exams/attempts/{attempt['attempt_id']}/finish", headers=student).json()
    second = client.post(f"/api/v1/exams/attempts/{attempt['attempt_id']}/finish", headers=student).json()
    assert first["ended_at"] == second["ended_at"]
    assert first["status"] == second["status"]


# ---- server-side flag-threshold enforcement --------------------------------------------------

def test_flag_threshold_auto_terminates_via_sync_check_path(client, upload, stub_embed, stub_monitoring):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, max_flags=2)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    stub_monitoring.update(no_face=True)  # every check now comes back flagged

    for _ in range(2):
        uid = upload(student, png_bytes(), "frame.png", "image/png")
        r = client.post(
            f"/api/v1/identity/sessions/{attempt['session_id']}/checks", json={"upload_id": uid}, headers=student,
        )
        assert r.status_code == 201, r.text

    status = client.get(f"/api/v1/exams/attempts/{attempt['attempt_id']}", headers=student).json()
    assert status["status"] == "terminated_for_flags"
    assert status["ended_at"]

    sess = client.get(f"/api/v1/identity/sessions/{attempt['session_id']}", headers=student).json()
    assert sess["ended_at"]

    # No race window: further answers are rejected now that the server has already ended it.
    q = _add_question(client, org_headers, exam["exam_id"])
    late = client.post(
        f"/api/v1/exams/attempts/{attempt['attempt_id']}/answers",
        json={"question_id": q["question_id"], "selected_option_index": 0}, headers=student,
    )
    assert late.status_code == 400


def test_flag_threshold_auto_terminates_via_background_job_check_path(client, upload, stub_embed, stub_monitoring):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, max_flags=2)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    stub_monitoring.update(no_face=True)

    def _run_job():
        uid = upload(student, png_bytes(), "frame.png", "image/png")
        job = client.post(
            f"/api/v1/identity/sessions/{attempt['session_id']}/checks/jobs", json={"upload_id": uid}, headers=student,
        ).json()
        deadline = time.time() + 10
        while time.time() < deadline:
            polled = client.get(
                f"/api/v1/identity/sessions/{attempt['session_id']}/checks/jobs/{job['job_id']}", headers=student,
            ).json()
            if polled["state"] in ("done", "failed"):
                return polled
            time.sleep(0.02)
        raise AssertionError("job stuck")

    for _ in range(2):
        result = _run_job()
        assert result["state"] == "done", result

    status = client.get(f"/api/v1/exams/attempts/{attempt['attempt_id']}", headers=student).json()
    assert status["status"] == "terminated_for_flags"


def test_attempt_status_reflects_flag_count_before_threshold_is_reached(client, upload, stub_embed, stub_monitoring):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, max_flags=3)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    stub_monitoring.update(no_face=True)
    uid = upload(student, png_bytes(), "frame.png", "image/png")
    client.post(f"/api/v1/identity/sessions/{attempt['session_id']}/checks", json={"upload_id": uid}, headers=student)

    status = client.get(f"/api/v1/exams/attempts/{attempt['attempt_id']}", headers=student).json()
    assert status["status"] == "in_progress"
    assert status["flagged_checks"] == 1
    assert status["max_flags"] == 3


# ---- cross-tenant scoping --------------------------------------------------------------------

def test_org_cannot_see_another_orgs_exam_or_attempts(client, upload, stub_embed):
    org_a, _ = _signup_org(client, "orga")
    org_b, _ = _signup_org(client, "orgb")
    exam = _create_exam(client, org_a, pass_threshold=0)
    join_code_a = _join_code_for(client, org_a)
    student, _ = _ready_student(client, upload, stub_embed, join_code_a)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    assert client.get(f"/api/v1/exams/{exam['exam_id']}/questions/admin", headers=org_b).status_code == 404
    assert client.get(f"/api/v1/admin/exams/{exam['exam_id']}/attempts", headers=org_b).status_code == 404

    own = client.get(f"/api/v1/admin/exams/{exam['exam_id']}/attempts", headers=org_a).json()
    assert any(a["attempt_id"] == attempt["attempt_id"] for a in own)


def test_admin_attempt_list_includes_session_id_for_report_download(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=0)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)
    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    rows = client.get(f"/api/v1/admin/exams/{exam['exam_id']}/attempts", headers=org_headers).json()
    row = next(r for r in rows if r["attempt_id"] == attempt["attempt_id"])
    assert row["session_id"] == attempt["session_id"]

    report = client.get(f"/api/v1/admin/sessions/{row['session_id']}/report", headers=org_headers)
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"


# ---- student-facing exam discovery (GET /exams/mine) ------------------------------------------

def test_list_my_exams_is_empty_before_joining_an_organization(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    _create_exam(client, org_headers)
    headers, _email2 = _signup(client, "unjoinedbrowser")
    _verify_id(client, headers, upload, stub_embed)
    # Deliberately not joined to any organisation.

    r = client.get("/api/v1/exams/mine", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json() == []


def test_list_my_exams_only_shows_joined_orgs_exams_with_no_attempt_yet(client, upload, stub_embed):
    org_a, _ = _signup_org(client, "orga2")
    org_b, _ = _signup_org(client, "orgb2")
    exam_a = _create_exam(client, org_a, title="Org A Quiz")
    _create_exam(client, org_b, title="Org B Quiz")
    join_code_a = _join_code_for(client, org_a)
    student, _ = _ready_student(client, upload, stub_embed, join_code_a)

    r = client.get("/api/v1/exams/mine", headers=student)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body) == 1
    assert body[0]["exam_id"] == exam_a["exam_id"]
    assert body[0]["my_attempt_id"] is None
    assert body[0]["my_attempt_status"] is None


def test_list_my_exams_reflects_in_progress_then_completed_status(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=0)
    join_code = _join_code_for(client, org_headers)
    student, _ = _ready_student(client, upload, stub_embed, join_code)

    attempt = client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student).json()

    mid = client.get("/api/v1/exams/mine", headers=student).json()
    assert mid[0]["my_attempt_id"] == attempt["attempt_id"]
    assert mid[0]["my_attempt_status"] == "in_progress"

    client.post(f"/api/v1/exams/attempts/{attempt['attempt_id']}/finish", headers=student)

    after = client.get("/api/v1/exams/mine", headers=student).json()
    assert after[0]["my_attempt_status"] in ("passed", "failed")


def test_list_my_exams_does_not_leak_another_students_attempt_status(client, upload, stub_embed):
    org_headers, _email = _signup_org(client)
    exam = _create_exam(client, org_headers, pass_threshold=0)
    join_code = _join_code_for(client, org_headers)
    student_a, _ = _ready_student(client, upload, stub_embed, join_code, name="peeka")
    student_b, _ = _ready_student(client, upload, stub_embed, join_code, name="peekb")

    client.post(f"/api/v1/exams/{exam['exam_id']}/attempts", json={}, headers=student_a)

    # student_a has an in-progress attempt, but student_b -- who never started one -- must not
    # see it reflected as their own status.
    b_view = client.get("/api/v1/exams/mine", headers=student_b).json()
    assert b_view[0]["my_attempt_id"] is None
    assert b_view[0]["my_attempt_status"] is None
