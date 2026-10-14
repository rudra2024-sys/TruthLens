"""Organisation join-code tests (added 2026-10-12) -- the self-serve link a student uses to tie
their account to a specific organisation before enrolling, and the organisation-side endpoints
that issue/regenerate the code. See admin.py/identity.py for the routes, models.py for the
User.organization_id/join_code columns.
"""

import uuid

from tests.conftest import PASSWORD, png_bytes, _signup


def _signup_org(client, name="testorg"):
    email = f"{name}-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/signup", json={
        "name": name, "email": email, "password": PASSWORD, "account_type": "organization",
    })
    assert r.status_code == 201, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}, email


def test_invalid_join_code_is_rejected(client):
    headers, _email = _signup(client, "joinertest")
    r = client.post("/api/v1/identity/organization", json={"join_code": "NOTREAL1"}, headers=headers)
    assert r.status_code == 400


def test_valid_join_code_links_student_to_organization(client):
    org_headers, _org_email = _signup_org(client)
    join_code = client.get("/api/v1/admin/join-code", headers=org_headers).json()["join_code"]

    student_headers, _student_email = _signup(client, "linker")
    r = client.post("/api/v1/identity/organization", json={"join_code": join_code}, headers=student_headers)
    assert r.status_code == 200, r.text
    assert r.json()["organization_id"]
    assert r.json()["organization_name"]

    me = client.get("/api/v1/identity/organization", headers=student_headers)
    assert me.status_code == 200
    assert me.json()["organization_id"] == r.json()["organization_id"]


def test_unjoined_student_sees_null_organization(client):
    headers, _email = _signup(client, "neverjoined")
    r = client.get("/api/v1/identity/organization", headers=headers)
    assert r.status_code == 200
    assert r.json() is None


def test_join_code_is_lazily_generated_once_and_persists(client):
    org_headers, _email = _signup_org(client)
    first = client.get("/api/v1/admin/join-code", headers=org_headers).json()["join_code"]
    second = client.get("/api/v1/admin/join-code", headers=org_headers).json()["join_code"]
    assert first == second


def test_regenerate_issues_new_code_without_unlinking_existing_students(client):
    org_headers, _email = _signup_org(client)
    old_code = client.get("/api/v1/admin/join-code", headers=org_headers).json()["join_code"]

    student_headers, _student_email = _signup(client, "staylinked")
    client.post("/api/v1/identity/organization", json={"join_code": old_code}, headers=student_headers)

    new_code = client.post("/api/v1/admin/join-code/regenerate", headers=org_headers).json()["join_code"]
    assert new_code != old_code

    # Already-linked student is unaffected.
    me = client.get("/api/v1/identity/organization", headers=student_headers).json()
    assert me is not None

    # The old code no longer works; the new one does.
    other_headers, _other_email = _signup(client, "latecomer")
    assert client.post("/api/v1/identity/organization", json={"join_code": old_code}, headers=other_headers).status_code == 400
    assert client.post("/api/v1/identity/organization", json={"join_code": new_code}, headers=other_headers).status_code == 200


def test_switching_organization_only_affects_future_sessions(client, upload, monkeypatch):
    """The whole point of snapshotting organization_id onto MonitoringSession instead of a live
    join: re-linking to a different org must not retroactively move past sessions."""
    monkeypatch.setattr("app.services.identity.detector._embed", lambda path: [1.0, 0.0])
    monkeypatch.setattr(
        "app.services.identity.detector._detect_faces",
        lambda path: {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)},
    )
    monkeypatch.setattr("app.services.identity.detector.assess_enrollment_quality", lambda path, detection: [])

    org_a_headers, org_a_email = _signup_org(client, "orga")
    org_b_headers, org_b_email = _signup_org(client, "orgb")
    code_a = client.get("/api/v1/admin/join-code", headers=org_a_headers).json()["join_code"]
    code_b = client.get("/api/v1/admin/join-code", headers=org_b_headers).json()["join_code"]

    student_headers, student_email = _signup(client, "switcher")

    # Join org A, enroll, start a session.
    client.post("/api/v1/identity/organization", json={"join_code": code_a}, headers=student_headers)
    uid = upload(student_headers, png_bytes(), "ref.png", "image/png")
    client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=student_headers)
    session_1 = client.post("/api/v1/identity/sessions", headers=student_headers).json()
    client.post(f"/api/v1/identity/sessions/{session_1['session_id']}/end", headers=student_headers)

    # Switch to org B -- re-enrolling isn't required, just re-joining and starting a new session.
    client.post("/api/v1/identity/organization", json={"join_code": code_b}, headers=student_headers)
    session_2 = client.post("/api/v1/identity/sessions", headers=student_headers).json()

    ids_seen_by_a = {row["session_id"] for row in client.get("/api/v1/admin/sessions", headers=org_a_headers).json()}
    ids_seen_by_b = {row["session_id"] for row in client.get("/api/v1/admin/sessions", headers=org_b_headers).json()}

    assert session_1["session_id"] in ids_seen_by_a
    assert session_2["session_id"] not in ids_seen_by_a
    assert session_2["session_id"] in ids_seen_by_b
    assert session_1["session_id"] not in ids_seen_by_b
