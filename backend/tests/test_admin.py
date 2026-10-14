"""Admin/proctor dashboard tests (added 2026-10-11) -- role gating, cross-user session listing,
the flagged_only filter, report download, and promote_admin.py's set_role().
"""

import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import MonitoringCheck, User
from tests.conftest import PASSWORD, png_bytes

from scripts.promote_admin import set_role


def _signup(client, name):
    email = f"{name}-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/signup", json={"name": name, "email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}, email


def _promote(email: str) -> None:
    async def _do():
        async with AsyncSessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalar_one()
            user.role = "organization"
            await db.commit()
    asyncio.run(_do())


@pytest.fixture()
def admin_auth(client):
    headers, email = _signup(client, "admin")
    _promote(email)
    return headers


@pytest.fixture()
def admin_join_code(client, admin_auth):
    return client.get("/api/v1/admin/join-code", headers=admin_auth).json()["join_code"]


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


def _enroll_and_start_session(client, headers, upload, stub_embed, join_code):
    r0 = client.post("/api/v1/identity/organization", json={"join_code": join_code}, headers=headers)
    assert r0.status_code == 200, r0.text
    stub_embed((1.0, 0.0))
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    s = client.post("/api/v1/identity/sessions", headers=headers)
    assert s.status_code == 201, s.text
    return s.json()


def test_non_organization_gets_403(client, auth):
    r = client.get("/api/v1/admin/sessions", headers=auth)
    assert r.status_code == 403


def test_signup_with_organization_account_type_grants_admin_access(client):
    email = f"org-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/signup", json={
        "name": "Org", "email": email, "password": PASSWORD, "account_type": "organization",
    })
    assert r.status_code == 201, r.text
    assert r.json()["role"] == "organization"
    headers = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert client.get("/api/v1/admin/sessions", headers=headers).status_code == 200


def test_signup_defaults_to_student_account_type(client):
    headers, _email = _signup(client, "plainstudent")
    me = client.get("/api/v1/auth/me", headers=headers)
    assert me.json()["role"] == "student"
    assert client.get("/api/v1/admin/sessions", headers=headers).status_code == 403


def test_legacy_admin_role_value_still_grants_access(client):
    """A role value of "admin" (this column's first version, before account_type existed, e.g.
    from an older scripts/promote_admin.py run) must keep working -- see
    auth.py::get_current_organization's docstring."""
    headers, email = _signup(client, "legacyadmin")

    async def _set_legacy():
        async with AsyncSessionLocal() as db:
            user = (await db.execute(select(User).where(User.email == email))).scalar_one()
            user.role = "admin"
            await db.commit()
    asyncio.run(_set_legacy())

    assert client.get("/api/v1/admin/sessions", headers=headers).status_code == 200


def test_admin_sees_sessions_across_users(client, auth, other_auth, admin_auth, admin_join_code, upload, stub_embed):
    session_a = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)
    session_b = _enroll_and_start_session(client, other_auth, upload, stub_embed, admin_join_code)

    r = client.get("/api/v1/admin/sessions", headers=admin_auth)
    assert r.status_code == 200
    ids = {row["session_id"] for row in r.json()}
    assert session_a["session_id"] in ids
    assert session_b["session_id"] in ids


def test_admin_only_sees_sessions_from_its_own_joined_students(client, auth, other_auth, admin_auth, admin_join_code, upload, stub_embed):
    """The real multi-tenancy fix (added 2026-10-12): org A's dashboard must not show org B's
    students, even though both orgs exist in the same DB."""
    org_b_headers, org_b_email = _signup(client, "orgb")
    _promote(org_b_email)
    org_b_join_code = client.get("/api/v1/admin/join-code", headers=org_b_headers).json()["join_code"]

    session_a = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)
    session_b = _enroll_and_start_session(client, other_auth, upload, stub_embed, org_b_join_code)

    ids_seen_by_a = {row["session_id"] for row in client.get("/api/v1/admin/sessions", headers=admin_auth).json()}
    ids_seen_by_b = {row["session_id"] for row in client.get("/api/v1/admin/sessions", headers=org_b_headers).json()}

    assert session_a["session_id"] in ids_seen_by_a
    assert session_b["session_id"] not in ids_seen_by_a
    assert session_b["session_id"] in ids_seen_by_b
    assert session_a["session_id"] not in ids_seen_by_b


def test_admin_cannot_view_another_orgs_session_by_id(client, auth, admin_auth, admin_join_code, upload, stub_embed):
    """Before 2026-10-12, _any_session had no ownership filter at all -- any organisation account
    could view or download any student's session by ID even though the list view only showed its
    own. This is the fix for that leak, on both the detail and report-download routes."""
    org_b_headers, org_b_email = _signup(client, "orgc")
    _promote(org_b_email)

    session = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)

    assert client.get(f"/api/v1/admin/sessions/{session['session_id']}", headers=org_b_headers).status_code == 404
    assert client.get(f"/api/v1/admin/sessions/{session['session_id']}/report", headers=org_b_headers).status_code == 404
    # Sanity check: the owning org can still see it (proves the 404 above is scoping, not a bug).
    assert client.get(f"/api/v1/admin/sessions/{session['session_id']}", headers=admin_auth).status_code == 200


def test_flagged_only_filter(client, auth, admin_auth, admin_join_code, upload, stub_embed, monkeypatch):
    session = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)

    # Force the check to flag via a stubbed no-face frame.
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_identity_pipeline",
        lambda: type("_P", (), {"detect_faces": staticmethod(lambda path: {
            "count": 0, "landmarks": None, "box": None, "image_size": (480, 640),
        })})(),
    )
    monkeypatch.setattr("app.services.identity.monitoring._run_deepfake_check", lambda path: (0.1, "REAL"))
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_object_pipeline",
        lambda: type("_O", (), {"detect": staticmethod(lambda path, conf_thresh=0.4: [])})(),
    )
    uid = upload(auth, png_bytes(), "frame.png", "image/png")
    r = client.post(f"/api/v1/identity/sessions/{session['session_id']}/checks", json={"upload_id": uid}, headers=auth)
    assert r.status_code == 201, r.text

    flagged = client.get("/api/v1/admin/sessions", params={"flagged_only": True}, headers=admin_auth).json()
    assert any(row["session_id"] == session["session_id"] for row in flagged)
    for row in flagged:
        assert row["flagged_checks"] >= 1


def test_admin_session_detail_includes_owner(client, auth, admin_auth, admin_join_code, upload, stub_embed):
    session = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)
    r = client.get(f"/api/v1/admin/sessions/{session['session_id']}", headers=admin_auth)
    assert r.status_code == 200
    body = r.json()
    assert body["user_email"]
    assert body["checks"] == []


def test_admin_session_detail_404_for_unknown_session(client, admin_auth):
    r = client.get("/api/v1/admin/sessions/does-not-exist", headers=admin_auth)
    assert r.status_code == 404


def test_admin_can_download_report_for_any_users_session(client, auth, admin_auth, admin_join_code, upload, stub_embed):
    session = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)
    r = client.get(f"/api/v1/admin/sessions/{session['session_id']}/report", headers=admin_auth)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"


def test_promote_admin_script_sets_and_revokes_role(client):
    """Exercises scripts/promote_admin.py's own set_role() directly, in-process -- a subprocess
    invocation was tried first and dropped: on this Windows/aiosqlite combination a child
    process opening the same DATABASE_URL string did not reliably see the parent's writes
    (confirmed via direct file inspection, not assumed), a platform/driver quirk unrelated to
    the script's own logic. Testing set_role() directly covers that logic without depending on
    cross-process SQLite file visibility."""
    _headers, email = _signup(client, "scripttarget")

    async def _run():
        async with AsyncSessionLocal() as db:
            outcome = await set_role(db, email, "organization")
            user = (await db.execute(select(User).where(User.email == email))).scalar_one()
            return outcome, user.role

    outcome, role = asyncio.run(_run())
    assert "organization" in outcome
    assert role == "organization"

    # Idempotent: promoting an already-organization account says so and changes nothing.
    async def _run_again():
        async with AsyncSessionLocal() as db:
            return await set_role(db, email, "organization")
    assert "already has role" in asyncio.run(_run_again())

    # Revoke path.
    async def _revoke():
        async with AsyncSessionLocal() as db:
            await set_role(db, email, "student")
            user = (await db.execute(select(User).where(User.email == email))).scalar_one()
            return user.role
    assert asyncio.run(_revoke()) == "student"


def test_promote_admin_unknown_email_returns_none(client):
    async def _run():
        async with AsyncSessionLocal() as db:
            return await set_role(db, "nobody-here@example.com", "organization")
    assert asyncio.run(_run()) is None


def test_roster_lists_only_this_orgs_linked_students(client, auth, admin_auth, admin_join_code):
    """Roster (added 2026-10-12) is a separate view from the session list -- a student who has
    joined but never started a session still belongs on it."""
    org_b_headers, org_b_email = _signup(client, "orgd")
    _promote(org_b_email)

    r0 = client.post("/api/v1/identity/organization", json={"join_code": admin_join_code}, headers=auth)
    assert r0.status_code == 200, r0.text

    roster = client.get("/api/v1/admin/roster", headers=admin_auth).json()
    assert any(s["user_id"] for s in roster)
    roster_org_b = client.get("/api/v1/admin/roster", headers=org_b_headers).json()
    assert roster_org_b == []


def test_remove_from_roster_unlinks_student_but_keeps_past_sessions_visible(client, auth, admin_auth, admin_join_code, upload, stub_embed):
    session = _enroll_and_start_session(client, auth, upload, stub_embed, admin_join_code)

    roster = client.get("/api/v1/admin/roster", headers=admin_auth).json()
    student_id = next(s["user_id"] for s in roster if s["email"])
    assert any(s["user_id"] == student_id for s in roster)

    r = client.delete(f"/api/v1/admin/roster/{student_id}", headers=admin_auth)
    assert r.status_code == 204, r.text

    roster_after = client.get("/api/v1/admin/roster", headers=admin_auth).json()
    assert not any(s["user_id"] == student_id for s in roster_after)

    # The already-created session stays visible -- only future sessions require rejoining.
    sessions_after = client.get("/api/v1/admin/sessions", headers=admin_auth).json()
    assert any(row["session_id"] == session["session_id"] for row in sessions_after)


def test_remove_from_roster_rejects_another_orgs_student(client, auth, admin_auth, admin_join_code):
    org_b_headers, org_b_email = _signup(client, "orge")
    _promote(org_b_email)

    client.post("/api/v1/identity/organization", json={"join_code": admin_join_code}, headers=auth)
    roster = client.get("/api/v1/admin/roster", headers=admin_auth).json()
    student_id = roster[0]["user_id"]

    r = client.delete(f"/api/v1/admin/roster/{student_id}", headers=org_b_headers)
    assert r.status_code == 404
