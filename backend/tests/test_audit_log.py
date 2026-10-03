"""Audit log (app/services/audit_log.py, AuditLogEntry, added 2026-10-03) - closes the gap
docs/ETHICS_AND_LIMITATIONS.md section 5 has flagged since it was written ("no audit log")."""
import asyncio

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import AuditLogEntry
from app.services.audit_log import log_event


def _entries(action=None, user_id=None):
    async def _query():
        async with AsyncSessionLocal() as db:
            stmt = select(AuditLogEntry)
            if action:
                stmt = stmt.where(AuditLogEntry.action == action)
            if user_id:
                stmt = stmt.where(AuditLogEntry.user_id == user_id)
            return (await db.execute(stmt)).scalars().all()
    return asyncio.run(_query())


def test_signup_and_login_are_logged(client):
    email = f"audit_{__import__('uuid').uuid4().hex[:8]}@example.com"
    signup = client.post("/api/v1/auth/signup", json={"name": "Audit", "email": email, "password": "testpass123"})
    uid = signup.json()["user_id"]
    assert any(e.action == "signup" and e.user_id == uid for e in _entries())

    client.post("/api/v1/auth/login", json={"email": email, "password": "testpass123"})
    assert any(e.action == "login" and e.user_id == uid for e in _entries())


def test_failed_login_is_logged_with_the_attempted_email_not_a_user_id(client):
    email = f"audit_fail_{__import__('uuid').uuid4().hex[:8]}@example.com"
    client.post("/api/v1/auth/signup", json={"name": "Audit", "email": email, "password": "testpass123"})
    client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password"})
    hits = [e for e in _entries(action="login_failed") if e.detail == email]
    assert len(hits) == 1
    assert hits[0].user_id is None


def test_feedback_withdrawal_is_logged(client, scan, auth):
    uid = scan(auth)
    client.put(f"/api/v1/detect/{uid}/feedback", json={"agrees": True}, headers=auth)
    client.delete(f"/api/v1/detect/{uid}/feedback", headers=auth)
    assert any(e.action == "feedback_withdrawn" and e.detail == uid for e in _entries(action="feedback_withdrawn"))


def test_account_deletion_is_logged_and_survives_the_deleted_account(client):
    email = f"audit_del_{__import__('uuid').uuid4().hex[:8]}@example.com"
    signup = client.post("/api/v1/auth/signup", json={"name": "Audit", "email": email, "password": "testpass123"})
    uid = signup.json()["user_id"]
    token = signup.json()["access_token"]
    client.delete("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})

    hits = [e for e in _entries(action="account_deleted") if e.user_id == uid]
    assert len(hits) == 1   # the log entry exists even though the user row it refers to is now gone


def test_log_event_failure_does_not_raise():
    """A broken audit write must never break the action it's logging."""
    class _BrokenSession:
        def add(self, *a, **k):
            raise RuntimeError("simulated DB failure")

        async def commit(self):
            pass

    asyncio.run(log_event(_BrokenSession(), "test_action"))  # must not raise


def test_log_event_survives_the_callers_request_failing(client):
    """Regression for the bug this design fixes: a failed-login 401 used to roll back the whole request
    session, silently discarding the login_failed entry logged just before the exception was raised."""
    email = f"audit_survive_{__import__('uuid').uuid4().hex[:8]}@example.com"
    client.post("/api/v1/auth/signup", json={"name": "Audit", "email": email, "password": "testpass123"})
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "wrong-password"})
    assert r.status_code == 401
    assert any(e.action == "login_failed" and e.detail == email for e in _entries(action="login_failed"))
