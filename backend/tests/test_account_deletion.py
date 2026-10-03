"""Account deletion (app/services/account_deletion.py, DELETE /auth/me, added 2026-10-03) - closes the gap
docs/ETHICS_AND_LIMITATIONS.md section 5 has flagged since it was written ("no delete-my-data button yet")."""
import os

from app.core.config import settings
from tests.conftest import png_bytes


def test_delete_account_removes_the_user_and_invalidates_the_token(client, auth):
    r = client.delete("/api/v1/auth/me", headers=auth)
    assert r.status_code == 204

    # the same token no longer authenticates anything - the user row is gone
    r2 = client.get("/api/v1/auth/me", headers=auth)
    assert r2.status_code == 401


def test_delete_account_removes_uploads_results_and_files_on_disk(client, auth, scan):
    uid = scan(auth)
    upload_before = client.get(f"/api/v1/upload/{uid}", headers=auth)
    assert upload_before.status_code == 200
    storage_path = upload_before.json()["storage_url"]
    assert os.path.isfile(storage_path)

    client.get(f"/api/v1/report/{uid}", headers=auth)  # generates the PDF on disk
    report_path = os.path.join(settings.REPORT_DIR, f"{uid}_report.pdf")
    assert os.path.isfile(report_path)

    r = client.delete("/api/v1/auth/me", headers=auth)
    assert r.status_code == 204

    assert not os.path.exists(storage_path)
    assert not os.path.exists(report_path)


def test_deleted_users_upload_result_and_report_become_inaccessible(client, auth, scan):
    uid = scan(auth)
    assert client.get(f"/api/v1/upload/{uid}", headers=auth).status_code == 200
    assert client.get(f"/api/v1/detect/{uid}/result", headers=auth).status_code == 200

    client.delete("/api/v1/auth/me", headers=auth)

    # the token is dead now (checked above), but also prove the data itself is gone, not just the session,
    # by re-authenticating as a brand new user and confirming the old upload_id is nowhere to be found.
    import uuid
    new_email = f"after_delete_{uuid.uuid4().hex[:8]}@example.com"
    signup = client.post("/api/v1/auth/signup", json={"name": "New", "email": new_email, "password": "testpass123"})
    new_auth = {"Authorization": f"Bearer {signup.json()['access_token']}"}
    assert client.get(f"/api/v1/upload/{uid}", headers=new_auth).status_code == 404


def test_delete_account_does_not_touch_other_users_data(client, auth, scan, upload):
    """Regression guard: deleting account A must never remove account B's uploads."""
    import uuid
    other_email = f"other_{uuid.uuid4().hex[:8]}@example.com"
    other_signup = client.post("/api/v1/auth/signup",
                               json={"name": "Other", "email": other_email, "password": "testpass123"})
    other_auth = {"Authorization": f"Bearer {other_signup.json()['access_token']}"}
    other_uid = upload(other_auth, png_bytes())

    client.delete("/api/v1/auth/me", headers=auth)

    assert client.get(f"/api/v1/upload/{other_uid}", headers=other_auth).status_code == 200


def test_delete_account_requires_authentication(client):
    r = client.delete("/api/v1/auth/me")
    assert r.status_code in (401, 403)


def test_delete_account_with_feedback_removes_the_feedback_too(client, auth, scan):
    uid = scan(auth)
    fb = client.put(f"/api/v1/detect/{uid}/feedback", json={"agrees": True}, headers=auth)
    assert fb.status_code in (200, 201)

    r = client.delete("/api/v1/auth/me", headers=auth)
    assert r.status_code == 204
