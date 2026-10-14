"""Session-check media retention script tests (added 2026-10-12) -- see
scripts/purge_old_session_media.py.
"""

import asyncio
import os

import pytest
from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import MonitoringCheck, Upload
from scripts.purge_old_session_media import purge_old_session_media
from tests.conftest import png_bytes


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
    class _FakePipeline:
        def detect_faces(self, path):
            return {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)}

        def embed(self, path):
            return [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.monitoring.get_identity_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr("app.services.identity.monitoring._run_deepfake_check", lambda path: (0.1, "REAL"))
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_object_pipeline",
        lambda: type("_O", (), {"detect": staticmethod(lambda path, conf_thresh=0.4: [])})(),
    )


def _make_old_check(client, auth, upload, stub_embed, stub_monitoring):
    """Enrolls, starts a session, submits one real check (a real Upload row + real on-disk file
    get created), then backdates that check's checked_at so it's eligible for purge. Returns
    (check_id, the Upload's storage_url) so the test can confirm the file actually existed."""
    stub_embed((1.0, 0.0))
    uid = upload(auth, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 201, r.text
    session = client.post("/api/v1/identity/sessions", headers=auth).json()

    frame_uid = upload(auth, png_bytes(), "frame.png", "image/png")
    check_r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks",
        json={"upload_id": frame_uid}, headers=auth,
    )
    assert check_r.status_code == 201, check_r.text
    check_id = check_r.json()["check_id"]

    async def _backdate():
        async with AsyncSessionLocal() as db:
            check = (await db.execute(select(MonitoringCheck).where(MonitoringCheck.check_id == check_id))).scalar_one()
            check.checked_at = "2000-01-01T00:00:00"
            upload_row = (await db.execute(select(Upload).where(Upload.upload_id == check.upload_id))).scalar_one()
            storage_url = upload_row.storage_url
            await db.commit()
            return storage_url
    storage_url = asyncio.run(_backdate())
    return check_id, storage_url


def test_dry_run_reports_counts_without_deleting(client, auth, upload, stub_embed, stub_monitoring):
    _check_id, storage_url = _make_old_check(client, auth, upload, stub_embed, stub_monitoring)
    assert os.path.exists(storage_url)

    async def _run():
        async with AsyncSessionLocal() as db:
            return await purge_old_session_media(db, older_than_days=1, dry_run=True)
    result = asyncio.run(_run())

    assert result["checks_considered"] >= 1
    assert result["uploads_deleted"] == 0
    assert os.path.exists(storage_url)  # untouched


def test_purge_deletes_old_upload_row_and_file_keeps_check_metadata(client, auth, upload, stub_embed, stub_monitoring):
    check_id, storage_url = _make_old_check(client, auth, upload, stub_embed, stub_monitoring)
    assert os.path.exists(storage_url)

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await purge_old_session_media(db, older_than_days=1, dry_run=False)
            await db.commit()
            return result
    result = asyncio.run(_run())

    assert result["uploads_deleted"] >= 1
    assert result["files_removed"] >= 1
    assert not os.path.exists(storage_url)

    # The MonitoringCheck row itself, and its derived fields, survive the purge untouched.
    async def _check_survives():
        async with AsyncSessionLocal() as db:
            check = (await db.execute(select(MonitoringCheck).where(MonitoringCheck.check_id == check_id))).scalar_one_or_none()
            return check
    check = asyncio.run(_check_survives())
    assert check is not None
    assert check.face_count == 1


def test_purge_is_idempotent_on_rerun(client, auth, upload, stub_embed, stub_monitoring):
    _check_id, storage_url = _make_old_check(client, auth, upload, stub_embed, stub_monitoring)

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await purge_old_session_media(db, older_than_days=1, dry_run=False)
            await db.commit()
            return result
    first = asyncio.run(_run())
    second = asyncio.run(_run())

    assert first["uploads_deleted"] >= 1
    assert second["uploads_deleted"] == 0
    assert second["files_removed"] == 0


def test_recent_checks_are_not_purged(client, auth, upload, stub_embed, stub_monitoring):
    stub_embed((1.0, 0.0))
    uid = upload(auth, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 201, r.text
    session = client.post("/api/v1/identity/sessions", headers=auth).json()
    frame_uid = upload(auth, png_bytes(), "frame.png", "image/png")
    check_r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks",
        json={"upload_id": frame_uid}, headers=auth,
    )
    assert check_r.status_code == 201, check_r.text

    async def _run():
        async with AsyncSessionLocal() as db:
            result = await purge_old_session_media(db, older_than_days=30, dry_run=False)
            await db.commit()
            return result
    result = asyncio.run(_run())
    # This fresh check must not be among whatever (if anything) got purged.
    upload_row_id = check_r.json()["check_id"]

    async def _still_has_upload():
        async with AsyncSessionLocal() as db:
            check = (await db.execute(select(MonitoringCheck).where(MonitoringCheck.check_id == upload_row_id))).scalar_one()
            upload_row = (await db.execute(select(Upload).where(Upload.upload_id == check.upload_id))).scalar_one_or_none()
            return upload_row
    assert asyncio.run(_still_has_upload()) is not None
