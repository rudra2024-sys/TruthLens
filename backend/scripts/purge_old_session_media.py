"""Session-check media retention (added 2026-10-12) -- deletes the raw captured-frame images
backing old MonitoringCheck rows, past a configurable age. A data-minimization practice for
session-monitoring images (biometric-adjacent data, same sensitivity class as the enrolled
reference photo): the derived signals already stored directly on MonitoringCheck (flags, scores,
face_box_json/landmarks_json) are untouched, so PDF report generation keeps working exactly as
before -- session_report.py never re-reads a check's raw image, only these already-denormalized
fields. Only the Upload row and its on-disk file are removed.

Manual script, matching scripts/promote_admin.py / scripts/export_feedback.py's convention -- no
scheduler/cron infrastructure exists in this project, and this doesn't introduce one; run it
periodically (e.g. from an OS-level scheduled task) however the deployment already runs things.

  cd backend && ../.venv/Scripts/python.exe scripts/purge_old_session_media.py --older-than-days 30
  (add --dry-run to preview counts without deleting anything)

Idempotent: re-running finds the same old MonitoringCheck rows (checked_at doesn't change), but
their Upload rows are already gone, so a second run just reports 0 deleted -- safe to schedule on
a loop without needing to track what was already purged.

Reads/writes the database configured by DATABASE_URL (the same one the app uses).
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select                # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession       # noqa: E402

from app.core.database import AsyncSessionLocal       # noqa: E402
from app.models.models import MonitoringCheck, Upload  # noqa: E402

logger = logging.getLogger(__name__)


def _remove_file(path: str | None) -> bool:
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Could not remove file during session-media purge: %s", path)
        return False


async def purge_old_session_media(db: AsyncSession, older_than_days: int, dry_run: bool = False) -> dict:
    """Deletes the Upload row + on-disk file for every MonitoringCheck older than
    `older_than_days`. Returns counts for the caller to report. Kept separate from main() so
    tests can exercise this directly, same convention as scripts/promote_admin.py::set_role."""
    cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
    checks = (
        await db.execute(select(MonitoringCheck).where(MonitoringCheck.checked_at < cutoff))
    ).scalars().all()

    upload_ids = [c.upload_id for c in checks if c.upload_id]
    if not upload_ids:
        return {"checks_considered": len(checks), "uploads_deleted": 0, "files_removed": 0}

    uploads = (await db.execute(select(Upload).where(Upload.upload_id.in_(upload_ids)))).scalars().all()

    files_removed = 0
    if not dry_run:
        for u in uploads:
            if _remove_file(u.storage_url):
                files_removed += 1
        if uploads:
            await db.execute(delete(Upload).where(Upload.upload_id.in_([u.upload_id for u in uploads])))
        await db.flush()

    return {
        "checks_considered": len(checks),
        "uploads_deleted": 0 if dry_run else len(uploads),
        "files_removed": files_removed,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--older-than-days", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting anything.")
    args = ap.parse_args()

    async with AsyncSessionLocal() as db:
        result = await purge_old_session_media(db, args.older_than_days, dry_run=args.dry_run)
        if not args.dry_run:
            await db.commit()
        label = "Would purge" if args.dry_run else "Purged"
        print(
            f"{label}: {result['uploads_deleted']} upload(s) / {result['files_removed']} file(s) "
            f"from {result['checks_considered']} check(s) older than {args.older_than_days} days."
        )


if __name__ == "__main__":
    asyncio.run(main())
