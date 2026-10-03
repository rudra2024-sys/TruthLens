"""Full account deletion (added 2026-10-03) -- closes the gap docs/ETHICS_AND_LIMITATIONS.md section 5 has
flagged since it was written: "no automatic deletion and no 'delete my data' button yet."

Deletes everything belonging to a user: every upload's stored file and generated PDF report on disk, every
analysis/feedback/detection-result row, then the user row itself. There is no cascade configured at the
ORM/FK level (SQLite doesn't enforce FK actions here), so this walks the dependency order explicitly rather
than relying on one. Irreversible -- this is the real "delete my data" action, not a soft-delete.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import AudioAnalysis, DetectionResult, Feedback, ImageAnalysis, Upload, User, VideoAnalysis

logger = logging.getLogger(__name__)


def _report_path(upload_id: str) -> str:
    return os.path.join(settings.REPORT_DIR, f"{upload_id}_report.pdf")


def _remove_file(path: str | None) -> bool:
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.exception("Could not remove file during account deletion: %s", path)
        return False


async def delete_account(db: AsyncSession, user: User) -> dict:
    """Deletes every row and file belonging to `user`, then the user row itself. Returns counts for the
    caller to report back (the response is the only record of what existed, once this returns the rows are
    gone)."""
    uploads = (await db.execute(select(Upload).where(Upload.user_id == user.user_id))).scalars().all()
    upload_ids = [u.upload_id for u in uploads]

    files_removed = 0
    for u in uploads:
        if _remove_file(u.storage_url):
            files_removed += 1
        if _remove_file(_report_path(u.upload_id)):
            files_removed += 1

    result_ids: list[str] = []
    if upload_ids:
        result_ids = list(
            (await db.execute(select(DetectionResult.result_id).where(DetectionResult.upload_id.in_(upload_ids))))
            .scalars().all()
        )

    feedback_count = 0
    if upload_ids:
        feedback_count = len(
            (await db.execute(select(Feedback.feedback_id).where(Feedback.upload_id.in_(upload_ids))))
            .scalars().all()
        )
        await db.execute(delete(Feedback).where(Feedback.upload_id.in_(upload_ids)))

    if result_ids:
        await db.execute(delete(ImageAnalysis).where(ImageAnalysis.result_id.in_(result_ids)))
        await db.execute(delete(VideoAnalysis).where(VideoAnalysis.result_id.in_(result_ids)))
        await db.execute(delete(AudioAnalysis).where(AudioAnalysis.result_id.in_(result_ids)))
        await db.execute(delete(DetectionResult).where(DetectionResult.result_id.in_(result_ids)))

    if upload_ids:
        await db.execute(delete(Upload).where(Upload.upload_id.in_(upload_ids)))

    await db.execute(delete(User).where(User.user_id == user.user_id))
    await db.flush()

    return {
        "uploads_deleted": len(upload_ids),
        "results_deleted": len(result_ids),
        "feedback_deleted": feedback_count,
        "files_removed": files_removed,
    }
