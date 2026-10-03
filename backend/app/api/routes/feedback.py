from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.models import DetectionResult, Feedback, Upload, User
from app.schemas.schemas import FeedbackIn, FeedbackOut
from app.services.audit_log import log_event

router = APIRouter(prefix="/detect", tags=["Feedback"])


async def _owned_upload(db: AsyncSession, upload_id: str, user: User) -> Upload:
    upload = (await db.execute(select(Upload).where(Upload.upload_id == upload_id))).scalar_one_or_none()
    # 404 (not 403) for "unknown" and "not yours" alike, as everywhere else in the API.
    if not upload or upload.user_id != user.user_id:
        raise HTTPException(404, "Upload not found.")
    return upload


async def _feedback_for(db: AsyncSession, upload_id: str, user: User) -> Optional[Feedback]:
    r = await db.execute(select(Feedback).where(Feedback.upload_id == upload_id, Feedback.user_id == user.user_id))
    return r.scalars().first()


@router.get("/{upload_id}/feedback", response_model=Optional[FeedbackOut])
async def get_feedback(upload_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    """The current user's feedback on this result, or null if none has been given."""
    await _owned_upload(db, upload_id, current_user)
    return await _feedback_for(db, upload_id, current_user)


@router.put("/{upload_id}/feedback", response_model=FeedbackOut)
async def put_feedback(
    upload_id: str,
    body: FeedbackIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Say whether a detection result was correct. Idempotent: one feedback per upload, resubmitting replaces it."""
    await _owned_upload(db, upload_id, current_user)
    result = (await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))).scalars().first()
    if result is None:
        raise HTTPException(404, "There is no detection result to give feedback on yet.")

    comment = (body.comment or "").strip() or None
    # Agreeing needs no label; disagreeing without saying what the file is counts as "unsure".
    true_label = None if body.agrees else (body.true_label or "unsure")
    fb = await _feedback_for(db, upload_id, current_user)
    now = datetime.utcnow().isoformat()
    if fb is None:
        fb = Feedback(upload_id=upload_id, user_id=current_user.user_id, created_at=now)
        db.add(fb)
    fb.result_id, fb.verdict = result.result_id, result.verdict
    fb.agrees, fb.true_label, fb.comment, fb.allow_reuse, fb.updated_at = body.agrees, true_label, comment, body.allow_reuse, now
    await db.flush()
    return fb


@router.delete("/{upload_id}/feedback", status_code=204)
async def delete_feedback(
    upload_id: str,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Withdraw feedback (and with it any consent to reuse the file)."""
    await _owned_upload(db, upload_id, current_user)
    fb = await _feedback_for(db, upload_id, current_user)
    if fb is not None:
        await db.delete(fb)
        await log_event(db, "feedback_withdrawn", user_id=current_user.user_id, detail=upload_id, request=http_request)
