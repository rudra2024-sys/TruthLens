import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.models import FaceReference, MonitoringCheck, MonitoringEvent, MonitoringSession, Upload, User
from app.schemas.schemas import (
    IdentityMatchOut,
    IdentityReferenceOut,
    MonitoringCheckOut,
    MonitoringEventIn,
    MonitoringEventOut,
    MonitoringSessionDetailOut,
    MonitoringSessionOut,
    UploadIdIn,
)
from app.services.audit_log import log_event
from app.services.detection_errors import UnprocessableMediaError
from app.services.identity.detector import enroll_reference, match_against_reference
from app.services.identity.monitoring import run_check
from app.services.report.session_report import generate_session_report

router = APIRouter(prefix="/identity", tags=["Identity"])


async def _owned_upload(db: AsyncSession, upload_id: str, user: User) -> Upload:
    upload = (await db.execute(select(Upload).where(Upload.upload_id == upload_id))).scalar_one_or_none()
    # 404 (not 403) for "unknown" and "not yours" alike, as everywhere else in the API.
    if not upload or upload.user_id != user.user_id:
        raise HTTPException(404, "Upload not found.")
    return upload


async def _reference_for(db: AsyncSession, user: User) -> Optional[FaceReference]:
    r = await db.execute(select(FaceReference).where(FaceReference.user_id == user.user_id))
    return r.scalar_one_or_none()


@router.post("/reference", response_model=IdentityReferenceOut, status_code=201)
async def post_reference(
    body: UploadIdIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enroll (or re-enroll, replacing any existing one) a reference photo for identity-match
    verification. The raw face embedding is never returned in the response."""
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "Reference photo must be an image.")
    try:
        reference = await enroll_reference(upload, db)
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    await log_event(db, "identity_reference_enrolled", user_id=current_user.user_id, detail=upload.upload_id, request=http_request)
    return reference


@router.get("/reference", response_model=Optional[IdentityReferenceOut])
async def get_reference(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The current user's enrolled reference, or null if none has been enrolled yet."""
    return await _reference_for(db, current_user)


@router.delete("/reference", status_code=204)
async def delete_reference(
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Withdraw the enrolled reference (idempotent). A subsequent match attempt will 404 until
    a new reference is enrolled."""
    reference = await _reference_for(db, current_user)
    if reference is not None:
        await db.delete(reference)
        await log_event(db, "identity_reference_deleted", user_id=current_user.user_id, request=http_request)


@router.post("/match", response_model=IdentityMatchOut, status_code=201)
async def post_match(
    body: UploadIdIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compare an already-uploaded live-captured frame against the current user's enrolled
    reference. 404 if nothing has been enrolled yet."""
    reference = await _reference_for(db, current_user)
    if reference is None:
        raise HTTPException(404, "Enroll a reference photo first.")
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "Live capture must be an image.")
    try:
        result = await match_against_reference(reference, upload, db)
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    return result


async def _owned_session(db: AsyncSession, session_id: str, user: User) -> MonitoringSession:
    session = (
        await db.execute(select(MonitoringSession).where(MonitoringSession.session_id == session_id))
    ).scalar_one_or_none()
    if not session or session.user_id != user.user_id:
        raise HTTPException(404, "Monitoring session not found.")
    return session


@router.post("/sessions", response_model=MonitoringSessionOut, status_code=201)
async def post_session(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start a continuous-monitoring session: periodic checks against the current user's
    enrolled reference. 404 if nothing has been enrolled yet."""
    reference = await _reference_for(db, current_user)
    if reference is None:
        raise HTTPException(404, "Enroll a reference photo first.")
    session = MonitoringSession(
        session_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        reference_id=reference.reference_id,
        started_at=datetime.utcnow().isoformat(),
    )
    db.add(session)
    await db.flush()
    return session


@router.post("/sessions/{session_id}/checks", response_model=MonitoringCheckOut, status_code=201)
async def post_session_check(
    session_id: str,
    body: UploadIdIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run one periodic check (identity drift + deepfake signal + presence) against an
    already-uploaded live-captured frame. 400 if the session has already ended."""
    session = await _owned_session(db, session_id, current_user)
    if session.ended_at is not None:
        raise HTTPException(400, "This monitoring session has already ended.")
    reference = (
        await db.execute(select(FaceReference).where(FaceReference.reference_id == session.reference_id))
    ).scalar_one_or_none()
    if reference is None:
        raise HTTPException(404, "The reference this session was started against no longer exists.")
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "Live capture must be an image.")
    try:
        check = await run_check(session, reference, upload, db)
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    return check


@router.post("/sessions/{session_id}/end", response_model=MonitoringSessionOut)
async def post_session_end(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """End a monitoring session (idempotent)."""
    session = await _owned_session(db, session_id, current_user)
    if session.ended_at is None:
        session.ended_at = datetime.utcnow().isoformat()
        await db.flush()
    return session


@router.post("/sessions/{session_id}/events", response_model=MonitoringEventOut, status_code=201)
async def post_session_event(
    session_id: str,
    body: MonitoringEventIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Record a behavioral event (tab/focus loss, clipboard paste, devtools heuristic, camera
    interruption) -- no camera frame or model inference involved, so this is cheap and synchronous.
    400 if the session has already ended, same rule as checks."""
    session = await _owned_session(db, session_id, current_user)
    if session.ended_at is not None:
        raise HTTPException(400, "This monitoring session has already ended.")
    event = MonitoringEvent(
        event_id=str(uuid.uuid4()),
        session_id=session.session_id,
        event_type=body.event_type,
        detail=body.detail,
        occurred_at=datetime.utcnow().isoformat(),
    )
    db.add(event)
    await db.flush()
    return event


@router.get("/sessions/{session_id}", response_model=MonitoringSessionDetailOut)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """A session plus its full check + event timeline, each newest first."""
    session = await _owned_session(db, session_id, current_user)
    checks = (
        await db.execute(
            select(MonitoringCheck)
            .where(MonitoringCheck.session_id == session_id)
            .order_by(MonitoringCheck.checked_at.desc())
        )
    ).scalars().all()
    events = (
        await db.execute(
            select(MonitoringEvent)
            .where(MonitoringEvent.session_id == session_id)
            .order_by(MonitoringEvent.occurred_at.desc())
        )
    ).scalars().all()
    return MonitoringSessionDetailOut(
        session_id=session.session_id,
        reference_id=session.reference_id,
        started_at=session.started_at,
        ended_at=session.ended_at,
        checks=checks,
        events=events,
    )


@router.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """A PDF summary of the session (works for an active or an ended session -- a snapshot so
    far is still useful)."""
    session = await _owned_session(db, session_id, current_user)
    checks = (
        await db.execute(
            select(MonitoringCheck)
            .where(MonitoringCheck.session_id == session_id)
            .order_by(MonitoringCheck.checked_at.asc())
        )
    ).scalars().all()
    events = (
        await db.execute(
            select(MonitoringEvent)
            .where(MonitoringEvent.session_id == session_id)
            .order_by(MonitoringEvent.occurred_at.asc())
        )
    ).scalars().all()
    path = generate_session_report(session, checks, events)
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"TruthLens_Session_Report_{session.session_id}.pdf",
    )
