import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.models import FaceReference, MonitoringCheck, MonitoringEvent, MonitoringSession, Upload, User
from app.schemas.schemas import (
    CheckJobOut,
    EnrollGovernmentIdIn,
    EnrollOrganizationIn,
    EnrollReferenceIn,
    IdentityMatchOut,
    IdentityReferenceOut,
    MonitoringCheckIn,
    MonitoringCheckOut,
    MonitoringEventIn,
    MonitoringEventOut,
    MonitoringSessionDetailOut,
    MonitoringSessionOut,
    MySessionListItemOut,
    OrganizationOut,
    UploadIdIn,
)
from app.services.account_deletion import _remove_file
from app.services.audit_log import log_event
from app.services.detection_errors import UnprocessableMediaError
from app.services.identity.detector import enroll_reference, match_against_reference, verify_government_id
from app.services.exams.enforcement import enforce_flag_threshold
from app.services.identity.monitoring import run_check
from app.services.join_code import normalize_join_code
from app.services.monitoring_jobs import manager as check_job_manager
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


async def _organization_for(db: AsyncSession, join_code: str) -> Optional[User]:
    normalized = normalize_join_code(join_code)
    r = await db.execute(
        select(User).where(User.join_code == normalized, User.role.in_(("organization", "admin")))
    )
    return r.scalar_one_or_none()


@router.post("/organization", response_model=OrganizationOut)
async def post_organization(
    body: EnrollOrganizationIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Link the current user to an organisation via its private join code (added 2026-10-12).
    Callable again later on purpose -- a student can switch organisations at any time; past
    monitoring sessions keep whichever organisation was actually linked when they ran (see
    MonitoringSession.organization_id), so switching never retroactively moves session history."""
    organization = await _organization_for(db, body.join_code)
    if organization is None:
        raise HTTPException(400, "Invalid join code.")
    current_user.organization_id = organization.user_id
    await db.flush()
    await log_event(db, "organization_joined", user_id=current_user.user_id, detail=organization.user_id, request=http_request)
    return OrganizationOut(organization_id=organization.user_id, organization_name=organization.name)


@router.get("/organization", response_model=Optional[OrganizationOut])
async def get_organization(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The organisation the current user is linked to, or null if none yet."""
    if current_user.organization_id is None:
        return None
    organization = (
        await db.execute(select(User).where(User.user_id == current_user.organization_id))
    ).scalar_one_or_none()
    if organization is None:
        return None
    return OrganizationOut(organization_id=organization.user_id, organization_name=organization.name)


@router.post("/reference", response_model=IdentityReferenceOut, status_code=201)
async def post_reference(
    body: EnrollReferenceIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enroll (or re-enroll, replacing any existing one) a reference photo for identity-match
    verification. The raw face embedding is never returned in the response. Requires explicit
    consent (body.consent == True) -- a face embedding is biometric data, and this is the point
    it's first created and stored.

    No longer requires having joined an organisation (that gate, added 2026-10-11, was relocated
    2026-10-12 to exam-attempt-start instead -- see services/exams/attempts.py::start_attempt --
    since ID verification/enrollment now deliberately happens before a student picks an
    organisation, not after)."""
    if body.consent is not True:
        raise HTTPException(400, "Consent is required to enroll a reference photo.")
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "Reference photo must be an image.")
    try:
        reference = await enroll_reference(upload, db)
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    await log_event(db, "identity_reference_enrolled", user_id=current_user.user_id, detail=upload.upload_id, request=http_request)
    return reference


@router.post("/verify-government-id", response_model=IdentityReferenceOut, status_code=201)
async def post_verify_government_id(
    body: EnrollGovernmentIdIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Transient-use government-ID enrollment (added 2026-10-12): verifies face match only, NOT
    document authenticity -- there is no way to detect a forged/fake ID card with this design.
    The uploaded ID-card photo is deleted (file + row) immediately after use -- always, success
    or failure, confirmed with the user as a deliberate choice over keeping a rejected photo
    around for a retry window. Nothing about the ID is ever retained beyond that moment; only the
    resulting face embedding is kept, same as any other FaceReference. Does NOT require an
    organisation link -- that gate lives on exam-attempt-start instead (see
    services/exams/attempts.py::start_attempt)."""
    if body.consent is not True:
        raise HTTPException(400, "Consent is required to verify a government ID.")
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "ID photo must be an image.")
    try:
        reference = await verify_government_id(upload, db)
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    finally:
        _remove_file(upload.storage_url)
        await db.delete(upload)
        await db.flush()
    await log_event(db, "government_id_verified", user_id=current_user.user_id, request=http_request)
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


async def _my_check_counts(db: AsyncSession, session_ids: list[str]) -> dict[str, tuple[int, int]]:
    """session_id -> (total_checks, flagged_checks), same aggregation shape as admin.py's
    _check_counts -- kept as its own small copy rather than a shared import, matching this
    project's existing "deliberately separate modules" convention for identity.py vs admin.py."""
    if not session_ids:
        return {}
    rows = (
        await db.execute(
            select(MonitoringCheck.session_id, MonitoringCheck.flagged)
            .where(MonitoringCheck.session_id.in_(session_ids))
        )
    ).all()
    counts: dict[str, list[int]] = {}
    for sid, flagged in rows:
        bucket = counts.setdefault(sid, [0, 0])
        bucket[0] += 1
        if flagged:
            bucket[1] += 1
    return {sid: (t, f) for sid, (t, f) in counts.items()}


@router.get("/sessions", response_model=list[MySessionListItemOut])
async def list_my_sessions(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The current user's own past monitoring sessions, newest first (added 2026-10-12) -- a
    dashboard-style list, separate from the per-session detail/report routes below."""
    sessions = (
        await db.execute(
            select(MonitoringSession)
            .where(MonitoringSession.user_id == current_user.user_id)
            .order_by(desc(MonitoringSession.started_at))
            .limit(limit).offset(offset)
        )
    ).scalars().all()
    counts = await _my_check_counts(db, [s.session_id for s in sessions])
    return [
        MySessionListItemOut(
            session_id=s.session_id, started_at=s.started_at, ended_at=s.ended_at,
            total_checks=counts.get(s.session_id, (0, 0))[0],
            flagged_checks=counts.get(s.session_id, (0, 0))[1],
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=MonitoringSessionOut, status_code=201)
async def post_session(
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start a continuous-monitoring session: periodic checks against the current user's
    enrolled reference. 404 if nothing has been enrolled yet.

    Idempotent (added 2026-10-11, item 7): if the caller already has an active session (one
    whose ended_at is still null), that session is returned instead of a new one -- same
    "don't duplicate in-flight work" philosophy as detection-job submission elsewhere in this
    API. Prevents a user from quietly starting several untracked, parallel sessions; status is
    200 for an existing session, the route's default 201 only for a genuinely new one."""
    reference = await _reference_for(db, current_user)
    if reference is None:
        raise HTTPException(404, "Enroll a reference photo first.")

    existing = (
        await db.execute(
            select(MonitoringSession)
            .where(MonitoringSession.user_id == current_user.user_id)
            .where(MonitoringSession.ended_at.is_(None))
        )
    ).scalar_one_or_none()
    if existing is not None:
        response.status_code = 200
        return existing

    session = MonitoringSession(
        session_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        reference_id=reference.reference_id,
        started_at=datetime.utcnow().isoformat(),
        organization_id=current_user.organization_id,
    )
    db.add(session)
    await db.flush()
    return session


@router.post("/sessions/{session_id}/checks", response_model=MonitoringCheckOut, status_code=201)
async def post_session_check(
    session_id: str,
    body: MonitoringCheckIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Run one periodic check (identity drift + deepfake signal + presence + gaze/objects, and
    optionally speech/talking if an audio clip is included) against an already-uploaded
    live-captured frame. 400 if the session has already ended."""
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
    audio_upload = None
    if body.audio_upload_id is not None:
        audio_upload = await _owned_upload(db, body.audio_upload_id, current_user)
        if audio_upload.media_type != "audio":
            raise HTTPException(400, "Audio clip must be an audio file.")
    try:
        check = await run_check(
            session, reference, upload, db, audio_upload=audio_upload,
            user_agent=http_request.headers.get("user-agent"),
        )
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    # Server-side exam flag-threshold enforcement (added 2026-10-12) -- atomic with the check
    # just inserted above, same transaction, before this response is returned. No-op if this
    # session isn't an in-progress exam attempt. See services/exams/enforcement.py.
    await enforce_flag_threshold(db, session.session_id)
    return check


@router.post("/sessions/{session_id}/checks/jobs", response_model=CheckJobOut, status_code=202)
async def post_session_check_job(
    session_id: str,
    body: MonitoringCheckIn,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Same check as POST .../checks (added 2026-10-11, item 6), run in the background instead of
    inline: returns immediately, poll GET .../checks/jobs/{job_id} for the result. The existing
    synchronous endpoint above is unchanged and still works -- this is additive, same precedent as
    video detection's job endpoint coexisting with its older synchronous one."""
    session = await _owned_session(db, session_id, current_user)
    if session.ended_at is not None:
        raise HTTPException(400, "This monitoring session has already ended.")
    upload = await _owned_upload(db, body.upload_id, current_user)
    if upload.media_type != "image":
        raise HTTPException(400, "Live capture must be an image.")
    if body.audio_upload_id is not None:
        audio_upload = await _owned_upload(db, body.audio_upload_id, current_user)
        if audio_upload.media_type != "audio":
            raise HTTPException(400, "Audio clip must be an audio file.")
    job = check_job_manager.submit(
        session_id=session_id, user_id=current_user.user_id, upload_id=body.upload_id,
        audio_upload_id=body.audio_upload_id, user_agent=http_request.headers.get("user-agent"),
    )
    return CheckJobOut(job_id=job.job_id, session_id=job.session_id, state=job.state)


@router.get("/sessions/{session_id}/checks/jobs/{job_id}", response_model=CheckJobOut)
async def get_session_check_job(
    session_id: str,
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll a background check job. `result` is populated once state == "done"."""
    job = check_job_manager.get(job_id)
    # 404 (not 403) for both "unknown" and "not yours"/"not this session's", as everywhere else.
    if job is None or job.user_id != current_user.user_id or job.session_id != session_id:
        raise HTTPException(404, "Check job not found. It may have expired.")
    return CheckJobOut(job_id=job.job_id, session_id=job.session_id, state=job.state, result=job.result, error=job.error)


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
