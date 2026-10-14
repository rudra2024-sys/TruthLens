"""Admin/proctor dashboard (added 2026-10-11, scoped to a real organisation 2026-10-12) --
read-only, cross-student visibility into monitoring sessions for an operator reviewing flagged
activity (e.g. an exam proctor), gated by auth.py::get_current_organization. Deliberately a
separate router/query layer from routes/identity.py's owner-scoped endpoints rather than reusing
`_owned_session` with an admin bypass -- keeps both modules independently readable (one always
filters by the caller's own user_id, this one by the caller's own organization_id, and an admin
accidentally seeing another user's "owned" endpoint behavior is not a mistake this structure can
make).

Every query below filters on MonitoringSession.organization_id -- a snapshot taken at
session-creation time of the student's User.organization_id, not a live join (see models.py).
Before 2026-10-12 this router had NO scoping at all: any organisation account could list, view,
or download any student's session, a real cross-tenant leak, not just a cosmetic "shows too much"
problem -- confirmed live (a brand-new organisation account's dashboard showed a student with no
relationship to it).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import desc, exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_organization
from app.core.database import get_db
from app.models.models import Exam, ExamAttempt, MonitoringCheck, MonitoringEvent, MonitoringSession, User
from app.schemas.schemas import (
    AdminSessionDetailOut,
    AdminSessionListItemOut,
    ExamAttemptListItemOut,
    JoinCodeOut,
    RosterStudentOut,
)
from app.services.audit_log import log_event
from app.services.join_code import generate_join_code
from app.services.report.session_report import generate_session_report

router = APIRouter(prefix="/admin", tags=["Admin"])


async def _check_counts(db: AsyncSession, session_ids: list[str]) -> dict[str, tuple[int, int]]:
    """session_id -> (total_checks, flagged_checks). Aggregated in Python rather than via a SQL
    COUNT/SUM-on-boolean, which isn't portable the same way across SQLite/Postgres -- fine at
    this project's scale (same pragmatic trade-off as session_report.py's Counter-based
    flag-breakdown, not a SQL aggregate)."""
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


@router.get("/join-code", response_model=JoinCodeOut)
async def get_join_code(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """This organisation's private join code, lazily generated on first call (added 2026-10-12)
    -- handles both brand-new organisation signups and organisation accounts that signed up
    before this feature existed with one code path, rather than special-casing either."""
    if _admin.join_code is None:
        for _ in range(5):  # pragmatic retry on a uniqueness collision, not a hard guarantee
            candidate = generate_join_code()
            clash = (await db.execute(select(User).where(User.join_code == candidate))).scalar_one_or_none()
            if clash is None:
                _admin.join_code = candidate
                break
        await db.flush()
    return JoinCodeOut(join_code=_admin.join_code)


@router.post("/join-code/regenerate", response_model=JoinCodeOut)
async def regenerate_join_code(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """Replace this organisation's join code. Does NOT unlink any already-joined student --
    linkage lives on the student's own User.organization_id, not derived from the code
    afterwards, so only future join attempts are affected."""
    for _ in range(5):
        candidate = generate_join_code()
        clash = (await db.execute(select(User).where(User.join_code == candidate))).scalar_one_or_none()
        if clash is None:
            _admin.join_code = candidate
            break
    await db.flush()
    await log_event(db, "join_code_regenerated", user_id=_admin.user_id)
    return JoinCodeOut(join_code=_admin.join_code)


@router.get("/roster", response_model=list[RosterStudentOut])
async def get_roster(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """Students currently linked to this organisation (added 2026-10-12) -- separate from the
    session list, since a student with zero sessions yet still belongs on a roster view."""
    rows = (await db.execute(select(User).where(User.organization_id == _admin.user_id))).scalars().all()
    return [RosterStudentOut(user_id=u.user_id, name=u.name, email=u.email) for u in rows]


@router.delete("/roster/{user_id}", status_code=204)
async def remove_from_roster(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """Unlink a student from this organisation. 404 (not 403) for both "unknown" and "not your
    student", same ownership convention as everywhere else in this API. Clears only
    User.organization_id -- does NOT touch any MonitoringSession.organization_id snapshot, so the
    organisation keeps seeing sessions it already reviewed; only future sessions require the
    student to rejoin."""
    student = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if student is None or student.organization_id != _admin.user_id:
        raise HTTPException(404, "Student not found.")
    student.organization_id = None
    await db.flush()
    await log_event(db, "removed_from_roster", user_id=_admin.user_id, detail=user_id)


@router.get("/sessions", response_model=list[AdminSessionListItemOut])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    flagged_only: bool = False,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """Every MonitoringSession belonging to a student linked to the calling organisation,
    newest first. flagged_only narrows to sessions with at least one flagged check, applied
    before pagination (not a post-filter) so limit/offset stay exact."""
    query = (
        select(MonitoringSession, User)
        .join(User, User.user_id == MonitoringSession.user_id)
        .where(MonitoringSession.organization_id == _admin.user_id)
    )
    if flagged_only:
        query = query.where(
            exists().where(
                (MonitoringCheck.session_id == MonitoringSession.session_id)
                & (MonitoringCheck.flagged == True)  # noqa: E712 (SQLAlchemy boolean comparison)
            )
        )
    query = query.order_by(desc(MonitoringSession.started_at)).limit(limit).offset(offset)
    rows = (await db.execute(query)).all()

    counts = await _check_counts(db, [s.session_id for s, _ in rows])
    items = []
    for session, user in rows:
        total, flagged = counts.get(session.session_id, (0, 0))
        items.append(AdminSessionListItemOut(
            session_id=session.session_id, user_id=user.user_id, user_name=user.name, user_email=user.email,
            started_at=session.started_at, ended_at=session.ended_at, total_checks=total, flagged_checks=flagged,
        ))
    return items


async def _any_session(db: AsyncSession, session_id: str, admin: User) -> tuple[MonitoringSession, User]:
    """Organisation-scoped session lookup -- filtered to sessions whose organization_id snapshot
    matches the calling organisation (added 2026-10-12; previously had no ownership filter at
    all, see module docstring). 404 (not 403) for both "unknown" and "not your organisation's",
    matching this codebase's existing ownership convention everywhere else."""
    row = (
        await db.execute(
            select(MonitoringSession, User)
            .join(User, User.user_id == MonitoringSession.user_id)
            .where(MonitoringSession.session_id == session_id)
            .where(MonitoringSession.organization_id == admin.user_id)
        )
    ).first()
    if row is None:
        raise HTTPException(404, "Monitoring session not found.")
    return row


@router.get("/sessions/{session_id}", response_model=AdminSessionDetailOut)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    session, user = await _any_session(db, session_id, _admin)
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
    return AdminSessionDetailOut(
        session_id=session.session_id, reference_id=session.reference_id,
        started_at=session.started_at, ended_at=session.ended_at,
        user_id=user.user_id, user_name=user.name, user_email=user.email,
        checks=checks, events=events,
    )


@router.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    session, _user = await _any_session(db, session_id, _admin)
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


@router.get("/exams/{exam_id}/attempts", response_model=list[ExamAttemptListItemOut])
async def list_exam_attempts(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """An exam's attempts, newest first, scoped to the calling organisation via ExamAttempt.
    organization_id -- the same denormalized-snapshot filtering pattern as list_sessions above,
    not a join through Exam (added 2026-10-12, see models.py's ExamAttempt docstring for why)."""
    exam = (await db.execute(select(Exam).where(Exam.exam_id == exam_id))).scalar_one_or_none()
    if exam is None or exam.organization_id != _admin.user_id:
        raise HTTPException(404, "Exam not found.")
    rows = (
        await db.execute(
            select(ExamAttempt, User)
            .join(User, User.user_id == ExamAttempt.user_id)
            .where(ExamAttempt.exam_id == exam_id)
            .where(ExamAttempt.organization_id == _admin.user_id)
            .order_by(desc(ExamAttempt.started_at))
        )
    ).all()
    return [
        ExamAttemptListItemOut(
            attempt_id=attempt.attempt_id, exam_id=attempt.exam_id, exam_title=exam.title,
            user_id=user.user_id, user_name=user.name, user_email=user.email,
            session_id=attempt.session_id, status=attempt.status, score=attempt.score,
            started_at=attempt.started_at, ended_at=attempt.ended_at,
        )
        for attempt, user in rows
    ]
