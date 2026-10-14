"""Server-side flag-threshold enforcement (added 2026-10-12) -- called from the exact two places
run_check() itself is called (identity.py::post_session_check, monitoring_jobs.py::
CheckJobManager._run), in the same DB transaction as the triggering check's own insert, so there
is no window for a compromised client to submit more answers after crossing the threshold: the
server has already marked the attempt terminal before that request even returns.

Deliberately NOT inside monitoring.py -- keeps that module exam-agnostic, same separation
discipline admin.py/identity.py already use for their own domains. A no-op for any session that
isn't actually an in-progress exam attempt (e.g. a plain "just monitor me" session), so this is
safe to call unconditionally from both check paths.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Exam, ExamAttempt, MonitoringCheck, MonitoringSession


async def enforce_flag_threshold(db: AsyncSession, session_id: str) -> None:
    attempt = (
        await db.execute(
            select(ExamAttempt)
            .where(ExamAttempt.session_id == session_id)
            .where(ExamAttempt.status == "in_progress")
        )
    ).scalar_one_or_none()
    if attempt is None:
        return  # not an exam session, or already terminal -- nothing to enforce

    exam = (await db.execute(select(Exam).where(Exam.exam_id == attempt.exam_id))).scalar_one()

    flagged_count = (
        await db.execute(
            select(func.count())
            .select_from(MonitoringCheck)
            .where(MonitoringCheck.session_id == session_id)
            .where(MonitoringCheck.flagged == True)  # noqa: E712 (SQLAlchemy boolean comparison)
        )
    ).scalar_one()

    if flagged_count < exam.max_flags:
        return

    now = datetime.utcnow().isoformat()
    attempt.status = "terminated_for_flags"
    attempt.ended_at = now

    session = (
        await db.execute(select(MonitoringSession).where(MonitoringSession.session_id == session_id))
    ).scalar_one_or_none()
    if session is not None and session.ended_at is None:
        session.ended_at = now

    await db.flush()
