"""Exam attempt lifecycle (added 2026-10-12): starting an attempt (which also creates the
MonitoringSession that proctors it -- see models.py's ExamAttempt docstring for why this 1:1
link is the whole point), recording answers, and finishing/scoring. Server-side flag-threshold
termination lives separately in enforcement.py, called from the check-insertion call sites, not
from here.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Exam, ExamAnswer, ExamAttempt, ExamQuestion, FaceReference, MonitoringSession, User


class AttemptStartError(Exception):
    """Raised for any reason an attempt can't start -- the route layer maps this to a 400/409,
    same separation as UnprocessableMediaError elsewhere in this codebase (a plain exception
    carrying a user-facing message, not an HTTPException, so this module stays framework-free)."""


class AttemptClosedError(Exception):
    """Raised when an action is attempted against an attempt that's no longer in_progress (e.g.
    terminated for flags mid-exam) -- covers the race where a student's client tries to submit
    one more answer after the server already ended the attempt."""


async def start_attempt(exam: Exam, current_user: User, db: AsyncSession, details: dict[str, str]) -> ExamAttempt:
    if current_user.organization_id != exam.organization_id:
        raise AttemptStartError("Join this exam's organisation before starting.")

    reference = (
        await db.execute(select(FaceReference).where(FaceReference.user_id == current_user.user_id))
    ).scalar_one_or_none()
    if reference is None or not reference.government_id_verified:
        raise AttemptStartError("Verify your government ID before starting an exam.")

    existing_in_progress = (
        await db.execute(
            select(ExamAttempt)
            .where(ExamAttempt.user_id == current_user.user_id)
            .where(ExamAttempt.status == "in_progress")
        )
    ).scalar_one_or_none()
    if existing_in_progress is not None:
        raise AttemptStartError("You already have an exam in progress. Finish or it must end before starting another.")

    now = datetime.utcnow().isoformat()
    session = MonitoringSession(
        session_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        reference_id=reference.reference_id,
        started_at=now,
        organization_id=current_user.organization_id,
    )
    db.add(session)
    await db.flush()

    attempt = ExamAttempt(
        attempt_id=str(uuid.uuid4()),
        exam_id=exam.exam_id,
        user_id=current_user.user_id,
        session_id=session.session_id,
        organization_id=exam.organization_id,
        started_at=now,
        details_json=json.dumps(details),
    )
    db.add(attempt)
    await db.flush()
    return attempt


async def submit_answer(attempt: ExamAttempt, question: ExamQuestion, selected_option_index: int, db: AsyncSession) -> ExamAnswer:
    if attempt.status != "in_progress":
        raise AttemptClosedError("This attempt has already ended.")

    existing = (
        await db.execute(
            select(ExamAnswer)
            .where(ExamAnswer.attempt_id == attempt.attempt_id)
            .where(ExamAnswer.question_id == question.question_id)
        )
    ).scalar_one_or_none()
    now = datetime.utcnow().isoformat()
    if existing is not None:
        existing.selected_option_index = selected_option_index
        existing.answered_at = now
        await db.flush()
        return existing

    answer = ExamAnswer(
        answer_id=str(uuid.uuid4()),
        attempt_id=attempt.attempt_id,
        question_id=question.question_id,
        selected_option_index=selected_option_index,
        answered_at=now,
    )
    db.add(answer)
    await db.flush()
    return answer


async def finish_attempt(attempt: ExamAttempt, db: AsyncSession) -> ExamAttempt:
    """Idempotent (mirrors identity.py::post_session_end) -- scores against ExamQuestion.
    correct_option_index and ends the linked MonitoringSession too."""
    if attempt.status != "in_progress":
        return attempt

    questions = (
        await db.execute(select(ExamQuestion).where(ExamQuestion.exam_id == attempt.exam_id))
    ).scalars().all()
    answers = (
        await db.execute(select(ExamAnswer).where(ExamAnswer.attempt_id == attempt.attempt_id))
    ).scalars().all()
    answer_by_question = {a.question_id: a.selected_option_index for a in answers}

    correct = sum(
        1 for q in questions
        if answer_by_question.get(q.question_id) == q.correct_option_index
    )

    exam = (await db.execute(select(Exam).where(Exam.exam_id == attempt.exam_id))).scalar_one()

    attempt.score = correct
    attempt.status = "passed" if correct >= exam.pass_threshold else "failed"
    attempt.ended_at = datetime.utcnow().isoformat()

    session = (
        await db.execute(select(MonitoringSession).where(MonitoringSession.session_id == attempt.session_id))
    ).scalar_one_or_none()
    if session is not None and session.ended_at is None:
        session.ended_at = attempt.ended_at

    await db.flush()
    return attempt
