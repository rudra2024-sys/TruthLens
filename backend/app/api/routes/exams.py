"""Proctored MCQ exams (added 2026-10-12) -- organisation-authored multiple-choice exams, taken
by students inside TruthLens itself (not redirected to an external test platform), continuously
monitored via the existing proctoring pipeline. An ExamAttempt is 1:1 with the MonitoringSession
that proctors it -- see models.py's ExamAttempt docstring. Server-side flag-threshold
auto-termination lives in services/exams/enforcement.py, wired into the check-insertion call
sites in identity.py/monitoring_jobs.py, not here.

Org-side routes (gated by get_current_organization) manage exams/questions. Student-side routes
(gated by get_current_user) start/take an attempt. 404-not-403 ownership convention throughout,
matching the rest of this API.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_organization, get_current_user
from app.core.database import get_db
from app.models.models import Exam, ExamAttempt, ExamQuestion, MonitoringCheck, User
from app.schemas.schemas import (
    ExamAnswerIn,
    ExamAnswerOut,
    ExamAttemptOut,
    ExamAttemptStartIn,
    ExamIn,
    ExamOut,
    ExamQuestionAdminOut,
    ExamQuestionIn,
    ExamQuestionPublicOut,
)
from app.services.exams.attempts import AttemptClosedError, AttemptStartError, finish_attempt, start_attempt, submit_answer

router = APIRouter(prefix="/exams", tags=["Exams"])


# ---- org-side: exam + question management -----------------------------------------------------

async def _owned_exam(db: AsyncSession, exam_id: str, organization_id: str) -> Exam:
    exam = (await db.execute(select(Exam).where(Exam.exam_id == exam_id))).scalar_one_or_none()
    if exam is None or exam.organization_id != organization_id:
        raise HTTPException(404, "Exam not found.")
    return exam


async def _question_count(db: AsyncSession, exam_id: str) -> int:
    return (
        await db.execute(select(func.count()).select_from(ExamQuestion).where(ExamQuestion.exam_id == exam_id))
    ).scalar_one()


@router.post("", response_model=ExamOut, status_code=201)
async def create_exam(
    body: ExamIn,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exam = Exam(
        exam_id=str(uuid.uuid4()), organization_id=_admin.user_id, title=body.title,
        pass_threshold=body.pass_threshold, max_flags=body.max_flags,
    )
    db.add(exam)
    await db.flush()
    return ExamOut(
        exam_id=exam.exam_id, organization_id=exam.organization_id, title=exam.title,
        pass_threshold=exam.pass_threshold, max_flags=exam.max_flags, is_active=exam.is_active,
        created_at=exam.created_at, question_count=0,
    )


@router.get("", response_model=list[ExamOut])
async def list_exams(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exams = (
        await db.execute(select(Exam).where(Exam.organization_id == _admin.user_id).order_by(Exam.created_at.desc()))
    ).scalars().all()
    out = []
    for exam in exams:
        count = await _question_count(db, exam.exam_id)
        out.append(ExamOut(
            exam_id=exam.exam_id, organization_id=exam.organization_id, title=exam.title,
            pass_threshold=exam.pass_threshold, max_flags=exam.max_flags, is_active=exam.is_active,
            created_at=exam.created_at, question_count=count,
        ))
    return out


@router.patch("/{exam_id}", response_model=ExamOut)
async def update_exam(
    exam_id: str,
    body: ExamIn,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exam = await _owned_exam(db, exam_id, _admin.user_id)
    exam.title = body.title
    exam.pass_threshold = body.pass_threshold
    exam.max_flags = body.max_flags
    await db.flush()
    count = await _question_count(db, exam.exam_id)
    return ExamOut(
        exam_id=exam.exam_id, organization_id=exam.organization_id, title=exam.title,
        pass_threshold=exam.pass_threshold, max_flags=exam.max_flags, is_active=exam.is_active,
        created_at=exam.created_at, question_count=count,
    )


@router.post("/{exam_id}/questions", response_model=ExamQuestionAdminOut, status_code=201)
async def add_question(
    exam_id: str,
    body: ExamQuestionIn,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exam = await _owned_exam(db, exam_id, _admin.user_id)
    if body.correct_option_index < 0 or body.correct_option_index >= len(body.options):
        raise HTTPException(400, "correct_option_index must point at one of the given options.")
    order_index = body.order_index
    if order_index is None:
        order_index = await _question_count(db, exam.exam_id)
    question = ExamQuestion(
        question_id=str(uuid.uuid4()), exam_id=exam.exam_id, order_index=order_index,
        text=body.text, options_json=json.dumps(body.options), correct_option_index=body.correct_option_index,
    )
    db.add(question)
    await db.flush()
    return ExamQuestionAdminOut(
        question_id=question.question_id, exam_id=question.exam_id, order_index=question.order_index,
        text=question.text, options=question.options, correct_option_index=question.correct_option_index,
    )


@router.get("/{exam_id}/questions/admin", response_model=list[ExamQuestionAdminOut])
async def list_questions_admin(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    """Organisation-facing view -- includes the answer key. Distinct path from the student-facing
    GET /{exam_id}/questions below (ExamQuestionPublicOut), which never includes it."""
    exam = await _owned_exam(db, exam_id, _admin.user_id)
    questions = (
        await db.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam.exam_id).order_by(ExamQuestion.order_index))
    ).scalars().all()
    return [
        ExamQuestionAdminOut(
            question_id=q.question_id, exam_id=q.exam_id, order_index=q.order_index,
            text=q.text, options=q.options, correct_option_index=q.correct_option_index,
        )
        for q in questions
    ]


async def _owned_question(db: AsyncSession, exam: Exam, question_id: str) -> ExamQuestion:
    question = (
        await db.execute(select(ExamQuestion).where(ExamQuestion.question_id == question_id))
    ).scalar_one_or_none()
    if question is None or question.exam_id != exam.exam_id:
        raise HTTPException(404, "Question not found.")
    return question


@router.patch("/{exam_id}/questions/{question_id}", response_model=ExamQuestionAdminOut)
async def update_question(
    exam_id: str,
    question_id: str,
    body: ExamQuestionIn,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exam = await _owned_exam(db, exam_id, _admin.user_id)
    question = await _owned_question(db, exam, question_id)
    if body.correct_option_index < 0 or body.correct_option_index >= len(body.options):
        raise HTTPException(400, "correct_option_index must point at one of the given options.")
    question.text = body.text
    question.options_json = json.dumps(body.options)
    question.correct_option_index = body.correct_option_index
    if body.order_index is not None:
        question.order_index = body.order_index
    await db.flush()
    return ExamQuestionAdminOut(
        question_id=question.question_id, exam_id=question.exam_id, order_index=question.order_index,
        text=question.text, options=question.options, correct_option_index=question.correct_option_index,
    )


@router.delete("/{exam_id}/questions/{question_id}", status_code=204)
async def delete_question(
    exam_id: str,
    question_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(get_current_organization),
):
    exam = await _owned_exam(db, exam_id, _admin.user_id)
    question = await _owned_question(db, exam, question_id)
    await db.delete(question)
    await db.flush()


# ---- student-side: viewing an exam + taking an attempt -----------------------------------------

@router.get("/{exam_id}", response_model=ExamOut)
async def get_exam(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Basic exam metadata for a student -- scoped to the student's own organisation_id, 404
    otherwise (same ownership convention as everywhere else), so a student can't browse exams
    belonging to an organisation they haven't joined."""
    exam = (await db.execute(select(Exam).where(Exam.exam_id == exam_id))).scalar_one_or_none()
    if exam is None or exam.organization_id != current_user.organization_id or not exam.is_active:
        raise HTTPException(404, "Exam not found.")
    count = await _question_count(db, exam.exam_id)
    return ExamOut(
        exam_id=exam.exam_id, organization_id=exam.organization_id, title=exam.title,
        pass_threshold=exam.pass_threshold, max_flags=exam.max_flags, is_active=exam.is_active,
        created_at=exam.created_at, question_count=count,
    )


@router.get("/{exam_id}/questions", response_model=list[ExamQuestionPublicOut])
async def list_questions_public(
    exam_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Student-facing view -- correct_option_index is never included. Only available once the
    student has an in_progress attempt on this exam, so the question list can't be browsed before
    officially starting (and therefore before monitoring is active)."""
    exam = (await db.execute(select(Exam).where(Exam.exam_id == exam_id))).scalar_one_or_none()
    if exam is None or exam.organization_id != current_user.organization_id:
        raise HTTPException(404, "Exam not found.")
    attempt = (
        await db.execute(
            select(ExamAttempt)
            .where(ExamAttempt.exam_id == exam_id)
            .where(ExamAttempt.user_id == current_user.user_id)
            .where(ExamAttempt.status == "in_progress")
        )
    ).scalar_one_or_none()
    if attempt is None:
        raise HTTPException(400, "Start an attempt before viewing questions.")
    questions = (
        await db.execute(select(ExamQuestion).where(ExamQuestion.exam_id == exam_id).order_by(ExamQuestion.order_index))
    ).scalars().all()
    return [
        ExamQuestionPublicOut(question_id=q.question_id, order_index=q.order_index, text=q.text, options=q.options)
        for q in questions
    ]


async def _flagged_checks_count(db: AsyncSession, session_id: str) -> int:
    return (
        await db.execute(
            select(func.count()).select_from(MonitoringCheck)
            .where(MonitoringCheck.session_id == session_id)
            .where(MonitoringCheck.flagged == True)  # noqa: E712
        )
    ).scalar_one()


def _attempt_out(attempt: ExamAttempt, exam: Exam, flagged_checks: int) -> ExamAttemptOut:
    return ExamAttemptOut(
        attempt_id=attempt.attempt_id, exam_id=attempt.exam_id, session_id=attempt.session_id,
        status=attempt.status, score=attempt.score, started_at=attempt.started_at, ended_at=attempt.ended_at,
        flagged_checks=flagged_checks, max_flags=exam.max_flags,
    )


@router.post("/{exam_id}/attempts", response_model=ExamAttemptOut, status_code=201)
async def post_attempt(
    exam_id: str,
    body: ExamAttemptStartIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exam = (await db.execute(select(Exam).where(Exam.exam_id == exam_id))).scalar_one_or_none()
    if exam is None or not exam.is_active:
        raise HTTPException(404, "Exam not found.")
    try:
        attempt = await start_attempt(exam, current_user, db, body.details)
    except AttemptStartError as e:
        raise HTTPException(400, str(e))
    return _attempt_out(attempt, exam, 0)


async def _owned_attempt(db: AsyncSession, attempt_id: str, user: User) -> ExamAttempt:
    attempt = (await db.execute(select(ExamAttempt).where(ExamAttempt.attempt_id == attempt_id))).scalar_one_or_none()
    if attempt is None or attempt.user_id != user.user_id:
        raise HTTPException(404, "Exam attempt not found.")
    return attempt


@router.get("/attempts/{attempt_id}", response_model=ExamAttemptOut)
async def get_attempt(
    attempt_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """The exam-taking page's polling endpoint -- includes the live flagged_checks count."""
    attempt = await _owned_attempt(db, attempt_id, current_user)
    exam = (await db.execute(select(Exam).where(Exam.exam_id == attempt.exam_id))).scalar_one()
    flagged = await _flagged_checks_count(db, attempt.session_id)
    return _attempt_out(attempt, exam, flagged)


@router.post("/attempts/{attempt_id}/answers", response_model=ExamAnswerOut, status_code=201)
async def post_answer(
    attempt_id: str,
    body: ExamAnswerIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    attempt = await _owned_attempt(db, attempt_id, current_user)
    question = (
        await db.execute(select(ExamQuestion).where(ExamQuestion.question_id == body.question_id))
    ).scalar_one_or_none()
    if question is None or question.exam_id != attempt.exam_id:
        raise HTTPException(404, "Question not found.")
    try:
        answer = await submit_answer(attempt, question, body.selected_option_index, db)
    except AttemptClosedError as e:
        raise HTTPException(400, str(e))
    return answer


@router.post("/attempts/{attempt_id}/finish", response_model=ExamAttemptOut)
async def post_finish(
    attempt_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    attempt = await _owned_attempt(db, attempt_id, current_user)
    attempt = await finish_attempt(attempt, db)
    exam = (await db.execute(select(Exam).where(Exam.exam_id == attempt.exam_id))).scalar_one()
    flagged = await _flagged_checks_count(db, attempt.session_id)
    return _attempt_out(attempt, exam, flagged)
