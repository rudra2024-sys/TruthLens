"""Background jobs for monitoring-session checks (added 2026-10-11, item 6).

Why: each check runs a CPU-bound stack (MTCNN + ConvNeXt + CLIP + YOLOv8n) that already gets
offloaded to a worker thread via run_in_threadpool inside monitoring.py::run_check, so the event
loop itself was never blocked -- but the HTTP request still waits synchronously for the whole
thing, with no concurrency cap. Under several proctored sessions each posting a check every
5-20s, that's unbounded CPU contention. This mirrors services/jobs.py's proven shape (dataclass
Job + semaphore-gated JobManager) but simpler: a check is one atomic computation, not a
multi-stage scan with a progress percentage, and the finished result (a small dict, unlike a
heavy video DetectionResult) is held directly on the job -- no separate fetch endpoint needed.

The existing synchronous POST .../checks endpoint is NOT removed or deprecated by this -- same
additive, non-breaking precedent as video's job endpoint coexisting with the older sync detect
endpoint (CLAUDE.md section 13).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.models import FaceReference, MonitoringSession, Upload
from app.schemas.schemas import MonitoringCheckOut
from app.services.detection_errors import UnprocessableMediaError
from app.services.exams.enforcement import enforce_flag_threshold
from app.services.identity.monitoring import run_check

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")


@dataclass
class CheckJob:
    job_id: str
    session_id: str
    user_id: str
    state: str = "queued"            # queued | running | done | failed
    result: dict | None = None       # MonitoringCheckOut-shaped, once state == "done"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class CheckJobManager:
    def __init__(self, concurrency: int | None = None, keep_seconds: int = 1800, max_jobs: int = 500):
        self.concurrency = max(1, concurrency or settings.MONITORING_CHECK_JOB_CONCURRENCY)
        self.keep_seconds = keep_seconds
        self.max_jobs = max_jobs
        self._jobs: dict[str, CheckJob] = {}
        self._tasks: set[asyncio.Task] = set()
        self._sem: asyncio.Semaphore | None = None

    def get(self, job_id: str) -> CheckJob | None:
        return self._jobs.get(job_id)

    def _purge(self) -> None:
        now = time.time()
        for jid in [j.job_id for j in self._jobs.values()
                    if j.state not in ACTIVE and j.finished_at and now - j.finished_at > self.keep_seconds]:
            self._jobs.pop(jid, None)
        if len(self._jobs) > self.max_jobs:
            finished = sorted((j for j in self._jobs.values() if j.state not in ACTIVE), key=lambda j: j.created_at)
            for j in finished[: len(self._jobs) - self.max_jobs]:
                self._jobs.pop(j.job_id, None)

    def submit(
        self, session_id: str, user_id: str, upload_id: str, audio_upload_id: str | None, user_agent: str | None,
    ) -> CheckJob:
        self._purge()
        job = CheckJob(job_id=str(uuid.uuid4()), session_id=session_id, user_id=user_id)
        self._jobs[job.job_id] = job
        task = asyncio.create_task(self._run(job, upload_id, audio_upload_id, user_agent))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _finish(self, job: CheckJob, state: str, result: dict | None = None, error: str | None = None) -> None:
        job.state, job.result, job.error = state, result, error
        job.finished_at = time.time()

    async def _run(self, job: CheckJob, upload_id: str, audio_upload_id: str | None, user_agent: str | None) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        async with self._sem:
            job.state, job.started_at = "running", time.time()
            try:
                async with AsyncSessionLocal() as db:
                    session = (
                        await db.execute(select(MonitoringSession).where(MonitoringSession.session_id == job.session_id))
                    ).scalar_one_or_none()
                    if session is None or session.user_id != job.user_id:
                        raise FileNotFoundError("session record missing")
                    if session.ended_at is not None:
                        raise UnprocessableMediaError("This monitoring session has already ended.")
                    reference = (
                        await db.execute(select(FaceReference).where(FaceReference.reference_id == session.reference_id))
                    ).scalar_one_or_none()
                    if reference is None:
                        raise FileNotFoundError("reference record missing")
                    upload = (await db.execute(select(Upload).where(Upload.upload_id == upload_id))).scalar_one_or_none()
                    if upload is None or upload.user_id != job.user_id:
                        raise FileNotFoundError("upload record missing")
                    audio_upload = None
                    if audio_upload_id is not None:
                        audio_upload = (
                            await db.execute(select(Upload).where(Upload.upload_id == audio_upload_id))
                        ).scalar_one_or_none()

                    check = await run_check(session, reference, upload, db, audio_upload=audio_upload, user_agent=user_agent)
                    result = MonitoringCheckOut.model_validate(check).model_dump()
                    # Server-side exam flag-threshold enforcement (added 2026-10-12) -- same
                    # transaction as the check insert above, before commit. See
                    # identity.py::post_session_check for why this is the sync-path counterpart.
                    await enforce_flag_threshold(db, session.session_id)
                    await db.commit()
                self._finish(job, "done", result=result)
            except UnprocessableMediaError as e:
                self._finish(job, "failed", error=str(e))
            except FileNotFoundError:
                logger.exception("Check job %s: referenced record missing", job.job_id)
                self._finish(job, "failed", error="The session, reference or upload is no longer available.")
            except Exception:
                logger.exception("Check job %s: failed unexpectedly", job.job_id)
                self._finish(job, "failed", error="The check failed due to an internal error. Please try again.")


manager = CheckJobManager()
