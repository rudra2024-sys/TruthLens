"""Background detection jobs: start a scan, poll its progress, optionally cancel it.

Why: a video scan is slow (decoding + a forward pass, seconds to minutes for large files) and used to run inside the
HTTP request, so the browser sat on a spinner with no feedback and, worse, blocked the server's event loop for every
other user. A job runs the same detector in the background; the client polls `GET /api/v1/jobs/{id}`.

Design notes
  * The detector functions are reused unchanged (they write the result rows); the job just owns its own DB session.
  * At most DETECTION_JOB_CONCURRENCY jobs run at once (default 1: inference is CPU-bound, so parallel jobs only make
    each one slower). The rest wait in `queued` and report their queue position.
  * Idempotent like the rest of the API: one upload has at most one result and at most one active job.
  * State lives in memory of a single server process (progress is polled, results are durable in the database). After
    a restart an unknown job id 404s; the client falls back to /detect/{upload_id}/result. Running multiple uvicorn
    workers would need a shared store; the shipped Docker setup runs one process.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from app.core.database import AsyncSessionLocal
from app.models.models import Upload
from app.services.audio.detector import run_audio_detection
from app.services.detection_errors import UnprocessableMediaError
from app.services.image.detector import run_image_detection
from app.services.video.detector import run_video_detection

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")


class JobCancelled(Exception):
    """Raised inside the progress callback to abort a running scan."""


@dataclass
class Job:
    job_id: str
    upload_id: str
    user_id: str
    media_type: str
    state: str = "queued"            # queued | running | done | failed | cancelled
    progress: float = 0.0            # 0..1
    stage: str = "Waiting in queue"
    result_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False


class JobManager:
    def __init__(self, concurrency: int | None = None, keep_seconds: int = 3600, max_jobs: int = 500):
        self.concurrency = max(1, concurrency or int(os.getenv("DETECTION_JOB_CONCURRENCY", "1")))
        self.keep_seconds = keep_seconds
        self.max_jobs = max_jobs
        self._jobs: dict[str, Job] = {}
        self._active_by_upload: dict[str, str] = {}
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()               # progress callbacks arrive from worker threads
        self._sem: asyncio.Semaphore | None = None

    # ---------------------------------------------------------------- queries
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int | None:
        """1 = next to run. None when the job is not waiting."""
        if job.state != "queued":
            return None
        ahead = sum(1 for j in self._jobs.values() if j.state == "queued" and j.created_at < job.created_at)
        return ahead + 1

    # ---------------------------------------------------------------- lifecycle
    def _purge(self) -> None:
        now = time.time()
        for jid in [j.job_id for j in self._jobs.values()
                    if j.state not in ACTIVE and j.finished_at and now - j.finished_at > self.keep_seconds]:
            self._jobs.pop(jid, None)
        if len(self._jobs) > self.max_jobs:
            finished = sorted((j for j in self._jobs.values() if j.state not in ACTIVE), key=lambda j: j.created_at)
            for j in finished[: len(self._jobs) - self.max_jobs]:
                self._jobs.pop(j.job_id, None)

    async def submit(self, upload: Upload, user_id: str, existing_result_id: str | None = None) -> Job:
        """Create a job for `upload`, or return the one that already covers it."""
        self._purge()
        if existing_result_id:                       # already scanned: a finished job, no work
            job = Job(job_id=str(uuid.uuid4()), upload_id=upload.upload_id, user_id=user_id, media_type=upload.media_type,
                      state="done", progress=1.0, stage="Done", result_id=existing_result_id)
            job.started_at = job.finished_at = job.created_at
            self._jobs[job.job_id] = job
            return job
        active_id = self._active_by_upload.get(upload.upload_id)
        if active_id and self._jobs.get(active_id) and self._jobs[active_id].state in ACTIVE:
            return self._jobs[active_id]

        job = Job(job_id=str(uuid.uuid4()), upload_id=upload.upload_id, user_id=user_id, media_type=upload.media_type)
        self._jobs[job.job_id] = job
        self._active_by_upload[upload.upload_id] = job.job_id
        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)                        # keep a reference so the task is not garbage-collected mid-run
        task.add_done_callback(self._tasks.discard)
        return job

    def cancel(self, job: Job) -> Job:
        if job.state in ACTIVE:
            job.cancel_requested = True
            if job.state == "queued":                # never started: cancel immediately
                self._finish(job, "cancelled", stage="Cancelled")
        return job

    def _finish(self, job: Job, state: str, stage: str, result_id: str | None = None, error: str | None = None) -> None:
        with self._lock:
            job.state, job.stage, job.error = state, stage, error
            if result_id:
                job.result_id = result_id
            if state == "done":
                job.progress = 1.0
            job.finished_at = time.time()
        if self._active_by_upload.get(job.upload_id) == job.job_id:
            self._active_by_upload.pop(job.upload_id, None)

    # ---------------------------------------------------------------- execution
    def _callback(self, job: Job):
        def cb(fraction: float, stage: str) -> None:
            if job.cancel_requested:
                raise JobCancelled()
            with self._lock:
                job.progress = max(job.progress, min(max(float(fraction), 0.0), 0.99))   # 1.0 is set on completion
                job.stage = stage
        return cb

    async def _run(self, job: Job) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        async with self._sem:
            if job.cancel_requested or job.state == "cancelled":
                return
            with self._lock:
                job.state, job.stage, job.started_at = "running", "Starting", time.time()
            try:
                async with AsyncSessionLocal() as db:
                    upload = (await db.execute(select(Upload).where(Upload.upload_id == job.upload_id))).scalar_one_or_none()
                    if upload is None:
                        raise FileNotFoundError("upload record missing")
                    if job.media_type == "video":
                        result_id = await run_video_detection(upload, db, progress=self._callback(job))
                    elif job.media_type == "image":
                        result_id = await run_image_detection(upload, db)
                    elif job.media_type == "audio":
                        result_id = await run_audio_detection(upload, db)
                    else:
                        raise UnprocessableMediaError(f"Unsupported media type: {job.media_type}")
                    if job.cancel_requested:
                        raise JobCancelled()
                    await db.commit()
                self._finish(job, "done", "Done", result_id=result_id)
            except JobCancelled:
                self._finish(job, "cancelled", "Cancelled")            # the session closes without commit -> rolled back
            except UnprocessableMediaError as e:
                self._finish(job, "failed", "Failed", error=str(e))
            except FileNotFoundError:
                logger.exception("Job %s: upload file missing", job.job_id)
                self._finish(job, "failed", "Failed",
                             error="The uploaded file is no longer available on the server. Please upload it again.")
            except Exception:
                logger.exception("Job %s: detection failed unexpectedly", job.job_id)
                self._finish(job, "failed", "Failed",
                             error="Detection failed due to an internal error. Please try again or contact support.")


manager = JobManager()
