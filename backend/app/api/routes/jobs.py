from fastapi import APIRouter, Depends, HTTPException
import time

from app.api.routes.auth import get_current_user
from app.models.models import User
from app.schemas.schemas import JobOut
from app.services.jobs import Job, manager

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def job_out(job: Job) -> JobOut:
    end = job.finished_at or time.time()
    return JobOut(
        job_id=job.job_id, upload_id=job.upload_id, media_type=job.media_type, state=job.state,
        progress=round(job.progress, 4), stage=job.stage, result_id=job.result_id, error=job.error,
        queue_position=manager.queue_position(job),
        elapsed_s=round(end - job.started_at, 2) if job.started_at else None,
    )


def _owned(job_id: str, user: User) -> Job:
    job = manager.get(job_id)
    # 404 (not 403) for both "unknown" and "not yours", as everywhere else in the API.
    if job is None or job.user_id != user.user_id:
        raise HTTPException(404, "Job not found. It may have expired.")
    return job


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Poll a background detection job. When state is "done", fetch the result from /detect/{upload_id}/result."""
    return job_out(_owned(job_id, current_user))


@router.delete("/{job_id}", response_model=JobOut)
async def cancel_job(job_id: str, current_user: User = Depends(get_current_user)):
    """Ask a queued or running job to stop. Nothing is stored for a cancelled scan."""
    return job_out(manager.cancel(_owned(job_id, current_user)))
