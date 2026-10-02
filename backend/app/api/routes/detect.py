import logging

from fastapi import APIRouter, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Upload, DetectionResult, User
from app.schemas.schemas import (
    DetectionResultOut, ExplanationOut, FrameExplanationOut, JobOut, ProvenanceOut, ProvenanceSignalOut,
    WindowExplanationOut,
)
from app.services.detection_errors import UnprocessableMediaError
from app.services.image.detector import run_image_detection
from app.services.video.detector import run_video_detection
from app.services.audio.detector import run_audio_detection
from app.api.routes.auth import get_current_user
from app.services.explain import core as explain_core
from app.services.explain.service import build_explanation
from app.services.provenance.service import build_provenance
from app.services.jobs import manager as job_manager
from app.api.routes.jobs import job_out

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Detection"])


@router.post("/{upload_id}", response_model=DetectionResultOut, status_code=201)
async def run_detection(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "Upload not found. It may have expired or the ID is invalid.")

    # Idempotent: an upload has at most one detection result. Without this, a repeated call (double click, client
    # retry) stored a second row and every read endpoint for that upload (/result, /report, /explain, /provenance)
    # then failed with "Multiple rows were found" - permanently. Return the existing result instead.
    already = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    existing = already.scalars().first()
    if existing is not None:
        return existing

    try:
        if upload.media_type == "image":
            result_id = await run_image_detection(upload, db)
        elif upload.media_type == "video":
            result_id = await run_video_detection(upload, db)
        elif upload.media_type == "audio":
            result_id = await run_audio_detection(upload, db)
        else:
            raise HTTPException(400, f"Unsupported media type: {upload.media_type}")
    except HTTPException:
        raise
    except UnprocessableMediaError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError:
        logger.exception("Upload record found but file missing on disk (upload_id=%s)", upload_id)
        raise HTTPException(500, "The uploaded file is no longer available on the server. Please upload it again.")
    except Exception:
        logger.exception("Detection pipeline failed unexpectedly (upload_id=%s)", upload_id)
        raise HTTPException(500, "Detection failed due to an internal error. Please try again or contact support.")

    # Re-fetch fully (relationships are lazy="selectin" so this is always safe)
    r2 = await db.execute(select(DetectionResult).where(DetectionResult.result_id == result_id))
    detection = r2.scalar_one_or_none()
    if not detection:
        raise HTTPException(500, "Detection completed but result could not be retrieved.")
    return detection


@router.get("/{upload_id}/result", response_model=DetectionResultOut)
async def get_result(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "No detection result found for this upload.")

    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")
    return result


@router.get("/{upload_id}/explain", response_model=ExplanationOut)
async def get_explanation(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Explainability for a scanned file: Grad-CAM heatmap (image), per-frame scores + face crops (video), or
    per-window scores + a time-saliency curve over a spectrogram (audio). Read-only; recomputed on demand with
    the deployed models. Never changes a stored result."""
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "No detection result found for this upload.")

    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")

    # Model forward/backward passes are CPU/GPU-bound; keep them off the event loop.
    ex = await run_in_threadpool(build_explanation, upload, result)

    out = ExplanationOut(
        upload_id=upload_id, media_type=ex.media_type, available=ex.available,
        reason=ex.reason, method=ex.method, note=ex.note,
    )
    if ex.image is not None:
        out.original = explain_core.data_uri(ex.image.original_jpeg)
        out.heatmap = explain_core.data_uri(ex.image.overlay_jpeg)
        out.fake_probability = ex.image.fake_probability
        out.evidence_strength = ex.image.evidence_strength
        out.convnext_fake_probability = ex.convnext_fake_probability
        out.clip_fake_probability = ex.clip_fake_probability
        out.verdict_driver = ex.verdict_driver
    if ex.video is not None:
        v = ex.video
        out.frames = [
            FrameExplanationOut(
                order=f.order, frame_index=f.frame_index, timestamp_s=f.timestamp_s, logit=f.logit,
                probability=f.probability, crop=explain_core.data_uri(f.crop_jpeg),
                heatmap=explain_core.data_uri(f.heatmap_jpeg) if f.heatmap_jpeg else None,
            )
            for f in v.frames
        ]
        out.mean_probability = v.probability
        out.threshold = v.threshold
        out.frames_above_threshold = v.frames_above_threshold
        out.duration_s = v.duration_s
    if ex.audio is not None:
        a = ex.audio
        out.windows = [
            WindowExplanationOut(order=w.order, start_s=w.start_s, end_s=w.end_s, probability=w.probability)
            for w in a.windows
        ]
        out.mean_probability = a.mean_probability
        out.threshold = a.threshold
        out.windows_above_threshold = a.windows_above_threshold
        out.duration_s = a.duration_s
        out.saliency_window_order = a.saliency_window_order
        out.spectrogram = explain_core.data_uri(a.spectrogram_jpeg)
        out.saliency = explain_core.data_uri(a.saliency_jpeg) if a.saliency_jpeg else None
    return out


@router.get("/{upload_id}/provenance", response_model=ProvenanceOut)
async def get_provenance(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Provenance evidence for a scanned file: C2PA Content Credentials, embedded metadata (EXIF / generation
    parameters / XMP), a generator-watermark check (image/video) and an ELA visual aid (image). Audio gets a
    C2PA-only check - audio has no EXIF/ELA/pixel-watermark equivalent in this codebase yet. Supplementary and
    read-only - never changes the stored verdict."""
    from dataclasses import asdict

    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "No detection result found for this upload.")
    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")

    pv = await run_in_threadpool(build_provenance, upload, result)
    out = ProvenanceOut(
        upload_id=upload_id, media_type=pv.media_type, available=pv.available, reason=pv.reason, caveat=pv.caveat,
        signals=[ProvenanceSignalOut(**asdict(s)) for s in pv.signals],
    )
    if pv.assessment:
        out.level, out.headline, out.conflict_note = pv.assessment.level, pv.assessment.headline, pv.assessment.conflict_note
    if pv.c2pa is not None:
        out.c2pa = asdict(pv.c2pa)
    if pv.metadata is not None:
        out.metadata = asdict(pv.metadata)
    out.container = pv.container
    if pv.watermark is not None:
        out.watermark = asdict(pv.watermark)
    if pv.ela is not None:
        ela = asdict(pv.ela)
        heat = ela.pop("heatmap_jpeg", None)
        ela["heatmap"] = explain_core.data_uri(heat) if heat else None
        out.ela = ela
    return out


@router.post("/{upload_id}/jobs", response_model=JobOut, status_code=202)
async def start_detection_job(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start detection in the background and return immediately; poll GET /api/v1/jobs/{job_id} for progress.
    Idempotent: an already-scanned upload returns a finished job, and an upload with a scan in flight returns that job."""
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload or upload.user_id != current_user.user_id:
        raise HTTPException(404, "Upload not found. It may have expired or the ID is invalid.")
    already = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    existing = already.scalars().first()
    job = await job_manager.submit(upload, current_user.user_id, existing.result_id if existing else None)
    return job_out(job)
