import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Upload, DetectionResult
from app.schemas.schemas import DetectionResultOut
from app.services.detection_errors import UnprocessableMediaError
from app.services.image.detector import run_image_detection
from app.services.video.detector import run_video_detection
from app.services.audio.detector import run_audio_detection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Detection"])


@router.post("/{upload_id}", response_model=DetectionResultOut, status_code=201)
async def run_detection(upload_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, "Upload not found. It may have expired or the ID is invalid.")

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
async def get_result(upload_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result found for this upload.")
    return result
