import uuid, time, os
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Upload, VideoAnalysis
from app.models.models import DetectionResult as DetectionResultRow
from app.services.video.backend import get_video_backend


async def run_video_detection(upload: Upload, db: AsyncSession) -> str:
    start = time.perf_counter()

    path = upload.storage_url
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded video not found at {path}")

    # Swappable backend (Priority 6 prep): today this is always
    # HeuristicVideoBackend with byte-for-byte identical behavior to before;
    # a future validated model can be selected via VIDEO_MODEL_BACKEND
    # without touching this function's DB-writing code. See
    # services/video/backend.py.
    backend = get_video_backend()
    result = backend.score(path)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result_id = str(uuid.uuid4())
    detection = DetectionResultRow(
        result_id=result_id, upload_id=upload.upload_id,
        confidence_score=result.confidence, verdict=result.verdict,
        model_used=result.model_used,
        processing_time_ms=elapsed_ms,
    )
    db.add(detection)
    await db.flush()

    db.add(VideoAnalysis(
        result_id=result_id,
        xception_score=result.raw_scores.get("structure"),
        face_voice_sync=result.raw_scores.get("consistency"),
        frames_analyzed=result.raw_scores.get("windows"),
    ))
    await db.flush()
    return result_id
