import uuid, time, os
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Upload, VideoAnalysis
from app.models.models import DetectionResult as DetectionResultRow
from app.services.video.backend import get_video_backend
from app.services.calibration import calibrate


async def run_video_detection(upload: Upload, db: AsyncSession, progress=None) -> str:
    """progress (optional): progress(fraction, stage), called from a worker thread while the model runs."""
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
    # The scoring is blocking, CPU-heavy work (decoding + a forward pass). Run it in a worker thread so it does
    # not freeze the event loop - previously every other request stalled for the whole duration of a video scan.
    result = await run_in_threadpool(backend.score, path, progress)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result_id = str(uuid.uuid4())
    # Post-hoc temperature-scaling calibration (additive, supplementary -- see app/services/calibration.py
    # and CLAUDE.md section 25). Unlike image, video's confidence_score IS already the raw P(FAKE) directly
    # (a pre-existing, documented quirk -- CLAUDE.md section 13), so no two-sided re-derivation is needed here.
    calibrated_confidence = calibrate(result.confidence, "video")
    detection = DetectionResultRow(
        result_id=result_id, upload_id=upload.upload_id,
        confidence_score=result.confidence, calibrated_confidence=calibrated_confidence,
        verdict=result.verdict,
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
        frames_with_face=result.raw_scores.get("frames_with_face"),
    ))
    await db.flush()
    return result_id
