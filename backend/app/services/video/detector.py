import uuid, time, os
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.models import Upload, DetectionResult, VideoAnalysis
from app.core.config import settings
from app.services.heuristics import analyze_video, clamp01


async def run_video_detection(upload: Upload, db: AsyncSession) -> str:
    start = time.perf_counter()

    path = upload.storage_url
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded video not found at {path}")

    structure, consistency, windows = analyze_video(path)
    confidence_score = clamp01(0.75 * structure + 0.25 * consistency)
    verdict = _verdict(confidence_score)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result_id = str(uuid.uuid4())
    detection = DetectionResult(
        result_id=result_id, upload_id=upload.upload_id,
        confidence_score=confidence_score, verdict=verdict,
        model_used="Video forensic heuristics v1",
        processing_time_ms=elapsed_ms,
    )
    db.add(detection)
    await db.flush()

    db.add(VideoAnalysis(
        result_id=result_id,
        xception_score=structure,
        face_voice_sync=consistency,
        frames_analyzed=windows,
    ))
    await db.flush()
    return result_id


def _verdict(score: float) -> str:
    if score >= settings.FAKE_THRESHOLD + 0.2:
        return "FAKE"
    if score <= settings.FAKE_THRESHOLD - 0.2:
        return "REAL"
    return "UNCERTAIN"
