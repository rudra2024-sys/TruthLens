import uuid
import time
import os

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Upload, AudioAnalysis
from app.models.models import DetectionResult as DetectionResultRow
from app.services.audio.backend import get_audio_backend


async def run_audio_detection(upload: Upload, db: AsyncSession) -> str:
    start = time.perf_counter()

    path = upload.storage_url

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Uploaded audio not found at {path}"
        )

    # Swappable backend (mirrors video/detector.py): today this is
    # AudioModelV1Backend (wav2vec2-base) by default; AUDIO_MODEL_BACKEND=aasist
    # selects the prior AASIST path for rollback. See services/audio/backend.py.
    backend = get_audio_backend()
    result = backend.score(path)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result_id = str(uuid.uuid4())

    detection = DetectionResultRow(
        result_id=result_id,
        upload_id=upload.upload_id,
        confidence_score=result.confidence,
        verdict=result.verdict,
        model_used=result.model_used,
        processing_time_ms=elapsed_ms,
    )

    db.add(detection)
    await db.flush()

    db.add(
        AudioAnalysis(
            result_id=result_id,
            wav2vec_score=result.raw_scores.get("mean", 0.0),
            lcnn_score=result.raw_scores.get("std", 0.0),
            duration_s=result.raw_scores.get("duration_s"),
            windows_analyzed=result.raw_scores.get("windows"),
        )
    )

    await db.flush()

    return result_id