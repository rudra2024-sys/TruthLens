import uuid
import time
import os

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Upload, DetectionResult, AudioAnalysis
from app.core.config import settings
from app.services.detection_errors import UnprocessableMediaError
from app.services.models.model_client import run_audio_model


async def run_audio_detection(upload: Upload, db: AsyncSession) -> str:
    start = time.perf_counter()

    path = upload.storage_url

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Uploaded audio not found at {path}"
        )

    # Run the real AASIST audio model. PyAV's decode errors
    # (av.error.FFmpegError and subclasses) are ValueError subclasses, so
    # this also covers the explicit ValueErrors raised by build_windows()
    # for empty/undecodable audio.
    try:
        model_result = run_audio_model(path)
    except ValueError as e:
        raise UnprocessableMediaError(
            "The uploaded file could not be read as audio. "
            "It may be corrupt or in an unsupported format."
        ) from e

    spoof_probability = float(
        model_result.get("spoof_probability", 0.0)
    )

    confidence_score = max(
        0.0,
        min(1.0, spoof_probability)
    )

    verdict = _verdict(confidence_score)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    result_id = str(uuid.uuid4())

    detection = DetectionResult(
        result_id=result_id,
        upload_id=upload.upload_id,
        confidence_score=confidence_score,
        verdict=verdict,
        model_used="AASIST",
        processing_time_ms=elapsed_ms,
    )

    db.add(detection)
    await db.flush()

    db.add(
        AudioAnalysis(
            result_id=result_id,
            wav2vec_score=spoof_probability,
            lcnn_score=float(
                model_result.get(
                    "spoof_probability_std",
                    0.0
                )
            ),
        )
    )

    await db.flush()

    return result_id


def _verdict(score: float) -> str:
    if score >= settings.FAKE_THRESHOLD + 0.2:
        return "FAKE"

    if score <= settings.FAKE_THRESHOLD - 0.2:
        return "REAL"

    return "UNCERTAIN"