import os
import time
import uuid

from PIL import UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Upload, DetectionResult, ImageAnalysis
from app.pipelines.image.inference import ImagePipeline
from app.services.detection_errors import UnprocessableMediaError


# Load the trained image model once when this module is imported.
# This prevents loading the 334 MB checkpoint for every image request.
_image_pipeline: ImagePipeline | None = None


def get_image_pipeline() -> ImagePipeline:
    global _image_pipeline

    if _image_pipeline is None:
        checkpoint = os.getenv("IMAGE_MODEL_CHECKPOINT")

        _image_pipeline = ImagePipeline(
            checkpoint_path=checkpoint
        )

    return _image_pipeline


async def run_image_detection(
    upload: Upload,
    db: AsyncSession,
) -> str:

    start = time.perf_counter()

    path = upload.storage_url

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Uploaded image not found at {path}"
        )

    # Run the NEW trained ConvNeXt-Tiny image pipeline.
    pipeline = get_image_pipeline()

    try:
        prediction = pipeline.predict(path)
    except (UnidentifiedImageError, OSError) as e:
        raise UnprocessableMediaError(
            "The uploaded file could not be read as an image. "
            "It may be corrupt or in an unsupported format."
        ) from e

    fake_probability = prediction["fake_probability"]
    real_probability = prediction["real_probability"]

    # Verdict banding: same +-0.2 margin around FAKE_THRESHOLD already used by
    # the audio and video detectors (services/audio/detector.py, services/
    # video/detector.py), applied here for the first time so a genuinely
    # close call comes back UNCERTAIN instead of a forced argmax. This is a
    # decision-boundary honesty change only — it does NOT and cannot fix a
    # confidently-wrong prediction (see CLAUDE.md / audit: the model
    # misclassifies real photos with high, not low, confidence), so it will
    # not catch that failure mode. No new threshold value is introduced.
    verdict = _verdict(fake_probability)

    confidence_score = prediction["confidence"]

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    result_id = str(uuid.uuid4())

    detection = DetectionResult(
        result_id=result_id,
        upload_id=upload.upload_id,
        confidence_score=confidence_score,
        verdict=verdict,
        model_used="ConvNeXt-Tiny",
        processing_time_ms=elapsed_ms,
    )

    db.add(detection)

    await db.flush()

    # Store the NEW model probabilities.
    #
    # Legacy efficientnet_score and fft_score are intentionally
    # left NULL for new detections.
    db.add(
        ImageAnalysis(
            result_id=result_id,
            efficientnet_score=None,
            fft_score=None,
            fake_probability=fake_probability,
            real_probability=real_probability,
        )
    )

    await db.flush()

    return result_id


def _verdict(fake_probability: float) -> str:
    if fake_probability >= settings.FAKE_THRESHOLD + 0.2:
        return "FAKE"
    if fake_probability <= settings.FAKE_THRESHOLD - 0.2:
        return "REAL"
    return "UNCERTAIN"