import os
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Upload, DetectionResult, ImageAnalysis
from app.pipelines.image.inference import ImagePipeline


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

    prediction = pipeline.predict(path)

    fake_probability = prediction["fake_probability"]
    real_probability = prediction["real_probability"]
    verdict = prediction["verdict"]

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