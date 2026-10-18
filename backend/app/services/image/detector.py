import logging
import math
import os
import time
import uuid

from PIL import UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import Upload, DetectionResult, ImageAnalysis
from app.pipelines.image.inference import ImagePipeline
from app.pipelines.image_clip.inference import ClipPipeline
from app.services.detection_errors import UnprocessableMediaError
from app.services.calibration import calibrate
from app.services.image import face_swap

logger = logging.getLogger(__name__)


# Load the trained image models once when this module is imported.
# This prevents loading the checkpoints for every image request.
_image_pipeline: ImagePipeline | None = None
_clip_pipeline: ClipPipeline | None = None


def get_image_pipeline() -> ImagePipeline:
    global _image_pipeline

    if _image_pipeline is None:
        checkpoint = os.getenv("IMAGE_MODEL_CHECKPOINT")

        _image_pipeline = ImagePipeline(
            checkpoint_path=checkpoint
        )

    return _image_pipeline


def get_clip_pipeline() -> ClipPipeline:
    global _clip_pipeline

    if _clip_pipeline is None:
        checkpoint = os.getenv("CLIP_MODEL_CHECKPOINT")

        _clip_pipeline = ClipPipeline(
            checkpoint_path=checkpoint
        )

    return _clip_pipeline


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

    # Ensemble: ConvNeXt-Tiny (trained end-to-end on our fake sources) +
    # a frozen-CLIP second opinion (features not shaped by any particular
    # generator's artifacts, so it catches some generators ConvNeXt misses --
    # see session notes 2026-09-11/12). Originally combined via a plain
    # average (2026-09-12), chosen over max because averaging won on both
    # overall and per-source accuracy at the time. Switched to max on
    # 2026-09-16 after CLIP got a round-3 fine-tune (added a curated
    # DiffusionDB portrait-app/vintage-costume source, see CLAUDE.md Sec 3)
    # that raised its accuracy on that genre from 26.7% to 93.3%, but
    # averaging with the still-unretrained ConvNeXt dragged the ensemble's
    # result on that same genre back down to 66.7% -- ConvNeXt's blind spot
    # was cancelling out most of CLIP's fix. Max lets either model's FAKE
    # catch through undiluted, at a measured cost: per the round-3 notebook's
    # comparison table, max scores ~0.9 points lower overall accuracy than
    # average across the four original sources (more false FAKE flags on
    # ordinary real photos), in exchange for ~40-80 points of recall on this
    # genre depending on model combination. Revisit this trade if ConvNeXt
    # is ever retrained on the same portrait-app source, which would let
    # averaging work again without the accuracy cliff on ordinary photos.
    convnext_pipeline = get_image_pipeline()
    clip_pipeline = get_clip_pipeline()

    try:
        convnext_prediction = convnext_pipeline.predict(path)
        clip_prediction = clip_pipeline.predict(path)
    except (UnidentifiedImageError, OSError) as e:
        raise UnprocessableMediaError(
            "The uploaded file could not be read as an image. "
            "It may be corrupt or in an unsupported format."
        ) from e

    convnext_fake_probability = convnext_prediction["fake_probability"]
    clip_fake_probability = clip_prediction["fake_probability"]

    fake_probability = max(
        convnext_fake_probability, clip_fake_probability
    )

    # Face-swap check (CLAUDE.md section 28). The whole-image models above are near chance on face swaps (a swap is
    # a real photo with one face replaced); Video Model v1 round 4, run on the detected face crop, is not. Applies
    # only when a frontal face >= 64 px is found; its Platt-calibrated P(FAKE) joins the same max rule. A failure
    # here must never break the scan, so it just means "check not applied".
    face_swap_probability, face_swap_face_px = _face_swap_check(path)
    ensemble_fake_probability = fake_probability
    if face_swap_probability is not None:
        fake_probability = max(fake_probability, face_swap_probability)
    real_probability = 1 - fake_probability

    # Verdict banding: +-0.2 margin around FAKE_THRESHOLD, same convention
    # already used by the audio and video detectors (services/audio/
    # detector.py, services/video/detector.py), so a genuinely close call
    # (including disagreement between the two models above) comes back
    # UNCERTAIN instead of a forced argmax.
    verdict = _verdict(fake_probability)

    confidence_score = max(fake_probability, real_probability)

    # Post-hoc temperature-scaling calibration (additive, supplementary -- see app/services/calibration.py
    # and CLAUDE.md section 25). Calibrates the underlying P(FAKE), then re-derives the two-sided confidence
    # the same way confidence_score itself is derived above, so the two numbers stay on the same convention.
    # When the face check drives the score it is already on a calibrated scale; the temperature fit belongs to the
    # whole-image ensemble only.
    if face_swap_probability is not None and face_swap_probability > ensemble_fake_probability:
        calibrated_fake = face_swap_probability
    else:
        calibrated_fake = calibrate(ensemble_fake_probability, "image")
    calibrated_confidence = max(calibrated_fake, 1 - calibrated_fake) if calibrated_fake is not None else None

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    result_id = str(uuid.uuid4())

    detection = DetectionResult(
        result_id=result_id,
        upload_id=upload.upload_id,
        confidence_score=confidence_score,
        calibrated_confidence=calibrated_confidence,
        verdict=verdict,
        model_used=(
            "ConvNeXt-Tiny + CLIP ViT-B/16 (ensemble)"
            if face_swap_probability is None
            else "ConvNeXt-Tiny + CLIP ViT-B/16 (ensemble) + face-swap check ("
            + ("photo model, round 2" if face_swap.checkpoint_path() else "Video Model v1 round 4")
            + ")"
        ),
        processing_time_ms=elapsed_ms,
    )

    db.add(detection)

    await db.flush()

    # Legacy efficientnet_score and fft_score are intentionally
    # left NULL for new detections.
    db.add(
        ImageAnalysis(
            result_id=result_id,
            efficientnet_score=None,
            fft_score=None,
            fake_probability=fake_probability,
            real_probability=real_probability,
            convnext_fake_probability=convnext_fake_probability,
            clip_fake_probability=clip_fake_probability,
            face_swap_fake_probability=face_swap_probability,
            face_swap_face_px=face_swap_face_px,
        )
    )

    await db.flush()

    return result_id


def _face_swap_check(path: str) -> tuple[float | None, int | None]:
    """(calibrated P(FAKE), face width px) or (None, None) when no usable face or the check is unavailable."""
    try:
        score = face_swap.score_image(path)
    except Exception:  # missing checkpoint, unreadable frame, ... -> the rest of the scan still stands
        logger.warning("face-swap check skipped", exc_info=True)
        return None, None
    if score is None:
        return None, None
    p = calibrate(1.0 / (1.0 + math.exp(-score.logit)), face_swap.calibration_key())
    if p is None:  # no calibration configured -> do not guess
        return None, None
    return p, score.face_px


def _verdict(fake_probability: float) -> str:
    if fake_probability >= settings.FAKE_THRESHOLD + 0.2:
        return "FAKE"
    if fake_probability <= settings.FAKE_THRESHOLD - 0.2:
        return "REAL"
    return "UNCERTAIN"