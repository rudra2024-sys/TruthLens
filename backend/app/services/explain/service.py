"""Explanation service shared by the API (`GET /detect/{id}/explain`) and the PDF report.

Read-only: recomputes attributions with the deployed models and touches neither the database nor any
stored result. If anything fails, callers get `available=False` with a reason instead of an exception, so an
explanation problem can never break a scan or a report download.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from app.models.models import DetectionResult, Upload
from app.services.explain.image_explainer import ImageExplanation, explain_image
from app.services.explain.video_explainer import VideoExplanation, explain_video

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Heatmaps show which regions the model's own FAKE score depends on (Grad-CAM). They are an explanation of "
    "the model's behaviour, not proof that a region was manipulated, and a model can be wrong."
)


@dataclass
class Explanation:
    available: bool
    media_type: str
    reason: str | None = None
    method: str | None = None
    note: str | None = None
    image: ImageExplanation | None = None
    video: VideoExplanation | None = None
    # image only: which sub-model produced the deployed (max) score
    convnext_fake_probability: float | None = None
    clip_fake_probability: float | None = None
    verdict_driver: str | None = None
    extras: dict = field(default_factory=dict)


def _unavailable(media_type: str, reason: str) -> Explanation:
    return Explanation(available=False, media_type=media_type, reason=reason)


def _explain_image(upload: Upload, result: DetectionResult) -> Explanation:
    from app.services.image.detector import get_image_pipeline

    pipe = get_image_pipeline()
    img = explain_image(upload.storage_url, pipe.model, pipe.device)

    ia = getattr(result, "image_analysis", None)
    cn = getattr(ia, "convnext_fake_probability", None)
    cl = getattr(ia, "clip_fake_probability", None)
    driver = None
    note = DISCLAIMER
    if cn is not None and cl is not None:
        driver = "CLIP" if cl > cn else "ConvNeXt-Tiny"
        if driver == "CLIP":
            note += (
                " This verdict was driven by the CLIP second-opinion model, which has no spatial heatmap; "
                "the map above shows what ConvNeXt-Tiny alone attended to."
            )
    if img.evidence_strength < 0.2:
        note += (
            " ConvNeXt-Tiny's own FAKE probability is low, so the heatmap is faded: no region raised its FAKE "
            "score meaningfully."
        )
    return Explanation(
        available=True, media_type="image", method="Grad-CAM on ConvNeXt-Tiny (last stage, FAKE logit)",
        note=note, image=img, convnext_fake_probability=cn, clip_fake_probability=cl, verdict_driver=driver,
    )


def _explain_video(upload: Upload, result: DetectionResult) -> Explanation:
    if os.getenv("VIDEO_MODEL_BACKEND", "model_v1") != "model_v1":
        return _unavailable(
            "video", "Explanations are only available for the neural video model (Video Model v1)."
        )
    try:
        vid = explain_video(upload.storage_url)
    except ValueError:
        return _unavailable("video", "This video could not be analysed frame by frame (too short or unreadable).")
    note = (
        "The verdict is the sigmoid of the mean of the 16 per-frame FAKE logits. Each bar is one frame's own "
        "score; the crops are the exact face/centre regions the model saw. " + DISCLAIMER
    )
    return Explanation(
        available=True, media_type="video",
        method="Per-frame scores + Grad-CAM on EfficientNet-B0 (Video Model v1)", note=note, video=vid,
    )


def build_explanation(upload: Upload, result: DetectionResult) -> Explanation:
    media = upload.media_type
    path = upload.storage_url
    if media == "audio":
        return _unavailable("audio", "Explanations are not available for audio yet.")
    if media not in ("image", "video"):
        return _unavailable(media, "Explanations are not available for this media type.")
    if not path or not os.path.isfile(path):
        return _unavailable(media, "The uploaded file is no longer available on the server.")
    try:
        return _explain_image(upload, result) if media == "image" else _explain_video(upload, result)
    except Exception:
        logger.exception("Explanation failed (upload_id=%s)", getattr(upload, "upload_id", "?"))
        return _unavailable(media, "The explanation could not be generated for this file.")
