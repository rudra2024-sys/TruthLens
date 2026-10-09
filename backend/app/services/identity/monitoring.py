"""Continuous monitoring: periodic checks within a MonitoringSession, each scoring three
independent signals against a single live-captured frame. A check can be flagged even when
identity matches -- that's the point (see CLAUDE.md's identity-match monitoring section).

Kept as a separate module from detector.py (which owns the one-shot enroll/match flow) so the
two call paths stay independently readable -- they share the identity pipeline singleton and
banding helpers but differ in an important way: a one-shot match treats "no face found" as a
hard error (there's nothing to compare), while a monitoring check treats it as a *flaggable
outcome to record*, not a failure.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import FaceReference, MonitoringCheck, MonitoringSession, Upload
from app.pipelines.identity.inference import NoFaceDetectedError
from app.services.detection_errors import UnprocessableMediaError
from app.services.identity.detector import (
    _cosine_similarity,
    _verdict as _identity_verdict,
    get_identity_pipeline,
)
from app.services.image.detector import (
    _verdict as _deepfake_verdict,
    get_clip_pipeline,
    get_image_pipeline,
)


# "multiple_faces" history -- first attempt (OpenCV Haar cascade, strict minNeighbors) was
# shipped and then found broken by real-webcam testing: 0 faces on an obvious, well-lit, single
# forward-facing person, while MTCNN (the detector this pipeline's own embedding step already
# uses) found that exact face correctly at 99.98% confidence. Haar was dropped, and presence
# detection was narrowed to 0-vs-1 only (via embed()/NoFaceDetectedError) rather than ship a
# broken count. Redone 2026-10-09 using IdentityPipeline.count_faces() (MTCNN's own .detect(),
# confidence-filtered + IoU-deduplicated -- see that method's docstring for the full validation
# story, now confirmed correct on a real two-person composite, real single-face frames, and
# faceless frames) -- restoring a real "more than one person in frame" signal.


def _run_deepfake_check(path: str) -> tuple[float, str]:
    """Same max(ConvNeXt, CLIP) ensemble as services/image/detector.py::run_image_detection,
    called directly against the already-loaded pipeline singletons -- deliberately does NOT
    write a DetectionResult/ImageAnalysis row (a 20s-cadence session would otherwise flood
    History/Reports with monitoring-frame entries that aren't real user-initiated scans)."""
    convnext_pipeline = get_image_pipeline()
    clip_pipeline = get_clip_pipeline()
    try:
        convnext_prediction = convnext_pipeline.predict(path)
        clip_prediction = clip_pipeline.predict(path)
    except (UnidentifiedImageError, OSError) as e:
        raise UnprocessableMediaError(
            "The uploaded file could not be read as an image. It may be corrupt or in an unsupported format."
        ) from e

    fake_probability = max(convnext_prediction["fake_probability"], clip_prediction["fake_probability"])
    return fake_probability, _deepfake_verdict(fake_probability)


def _compute_signals(path: str, reference: FaceReference) -> dict:
    """All the CPU-bound work (MTCNN count+embedding, ConvNeXt+CLIP) -- deliberately a plain
    sync function with no DB access, so the route layer can run it via run_in_threadpool
    without blocking the event loop (same reasoning as the existing /explain route)."""
    pipeline = get_identity_pipeline()
    face_count = pipeline.count_faces(path)

    similarity_score = None
    identity_verdict = None
    if face_count == 1:
        try:
            live_embedding = pipeline.embed(path)
        except NoFaceDetectedError:
            # count_faces and embed() disagree (MTCNN's own two internal calls, rare) -- treat
            # like no usable face rather than crashing the check.
            face_count = 0
        else:
            reference_embedding = json.loads(reference.embedding_json)
            similarity_score = _cosine_similarity(live_embedding, reference_embedding)
            identity_verdict = _identity_verdict(similarity_score, settings.IDENTITY_MATCH_THRESHOLD)

    fake_probability, deepfake_verdict = _run_deepfake_check(path)

    reasons: list[str] = []
    if face_count == 0:
        reasons.append("no_face")
    elif face_count > 1:
        reasons.append("multiple_faces")
    if identity_verdict == "NO_MATCH":
        reasons.append("identity_mismatch")
    elif identity_verdict == "UNCERTAIN":
        reasons.append("identity_uncertain")
    if deepfake_verdict == "FAKE":
        reasons.append("deepfake_signal")
    elif deepfake_verdict == "UNCERTAIN":
        reasons.append("deepfake_uncertain")

    return {
        "face_count": face_count,
        "similarity_score": similarity_score,
        "identity_verdict": identity_verdict,
        "fake_probability": fake_probability,
        "deepfake_verdict": deepfake_verdict,
        "flagged": bool(reasons),
        "flag_reasons_json": json.dumps(reasons),
    }


async def run_check(
    session: MonitoringSession,
    reference: FaceReference,
    upload: Upload,
    db: AsyncSession,
) -> MonitoringCheck:
    path = upload.storage_url
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded image not found at {path}")

    # Referenced as a plain module-global (not imported into a local variable) so tests can
    # monkeypatch app.services.identity.monitoring._compute_signals directly, same technique
    # used for detector.py's _embed (see tests/test_identity.py's stub_embed fixture).
    signals = await run_in_threadpool(_compute_signals, path, reference)

    check = MonitoringCheck(
        check_id=str(uuid.uuid4()),
        session_id=session.session_id,
        upload_id=upload.upload_id,
        checked_at=datetime.utcnow().isoformat(),
        **signals,
    )
    db.add(check)
    await db.flush()
    return check
