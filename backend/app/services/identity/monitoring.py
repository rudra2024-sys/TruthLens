"""Continuous monitoring: periodic checks within a MonitoringSession, each scoring several
independent signals against a single live-captured frame (and, optionally, a short accompanying
audio clip). A check can be flagged even when identity matches -- that's the point (see
CLAUDE.md's identity-match monitoring section).

Kept as a separate module from detector.py (which owns the one-shot enroll/match flow) so the
two call paths stay independently readable -- they share the identity pipeline singleton and
banding helpers but differ in an important way: a one-shot match treats "no face found" as a
hard error (there's nothing to compare), while a monitoring check treats it as a *flaggable
outcome to record*, not a failure.

"multiple_faces" history -- first attempt (OpenCV Haar cascade, strict minNeighbors) was shipped
and then found broken by real-webcam testing: 0 faces on an obvious, well-lit, single forward-
facing person, while MTCNN (the detector this pipeline's own embedding step already uses) found
that exact face correctly at 99.98% confidence. Haar was dropped, and presence detection was
narrowed to 0-vs-1 only (via embed()/NoFaceDetectedError) rather than ship a broken count. Redone
2026-10-09 using IdentityPipeline.detect_faces() (MTCNN's own .detect(), confidence-filtered +
IoU-deduplicated -- see that method's docstring for the full validation story, now confirmed
correct on a real two-person composite, real single-face frames, and faceless frames) --
restoring a real "more than one person in frame" signal.

Items 7/8/10 (added 2026-10-09): speech/talking detection, phone/book object detection, and
gaze/head-pose all extend _compute_signals below. Item 9 (lip-sync) is informational only --
mouth_width_px is recorded on every check where a single face was found, but deliberately does
NOT produce its own flag reason (see session_report.py's lip-sync section for why: MTCNN's
5-point landmarks give mouth-corner WIDTH, not true vertical aperture, a much weaker proxy for
talking than real lip-sync work would use -- mixing it into the same auto-flag confidence tier
as the better-validated signals would dilute trust in those).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
from sqlalchemy import select
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
from app.services.identity.head_pose import estimate_yaw_pitch, mouth_width
from app.services.image.detector import (
    _verdict as _deepfake_verdict,
    get_clip_pipeline,
    get_image_pipeline,
)

_object_pipeline = None


def get_object_pipeline():
    global _object_pipeline
    if _object_pipeline is None:
        from app.pipelines.objects.inference import ObjectPipeline
        _object_pipeline = ObjectPipeline()
    return _object_pipeline


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


def _compute_signals(path: str, reference: FaceReference, audio_path: str | None) -> dict:
    """All the CPU-bound work (MTCNN detect+embed, ConvNeXt+CLIP, YOLOv8n, audio VAD) --
    deliberately a plain sync function with no DB access, so the route layer can run it via
    run_in_threadpool without blocking the event loop (same reasoning as the existing /explain
    route). Returns raw signal values only -- session-state-dependent decisions (the head-pose
    baseline, "two consecutive off-baseline checks") need DB access and live in run_check below,
    not here.
    """
    pipeline = get_identity_pipeline()
    detection = pipeline.detect_faces(path)
    face_count = detection["count"]
    landmarks = detection["landmarks"]
    image_width, image_height = detection["image_size"]

    similarity_score = None
    identity_verdict = None
    yaw_deg = None
    pitch_deg = None
    mouth_width_px = None
    face_box = None
    landmarks_points = None

    if face_count == 1 and landmarks is not None:
        try:
            yaw_deg, pitch_deg = estimate_yaw_pitch(landmarks, detection["image_size"])
        except RuntimeError:
            pass  # solvePnP didn't converge on this frame -- leave yaw/pitch unset, not fatal
        mouth_width_px = mouth_width(landmarks)
        # Plain-list copies of the real MTCNN output for the monitoring UI's live face-overlay
        # visualization (item 3 of the follow-up FYP plan) -- genuine per-check detection
        # coordinates, not a client-side/decorative approximation.
        if detection["box"] is not None:
            face_box = [float(v) for v in detection["box"]]
        landmarks_points = [[float(x), float(y)] for x, y in landmarks]

        try:
            live_embedding = pipeline.embed(path)
        except NoFaceDetectedError:
            # detect_faces and embed() disagree (MTCNN's own two internal calls, rare) -- treat
            # like no usable face rather than crashing the check.
            face_count = 0
            yaw_deg = pitch_deg = mouth_width_px = None
            face_box = landmarks_points = None
        else:
            reference_embedding = json.loads(reference.embedding_json)
            similarity_score = _cosine_similarity(live_embedding, reference_embedding)
            identity_verdict = _identity_verdict(similarity_score, settings.IDENTITY_MATCH_THRESHOLD)

    fake_probability, deepfake_verdict = _run_deepfake_check(path)

    object_detections: list[str] = []
    try:
        object_detections = get_object_pipeline().detect(
            path, conf_thresh=settings.OBJECT_DETECTION_CONFIDENCE_THRESHOLD
        )
    except FileNotFoundError:
        pass  # checkpoint not present locally (e.g. CI) -- object detection just silently skips

    speech_ratio = None
    if audio_path is not None:
        from app.services.identity.audio_vad import speech_ratio as _speech_ratio
        speech_ratio = _speech_ratio(audio_path)

    return {
        "face_count": face_count,
        "similarity_score": similarity_score,
        "identity_verdict": identity_verdict,
        "fake_probability": fake_probability,
        "deepfake_verdict": deepfake_verdict,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "mouth_width_px": mouth_width_px,
        "speech_ratio": speech_ratio,
        "object_detections": object_detections,
        "face_box": face_box,
        "landmarks": landmarks_points,
        "image_width": image_width,
        "image_height": image_height,
    }


async def _previous_check(db: AsyncSession, session_id: str) -> MonitoringCheck | None:
    r = await db.execute(
        select(MonitoringCheck)
        .where(MonitoringCheck.session_id == session_id)
        .order_by(MonitoringCheck.checked_at.desc())
        .limit(1)
    )
    return r.scalar_one_or_none()


def _deviates_from_baseline(yaw: float, pitch: float, baseline_yaw: float, baseline_pitch: float) -> bool:
    return (
        abs(yaw - baseline_yaw) > settings.LOOKING_AWAY_YAW_THRESHOLD_DEG
        or abs(pitch - baseline_pitch) > settings.LOOKING_AWAY_PITCH_THRESHOLD_DEG
    )


async def run_check(
    session: MonitoringSession,
    reference: FaceReference,
    upload: Upload,
    db: AsyncSession,
    audio_upload: Upload | None = None,
) -> MonitoringCheck:
    path = upload.storage_url
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded image not found at {path}")

    audio_path = None
    if audio_upload is not None:
        audio_path = audio_upload.storage_url
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Uploaded audio not found at {audio_path}")

    # Referenced as a plain module-global (not imported into a local variable) so tests can
    # monkeypatch app.services.identity.monitoring._compute_signals directly, same technique
    # used for detector.py's _embed (see tests/test_identity.py's stub_embed fixture).
    signals = await run_in_threadpool(_compute_signals, path, reference, audio_path)

    reasons: list[str] = []
    if signals["face_count"] == 0:
        reasons.append("no_face")
    elif signals["face_count"] > 1:
        reasons.append("multiple_faces")
    if signals["identity_verdict"] == "NO_MATCH":
        reasons.append("identity_mismatch")
    elif signals["identity_verdict"] == "UNCERTAIN":
        reasons.append("identity_uncertain")
    if signals["deepfake_verdict"] == "FAKE":
        reasons.append("deepfake_signal")
    elif signals["deepfake_verdict"] == "UNCERTAIN":
        reasons.append("deepfake_uncertain")
    if signals["speech_ratio"] is not None and signals["speech_ratio"] >= settings.TALKING_SPEECH_RATIO_THRESHOLD:
        reasons.append("talking_detected")
    if signals["object_detections"]:
        reasons.append("object_detected")

    # Head-pose baseline + sustained-deviation logic -- needs session/DB state, so it lives here
    # rather than in the pure _compute_signals function above.
    yaw, pitch = signals["yaw_deg"], signals["pitch_deg"]
    if yaw is not None and pitch is not None:
        if session.baseline_yaw is None:
            session.baseline_yaw, session.baseline_pitch = yaw, pitch
        elif _deviates_from_baseline(yaw, pitch, session.baseline_yaw, session.baseline_pitch):
            previous = await _previous_check(db, session.session_id)
            previous_deviates = (
                previous is not None
                and previous.yaw_deg is not None
                and previous.pitch_deg is not None
                and _deviates_from_baseline(
                    previous.yaw_deg, previous.pitch_deg, session.baseline_yaw, session.baseline_pitch
                )
            )
            if previous_deviates:
                reasons.append("looking_away")

    check = MonitoringCheck(
        check_id=str(uuid.uuid4()),
        session_id=session.session_id,
        upload_id=upload.upload_id,
        face_count=signals["face_count"],
        similarity_score=signals["similarity_score"],
        identity_verdict=signals["identity_verdict"],
        fake_probability=signals["fake_probability"],
        deepfake_verdict=signals["deepfake_verdict"],
        yaw_deg=yaw,
        pitch_deg=pitch,
        mouth_width_px=signals["mouth_width_px"],
        speech_ratio=signals["speech_ratio"],
        object_detections_json=json.dumps(signals["object_detections"]),
        face_box_json=json.dumps(signals["face_box"]) if signals["face_box"] is not None else None,
        landmarks_json=json.dumps(signals["landmarks"]) if signals["landmarks"] is not None else None,
        image_width=signals["image_width"],
        image_height=signals["image_height"],
        flagged=bool(reasons),
        flag_reasons_json=json.dumps(reasons),
        checked_at=datetime.utcnow().isoformat(),
    )
    db.add(check)
    await db.flush()
    return check
