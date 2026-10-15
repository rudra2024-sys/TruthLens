"""Identity-match service: enroll a reference face embedding, and compare a later
live-captured frame against it. See CLAUDE.md's identity-match section for the full design
rationale (model choice, threshold caveats, weight-download guard).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime

from PIL import UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import FaceReference, IdentityMatchResult, Upload
from app.pipelines.identity.inference import IdentityPipeline, NoFaceDetectedError
from app.services.detection_errors import UnprocessableMediaError
from app.services.identity.enrollment_quality import assess_enrollment_quality
from app.services.identity.id_quality import assess_government_id_quality


# Loaded once per process on first use, like every other pipeline singleton in this codebase
# (services/image/detector.py, etc.) -- NOT constructed at import time, so a pytest run that
# never calls get_identity_pipeline() never triggers facenet-pytorch's weight-presence check.
_identity_pipeline: IdentityPipeline | None = None


def get_identity_pipeline() -> IdentityPipeline:
    global _identity_pipeline
    if _identity_pipeline is None:
        _identity_pipeline = IdentityPipeline()
    return _identity_pipeline


def _embed(path: str) -> list[float]:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded image not found at {path}")

    pipeline = get_identity_pipeline()
    try:
        return pipeline.embed(path)
    except NoFaceDetectedError as e:
        raise UnprocessableMediaError(str(e)) from e
    except (UnidentifiedImageError, OSError) as e:
        raise UnprocessableMediaError(
            "The uploaded file could not be read as an image. "
            "It may be corrupt or in an unsupported format."
        ) from e


def _detect_faces(path: str) -> dict:
    """Thin wrapper (same purpose as _embed above: a module-level function tests can monkeypatch
    without constructing the real pipeline -- see tests/test_identity.py's stub_embed fixture)."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Uploaded image not found at {path}")
    return get_identity_pipeline().detect_faces(path)


async def enroll_reference(upload: Upload, db: AsyncSession) -> FaceReference:
    # _embed() first (not the quality gate) -- it's the authoritative "is there a face at all"
    # check (raises NoFaceDetectedError otherwise). detect_faces() uses a stricter 0.95
    # confidence filter for its box/count (built for the monitoring overlay/face-count use case)
    # and can disagree with embed()'s own internal MTCNN threshold -- confirmed directly via live
    # testing: a heavily blurred photo found no confident box via detect_faces() (count=0) while
    # still embedding successfully, which let it slip past a quality gate that only ran its blur
    # check inside the count==1 branch. Running embed() first means a genuinely faceless photo
    # still gets the accurate "no face" error, while a blurry-but-embeddable one still reaches
    # the quality gate below.
    embedding = _embed(upload.storage_url)
    embedding_json = json.dumps(embedding)

    detection = _detect_faces(upload.storage_url)
    issues = assess_enrollment_quality(upload.storage_url, detection)
    if issues:
        raise UnprocessableMediaError(" ".join(issues))

    existing = (
        await db.execute(select(FaceReference).where(FaceReference.user_id == upload.user_id))
    ).scalar_one_or_none()

    now = datetime.utcnow().isoformat()

    if existing is not None:
        existing.upload_id = upload.upload_id
        existing.embedding_json = embedding_json
        existing.enrolled_at = now
        existing.consent_given_at = now
        await db.flush()
        return existing

    reference = FaceReference(
        reference_id=str(uuid.uuid4()),
        user_id=upload.user_id,
        upload_id=upload.upload_id,
        embedding_json=embedding_json,
        enrolled_at=now,
        consent_given_at=now,
    )
    db.add(reference)
    await db.flush()
    return reference


async def _upsert_government_id_reference(
    user_id: str, upload_id: str, embedding_json: str, db: AsyncSession,
) -> FaceReference:
    """Shared by verify_government_id and verify_identity_with_selfie -- both end up storing the
    ID-card photo's embedding as the user's FaceReference, marked government_id_verified=True."""
    existing = (
        await db.execute(select(FaceReference).where(FaceReference.user_id == user_id))
    ).scalar_one_or_none()

    now = datetime.utcnow().isoformat()

    if existing is not None:
        existing.upload_id = upload_id
        existing.embedding_json = embedding_json
        existing.enrolled_at = now
        existing.consent_given_at = now
        existing.government_id_verified = True
        await db.flush()
        return existing

    reference = FaceReference(
        reference_id=str(uuid.uuid4()),
        user_id=user_id,
        upload_id=upload_id,
        embedding_json=embedding_json,
        enrolled_at=now,
        consent_given_at=now,
        government_id_verified=True,
    )
    db.add(reference)
    await db.flush()
    return reference


async def verify_government_id(upload: Upload, db: AsyncSession) -> FaceReference:
    """Transient-use government-ID enrollment (added 2026-10-12): extracts a face embedding from
    a government-ID-card photo and stores/updates FaceReference exactly like enroll_reference,
    marked government_id_verified=True. Does NOT touch the uploaded file -- deleting it is the
    caller's job (routes/identity.py::post_verify_government_id), always, success or failure, so
    "transient" has no happy-path-only exception. Same embed-before-quality-gate ordering as
    enroll_reference, for the same reason (see that function's comment).

    Scope note, worth repeating here since it's easy to overclaim: this verifies the face on the
    photographed document matches, nothing more. There is no document-authenticity/OCR check --
    a forged or fake ID card with a real face on it passes exactly as well as a genuine one
    would. That is a deliberate, named limitation, not an oversight."""
    embedding = _embed(upload.storage_url)
    embedding_json = json.dumps(embedding)

    detection = _detect_faces(upload.storage_url)
    issues = assess_government_id_quality(upload.storage_url, detection)
    if issues:
        raise UnprocessableMediaError(" ".join(issues))

    return await _upsert_government_id_reference(upload.user_id, upload.upload_id, embedding_json, db)


async def verify_identity_with_selfie(
    id_upload: Upload, live_upload: Upload, db: AsyncSession,
) -> tuple[float, str, FaceReference | None]:
    """Per-attempt identity check (added 2026-10-14, see CLAUDE.md's ProctorLens section): a
    fresh government-ID photo plus a live selfie taken moments apart, compared for a similarity
    "confidence score" gated the same way as the existing live-match flow (match_against_reference
    below) -- this is the check that runs before every exam attempt, not a one-time enrollment
    step. Both uploads are the caller's to delete immediately after (routes/exams.py), same
    transient-use convention as verify_government_id.

    On a MATCH verdict, also refreshes the stored FaceReference to this attempt's ID photo (via
    the same upsert verify_government_id uses) so the exam's own periodic monitoring checks
    compare against the identity just confirmed moments ago, not a stale reference from an
    earlier session. On UNCERTAIN/NO_MATCH, the reference is left untouched and None is returned
    in its place -- the caller must not start the attempt."""
    id_embedding = _embed(id_upload.storage_url)
    detection = _detect_faces(id_upload.storage_url)
    issues = assess_government_id_quality(id_upload.storage_url, detection)
    if issues:
        raise UnprocessableMediaError(" ".join(issues))

    live_embedding = _embed(live_upload.storage_url)

    similarity = _cosine_similarity(id_embedding, live_embedding)
    verdict = _verdict(similarity, settings.IDENTITY_MATCH_THRESHOLD)

    reference = None
    if verdict == "MATCH":
        reference = await _upsert_government_id_reference(
            id_upload.user_id, id_upload.upload_id, json.dumps(id_embedding), db,
        )
    return similarity, verdict, reference


async def match_against_reference(
    reference: FaceReference,
    upload: Upload,
    db: AsyncSession,
) -> IdentityMatchResult:
    live_embedding = _embed(upload.storage_url)
    reference_embedding = json.loads(reference.embedding_json)

    similarity = _cosine_similarity(live_embedding, reference_embedding)
    verdict = _verdict(similarity, settings.IDENTITY_MATCH_THRESHOLD)

    result = IdentityMatchResult(
        match_id=str(uuid.uuid4()),
        user_id=upload.user_id,
        reference_id=reference.reference_id,
        upload_id=upload.upload_id,
        similarity_score=similarity,
        verdict=verdict,
        checked_at=datetime.utcnow().isoformat(),
    )
    db.add(result)
    await db.flush()
    return result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _verdict(similarity: float, threshold: float, margin: float = 0.1) -> str:
    """Same +-margin banding convention as the image/audio/video detectors' own local
    _verdict()/_band_verdict() helpers -- a genuinely close call comes back UNCERTAIN instead
    of a forced match/no-match. See config.py's IDENTITY_MATCH_THRESHOLD for the caveat that
    this threshold is an unvalidated literature-typical default, not measured in this repo."""
    if similarity >= threshold + margin:
        return "MATCH"
    if similarity <= threshold - margin:
        return "NO_MATCH"
    return "UNCERTAIN"
