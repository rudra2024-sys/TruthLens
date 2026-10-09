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


async def enroll_reference(upload: Upload, db: AsyncSession) -> FaceReference:
    embedding = _embed(upload.storage_url)
    embedding_json = json.dumps(embedding)

    existing = (
        await db.execute(select(FaceReference).where(FaceReference.user_id == upload.user_id))
    ).scalar_one_or_none()

    now = datetime.utcnow().isoformat()

    if existing is not None:
        existing.upload_id = upload.upload_id
        existing.embedding_json = embedding_json
        existing.enrolled_at = now
        await db.flush()
        return existing

    reference = FaceReference(
        reference_id=str(uuid.uuid4()),
        user_id=upload.user_id,
        upload_id=upload.upload_id,
        embedding_json=embedding_json,
        enrolled_at=now,
    )
    db.add(reference)
    await db.flush()
    return reference


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
