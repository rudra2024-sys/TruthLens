"""Deep readiness check (added 2026-10-03) -- GET /health/ready. `/health` itself stays a fast, dependency-free
liveness probe (useful for "is the process even up" during a transient DB hiccup, so an orchestrator doesn't
restart a process that would recover on its own); this is the separate, slower check for "can this instance
actually serve a real request right now."

Checks DB connectivity (a trivial SELECT 1) and whether each model's checkpoint file exists on disk -- NOT
whether the model is actually loaded into memory (that only happens lazily, on first real request per
detector, and loading every checkpoint just to answer a health check would be slow and wasteful; checkpoint
presence is what actually fails in practice, e.g. a bad volume mount or a forgotten fetch_checkpoints.py run).
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

BACKEND = Path(__file__).resolve().parents[2]


def _checkpoint_status(env_var: str, default: Path) -> dict:
    path = Path(os.getenv(env_var) or default)
    return {"path": str(path), "present": path.is_file()}


def _identity_checkpoint_status(default_torch_home: Path) -> dict:
    # Unlike the other models, InceptionResnetV1's pretrained weights are cached under a
    # TORCH_HOME/checkpoints/ directory (a facenet-pytorch/torch.hub convention) rather than
    # a single fixed file path -- see app/pipelines/identity/model.py's weights_present(),
    # which this mirrors without importing torch (readiness checks stay import-light).
    torch_home = Path(os.getenv("TORCH_HOME") or default_torch_home)
    checkpoints_dir = torch_home / "checkpoints"
    present = checkpoints_dir.is_dir() and any(checkpoints_dir.glob("*.pt"))
    return {"path": str(checkpoints_dir), "present": present}


def _default_checkpoints() -> dict[str, dict]:
    repo = BACKEND.parent
    return {
        "image_convnext": _checkpoint_status(
            "IMAGE_MODEL_CHECKPOINT",
            repo / "models" / "checkpoints" / "image" / "convnext_tiny_diversified_v2.pth",
        ),
        "image_clip": _checkpoint_status(
            "CLIP_MODEL_CHECKPOINT",
            repo / "models" / "checkpoints" / "image_clip" / "clip_head_round3_portrait_app.pth",
        ),
        "video_model_v1": _checkpoint_status(
            "VIDEO_MODEL_V1_CHECKPOINT",
            BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt",
        ),
        "audio_model_v1": _checkpoint_status(
            "AUDIO_MODEL_V1_CHECKPOINT",
            BACKEND / "checkpoints" / "audio" / "wav2vec2_v9.pt",
        ),
        "identity_v1": _identity_checkpoint_status(
            BACKEND / "checkpoints" / "identity" / "torch_cache",
        ),
    }


async def check_readiness(db: AsyncSession) -> dict:
    db_ok = True
    db_error: str | None = None
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001 - any DB failure means "not ready", report what it was
        db_ok = False
        db_error = f"{type(e).__name__}: {e}"

    checkpoints = _default_checkpoints()
    ready = db_ok and all(c["present"] for c in checkpoints.values())

    out = {"ready": ready, "database": {"ok": db_ok}, "checkpoints": checkpoints}
    if db_error:
        out["database"]["error"] = db_error
    return out
