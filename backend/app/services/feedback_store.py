"""Keep a retraining-ready copy of files users said the model got wrong.

When a user disagrees with a result, says what the file really is ("real" or "ai") and has ticked "allow reuse",
the file is copied to  <FEEDBACK_WRONG_DIR>/<real|ai>/<media_type>/<upload_id>.<ext>  with a small
<upload_id>.json beside it (what the model said, what the user said, their comment). Anything else - agreeing,
"unsure", no consent - keeps nothing. The copy follows the feedback: changing or withdrawing it, or deleting the
account, removes the copy again. Nothing here trains or changes a model; using these files to retrain needs explicit
approval (CLAUDE.md section 8 rule 1), and the labels are the user's word, so review them before use.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from app.core.config import settings
from app.models.models import Feedback, Upload

logger = logging.getLogger(__name__)

_LABELS = ("real", "ai")


def remove_wrong_feedback(upload_id: str) -> int:
    """Deletes every kept copy (any label folder) for this upload. Returns how many files were removed."""
    removed = 0
    root = Path(settings.FEEDBACK_WRONG_DIR)
    for label in _LABELS:
        for media_dir in (root / label).glob("*"):
            for f in media_dir.glob(f"{upload_id}.*"):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    logger.exception("Could not remove kept feedback file: %s", f)
    return removed


def sync_wrong_feedback(fb: Feedback, upload: Upload) -> bool:
    """Makes the kept copy match the feedback as it stands now. Returns True if a copy is kept. Never raises: a
    problem here must not stop the user's feedback from being saved."""
    try:
        remove_wrong_feedback(upload.upload_id)
        if fb.agrees or not fb.allow_reuse or fb.true_label not in _LABELS:
            return False
        src = Path(upload.storage_url)
        if not src.is_file():
            return False
        dest_dir = Path(settings.FEEDBACK_WRONG_DIR) / fb.true_label / upload.media_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest_dir / f"{upload.upload_id}{src.suffix}")
        (dest_dir / f"{upload.upload_id}.json").write_text(json.dumps({
            "upload_id": upload.upload_id, "media_type": upload.media_type, "verdict_shown": fb.verdict,
            "user_says_it_is": fb.true_label, "comment": fb.comment, "saved_at": fb.updated_at,
        }, indent=2), encoding="utf-8")
        return True
    except Exception:
        logger.exception("Could not keep the wrong-feedback copy for upload %s", upload.upload_id)
        return False
