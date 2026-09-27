"""Turn user feedback into evaluation material - respecting consent.

Only feedback with allow_reuse=True ever has its file copied or its comment exported. Everything else contributes
anonymous counts only. A label is derived only when it is unambiguous:
  * user disagreed and said "real" / "ai"     -> that label
  * user agreed and the shown verdict was FAKE -> ai; REAL -> real  (an agreed UNCERTAIN carries no label)
  * "unsure" answers carry no label.
The output manifest has the same columns as eval/build_manifest.py, so `python -m eval.run_predictions` can score the
deployed models on the cases users flagged. This is for EVALUATION; using it to train needs explicit approval (CLAUDE.md).
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Feedback, Upload


def derived_label(fb: Feedback) -> int | None:
    """1 = AI-generated, 0 = real, None = no reliable label."""
    if fb.agrees:
        return {"FAKE": 1, "REAL": 0}.get(fb.verdict)
    return {"ai": 1, "real": 0}.get(fb.true_label or "")


async def export_feedback(db: AsyncSession, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    (out / "files").mkdir(parents=True, exist_ok=True)
    rows = (await db.execute(select(Feedback, Upload).join(Upload, Upload.upload_id == Feedback.upload_id))).all()

    summary_rows, manifest_rows = [], []
    counts = Counter()
    confusion = Counter()                                   # (shown verdict, user's label) among unambiguous cases
    for fb, up in rows:
        label = derived_label(fb)
        counts["total"] += 1
        counts["agree" if fb.agrees else "disagree"] += 1
        counts[f"media_{up.media_type}"] += 1
        if label is not None:
            confusion[(fb.verdict, "ai" if label else "real")] += 1
        summary_rows.append({
            "feedback_id": fb.feedback_id, "media_type": up.media_type, "verdict_shown": fb.verdict,
            "agrees": int(fb.agrees), "user_label": fb.true_label or "", "derived_label": "" if label is None else label,
            "allow_reuse": int(fb.allow_reuse), "created_at": fb.created_at,
            "comment": (fb.comment or "") if fb.allow_reuse else "",         # comments only with consent
        })
        if fb.allow_reuse and label is not None and Path(up.storage_url).is_file():
            dest = out / "files" / f"{up.upload_id}{Path(up.storage_url).suffix}"
            shutil.copyfile(up.storage_url, dest)
            manifest_rows.append({"path": str(dest.resolve()), "label": label, "source": "user_feedback",
                                  "trained_on": "no", "media_type": up.media_type})
            counts["exported_files"] += 1

    with open(out / "feedback_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feedback_id", "media_type", "verdict_shown", "agrees", "user_label",
                                          "derived_label", "allow_reuse", "created_at", "comment"])
        w.writeheader()
        w.writerows(summary_rows)
    with open(out / "manifest_feedback.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "trained_on", "media_type"])
        w.writeheader()
        w.writerows(manifest_rows)

    wrong = sum(v for (shown, truth), v in confusion.items()
                if (shown == "FAKE" and truth == "real") or (shown == "REAL" and truth == "ai"))
    decided = sum(v for (shown, _), v in confusion.items() if shown in ("FAKE", "REAL"))
    return {
        "feedback_total": counts["total"], "agree": counts["agree"], "disagree": counts["disagree"],
        "agreement_rate": (counts["agree"] / counts["total"]) if counts["total"] else None,
        "exported_files": counts["exported_files"],
        "confident_verdicts_judged_wrong": wrong, "confident_verdicts_with_a_label": decided,
        "by_media": {k[6:]: v for k, v in counts.items() if k.startswith("media_")},
        "confusion": {f"{a}->{b}": v for (a, b), v in sorted(confusion.items())},
        "out_dir": str(out.resolve()),
    }
