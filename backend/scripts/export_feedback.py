"""Export user feedback for evaluation (consented files only).

  cd backend && ../.venv/Scripts/python.exe scripts/export_feedback.py --out feedback_export

Reads the database configured by DATABASE_URL (the same one the app uses). Writes feedback_summary.csv (anonymous, no file
paths), manifest_feedback.csv + files/ (ONLY feedback whose author ticked "allow reuse") and prints agreement statistics.
Score the deployed models on the flagged cases with:
  ../.venv/Scripts/python.exe -m eval.run_predictions --media image --manifest feedback_export/manifest_feedback.csv --out eval/data/pred_feedback.csv
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import AsyncSessionLocal          # noqa: E402
from app.services.audit_log import log_event              # noqa: E402
from app.services.feedback_export import export_feedback  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="feedback_export")
    args = ap.parse_args()
    async with AsyncSessionLocal() as db:
        result = await export_feedback(db, args.out)
        # No single "user" to attribute this to (an operator ran it, not an end user) - detail carries what
        # was actually exported so the log says more than just "someone ran this at some point".
        await log_event(db, "feedback_exported",
                        detail=f"exported_files={result.get('exported_files')} out_dir={result.get('out_dir')}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
