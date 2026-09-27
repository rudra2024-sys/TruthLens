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
from app.services.feedback_export import export_feedback  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="feedback_export")
    args = ap.parse_args()
    async with AsyncSessionLocal() as db:
        print(json.dumps(await export_feedback(db, args.out), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
