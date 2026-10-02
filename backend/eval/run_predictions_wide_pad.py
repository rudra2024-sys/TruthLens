"""Re-scores an existing video manifest with the EXPERIMENTAL wide-face-padding backend
(video/model_v1/wide_pad_aware.py, VIDEO_MODEL_BACKEND=model_v1_wide_pad), for a fair, apples-to-apples
comparison against the deployed Video Model v1's own predictions on the same files. Mirrors
run_predictions_rotation_aware.py / run_predictions_consensus.py exactly, just swapping in
wide_pad_aware.predict. Output is in the same CSV shape so the existing, unmodified make_report.py can score
it directly.

  python -m eval.run_predictions_wide_pad --manifest eval/data/manifest_video_rtfs.csv \
      --out eval/data/pred_video_rtfs_wide_pad.csv

Same resumable-append pattern as run_predictions.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

DEFAULT_VIDEO = BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt"


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_paths(out_path):
    if not Path(out_path).exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {r["path"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-ckpt", default=str(DEFAULT_VIDEO))
    args = ap.parse_args()

    from app.services.video.model_v1 import wide_pad_aware

    rows = read_manifest(args.manifest)
    skip = done_paths(args.out)
    todo = [r for r in rows if r["path"] not in skip]
    print(f"{len(rows)} in manifest, {len(skip)} already done, {len(todo)} to run")

    fields = ["path", "label", "source", "trained_on", "video_fake", "video_logit",
              "frame_logit_min", "frame_logit_max"]

    def predict(path):
        r = wide_pad_aware.predict(path, checkpoint_path=args.video_ckpt)
        return {"video_fake": r.probability, "video_logit": r.logit,
                "frame_logit_min": min(r.frame_logits), "frame_logit_max": max(r.frame_logits)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    failed = 0
    t0 = time.time()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            try:
                scores = predict(r["path"])
            except Exception as e:                      # keep going; report at the end
                failed += 1
                print(f"  FAILED {r['path']}: {type(e).__name__}: {e}")
                continue
            w.writerow({**{k: r[k] for k in ("path", "label", "source", "trained_on")}, **scores})
            f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/i:.2f}s/file  eta {el/i*(len(todo)-i)/60:.1f} min")
    print(f"done -> {out}   (failed: {failed})")


if __name__ == "__main__":
    main()
