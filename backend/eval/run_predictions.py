"""Run the deployed detectors over a manifest and save raw per-file scores.

This is the slow step, kept separate from make_report.py so metrics/plots can be
re-generated instantly. Re-running resumes: files already in the output CSV are skipped.

  python -m eval.run_predictions --media image --manifest eval/data/manifest_image.csv \
      --out eval/data/pred_image.csv
  python -m eval.run_predictions --media video --manifest eval/data/manifest_video.csv \
      --out eval/data/pred_video.csv

Checkpoints default to the ones documented in CLAUDE.md section 6; override with
--convnext-ckpt / --clip-ckpt / --video-ckpt (e.g. to evaluate a rollback checkpoint).
Devices default to CPU. The predictors are the exact production classes
(ImagePipeline, ClipPipeline, video model_v1 optimized.predict), not re-implementations.
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

os.environ.setdefault("IMAGE_MODEL_DEVICE", "cpu")
os.environ.setdefault("CLIP_MODEL_DEVICE", "cpu")
os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

DEFAULT_CONVNEXT = REPO / "models" / "checkpoints" / "image" / "convnext_tiny_diversified_v2.pth"
DEFAULT_CLIP = REPO / "models" / "checkpoints" / "image_clip" / "clip_head_round3_portrait_app.pth"
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
    ap.add_argument("--media", choices=["image", "video"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--convnext-ckpt", default=str(DEFAULT_CONVNEXT))
    ap.add_argument("--clip-ckpt", default=str(DEFAULT_CLIP))
    ap.add_argument("--video-ckpt", default=str(DEFAULT_VIDEO))
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    skip = done_paths(args.out)
    todo = [r for r in rows if r["path"] not in skip]
    print(f"{len(rows)} in manifest, {len(skip)} already done, {len(todo)} to run")

    if args.media == "image":
        from app.pipelines.image.inference import ImagePipeline
        from app.pipelines.image_clip.inference import ClipPipeline
        convnext = ImagePipeline(checkpoint_path=args.convnext_ckpt)
        clip = ClipPipeline(checkpoint_path=args.clip_ckpt)
        fields = ["path", "label", "source", "trained_on", "convnext_fake", "clip_fake"]

        def predict(path):
            return {"convnext_fake": convnext.predict(path)["fake_probability"],
                    "clip_fake": clip.predict(path)["fake_probability"]}
    else:
        from app.services.video.model_v1 import optimized
        fields = ["path", "label", "source", "trained_on", "video_fake", "video_logit",
                  "frame_logit_min", "frame_logit_max"]

        def predict(path):
            r = optimized.predict(path, checkpoint_path=args.video_ckpt)
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
