"""Re-test the deployed image models on degraded copies of the robustness subset.

  python -m eval.run_robustness            # resumes if interrupted
  python -m eval.make_robustness_report    # tables + charts from the saved scores

For every image in eval/data/manifest_robust.csv and every (family, level) in eval/perturbations.py the image is degraded
(deterministically), written to a temporary lossless PNG and scored by the same ImagePipeline / ClipPipeline classes the
API uses. One row per (image, setting) is appended to eval/data/robust_pred.csv, flushed after each image.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("IMAGE_MODEL_DEVICE", "cpu")
os.environ.setdefault("CLIP_MODEL_DEVICE", "cpu")

from eval import perturbations as P  # noqa: E402

DEFAULT_CONVNEXT = REPO / "models" / "checkpoints" / "image" / "convnext_tiny_diversified_v2.pth"
DEFAULT_CLIP = REPO / "models" / "checkpoints" / "image_clip" / "clip_head_round3_portrait_app.pth"
FIELDS = ["path", "label", "group", "status", "source", "family", "level", "convnext_fake", "clip_fake"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(BACKEND / "eval" / "data" / "manifest_robust.csv"))
    ap.add_argument("--out", default=str(BACKEND / "eval" / "data" / "robust_pred.csv"))
    ap.add_argument("--convnext-ckpt", default=str(DEFAULT_CONVNEXT))
    ap.add_argument("--clip-ckpt", default=str(DEFAULT_CLIP))
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            done = {(r["path"], r["family"], r["level"]) for r in csv.DictReader(f)}
    todo_settings = P.settings()
    print(f"{len(rows)} images x {len(todo_settings)} settings; {len(done)} rows already done")

    from app.pipelines.image.inference import ImagePipeline
    from app.pipelines.image_clip.inference import ClipPipeline

    convnext = ImagePipeline(checkpoint_path=args.convnext_ckpt)
    clip = ClipPipeline(checkpoint_path=args.clip_ckpt)

    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    tmp = Path(tempfile.mkdtemp(prefix="robust_"))
    t0 = time.time()
    written = errors = 0
    try:
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for i, r in enumerate(rows, 1):
                pending = [(fam, lvl) for fam, lvl in todo_settings if (r["path"], fam, str(lvl)) not in done]
                if not pending:
                    continue
                try:
                    with Image.open(r["path"]) as im:
                        src = im.convert("RGB")
                except Exception as e:
                    errors += 1
                    print(f"  cannot read {r['path']}: {e}")
                    continue
                key = Path(r["path"]).name
                for fam, lvl in pending:
                    try:
                        degraded = P.apply(fam, lvl, src, key)
                        p = tmp / "x.png"
                        degraded.save(p, format="PNG", compress_level=1)
                        row = {k: r[k] for k in ("path", "label", "group", "status", "source")}
                        row.update(family=fam, level=lvl, convnext_fake=convnext.predict(str(p))["fake_probability"],
                                   clip_fake=clip.predict(str(p))["fake_probability"])
                        w.writerow(row)
                        written += 1
                    except Exception as e:
                        errors += 1
                        print(f"  FAILED {key} {fam}={lvl}: {type(e).__name__}: {e}")
                f.flush()
                if i % 5 == 0 or i == len(rows):
                    el = time.time() - t0
                    left = len(rows) - i
                    print(f"  {i}/{len(rows)} images  {written} rows  {el/60:.1f} min elapsed  ~{el / max(i, 1) * left / 60:.1f} min left", flush=True)
    finally:
        for p in tmp.glob("*"):
            p.unlink(missing_ok=True)
        tmp.rmdir()
    print(f"done -> {out}  (rows written this run: {written}, errors: {errors})")


if __name__ == "__main__":
    main()
