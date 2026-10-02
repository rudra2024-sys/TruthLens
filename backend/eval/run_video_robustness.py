"""Re-tests the DEPLOYED Video Model v1 on per-frame-degraded copies of a balanced video subset, mirroring the
image robustness suite's methodology (eval/perturbations.py, same families/levels) but for video, which that
suite never covered (CLAUDE.md section 14 is images only). Read-only measurement: changes no model, weight or
threshold, and does not write any video file to disk -- each of the 16 sampled frames is degraded in memory
(as a PIL image, via the exact same eval.perturbations.apply() used for images) before the unmodified
common.preprocess_frame() Haar-crop-resize-normalize pipeline and the unmodified production model.

Scope kept deliberately smaller than the image suite (a subset of videos x a subset of settings, not the full
21): each (video, setting) pair costs a full 16-frame decode + Haar + EfficientNet forward pass (~2-3s), so the
full cross product would take much longer than the image suite's per-image cost allows for a background run.

  python -m eval.run_video_robustness --out eval/data/video_robust_pred.csv

Resumable (same append-and-skip-done pattern as run_robustness.py).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

from eval import perturbations as P  # noqa: E402

DEFAULT_VIDEO = BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt"
MANIFEST = BACKEND / "eval" / "data" / "manifest_video_rtfs.csv"
FIELDS = ["path", "label", "source", "family", "level", "video_fake"]

# A reduced settings list (not the full 21): one representative level per family plus the two sharing
# pipelines most relevant to real-world video (whatsapp-style re-compression, webp), to keep the (videos x
# settings) product small enough for a single background run. "clean" is always included.
SETTINGS = [
    ("clean", 0),
    ("jpeg", 90), ("jpeg", 50), ("jpeg", 10),
    ("resize", 0.5), ("resize", 0.25),
    ("blur", 1), ("blur", 3),
    ("noise", 10),
    ("social", "whatsapp"), ("social", "webp"),
]


def pick_subset(manifest_path, n_per_source: int):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = []
    for src, items in by_source.items():
        items = sorted(items, key=lambda r: r["path"])  # deterministic
        out += items[:n_per_source]
    return out


def done_keys(out_path):
    if not Path(out_path).exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {(r["path"], r["family"], r["level"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-ckpt", default=str(DEFAULT_VIDEO))
    ap.add_argument("--n-per-source", type=int, default=10,
                     help="videos sampled per manifest `source` (deterministic, sorted by path)")
    args = ap.parse_args()

    from app.services.video.model_v1 import common, optimized

    subset = pick_subset(args.manifest, args.n_per_source)
    todo_settings = SETTINGS
    skip = done_keys(args.out)
    print(f"{len(subset)} videos x {len(todo_settings)} settings = {len(subset) * len(todo_settings)} rows; "
          f"{len(skip)} already done")

    model, device, _ = optimized._get_model_and_device(args.video_ckpt)
    cascade = common.get_face_cascade()

    import numpy as np
    import torch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    failed = 0
    t0 = time.time()
    total = len(subset) * len(todo_settings)
    done_count = len(skip)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in subset:
            path = r["path"]
            try:
                frames_bgr = common.decode_selected_frames(path, use_grab_skip=True)
            except Exception as e:
                failed += 1
                print(f"  DECODE FAILED {path}: {type(e).__name__}: {e}")
                continue
            frames_rgb_pil = [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in frames_bgr]
            key = Path(path).name
            for family, level in todo_settings:
                if (path, family, str(level)) in skip:
                    continue
                degraded_bgr = []
                for pil_frame in frames_rgb_pil:
                    deg = P.apply(family, level, pil_frame, key=key)
                    degraded_bgr.append(cv2.cvtColor(np.asarray(deg), cv2.COLOR_RGB2BGR))
                processed = [common.preprocess_frame(fr, cascade) for fr in degraded_bgr]
                batch = np.stack(processed, axis=0)
                tensor = torch.from_numpy(batch).contiguous().float().to(device)
                with torch.inference_mode():
                    logits = model(tensor).squeeze(-1)
                    probability = float(torch.sigmoid(logits.mean()).item())
                w.writerow({"path": path, "label": r["label"], "source": r["source"],
                            "family": family, "level": level, "video_fake": probability})
                f.flush()
                done_count += 1
                if done_count % 20 == 0 or done_count == total:
                    el = time.time() - t0
                    rate = el / max(1, done_count - len(skip))
                    remaining = total - done_count
                    print(f"  {done_count}/{total}  {rate:.2f}s/row  eta {rate * remaining / 60:.1f} min")
    print(f"done -> {out}   (decode failures: {failed})")


if __name__ == "__main__":
    main()
