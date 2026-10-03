"""Re-tests the DEPLOYED Audio Model v1 on degraded copies of a labeled audio manifest, mirroring
run_video_robustness.py's approach (perturb in memory, score with the unmodified production pipeline, no
model/weight/threshold change).

**This script has NOT been run against real labeled audio** -- this project has no locally-available labeled
audio deepfake dataset (checked 2026-10-03; only image/video datasets exist locally). Running it for real
needs a manifest CSV (path,label -- label 1=deepfake/spoof, 0=genuine) pointing at a real dataset such as
ASVspoof2019-LA or In-The-Wild, which would need to be sourced first (Kaggle download or user-supplied files)
-- not attempted here without that direction, same reasoning as the module docstring in
eval/perturbations_audio.py.

What IS verified (see tests/test_audio_robustness_eval.py): the harness itself runs end-to-end without
crashing against synthetic placeholder audio (sine tones), producing sane, non-degenerate scores under each
perturbation. That confirms the mechanism works; it says nothing about the deployed model's real accuracy or
robustness, which can only be measured against real genuine/spoof audio.

  python -m eval.run_audio_robustness --manifest <path,label CSV> --out eval/data/audio_robust_pred.csv
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
os.environ.setdefault("AUDIO_MODEL_V1_DEVICE", "cpu")

from eval import perturbations_audio as PA  # noqa: E402

FIELDS = ["path", "label", "family", "level", "audio_fake"]


def done_keys(out_path):
    if not Path(out_path).exists():
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {(r["path"], r["family"], r["level"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns: path,label (1=spoof, 0=genuine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    from app.services.audio.model_v1 import common, optimized

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    todo_settings = PA.settings()
    skip = done_keys(args.out)
    print(f"{len(rows)} audio files x {len(todo_settings)} settings = {len(rows) * len(todo_settings)} rows; "
          f"{len(skip)} already done")

    import numpy as np
    import torch
    import torch.nn.functional as F

    model, device, _ = optimized._get_model_and_device(args.checkpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    failed = 0
    t0 = time.time()
    total = len(rows) * len(todo_settings)
    done_count = len(skip)
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            path = r["path"]
            try:
                from app.services.models.audio_model import decode_audio_mono_16k
                raw = decode_audio_mono_16k(path)
                normalized = common.rms_normalize(raw)
            except Exception as e:
                failed += 1
                print(f"  DECODE FAILED {path}: {type(e).__name__}: {e}")
                continue
            key = Path(path).name
            for family, level in todo_settings:
                if (path, family, str(level)) in skip:
                    continue
                degraded = PA.apply(family, level, normalized, common.TARGET_SR, key=key)
                windows = optimized._build_windows(degraded)
                tensor = torch.from_numpy(windows).contiguous().float().to(device)
                with torch.inference_mode():
                    logits = model(tensor)
                    prob = float(F.softmax(logits, dim=-1)[:, 1].mean().item())
                w.writerow({"path": path, "label": r["label"], "family": family, "level": level,
                            "audio_fake": prob})
                f.flush()
                done_count += 1
                if done_count % 20 == 0 or done_count == total:
                    el = time.time() - t0
                    rate = el / max(1, done_count - len(skip))
                    print(f"  {done_count}/{total}  {rate:.2f}s/row  eta {rate * (total - done_count) / 60:.1f} min")
    print(f"done -> {out}   (decode failures: {failed})")


if __name__ == "__main__":
    main()
