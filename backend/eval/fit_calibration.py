"""Fits a post-hoc temperature-scaling calibrator for each deployed detector's confidence score, using the
eval scores already sitting in eval/data/ from prior measurement runs. Read-only w.r.t. the model: this never
touches weights or thresholds, it only produces a single scalar (per media type) that rescales the existing
probability before it's shown as "confidence" -- see app/services/calibration.py for where that scalar is
applied, and CLAUDE.md section 25 for why this is additive (a separate `calibrated_confidence` field) rather
than silently replacing the existing `confidence_score`.

Temperature scaling (Guo et al. 2017): calibrated_p = sigmoid(logit(raw_p) / T), T fit by minimizing negative
log-likelihood on held-out (label, raw_p) pairs via a 1D scalar search. T > 1 softens over-confident scores
(pulls them toward 0.5); T < 1 sharpens under-confident ones. This script is dev-only tooling (uses scipy,
already available transitively via scikit-learn in requirements-dev.txt) -- nothing in the production
inference path needs scipy, see calibration.py's plain log/exp math.

  python -m eval.fit_calibration
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from eval import metrics as M  # noqa: E402

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    z = _logit(p)

    def nll(T: float) -> float:
        if T <= 0:
            return np.inf
        calibrated = np.clip(_sigmoid(z / T), EPS, 1 - EPS)
        return -float(np.mean(y * np.log(calibrated) + (1 - y) * np.log(1 - calibrated)))

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(result.x)


def _load_image_scores() -> tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for name in ("pred_b1.csv", "pred_b2.csv", "pred_b3.csv", "pred_b4.csv"):
        path = BACKEND / "eval" / "data" / name
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ys.append(int(r["label"]))
                ps.append(max(float(r["convnext_fake"]), float(r["clip_fake"])))  # deployed max-ensemble
    return np.array(ys), np.array(ps)


def _load_video_scores() -> tuple[np.ndarray, np.ndarray]:
    path = BACKEND / "eval" / "data" / "pred_video_rtfs.csv"
    ys, ps = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ys.append(int(r["label"]))
            ps.append(float(r["video_fake"]))
    return np.array(ys), np.array(ps)


def main():
    out = {}
    for media, loader in (("image", _load_image_scores), ("video", _load_video_scores)):
        y, p = loader()
        if len(y) == 0:
            print(f"{media}: no data found, skipping")
            continue
        ece_before, *_ = M.expected_calibration_error(y, p)
        T = fit_temperature(y, p)
        calibrated = _sigmoid(_logit(p) / T)
        ece_after, *_ = M.expected_calibration_error(y, calibrated)
        out[media] = {"temperature": round(T, 4), "n": len(y),
                      "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4)}
        print(f"{media}: n={len(y)} T={T:.4f} ECE {ece_before:.4f} -> {ece_after:.4f}")

    out_path = BACKEND / "app" / "core" / "calibration.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
