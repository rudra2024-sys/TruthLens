"""Turn an identity-match predictions CSV (from run_identity_predictions.py) into a calibration
report: the same/different-person similarity distributions, ROC/AUC, and the threshold that
best separates them -- the real, measured basis for IDENTITY_MATCH_THRESHOLD in config.py.

  python -m eval.make_identity_report --pred eval/data/pred_identity.csv --out eval/results/identity

Writes <out>/report.md and <out>/similarity_distributions.png. Nothing is re-run through the
model -- everything here is computed from the stored similarity scores.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import metrics as M


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("predictions file is empty")
    return rows


def best_accuracy_threshold(same_scores: np.ndarray, diff_scores: np.ndarray):
    """Scans every distinct observed similarity value as a candidate threshold (match if
    similarity >= threshold) and returns the one maximizing overall accuracy, plus its
    accuracy/false-accept-rate/false-reject-rate at that point."""
    candidates = np.unique(np.concatenate([same_scores, diff_scores]))
    best = None
    for t in candidates:
        true_accepts = int((same_scores >= t).sum())
        false_rejects = len(same_scores) - true_accepts
        false_accepts = int((diff_scores >= t).sum())
        true_rejects = len(diff_scores) - false_accepts
        accuracy = (true_accepts + true_rejects) / (len(same_scores) + len(diff_scores))
        if best is None or accuracy > best["accuracy"]:
            best = {
                "threshold": float(t), "accuracy": accuracy,
                "false_accept_rate": false_accepts / len(diff_scores),
                "false_reject_rate": false_rejects / len(same_scores),
            }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.pred)
    y = np.array([int(r["same_person"]) for r in rows])          # 1 = same person (match)
    s = np.array([float(r["similarity"]) for r in rows])
    same_scores, diff_scores = s[y == 1], s[y == 0]

    auc = M.roc_auc(y, s)
    best = best_accuracy_threshold(same_scores, diff_scores)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(min(s.min(), 0.0), max(s.max(), 1.0), 40)
    ax.hist(diff_scores, bins=bins, alpha=0.6, label=f"different person (n={len(diff_scores)})", color="#d62728")
    ax.hist(same_scores, bins=bins, alpha=0.6, label=f"same person (n={len(same_scores)})", color="#2ca02c")
    ax.axvline(best["threshold"], color="black", linestyle="--", linewidth=1,
               label=f"best-accuracy threshold = {best['threshold']:.3f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title("Identity-match similarity: same vs. different person (LFW DevTest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "similarity_distributions.png", dpi=130)
    plt.close(fig)

    report = f"""# Identity-match threshold calibration

Source: `jessicali9530/lfw-dataset` on Kaggle (LFW DevTest split, 500 same-person +
500 different-person labeled pairs). Scored with the real, deployed `IdentityPipeline.embed()`
and `detector.py::_cosine_similarity` (see `eval/run_identity_predictions.py`) -- not a
re-implementation.

- Pairs scored: {len(rows)} ({len(same_scores)} same-person, {len(diff_scores)} different-person)
- Same-person similarity: mean {same_scores.mean():.3f}, std {same_scores.std():.3f}, min {same_scores.min():.3f}, max {same_scores.max():.3f}
- Different-person similarity: mean {diff_scores.mean():.3f}, std {diff_scores.std():.3f}, min {diff_scores.min():.3f}, max {diff_scores.max():.3f}
- ROC AUC: {auc:.4f}

## Best-accuracy threshold

| Threshold | Accuracy | False-accept rate | False-reject rate |
|---|---|---|---|
| {best['threshold']:.4f} | {best['accuracy']:.4f} | {best['false_accept_rate']:.4f} | {best['false_reject_rate']:.4f} |

False-accept rate = fraction of different-person pairs scored as a match at this threshold.
False-reject rate = fraction of same-person pairs scored as NOT a match at this threshold.

## Caveats

- LFW is frontal, studio/press-photo-quality, mostly well-lit faces of public figures -- not
  webcam footage. A webcam enrollment/monitoring photo (worse lighting, off-angle, compression)
  will plausibly show more overlap between the two distributions than this measurement does, so
  treat this threshold as a real, data-grounded starting point, not a guaranteed production
  false-accept/false-reject rate.
- The monitoring-session thresholds (`LOOKING_AWAY_*`, `TALKING_SPEECH_RATIO_THRESHOLD`,
  `OBJECT_DETECTION_CONFIDENCE_THRESHOLD`) and item 9's `LIVENESS_MOTION_EPSILON_FRACTION` are
  NOT calibrated by this study -- no comparable labeled dataset was found for any of them, and
  they remain stated placeholders in `config.py`.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_dir / 'report.md'} and {out_dir / 'similarity_distributions.png'}")


if __name__ == "__main__":
    main()
