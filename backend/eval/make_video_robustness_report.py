"""Turns eval/data/video_robust_pred.csv (from run_video_robustness.py) into a markdown table -- accuracy/AUC
per perturbation setting for the deployed Video Model v1, mirroring the image robustness report's purpose
(CLAUDE.md section 14) but scoped to video's reduced settings list. Nothing re-run through a model.

  python -m eval.make_video_robustness_report --pred eval/data/video_robust_pred.csv \
      --out eval/results/video_robustness/report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M

THRESHOLD = 0.525
BAND = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.pred, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_setting = defaultdict(list)
    for r in rows:
        by_setting[(r["family"], r["level"])].append(r)

    def order_key(k):
        fam, lvl = k
        order = ["clean", "jpeg", "resize", "blur", "noise", "social"]
        return (order.index(fam) if fam in order else 99, str(lvl))

    lines = [
        "# Video Model v1 -- robustness under per-frame degradation (reduced settings, read-only measurement)",
        "",
        "Not part of the deployed pipeline; see CLAUDE.md section 23. Each row perturbs all 16 sampled frames "
        "of each video identically (same family/level) before the unmodified Haar-crop + model pipeline.",
        "",
        "| family | level | n | accuracy | balanced_acc | fake_recall | real_spec | auc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(by_setting, key=order_key):
        fam, lvl = key
        group = by_setting[key]
        y = np.array([int(r["label"]) for r in group])
        s = np.array([float(r["video_fake"]) for r in group])
        summ = M.summary(y, s, THRESHOLD)
        auc = M.roc_auc(y, s) if M.has_both_classes(y) else float("nan")
        lines.append(
            f"| {fam} | {lvl} | {len(group)} | {summ['accuracy']*100:.1f}% | "
            f"{summ['balanced_accuracy']*100:.1f}% | {summ['fake_recall']*100:.1f}% | "
            f"{summ['real_specificity']*100:.1f}% | {auc:.3f} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for line in lines[6:]:
        print(line)


if __name__ == "__main__":
    main()
