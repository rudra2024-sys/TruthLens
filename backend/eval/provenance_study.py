"""How much does provenance add on top of the detection models?  (and is ELA any good?)

For every file in the given predictions CSVs (from run_predictions.py) this reads C2PA Content Credentials and
embedded metadata, then reports per source:
  * how many carry an explicit AI declaration (Content Credentials / XMP / embedded generation parameters),
  * how many carry camera-style EXIF, and how many carry nothing at all,
and, for AI (label=1) images the deployed ensemble MISSES (max(ConvNeXt, CLIP) < 0.5), how many are nevertheless
caught by provenance. It also measures whether ELA summary statistics separate real from fake within JPEG sources.

  python -m eval.provenance_study --pred eval/data/pred_image.csv eval/data/pred_b1.csv ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.provenance.c2pa_reader import read_c2pa            # noqa: E402
from app.services.provenance.ela import error_level_analysis         # noqa: E402
from app.services.provenance.metadata import read_metadata           # noqa: E402
from eval import metrics as M                                        # noqa: E402


def classify(path: str) -> dict:
    c = read_c2pa(path)
    m = read_metadata(path)
    declared = bool(c.present and c.ai_declared) or m.xmp_declares_ai or bool(m.ai_param_fields)
    return {
        "c2pa": c.present, "c2pa_ai": bool(c.present and c.ai_declared), "meta_ai": m.xmp_declares_ai or bool(m.ai_param_fields),
        "declared_ai": declared, "camera": m.camera_capture_metadata, "any_exif": m.has_exif,
        "nothing": not (c.present or m.has_exif or m.ai_param_fields or m.xmp_declares_ai),
        "fmt": m.format,
    }


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", default=str(BACKEND / "eval" / "results" / "provenance_study.md"))
    args = ap.parse_args()

    rows = []
    for p in args.pred:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    print(f"{len(rows)} files")

    per_src = defaultdict(lambda: defaultdict(int))
    missed_total = missed_caught = 0
    fake_total = fake_declared = 0
    ela_by_src = defaultdict(lambda: {0: [], 1: []})
    for i, r in enumerate(rows):
        label = int(r["label"])
        c = classify(r["path"])
        d = per_src[r["source"]]
        d["n"] += 1
        d[f"label{label}"] += 1
        for k in ("c2pa", "declared_ai", "camera", "any_exif", "nothing"):
            d[k] += bool(c[k])
        if label == 1:
            fake_total += 1
            fake_declared += bool(c["declared_ai"])
            if max(float(r["convnext_fake"]), float(r["clip_fake"])) < 0.5:
                missed_total += 1
                missed_caught += bool(c["declared_ai"])
                d["missed"] += 1
                d["missed_caught"] += bool(c["declared_ai"])
        if c["fmt"] == "JPEG" and i % 1 == 0:
            e = error_level_analysis(r["path"])
            if e.applicable:
                ela_by_src[r["source"]][label].append((e.mean_error, e.p95_error, e.block_cv if e.block_cv is not None else np.nan))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(rows)}")

    L = ["# Provenance study\n",
         f"Files: {len(rows)} (from `{'`, `'.join(Path(p).name for p in args.pred)}`). AI = label 1.\n",
         "## Metadata / Content Credentials per source\n",
         "| source | files | AI declared | C2PA present | camera EXIF | any EXIF | nothing at all |",
         "|---|---|---|---|---|---|---|"]
    for s in sorted(per_src):
        d = per_src[s]
        L.append(f"| {s} | {d['n']} | {pct(d['declared_ai'], d['n'])} | {pct(d['c2pa'], d['n'])} | "
                 f"{pct(d['camera'], d['n'])} | {pct(d['any_exif'], d['n'])} | {pct(d['nothing'], d['n'])} |")
    L += ["", "## Does provenance add to the models?\n",
          f"- AI images overall: **{fake_declared} of {fake_total}** ({pct(fake_declared, fake_total)}) carry an explicit AI declaration.",
          f"- AI images the deployed ensemble **misses** (scored < 0.5): {missed_total}; of these, provenance declares "
          f"**{missed_caught}** as AI ({pct(missed_caught, missed_total)}).\n",
          "| source | AI images missed by models | ...declared AI by provenance |", "|---|---|---|"]
    for s in sorted(per_src):
        d = per_src[s]
        if d["missed"]:
            L.append(f"| {s} | {d['missed']} | {d['missed_caught']} ({pct(d['missed_caught'], d['missed'])}) |")

    L += ["", "## Is ELA a usable detector? (JPEG sources with both classes)\n",
          "AUC of each ELA summary statistic for FAKE vs REAL *within the same source* (0.5 = no information; the "
          "statistic direction is whichever gives AUC >= 0.5, so values below 0.5 cannot occur).\n",
          "| source | real | fake | mean error | p95 error | block CV |", "|---|---|---|---|---|---|"]
    any_ela = False
    for s in sorted(ela_by_src):
        r0, r1 = ela_by_src[s][0], ela_by_src[s][1]
        if len(r0) < 30 or len(r1) < 30:
            continue
        any_ela = True
        cells = []
        for j in range(3):
            y = np.array([0] * len(r0) + [1] * len(r1))
            v = np.array([t[j] for t in r0] + [t[j] for t in r1], dtype=float)
            ok = ~np.isnan(v)
            a = M.roc_auc(y[ok], v[ok])
            cells.append(f"{max(a, 1 - a):.3f}")
        L.append(f"| {s} | {len(r0)} | {len(r1)} | " + " | ".join(cells) + " |")
    if not any_ela:
        L.append("| (no JPEG source with >= 30 files of both classes) | | | | | |")
    L += ["", "Interpretation: values near 0.5 mean ELA carries little class information for that source, so it is "
          "shown in the product only as a labelled visual aid and never used as a score."]
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
