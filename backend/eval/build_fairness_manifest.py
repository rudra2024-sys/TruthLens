"""Builds a demographically-balanced manifest from the UTKFace dataset for the fairness measurement in
CLAUDE.md section 25 item 10. Every image is a genuine photo (label=0), so scoring it with
`eval.run_predictions --media image` and `eval.make_report` measures real-photo specificity per demographic
group (the `source` column, reused by make_report's existing per-source breakdown) -- not generation-side
fairness, which would need a demographic-labeled AI-generated set that wasn't found or attempted.

Dataset: jangedoo/utkface-new on Kaggle (23,708 images, CC-licensed, filename-encoded
`{age}_{gender}_{race}_{timestamp}.jpg.chip.jpg`; gender 0=male/1=female, race 0=white/1=black/2=asian/
3=indian/4=other -- the dataset's own documented convention, not invented here). Download it yourself (e.g.
via the Kaggle API/CLI or the mcp__kaggle-token__* tools) to a local path and pass that path in; this script
never downloads anything on its own.

  python -m eval.build_fairness_manifest --utkface-dir D:/eval_data/fairness/utkface/UTKFace \
      --out eval/data/manifest_fairness.csv --n-per-group 60 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from random import Random

RACE = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}
GENDER = {0: "male", 1: "female"}
_PATTERN = re.compile(r"^(\d+)_(\d)_(\d)_")


def build_manifest(utkface_dir: Path, n_per_group: int, seed: int) -> list[dict]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    skipped = 0
    for f in utkface_dir.iterdir():
        m = _PATTERN.match(f.name)
        if not m:
            skipped += 1
            continue
        gender, race = int(m.group(2)), int(m.group(3))
        if gender not in GENDER or race not in RACE:
            skipped += 1
            continue
        buckets[(GENDER[gender], race)].append(f.name)
    print(f"{sum(len(v) for v in buckets.values())} usable images, {skipped} unparseable filenames skipped")

    rows = []
    for key in sorted(buckets.keys(), key=lambda k: (k[0], k[1])):
        gender, race_code = key
        files = sorted(buckets[key])
        # A fresh, per-bucket-seeded RNG (not one shared Random advanced across buckets in loop order) so the
        # chosen sample is reproducible regardless of what order the buckets happen to be iterated in -- a
        # single shared Random instance was tried first and rejected once it was caught producing a different
        # sample than an earlier, differently-ordered version of this same logic despite the identical seed.
        Random(f"{seed}:{gender}:{race_code}").shuffle(files)
        chosen = files[:n_per_group]
        if len(chosen) < n_per_group:
            print(f"  warning: only {len(chosen)}/{n_per_group} available for {gender}/{RACE[race_code]}")
        for f in chosen:
            rows.append({
                "path": str(utkface_dir / f),
                "label": 0,
                "source": f"utkface_{gender}_{RACE[race_code]}",
                "trained_on": "no",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utkface-dir", required=True, type=Path, help="path to the extracted UTKFace/ folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-group", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_manifest(args.utkface_dir, args.n_per_group, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "trained_on"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
