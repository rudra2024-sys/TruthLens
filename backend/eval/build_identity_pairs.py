"""Builds a manifest of labeled same/different-person photo pairs from the LFW dataset, for
calibrating IDENTITY_MATCH_THRESHOLD (CLAUDE.md section 5/25-style data-driven threshold work;
see docs/identity calibration notes). Unlike the fairness manifest, which samples a dataset by
filename-encoded demographics, LFW already ships its own labeled verification pairs -- this just
reads them and resolves them to real file paths, no pair construction of our own needed.

Dataset: jessicali9530/lfw-dataset on Kaggle ("Labelled Faces in the Wild", 13,233 images,
79k+ downloads, MIT-adjacent "Other (specified in description)" license -- the standard LFW
redistribution terms). Download it yourself (e.g. via the Kaggle API/CLI or the
mcp__kaggle-token__* tools) and extract it to a local path; this script never downloads anything
on its own, same convention as build_fairness_manifest.py.

Uses the dataset's own matchpairsDevTest/mismatchpairsDevTest.csv (500 same-person + 500
different-person pairs, the official LFW "DevTest" split) -- real labels from the dataset's
creators, not pairs invented here.

  python -m eval.build_identity_pairs --lfw-dir D:/eval_data/identity/lfw \
      --out eval/data/manifest_identity_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _image_path(lfw_dir: Path, name: str, imagenum: str) -> Path:
    return lfw_dir / "lfw-deepfunneled" / "lfw-deepfunneled" / name / f"{name}_{int(imagenum):04d}.jpg"


def build_pairs(lfw_dir: Path) -> list[dict]:
    rows = []
    missing = 0

    with open(lfw_dir / "matchpairsDevTest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = _image_path(lfw_dir, row["name"], row["imagenum1"])
            b = _image_path(lfw_dir, row["name"], row["imagenum2"])
            if not (a.exists() and b.exists()):
                missing += 1
                continue
            rows.append({"image_a": str(a), "image_b": str(b), "same_person": 1})

    # mismatchpairs has two columns both literally named "name" in the header -- csv.DictReader
    # collapses duplicate keys, so this reads it positionally instead.
    with open(lfw_dir / "mismatchpairsDevTest.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for name1, imagenum1, name2, imagenum2 in reader:
            a = _image_path(lfw_dir, name1, imagenum1)
            b = _image_path(lfw_dir, name2, imagenum2)
            if not (a.exists() and b.exists()):
                missing += 1
                continue
            rows.append({"image_a": str(a), "image_b": str(b), "same_person": 0})

    print(f"{len(rows)} usable pairs ({sum(r['same_person'] for r in rows)} same-person, "
          f"{sum(1 - r['same_person'] for r in rows)} different-person), {missing} skipped (file not found)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw-dir", required=True, type=Path, help="path to the extracted LFW dataset root")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = build_pairs(args.lfw_dir)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_a", "image_b", "same_person"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
