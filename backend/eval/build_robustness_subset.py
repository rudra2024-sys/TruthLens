"""Copy a small, balanced, deterministic subset of the evaluation images into one self-contained folder.

The robustness suite re-tests the models on degraded copies of these images. Copying them out of the (large,
re-downloadable) dataset folders means the suite keeps working after the datasets are deleted.

  python -m eval.build_robustness_subset --out D:/eval_data/robustness_subset --per-class 30

Groups (all label-balanced): three sources the models trained on ("seen": AI-vs-Human, 140k Faces, DeepDetect) and two
they never saw ("unseen": OpenFake, GenImage). GenImage fakes are split evenly over its three generators.
Reads eval/data/manifest_b*.csv (produced by build_manifest.py); writes eval/data/manifest_robust.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "eval" / "data"

# group name -> list of (manifest source, label) to draw from, with weight (share of that class)
GROUPS = {
    "aivshuman": ("seen",   {"real": ["aivshuman_train"], "fake": ["aivshuman_train"]}),
    "faces140k": ("seen",   {"real": ["faces140k_test"], "fake": ["faces140k_test"]}),
    "deepdetect": ("seen",  {"real": ["deepdetect_test"], "fake": ["deepdetect_test"]}),
    "openfake": ("unseen",  {"real": ["openfake_test"], "fake": ["openfake_test"]}),
    "genimage": ("unseen",  {"real": ["genimage_real_imagenet"],
                             "fake": ["genimage_fake_sd", "genimage_fake_midjourney", "genimage_fake_biggan"]}),
}


def load_manifests():
    rows = []
    for p in sorted(DATA.glob("manifest_b*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_manifests()
    by = {}
    for r in rows:
        by.setdefault((r["source"], int(r["label"])), []).append(r)

    out = Path(args.out).resolve()                # absolute paths in the manifest, whatever the caller's cwd
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for group, (status, spec) in GROUPS.items():
        for label_name, label in (("real", 0), ("fake", 1)):
            sources = spec[label_name]
            per_src = [args.per_class // len(sources)] * len(sources)
            per_src[0] += args.per_class - sum(per_src)            # remainder to the first source
            for src, n in zip(sources, per_src):
                pool = sorted(by.get((src, label), []), key=lambda r: r["path"])
                if len(pool) < n:
                    raise SystemExit(f"not enough images for {src} label={label}: have {len(pool)}, need {n}")
                for r in rng.sample(pool, n):
                    dest_dir = out / group / label_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    src_path = Path(r["path"])
                    dest = dest_dir / f"{src}__{src_path.name}"
                    if not dest.exists():
                        shutil.copy2(src_path, dest)
                    result.append({"path": str(dest), "label": label, "group": group, "status": status,
                                   "source": src})

    with open(DATA / "manifest_robust.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "status", "source"])
        w.writeheader()
        w.writerows(result)

    size_mb = sum(Path(r["path"]).stat().st_size for r in result) / 1e6
    print(f"{len(result)} images copied to {out} ({size_mb:.0f} MB)  ->  {DATA / 'manifest_robust.csv'}")
    counts = {}
    for r in result:
        counts.setdefault((r["status"], r["group"]), [0, 0])[r["label"]] += 1
    for (status, group), (nr, nf) in counts.items():
        print(f"  {status:7s} {group:11s} real={nr:3d} fake={nf:3d}")


if __name__ == "__main__":
    main()
