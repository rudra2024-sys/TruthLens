"""Build an evaluation manifest (CSV) from folders of labeled media.

Each source is a folder. Two layouts are supported:

  --add  NAME=DIR         DIR contains  real/  and  fake/  subfolders
  --fake NAME=DIR         DIR is a flat folder of FAKE-only media
  --real NAME=DIR         DIR is a flat folder of REAL-only media
  --csv  NAME=FILE.csv    labels come from a CSV; extra options: root=DIR (paths are relative to it),
                          path_col=..., label_col=..., fake_value=1 (label value meaning FAKE)

Append ",trained_on=yes|no|unknown" to record whether the deployed model saw that
source during training (reported next to the metrics so contaminated numbers are
never mistaken for held-out ones), ",max=N" to randomly sample at most N files per
class, ",glob=PATTERN" to keep only matching filenames (e.g. glob=real*.mp4) and
",dedupe_name=yes" to keep one file per filename when a folder tree repeats names.

Example:
  python -m eval.build_manifest --media image --out eval/data/manifest_image.csv \
      --fake "blind_spot_portrait_app=test_data/blind_spot_images/portrait_app,trained_on=yes" \
      --fake "blind_spot_chatgpt=test_data/blind_spot_images/chatgpt_everyday,trained_on=no"
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import random
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_spec(spec: str):
    head, *opts = spec.split(",")
    if "=" not in head:
        raise SystemExit(f"bad spec (need NAME=DIR): {spec}")
    name, directory = head.split("=", 1)
    options = {}
    for o in opts:
        k, _, v = o.partition("=")
        options[k.strip()] = v.strip()
    return name.strip(), Path(directory.strip()), options


def list_media(directory: Path, exts: set[str]):
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", choices=["image", "video"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--fake", action="append", default=[])
    ap.add_argument("--real", action="append", default=[])
    ap.add_argument("--csv", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exts = IMAGE_EXT if args.media == "image" else VIDEO_EXT
    rng = random.Random(args.seed)
    rows = []

    def take(name, files, label, opts):
        if "glob" in opts:                                   # filename filter, e.g. glob=real*.mp4
            files = [f for f in files if fnmatch.fnmatch(f.name.lower(), opts["glob"].lower())]
        if opts.get("dedupe_name") == "yes":                 # same filename in several dirs -> keep one
            seen, unique = set(), []
            for f in files:
                if f.name.lower() not in seen:
                    seen.add(f.name.lower())
                    unique.append(f)
            files = unique
        if "max" in opts and len(files) > int(opts["max"]):
            files = rng.sample(files, int(opts["max"]))
        for f in sorted(files):
            rows.append({"path": str(f.resolve()), "label": label, "source": name,
                         "trained_on": opts.get("trained_on", "unknown")})

    for spec in args.add:
        name, d, opts = parse_spec(spec)
        take(name, list_media(d / "real", exts), 0, opts)
        take(name, list_media(d / "fake", exts), 1, opts)
    for spec in args.fake:
        name, d, opts = parse_spec(spec)
        take(name, list_media(d, exts), 1, opts)
    for spec in args.real:
        name, d, opts = parse_spec(spec)
        take(name, list_media(d, exts), 0, opts)

    for spec in args.csv:
        name, csv_path, opts = parse_spec(spec)
        root = Path(opts.get("root", csv_path.parent))
        pcol, lcol = opts.get("path_col", "file_name"), opts.get("label_col", "label")
        fake_value = opts.get("fake_value", "1")
        by_label = {0: [], 1: []}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                p = root / r[pcol]
                if p.suffix.lower() in exts and p.is_file():
                    by_label[1 if str(r[lcol]).strip() == fake_value else 0].append(p)
        for label, files in by_label.items():
            take(name, files, label, opts)

    if not rows:
        raise SystemExit("no media found - check the paths")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "trained_on"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows)} files)")
    by = {}
    for r in rows:
        by.setdefault(r["source"], [0, 0])[r["label"]] += 1
    for s, (nr, nf) in by.items():
        print(f"  {s:35s} real={nr:5d} fake={nf:5d}")


if __name__ == "__main__":
    main()
