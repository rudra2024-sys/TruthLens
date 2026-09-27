"""Download the model checkpoints a deployment needs and verify them against models/CHECKSUMS.sha256.

The checkpoints (~335 MB in total) are gitignored, so a fresh clone/server does not have them. Host the three files somewhere
you control (a GitHub Release, an S3/R2 bucket, your own server) under their file names, then on the server:

    python backend/scripts/fetch_checkpoints.py --base-url https://example.com/truthlens-models/

Each file is fetched from <base-url>/<file name>, written atomically, and its SHA-256 must match the checksum list, otherwise
it is discarded (a corrupted or tampered model is never left in place). Files already present and correct are skipped.
Use --check to only verify what is on disk. Pure standard library: works in a bare Docker image.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECKSUMS = REPO / "models" / "CHECKSUMS.sha256"


def read_checksums(path: Path = CHECKSUMS) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest.lower()
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(base_url: str, rel: str, expected: str, root: Path = REPO) -> str:
    """Returns 'present' | 'downloaded'. Raises RuntimeError on a checksum mismatch (nothing is left on disk)."""
    dest = root / rel
    if dest.is_file() and sha256_of(dest) == expected:
        return "present"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = base_url.rstrip("/") + "/" + dest.name
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    got = sha256_of(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name}: checksum mismatch (expected {expected[:12]}..., got {got[:12]}...); discarded")
    os.replace(tmp, dest)
    return "downloaded"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", help="folder URL that contains the checkpoint files")
    ap.add_argument("--check", action="store_true", help="only verify the files already on disk")
    args = ap.parse_args(argv)

    problems = 0
    for rel, digest in read_checksums().items():
        dest = REPO / rel
        if args.check or not args.base_url:
            ok = dest.is_file() and sha256_of(dest) == digest
            print(("OK      " if ok else "MISSING " if not dest.is_file() else "CORRUPT ") + rel)
            problems += not ok
            continue
        try:
            print(f"{fetch(args.base_url, rel, digest):10s} {rel}")
        except Exception as e:                       # network error, 404, checksum mismatch
            print(f"FAILED     {rel}: {e}")
            problems += 1
    if not args.check and not args.base_url:
        print("(no --base-url given: verified only; nothing downloaded)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
