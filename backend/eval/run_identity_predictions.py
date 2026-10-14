"""Run the deployed identity-match pipeline over a pairs manifest and save raw similarity scores.

Uses the exact production code (IdentityPipeline.embed, detector.py's own _cosine_similarity),
pinning this evaluation to what actually runs in the app -- same principle as every other eval
harness here (run_predictions.py's own docstring says so explicitly). Device defaults to CPU.
Embeddings are cached per image path within one run (pairs commonly reuse an image across rows).
Resumable: pairs already scored in --out are skipped on a re-run.

  python -m eval.run_identity_predictions --manifest eval/data/manifest_identity_pairs.csv \
      --out eval/data/pred_identity.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("IDENTITY_MODEL_DEVICE", "cpu")

from app.pipelines.identity.inference import NoFaceDetectedError  # noqa: E402
from app.services.identity.detector import _cosine_similarity, get_identity_pipeline  # noqa: E402


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_scored(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["image_a"], r["image_b"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    done = already_scored(args.out)
    todo = [r for r in rows if (r["image_a"], r["image_b"]) not in done]
    print(f"{len(rows)} pairs total, {len(done)} already scored, {len(todo)} to run")

    pipeline = get_identity_pipeline()
    embedding_cache: dict[str, list[float] | None] = {}
    no_face_count = 0

    def embed(path: str):
        if path not in embedding_cache:
            try:
                embedding_cache[path] = pipeline.embed(path)
            except NoFaceDetectedError:
                embedding_cache[path] = None
        return embedding_cache[path]

    write_header = not os.path.exists(args.out) or not done
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_a", "image_b", "same_person", "similarity"])
        if write_header:
            w.writeheader()
        for i, row in enumerate(todo, 1):
            emb_a, emb_b = embed(row["image_a"]), embed(row["image_b"])
            if emb_a is None or emb_b is None:
                no_face_count += 1
                continue
            similarity = _cosine_similarity(emb_a, emb_b)
            w.writerow({
                "image_a": row["image_a"], "image_b": row["image_b"],
                "same_person": row["same_person"], "similarity": similarity,
            })
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
                f.flush()

    print(f"done. {no_face_count} pairs skipped (no face found in at least one image)")


if __name__ == "__main__":
    main()
