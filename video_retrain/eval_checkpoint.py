"""Authoritative before/after test of a fine-tuned video checkpoint against the deployed one, on the held-out TEST
split only (never used for training or epoch selection), through the exact production predict().

usage (from backend/):  ../.venv/bin/python ../video_retrain/eval_checkpoint.py <candidate best.pt> [out.csv]

Sources: step-1 test split (RTFS + FakeAVCeleb, video_retrain/work/step1_calibration/videos) and the Celeb-DF test
rows (video_retrain/work/celebdf_test/videos). Reports AUC per source (threshold-free, the fair comparison since a new
checkpoint needs its own calibration refit) and accuracy at the raw 0.5 cut for reference."""
import csv, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from app.services.video.model_v1 import optimized  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
DEPLOYED = os.path.join(HERE, "..", "backend", "checkpoints", "video", "epoch_11_model_only.pt")


def candidate_path(p):
    """best.pt from the notebook stores {'model_state_dict': ...}; production load_checkpoint reads the same key."""
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ck, "unexpected checkpoint format"
    return p


def test_items():
    items = []
    for r in csv.DictReader(open(os.path.join(WORK, "step1_calibration", "manifest.csv"))):
        if r["split"] == "test":
            items.append((r["dataset"], int(r["label"]), os.path.join(WORK, "step1_calibration", "videos", r["local"])))
    for r in csv.DictReader(open(os.path.join(HERE, "manifest.csv"))):
        if r["dataset"] == "celebdf":
            items.append(("celebdf", int(r["label"]), os.path.join(WORK, "celebdf_test", "videos", os.path.basename(r["key"]) + ".mp4")))
    return [i for i in items if os.path.isfile(i[2])]


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def main():
    cand = candidate_path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(WORK, "checkpoint_eval.csv")
    items = test_items()
    rows = []
    for name, ck in (("deployed", DEPLOYED), ("candidate", cand)):
        optimized.clear_model_cache()
        for ds, y, path in items:
            r = optimized.predict(path, checkpoint_path=ck)
            rows.append(dict(model=name, dataset=ds, label=y, file=os.path.relpath(path, WORK), prob=r.probability,
                             mean_logit=r.logit))
            print(name, ds, y, f"{r.probability:.4f}", os.path.basename(path), flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n| source | n (fake/real) | deployed AUC | candidate AUC | deployed acc@0.5 | candidate acc@0.5 |")
    print("|---|---|---|---|---|---|")
    for ds in sorted({r["dataset"] for r in rows}) + ["ALL"]:
        res = {}
        for m in ("deployed", "candidate"):
            sub = [r for r in rows if r["model"] == m and (ds == "ALL" or r["dataset"] == ds)]
            s = [r["mean_logit"] for r in sub]; y = [r["label"] for r in sub]
            acc = np.mean([(r["prob"] >= 0.5) == bool(r["label"]) for r in sub])
            res[m] = (auc(s, y), acc, sum(y), len(y) - sum(y))
        d, c = res["deployed"], res["candidate"]
        print(f"| {ds} | {d[2]}/{d[3]} | {d[0]:.3f} | {c[0]:.3f} | {d[1]*100:.1f}% | {c[1]*100:.1f}% |")


if __name__ == "__main__":
    main()
