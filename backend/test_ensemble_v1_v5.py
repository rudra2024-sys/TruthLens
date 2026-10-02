"""Standalone, read-only test: does combining Video Model v1 (deployed) with the CNN+GRU v5
head improve real-world accuracy, or not? Scores the 10 local ground-truth videos
(C:\\real photos, C:\\fake photos) with both models independently, then evaluates several
ensemble strategies (average, max, min) against each model alone.

Not wired into production. Not collected by pytest. Run manually:
    ../.venv/Scripts/python.exe test_ensemble_v1_v5.py
"""
import glob
import os

os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

from app.services.video.model_v1 import optimized as v1
from app.services.video import cnn_gru_v5 as v5

THRESHOLD = 0.525

candidates = sorted(set(glob.glob(r"C:\real photos\*.mp4")) | set(glob.glob(r"C:\fake photos\*.mp4")))
videos = []
seen = set()
for p in candidates:
    name = os.path.basename(p).lower()
    if name in seen:
        continue
    seen.add(name)
    if "fake" in name:
        videos.append((p, 1))
    elif "real" in name:
        videos.append((p, 0))

def pairwise_auc(scores_labels):
    pos = [s for s, l in scores_labels if l == 1]
    neg = [s for s, l in scores_labels if l == 0]
    if not pos or not neg:
        return float("nan")
    count = total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / total

def acc_at_threshold(scores_labels, thr):
    correct = sum(1 for s, l in scores_labels if (s >= thr) == bool(l))
    return correct / len(scores_labels)

rows = []
for path, label in videos:
    name = os.path.basename(path)
    p1 = v1.predict(path).probability
    p5 = v5.predict(path).probability
    rows.append((name, label, p1, p5))

print(f"{'file':22s} {'label':5s} {'v1':>8s} {'v5':>8s} {'avg':>8s} {'max':>8s} {'min':>8s}")
sl_v1, sl_v5, sl_avg, sl_max, sl_min = [], [], [], [], []
for name, label, p1, p5 in rows:
    avg = (p1 + p5) / 2
    mx = max(p1, p5)
    mn = min(p1, p5)
    sl_v1.append((p1, label))
    sl_v5.append((p5, label))
    sl_avg.append((avg, label))
    sl_max.append((mx, label))
    sl_min.append((mn, label))
    lab = "FAKE" if label == 1 else "REAL"
    print(f"{name:22s} {lab:5s} {p1:8.4f} {p5:8.4f} {avg:8.4f} {mx:8.4f} {mn:8.4f}")

print()
for strat_name, sl in [("v1 alone (deployed)", sl_v1), ("v5 alone", sl_v5),
                        ("average(v1,v5)", sl_avg), ("max(v1,v5)", sl_max), ("min(v1,v5)", sl_min)]:
    auc = pairwise_auc(sl)
    acc = acc_at_threshold(sl, THRESHOLD)
    print(f"{strat_name:22s}  pairwise AUC = {auc:.4f}   acc@{THRESHOLD} = {acc:.4f} ({int(acc*len(sl))}/{len(sl)})")
