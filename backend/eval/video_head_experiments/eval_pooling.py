import sys, glob, os
sys.path.insert(0, r"C:\TrueLense\tl\backend")
os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

import numpy as np
import torch
import torch.nn as nn

from app.services.video.model_v1 import common

cascade = common.get_face_cascade()

# --- frozen CNN backbone (same as production) ---
cnn = common.build_model()
common.load_checkpoint(cnn)
cnn.eval()
for p in cnn.parameters():
    p.requires_grad = False


def cnn_per_frame_logits(video_path):
    frames = common.decode_selected_frames(video_path, use_grab_skip=False)
    batch = np.stack([common.preprocess_frame(f, cascade) for f in frames], axis=0)
    tensor = torch.from_numpy(batch).contiguous().float()
    with torch.inference_mode():
        logits = cnn(tensor).squeeze(-1)  # (16,)
    return logits.numpy()


def pool_mean(logits):
    return 1 / (1 + np.exp(-logits.mean()))


def pool_max(logits):
    return 1 / (1 + np.exp(-logits.max()))


def pool_topk_mean(logits, k):
    top = np.sort(logits)[-k:]
    return 1 / (1 + np.exp(-top.mean()))


candidates = sorted(set(glob.glob(r"C:\real photos\*.mp4")) | set(glob.glob(r"C:\fake photos\*.mp4")))
videos = []
seen_names = set()
for p in candidates:
    name = os.path.basename(p).lower()
    if name in seen_names:
        continue
    seen_names.add(name)
    if "fake" in name:
        videos.append((p, "FAKE"))
    elif "real" in name:
        videos.append((p, "REAL"))
    else:
        print(f"WARNING: could not infer label from filename, skipping: {p}")


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


K = 4  # top-k frames out of 16

rows = []
for path, label in videos:
    name = os.path.basename(path)
    logits = cnn_per_frame_logits(path)
    lab = 1 if label == "FAKE" else 0
    scores = {
        "mean": pool_mean(logits),
        "max": pool_max(logits),
        f"top{K}_mean": pool_topk_mean(logits, K),
    }
    rows.append((name, label, lab, logits, scores))

print(f"{'file':22s} {'label':5s} {'mean(dep)':>10s} {'max':>10s} {f'top{K}mean':>10s}   per-frame logits")
by_method = {"mean": [], "max": [], f"top{K}_mean": []}
for name, label, lab, logits, scores in rows:
    by_method["mean"].append((scores["mean"], lab))
    by_method["max"].append((scores["max"], lab))
    by_method[f"top{K}_mean"].append((scores[f"top{K}_mean"], lab))
    logit_str = " ".join(f"{v:5.2f}" for v in logits)
    print(f"{name:22s} {label:5s} {scores['mean']:10.4f} {scores['max']:10.4f} {scores[f'top{K}_mean']:10.4f}   [{logit_str}]")

print()
for method, sl in by_method.items():
    auc = pairwise_auc(sl)
    print(f"pooling={method:10s} pairwise AUC on this 10-video set: {auc:.4f}")

print()
print("Deployed threshold is 0.525 (sigmoid space). Scores above are already sigmoided.")
print("Per-video pass/fail at 0.525 for each pooling method:")
for name, label, lab, logits, scores in rows:
    verdicts = {m: ("FAKE" if s >= 0.525 else "REAL") for m, s in scores.items()}
    correct = {m: (v == label) for m, v in verdicts.items()}
    print(f"{name:22s} true={label:5s} " + "  ".join(f"{m}={verdicts[m]}({'OK' if correct[m] else 'X'})" for m in verdicts))
