import sys, glob, os
sys.path.insert(0, r"C:\TrueLense\tl\backend")
os.environ.setdefault("VIDEO_MODEL_V1_DEVICE", "cpu")

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from app.services.video.model_v1 import common

cascade = common.get_face_cascade()

# --- frozen CNN backbone (same as production) ---
cnn = common.build_model()
common.load_checkpoint(cnn)
cnn.eval()
for p in cnn.parameters():
    p.requires_grad = False

FEATURE_DIM = cnn.classifier[1].in_features

def extract_cnn_features(x):
    feats = cnn.features(x)
    feats = cnn.avgpool(feats)
    return torch.flatten(feats, 1)

# --- GRU head ---
class TemporalHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        _, h = self.gru(x)
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h_last)).squeeze(-1)

ckpt = torch.load(r"C:\Users\admin\Downloads\truthlens_video_cnn_rnn_head_v4.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
head = TemporalHead(input_dim=cfg["feature_dim"], hidden_dim=cfg["gru_hidden_dim"])
head.load_state_dict(ckpt["model_state_dict"])
head.eval()

print("Loaded GRU head. Training config:", cfg)
print("Training/val metrics reported by notebook:", ckpt["metrics"])
print()

def cnn_rnn_predict(video_path):
    frames = common.decode_selected_frames(video_path, use_grab_skip=False)
    batch = np.stack([common.preprocess_frame(f, cascade) for f in frames], axis=0)
    tensor = torch.from_numpy(batch).contiguous().float()
    with torch.inference_mode():
        feats = extract_cnn_features(tensor)          # (16, 1280)
        logit = head(feats.unsqueeze(0))               # (1,)
        prob = torch.sigmoid(logit).item()
    return prob

def cnn_only_predict(video_path):
    frames = common.decode_selected_frames(video_path, use_grab_skip=False)
    batch = np.stack([common.preprocess_frame(f, cascade) for f in frames], axis=0)
    tensor = torch.from_numpy(batch).contiguous().float()
    with torch.inference_mode():
        logits = cnn(tensor).squeeze(-1)
    prob = torch.sigmoid(logits.mean()).item()
    return prob

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

print(f"{'file':22s} {'label':5s} {'CNN-only':>10s} {'CNN+GRU':>10s}")
cnn_only_sl, cnn_rnn_sl = [], []
for path, label in videos:
    name = os.path.basename(path)
    p_cnn = cnn_only_predict(path)
    p_rnn = cnn_rnn_predict(path)
    lab = 1 if label == "FAKE" else 0
    cnn_only_sl.append((p_cnn, lab))
    cnn_rnn_sl.append((p_rnn, lab))
    print(f"{name:22s} {label:5s} {p_cnn:10.4f} {p_rnn:10.4f}")

auc_cnn_only = pairwise_auc(cnn_only_sl)
auc_cnn_rnn = pairwise_auc(cnn_rnn_sl)
print()
print(f"CNN-only  (deployed, threshold 0.525) pairwise AUC on this 10-video set: {auc_cnn_only:.4f}")
print(f"CNN+GRU   (new)                       pairwise AUC on this 10-video set: {auc_cnn_rnn:.4f}")
