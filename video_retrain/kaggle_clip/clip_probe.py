"""Round 4, track B: does a different model beat the fine-tuned EfficientNet-B0 on the same data?

Frozen CLIP image encoders (ViT-B/16 and ViT-L/14, OpenAI weights) on the exact production face crops round 3
extracted (attached from the truthlens-video-finetune-v2 kernel output), with a small trained head on top:
  - logistic regression on per-frame embeddings (video score = mean frame logit, like production)
  - a 2-layer MLP on the same embeddings
Train embeddings include one degraded copy per frame (JPEG/resize/blur/noise, same recipe as round 3) so the head
sees compressed footage. Splits are manifest_r3's (no person / source clip crosses train/val/test).
Outputs in /kaggle/working: feats_<model>.npz (per-frame embeddings, for reuse), heads_<model>.pt, report.json."""
import csv, glob, io, json, os, random, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==2.24.0"], check=True)
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

W = "/kaggle/working"
N = 16
manifest = next(os.path.join(d, "manifest_r3.csv") for d, _, f in os.walk("/kaggle/input") if "manifest_r3.csv" in f)
crops_dir = sorted((d for d, sub, _ in os.walk("/kaggle/input") if d.count("/") <= 7 and os.path.basename(d) == "crops"
                    and os.path.isdir(f"{d}/rtfs")), key=len)[0]   # Kaggle mounts inputs at varying depths
ROWS = [r for r in csv.DictReader(open(manifest)) if os.path.exists(f"{crops_dir}/{r['key']}.png")]
print(len(ROWS), "videos with crops in", crops_dir, flush=True)
dev = torch.device("cuda")


def strip(r):
    a = np.asarray(Image.open(f"{crops_dir}/{r['key']}.png").convert("RGB"))
    return [a[:, i * 224:(i + 1) * 224] for i in range(N)]


def degrade(img, rnd):   # identical to round 3's training augmentation
    if rnd.random() < 0.5: img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rnd.random() < 0.35:
        s = rnd.uniform(0.25, 0.75); small = img.resize((max(16, int(224 * s)),) * 2, Image.Resampling.BILINEAR)
        img = small.resize((224, 224), Image.Resampling.BILINEAR)
    if rnd.random() < 0.25: img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.8)))
    if rnd.random() < 0.25:
        a = np.asarray(img, dtype=np.float32) + np.random.normal(0, rnd.uniform(2, 10), (224, 224, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rnd.random() < 0.6:
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=rnd.randint(15, 90)); buf.seek(0); img = Image.open(buf).convert("RGB")
    return img


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); p, n = s[y == 1], s[y == 0]
    return float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :]))) if len(p) and len(n) else float("nan")


def per_source(scores, rows):
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == ds]
        out[ds] = auc([scores[i] for i in idx], [int(rows[i]["label"]) for i in idx])
    out["mean"] = float(np.nanmean(list(out.values())))
    return out


report = {}
for arch, pre in (("ViT-B-16", "openai"), ("ViT-L-14", "openai")):
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre)
    model = model.to(dev).eval().half()
    mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(frames):
        x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        f = model.encode_image(((x - mean) / std).half()).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16)

    feats, aug = [], []
    for k, r in enumerate(ROWS):
        fr = strip(r)
        feats.append(embed(fr))
        if r["split"] == "train":
            rnd = random.Random(k)
            aug.append(embed([np.asarray(degrade(Image.fromarray(a), rnd)) for a in fr]))
        else:
            aug.append(None)
        if k % 1000 == 0:
            print(arch, k, "/", len(ROWS), f"{time.time() - t0:.0f}s", flush=True)
    F = np.stack(feats)                                   # (videos, 16, d)
    np.savez_compressed(f"{W}/feats_{arch}.npz", keys=np.array([r["key"] for r in ROWS]), feats=F)
    del model; torch.cuda.empty_cache()

    tr = [i for i, r in enumerate(ROWS) if r["split"] == "train"]
    va = [i for i, r in enumerate(ROWS) if r["split"] == "val"]
    te = [i for i, r in enumerate(ROWS) if r["split"] == "test"]
    # per-source class-balanced weights so no single dataset dominates (same idea as round 3's per-source cap)
    from collections import Counter
    cnt = Counter((ROWS[i]["dataset"], ROWS[i]["label"]) for i in tr)
    Xtr = np.concatenate([np.concatenate([F[i], aug[i]]) for i in tr]).astype(np.float32)
    ytr = np.concatenate([[float(ROWS[i]["label"])] * 32 for i in tr]).astype(np.float32)
    wtr = np.concatenate([[1.0 / cnt[(ROWS[i]["dataset"], ROWS[i]["label"])]] * 32 for i in tr]).astype(np.float32)
    wtr *= len(wtr) / wtr.sum()
    Xt, yt, wt = (torch.from_numpy(a).to(dev) for a in (Xtr, ytr, wtr))

    def video_scores(head, idx):
        with torch.no_grad():
            x = torch.from_numpy(F[idx].astype(np.float32)).to(dev)
            return head(x.view(-1, x.shape[-1])).view(len(idx), N).mean(1).cpu().numpy()

    def fit(head, epochs, lr, wd):
        head = head.to(dev); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        best, best_state, hist = -1, None, []
        for ep in range(epochs):
            head.train(); perm = torch.randperm(len(Xt), device=dev)
            for b in range(0, len(perm), 4096):
                j = perm[b:b + 4096]
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    head(Xt[j]).squeeze(-1), yt[j] * 0.9 + 0.05, reduction="none") * wt[j]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval(); v = per_source(video_scores(head, va), [ROWS[i] for i in va]); hist.append(v["mean"])
            if v["mean"] > best:
                best, best_state, best_v = v["mean"], {k: t.detach().clone() for k, t in head.state_dict().items()}, v
        head.load_state_dict(best_state)
        return head, best_v, hist

    d = F.shape[-1]
    res = {}
    for name, mk, ep, lr, wd in (
            ("linear", lambda: nn.Linear(d, 1), 30, 1e-3, 1e-4),
            ("mlp", lambda: nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 1)), 30, 3e-4, 1e-2)):
        head, v, hist = fit(mk(), ep, lr, wd)
        t = per_source(video_scores(head, te), [ROWS[i] for i in te])
        torch.save({"arch": arch, "pretrained": pre, "head": name, "state_dict": head.state_dict()}, f"{W}/head_{arch}_{name}.pt")
        with open(f"{W}/val_scores_{arch}_{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["dataset", "label", "key", "mean_logit"])
            for i, s in zip(va, video_scores(head, va)):
                w.writerow([ROWS[i]["dataset"], ROWS[i]["label"], ROWS[i]["key"], f"{s:.6f}"])
        res[name] = {"val": v, "test": t, "val_curve": hist}
        print(f"== {arch} {name}: val {v}\n   test {t}", flush=True)
    report[arch] = res
    json.dump(report, open(f"{W}/report.json", "w"), indent=1)
    print(arch, "done in", f"{time.time() - t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
