"""Image round 1, LOCAL (Apple MPS) run on the reduced sample from fetch_local_data.py. Same recipe as
kaggle_kernel/train_image_r1.py (fine-tune ConvNeXt-Tiny from convnext_tiny_diversified_v2.pth; FAKE=0/REAL=1; direct
224 bicubic resize; JPEG/blur/noise/downscale augmentation on both classes; per-source balanced sampling; checkpoint
chosen on validation only), but smaller and fp32. CLIP round 3 is only scored, never trained.
usage: python train_local.py            (writes work/run_local/{best.pt,results.json,scores.csv,train_log.txt})"""
import csv, io, json, math, os, random, sys, time
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from torchvision.transforms import InterpolationMode

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "work", "local_data"); OUT = os.path.join(HERE, "work", "run_local")
EXT = os.path.join(HERE, "work", "ext_eval")
V2 = os.path.join(REPO, "models", "checkpoints", "image", "convnext_tiny_diversified_v2.pth")
CLIPCK = os.path.join(REPO, "models", "checkpoints", "image_clip", "clip_head_round3_portrait_app.pth")
EPOCHS = int(os.environ.get("EPOCHS", 3)); BATCH = 32
LR, LR_HEAD, WD, WARMUP = 2e-5, 1e-4, 0.05, 150
QUOTA = {"genimage": 6000, "sut": 3600, "dff": 4800, "cifake": 1200, "avh": 1200, "faces140k": 1200, "deepdetect": 1200}
if os.environ.get("SMOKE"): EPOCHS, QUOTA = 1, {k: 64 for k in QUOTA}
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NW = 6
TF = transforms.Compose([transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=int(q)); b.seek(0); return Image.open(b).convert("RGB")


class DS(Dataset):
    def __init__(self, items, train=False, jpeg_q=None, tf=TF):
        self.items, self.train, self.jpeg_q, self.tf = items, train, jpeg_q, tf

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        for k in range(5):
            it = self.items[(i + k) % len(self.items)]
            try:
                im = Image.open(it["path"]); im.load(); im = im.convert("RGB"); break
            except Exception:
                continue
        else:
            raise RuntimeError("unreadable images")
        if self.train:
            if random.random() < 0.5:
                s = random.uniform(0.4, 1.0); w, h = im.size
                small = im.resize((max(32, int(w * s)), max(32, int(h * s))), random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
                im = small.resize((w, h), Image.BICUBIC) if random.random() < 0.5 else small
            if random.random() < 0.75:
                im = jpeg(im, random.randint(30, 95))
            if random.random() < 0.2:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0)))
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.jpeg_q:
            im = jpeg(im, self.jpeg_q)
        t = self.tf(im)
        if self.train and random.random() < 0.2:
            t = t + torch.randn_like(t) * (random.uniform(2, 8) / 255 / 0.226)
        return t, (0 if it["fake"] else 1), i


def loader(items, train=False, bs=64, jpeg_q=None, tf=TF, shuffle=False):
    return DataLoader(DS(items, train, jpeg_q, tf), batch_size=bs, shuffle=shuffle, num_workers=NW,
                      persistent_workers=False, drop_last=train)


def build():
    m = convnext_tiny(weights=None); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2); return m


@torch.inference_mode()
def p_fake(model, items, jpeg_q=None, tf=TF, head=None):
    model.eval(); out = np.zeros(len(items), dtype=np.float32)
    for x, _, idx in loader(items, False, 64, jpeg_q, tf):
        x = x.to(DEV)
        if head is not None:
            e = model.encode_image(x); e = e / e.norm(dim=-1, keepdim=True); lg = head(e)
        else:
            lg = model(x)
        out[idx.numpy()] = torch.softmax(lg.float(), 1)[:, 0].cpu().numpy()
    return out


def take(lst, n, seed):
    rr = random.Random(seed); lst = list(lst); rr.shuffle(lst); return lst[:n]


def metrics(y_fake, p):
    y = np.asarray(y_fake, bool); pred = np.asarray(p) >= 0.5
    tpr = float(pred[y].mean()) if y.any() else float("nan"); tnr = float((~pred[~y]).mean()) if (~y).any() else float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if y.any() and (~y).any() else float("nan")
    except Exception:
        auc = float("nan")
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "bal_acc": float(np.nanmean([tpr, tnr])), "recall_fake": tpr,
            "spec_real": tnr, "auc": auc}


def main():
    os.makedirs(OUT, exist_ok=True)
    LOG = open(f"{OUT}/train_log.txt", "a")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    items = []
    for r in csv.DictReader(open(os.path.join(HERE, "local_manifest.csv"))):
        items.append({"path": os.path.join(DATA, r["path"]), "fake": bool(int(r["fake"])), "group": r["group"], "sub": r["sub"],
                      "split": r["split"], "fmt": r["fmt"]})
    groups = sorted({it["group"] for it in items})
    cnt = defaultdict(int)
    for it in items: cnt[(it["group"], it["split"], "fake" if it["fake"] else "real")] += 1
    log("device", DEV, "| items:", len(items)); [log("  ", k, cnt[k]) for k in sorted(cnt)]

    def sel(split, cap, gs=None):
        out = []
        for g in gs or groups:
            for fake in (True, False):
                out += take(sorted([it for it in items if it["group"] == g and it["split"] == split and it["fake"] == fake], key=lambda z: z["path"]), cap, 31)
        return out

    def group_bal(model, its):
        p = p_fake(model, its); res = {}
        for g in groups:
            ix = [i for i, it in enumerate(its) if it["group"] == g]
            if ix: res[g] = metrics([its[i]["fake"] for i in ix], p[ix])["bal_acc"]
        return res

    ck = torch.load(V2, map_location="cpu")
    old = build(); old.load_state_dict(ck["model_state_dict"]); old.to(DEV).eval()
    val = sel("val", 150)
    log("baseline v2 val balanced acc:", json.dumps({k: round(v, 3) for k, v in group_bal(old, val).items()}))

    tr = defaultdict(lambda: defaultdict(list))
    for it in items:
        if it["split"] == "train": tr[(it["group"], it["fake"])][it["sub"]].append(it)

    def epoch_list(rr):
        out = []
        for g, q in QUOTA.items():
            for fake in (True, False):
                subs = tr.get((g, fake))
                if subs:
                    keys = sorted(subs); out += [rr.choice(subs[rr.choice(keys)]) for _ in range(q // 2)]
        rr.shuffle(out); return out

    model = build(); model.load_state_dict(ck["model_state_dict"]); model.to(DEV)
    hp = list(model.classifier.parameters()); ids = {id(p) for p in hp}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in ids], "lr": LR}, {"params": hp, "lr": LR_HEAD}], weight_decay=WD)
    base = [LR, LR_HEAD]; spe = sum(q // 2 * 2 for g, q in QUOTA.items() if (g, True) in tr) // BATCH; total = spe * EPOCHS
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, hist, step, rr, t0 = -1, [], 0, random.Random(1234), time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); run = n = 0
        for x, y, _ in loader(epoch_list(rr), True, BATCH, shuffle=True):
            lrm = min(1, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1, step / max(1, total))))
            for gi, g in enumerate(opt.param_groups): g["lr"] = base[gi] * lrm
            loss = crit(model(x.to(DEV)), y.to(DEV))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += float(loss) * len(y); n += len(y); step += 1
            if step % 50 == 0: log(f"  ep{ep} step{step}/{total} loss {run / n:.4f} {time.time() - t0:.0f}s")
        ba = group_bal(model, val); score = float(np.mean(list(ba.values())))
        hist.append({"epoch": ep, "train_loss": run / max(1, n), "val_mean_bal_acc": score, "val": ba})
        log(f"epoch {ep}: loss {run / max(1, n):.4f} val mean bal acc {score:.4f}", json.dumps({k: round(v, 3) for k, v in ba.items()}))
        if score > best:
            best = score
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "best_val_accuracy": score,
                        "note": "round-1 LOCAL fine-tune of convnext_tiny_diversified_v2; class 0=FAKE 1=REAL"}, f"{OUT}/best.pt"); log("  saved best.pt")

    # ---------------- final evaluation
    new = build(); new.load_state_dict(torch.load(f"{OUT}/best.pt", map_location="cpu")["model_state_dict"]); new.to(DEV).eval()
    test = sel("test", 300) + [dict(it, group="genimage_ADM_heldout_generator") for it in items if it["split"] == "test_gen"]
    for it in sel("val", 150, ["dff"]): pass
    # never-trained external sets (see README): ChatGPT 8, portrait-app 15 (CLIP's training genre), GenImage 10 (may overlap sampled train), face-swap 3, UTKFace reals
    ext = []
    def add_ext(name, d, fake):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")): ext.append({"path": os.path.join(d, f), "fake": fake, "group": name, "sub": name, "split": "ext", "fmt": f.rsplit(".", 1)[-1]})
    bt = os.path.join(REPO, "backend", "test_data", "blind_spot_images")
    add_ext("ext_chatgpt8", f"{bt}/chatgpt_everyday", True); add_ext("ext_portrait_app15(CLIP trained on genre)", f"{bt}/portrait_app", True)
    add_ext("ext_genimage10(may overlap training)", os.path.join(REPO, "testing/ai_generated/images"), True)
    add_ext("ext_faceswap3", os.path.join(REPO, "testing/deepfake/images"), True); add_ext("ext_utkface_real", f"{EXT}/utk_real", False)
    test += ext
    lab = np.array([it["fake"] for it in test]); sets = np.array([it["group"] for it in test])
    P = {"old": p_fake(old, test), "new": p_fake(new, test)}
    try:
        import open_clip
        cck = torch.load(CLIPCK, map_location="cpu")
        clip, _, ctf = open_clip.create_model_and_transforms(cck.get("clip_backbone", "ViT-B-16"), pretrained=cck.get("clip_pretrained", "openai"))
        clip = clip.to(DEV).eval()
        head = nn.Sequential(nn.Linear(cck["embed_dim"], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        head.load_state_dict({k.replace("net.", ""): v for k, v in cck["head_state_dict"].items()}); head.to(DEV).eval()
        P["clip"] = p_fake(clip, test, tf=ctf, head=head)
        P["max_old_clip"] = np.maximum(P["old"], P["clip"]); P["max_new_clip"] = np.maximum(P["new"], P["clip"]); P["avg_new_clip"] = (P["new"] + P["clip"]) / 2
    except Exception as e:
        log("CLIP unavailable:", repr(e))
    res = {"history": hist, "best_val_mean_bal_acc": best, "sets": {s: {m: metrics(lab[sets == s], p[sets == s]) for m, p in P.items()} for s in sorted(set(sets))}}
    trained = ~np.isin(sets, [s for s in set(sets) if s.startswith("ext_")] + ["genimage_ADM_heldout_generator"])
    res["pooled_in_distribution_test"] = {m: metrics(lab[trained], p[trained]) for m, p in P.items()}
    sut = [it for it in test if it["group"] == "sut"]
    res["sut_format_check"] = {}
    for q in (None, 90, 60):
        for nm, mdl in (("old", old), ("new", new)):
            p = p_fake(mdl, sut, jpeg_q=q)
            res["sut_format_check"][f"{nm}_q{q}"] = {"fake_recall": float(np.mean([p[i] >= .5 for i, it in enumerate(sut) if it["fake"]])),
                                                     "real_spec": float(np.mean([p[i] < .5 for i, it in enumerate(sut) if not it["fake"]]))}
    sub = take(test, 300, 3); sl = np.array([it["fake"] for it in sub])
    res["robust_jpeg_q40"] = {n: metrics(sl, p_fake(m, sub, jpeg_q=40)) for n, m in (("old", old), ("new", new))}
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    with open(f"{OUT}/scores.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "set", "fake"] + list(P))
        for i, it in enumerate(test): w.writerow([it["path"], sets[i], int(it["fake"])] + [f"{P[m][i]:.5f}" for m in P])
    log("DONE", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
