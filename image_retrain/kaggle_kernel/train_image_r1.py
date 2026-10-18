"""TruthLens image round 1: fine-tune ConvNeXt-Tiny (from convnext_tiny_diversified_v2.pth) on the manifest_r1 sources
+ rehearsal of the four original sources. Runs as a Kaggle script kernel. CLIP head is NOT retrained (the portrait-app
source it depends on is not available here); CLIP round 3 is only scored for the ensemble comparison.

Conventions kept identical to production: class index 0 = FAKE, 1 = REAL; direct 224x224 bicubic resize + ImageNet
normalisation (backend/app/pipelines/image/preprocessing.py); checkpoint key "model_state_dict".
Outputs (/kaggle/working): best.pt, results.json, scores.csv, train_log.txt
"""
import os, sys, io, csv, json, math, random, subprocess, time, hashlib
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from torchvision.transforms import InterpolationMode

SMOKE = True  # tiny dry run to verify paths/plumbing
OUT = "/kaggle/working"
EPOCHS = 1 if SMOKE else 6
BATCH = 48
LR, LR_HEAD, WD, WARMUP = 2e-5, 1e-4, 0.05, 300
QUOTA = {"genimage": 12000, "sut": 6000, "dff": 9000, "cifake": 3000, "avh": 4000, "faces140k": 4000, "deepdetect": 2000}
VAL_CAP, TEST_CAP = (20, 20) if SMOKE else (300, 600)
if SMOKE:
    QUOTA = {k: 96 for k in QUOTA}
SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = open(f"{OUT}/train_log.txt", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ------------------------------------------------------------------ locating inputs
def locate(rel, max_depth=4):
    base = "/kaggle/input"
    for d, dirs, _ in os.walk(base):
        depth = d[len(base):].count(os.sep)
        if os.path.exists(os.path.join(d, rel)):
            return d
        if depth >= max_depth:
            dirs[:] = []
    return None


ROOT = {k: locate(v) for k, v in {
    "inputs": "manifest_r1.csv", "genimage": "ADM", "sut": "Super_GenAI_Dataset/Fake",
    "dff": "deepfakefusion_v2_sixsource_dataset/train", "cifake": "train/FAKE", "avh": "train.csv",
    "faces140k": "real_vs_fake/real-vs-fake/train/fake", "deepdetect": "ddata/train/fake"}.items()}
log("roots:", json.dumps(ROOT, indent=1))
assert ROOT["inputs"], "inputs dataset (manifest + checkpoints) not attached"
IN = ROOT["inputs"]

# ------------------------------------------------------------------ items
# item: dict(path, fake(bool), group, sub, split, fmt)
def item(path, fake, group, sub, split):
    return {"path": path, "fake": bool(fake), "group": group, "sub": sub, "split": split,
            "fmt": path.rsplit(".", 1)[-1].lower()}


ITEMS = []
for r in csv.DictReader(open(f"{IN}/manifest_r1.csv")):
    g = r["dataset"]
    if not ROOT.get(g):
        log("WARNING: source missing, skipped:", g); continue
    ITEMS.append(item(os.path.join(ROOT[g], r["path"]), int(r["label"]), g, r["source"], r["split"]))


def listdir_imgs(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))


def take(lst, n, seed):
    rr = random.Random(seed); lst = list(lst); rr.shuffle(lst); return lst[:n]


def add_rehearsal(group, per_class_train, per_class_val, per_class_test, pools):
    """pools: {(fake?): (train_pool, held_out_pool)}; held-out pool is split into val/test."""
    for fake, (tr, ho) in pools.items():
        ho = take(ho, per_class_val + per_class_test, 77)
        for p in take(tr, per_class_train, 78): ITEMS.append(item(p, fake, group, "fake" if fake else "real", "train"))
        for p in ho[:per_class_val]: ITEMS.append(item(p, fake, group, "fake" if fake else "real", "val"))
        for p in ho[per_class_val:]: ITEMS.append(item(p, fake, group, "fake" if fake else "real", "test"))


N_TR = 40 if SMOKE else 3000
if ROOT["cifake"]:
    r = ROOT["cifake"]
    add_rehearsal("cifake", N_TR, VAL_CAP, TEST_CAP, {True: (listdir_imgs(f"{r}/train/FAKE"), listdir_imgs(f"{r}/test/FAKE")),
                                                      False: (listdir_imgs(f"{r}/train/REAL"), listdir_imgs(f"{r}/test/REAL"))})
if ROOT["faces140k"]:
    r = ROOT["faces140k"] + "/real_vs_fake/real-vs-fake"
    add_rehearsal("faces140k", N_TR, VAL_CAP, TEST_CAP, {True: (listdir_imgs(f"{r}/train/fake"), listdir_imgs(f"{r}/test/fake")),
                                                         False: (listdir_imgs(f"{r}/train/real"), listdir_imgs(f"{r}/test/real"))})
if ROOT["deepdetect"]:
    r = ROOT["deepdetect"] + "/ddata"
    add_rehearsal("deepdetect", 40 if SMOKE else 2000, VAL_CAP, TEST_CAP,
                  {True: (listdir_imgs(f"{r}/train/fake"), listdir_imgs(f"{r}/test/fake")),
                   False: (listdir_imgs(f"{r}/train/real"), listdir_imgs(f"{r}/test/real"))})
AVH_LABEL_ONE_IS_FAKE = None
if ROOT["avh"]:
    r = ROOT["avh"]
    rows = list(csv.DictReader(open(f"{r}/train.csv")))
    by = defaultdict(list)
    for x in rows: by[int(x["label"])].append(os.path.join(r, x["file_name"]))
    AVH_BY = by  # label mapping resolved after the baseline model is loaded

# ------------------------------------------------------------------ data
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
            raise RuntimeError("5 unreadable images in a row")
        if self.train:
            if random.random() < 0.5:  # downscale (optionally restore) - resolution degradation
                s = random.uniform(0.4, 1.0); w, h = im.size
                small = im.resize((max(32, int(w * s)), max(32, int(h * s))), random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
                im = small.resize((w, h), Image.BICUBIC) if random.random() < 0.5 else small
            if random.random() < 0.75:  # JPEG for BOTH classes: removes PNG-vs-JPEG as a shortcut
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
        return t, (0 if it["fake"] else 1), i  # production convention: FAKE=0, REAL=1


def loader(items, train=False, bs=128, jpeg_q=None, tf=TF, shuffle=False):
    return DataLoader(DS(items, train, jpeg_q, tf), batch_size=bs, shuffle=shuffle, num_workers=4, pin_memory=True,
                      persistent_workers=False, drop_last=train)


def build():
    m = convnext_tiny(weights=None); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2); return m


@torch.inference_mode()
def p_fake(model, items, jpeg_q=None, tf=TF, clip=None):
    model.eval(); out = np.zeros(len(items), dtype=np.float32)
    for x, _, idx in loader(items, False, 128, jpeg_q, tf):
        x = x.to(DEV, non_blocking=True)
        with torch.autocast("cuda", enabled=DEV.type == "cuda"):
            if clip is not None:
                e = model.encode_image(x); e = e / e.norm(dim=-1, keepdim=True); lg = clip(e.float())
            else:
                lg = model(x)
        out[idx.numpy()] = torch.softmax(lg.float(), 1)[:, 0].cpu().numpy()
    return out


# ------------------------------------------------------------------ models
OLD = build(); ck = torch.load(f"{IN}/convnext_tiny_diversified_v2.pth", map_location="cpu")
OLD.load_state_dict(ck["model_state_dict"]); OLD.to(DEV).eval()

if ROOT["avh"]:  # resolve which csv label means AI using the already-trained v2 model (it scored ~99% on this source)
    s0 = take(AVH_BY[0], 150, 5); s1 = take(AVH_BY[1], 150, 6)
    it0 = [item(p, 0, "avh", "x", "t") for p in s0]; it1 = [item(p, 0, "avh", "x", "t") for p in s1]
    m0, m1 = p_fake(OLD, it0).mean(), p_fake(OLD, it1).mean()
    AVH_LABEL_ONE_IS_FAKE = bool(m1 > m0)
    log(f"avh label check: mean P(fake) label0={m0:.3f} label1={m1:.3f} -> label 1 is {'FAKE' if AVH_LABEL_ONE_IS_FAKE else 'REAL'}")
    assert abs(m1 - m0) > 0.5, "avh label direction ambiguous - abort rather than train on flipped labels"
    fk, rl = (1, 0) if AVH_LABEL_ONE_IS_FAKE else (0, 1)
    for fake, lab in ((True, fk), (False, rl)):
        pool = sorted(AVH_BY[lab]); rr = random.Random(9); rr.shuffle(pool)
        ho, tr = pool[:VAL_CAP + TEST_CAP], pool[VAL_CAP + TEST_CAP:]
        for p in tr[:(40 if SMOKE else 3000) * 2 // 2]: ITEMS.append(item(p, fake, "avh", "fake" if fake else "real", "train"))
        for p in ho[:VAL_CAP]: ITEMS.append(item(p, fake, "avh", "fake" if fake else "real", "val"))
        for p in ho[VAL_CAP:]: ITEMS.append(item(p, fake, "avh", "fake" if fake else "real", "test"))

GROUPS = sorted({it["group"] for it in ITEMS})
log("items per group/split/class:")
cnt = defaultdict(int)
for it in ITEMS: cnt[(it["group"], it["split"], "fake" if it["fake"] else "real")] += 1
for k in sorted(cnt): log("  ", k, cnt[k])

# ------------------------------------------------------------------ evaluation helpers
def capped(split, cap, groups=None, extra_splits=()):
    out = []
    for g in groups or GROUPS:
        for fake in (True, False):
            pool = sorted([it for it in ITEMS if it["group"] == g and it["split"] in (split,) + tuple(extra_splits) and it["fake"] == fake],
                          key=lambda z: z["path"])
            out += take(pool, cap, 31)
    return out


def metrics(y_fake, p):
    y = np.asarray(y_fake, bool); pred = np.asarray(p) >= 0.5
    tpr = float((pred[y]).mean()) if y.any() else float("nan"); tnr = float((~pred[~y]).mean()) if (~y).any() else float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p)) if y.any() and (~y).any() else float("nan")
    except Exception:
        auc = float("nan")
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "bal_acc": float(np.nanmean([tpr, tnr])), "recall_fake": tpr, "spec_real": tnr, "auc": auc}


def group_bal_acc(model, items):
    p = p_fake(model, items); res = {}
    for g in GROUPS:
        idx = [i for i, it in enumerate(items) if it["group"] == g]
        if idx: res[g] = metrics([items[i]["fake"] for i in idx], p[idx])["bal_acc"]
    return res


VAL = capped("val", VAL_CAP)
log("baseline (v2) val balanced accuracy per group:", json.dumps({k: round(v, 4) for k, v in group_bal_acc(OLD, VAL).items()}))

# ------------------------------------------------------------------ sampling + training
# pre-index train items per (group, class, sub) once for speed
_TR = defaultdict(lambda: defaultdict(list))
for it in ITEMS:
    if it["split"] == "train": _TR[(it["group"], it["fake"])][it["sub"]].append(it)


def epoch_list(rr):
    out = []
    for g, quota in QUOTA.items():
        for fake in (True, False):
            subs = _TR.get((g, fake))
            if not subs: continue
            keys = sorted(subs)
            out += [rr.choice(subs[rr.choice(keys)]) for _ in range(quota // 2)]
    rr.shuffle(out); return out


model = build(); model.load_state_dict(ck["model_state_dict"]); model.to(DEV).to(memory_format=torch.channels_last)
head_params = list(model.classifier.parameters()); hp = {id(p) for p in head_params}
opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in hp], "lr": LR},
                         {"params": head_params, "lr": LR_HEAD}], weight_decay=WD)
steps_per_epoch = sum(QUOTA[g] // 2 * 2 for g in QUOTA if (g, True) in _TR) // BATCH
total = steps_per_epoch * EPOCHS
base = [LR, LR_HEAD]
scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
best, history, step = -1, [], 0
rr = random.Random(SEED)
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    model.train(); items = epoch_list(rr); run = 0.0; n = 0
    for x, y, _ in loader(items, True, BATCH, shuffle=True):
        lrm = min(1, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1, step / max(1, total))))
        for gi, g in enumerate(opt.param_groups): g["lr"] = base[gi] * lrm
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEV)
        with torch.autocast("cuda", enabled=DEV.type == "cuda"):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run += float(loss) * len(y); n += len(y); step += 1
        if step % 100 == 0: log(f"  ep{ep} step{step}/{total} loss {run / n:.4f} {time.time() - t0:.0f}s")
    ba = group_bal_acc(model, VAL); score = float(np.mean(list(ba.values())))
    history.append({"epoch": ep, "train_loss": run / max(1, n), "val_mean_bal_acc": score, "val": ba})
    log(f"epoch {ep}: loss {run / max(1, n):.4f} val mean bal acc {score:.4f}", json.dumps({k: round(v, 3) for k, v in ba.items()}))
    if score > best:
        best = score
        torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "best_val_accuracy": score,
                    "note": "round-1 fine-tune of convnext_tiny_diversified_v2; class 0=FAKE 1=REAL"}, f"{OUT}/best.pt")
        log("  saved best.pt")

# ------------------------------------------------------------------ final evaluation (best checkpoint)
NEW = build(); NEW.load_state_dict(torch.load(f"{OUT}/best.pt", map_location="cpu")["model_state_dict"]); NEW.to(DEV).eval()
CLIP = CLIP_HEAD = CLIP_TF = None
try:
    try:
        import open_clip
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"]); import open_clip
    cck = torch.load(f"{IN}/clip_head_round3_portrait_app.pth", map_location="cpu")
    CLIP, _, CLIP_TF = open_clip.create_model_and_transforms(cck.get("clip_backbone", "ViT-B-16"), pretrained=cck.get("clip_pretrained", "openai"))
    CLIP = CLIP.to(DEV).eval()
    CLIP_HEAD = nn.Sequential(nn.Linear(cck["embed_dim"], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    CLIP_HEAD.load_state_dict({k.replace("net.", ""): v for k, v in cck["head_state_dict"].items()}); CLIP_HEAD.to(DEV).eval()
except Exception as e:
    log("CLIP unavailable, ensemble comparison skipped:", repr(e)); CLIP = None

TEST = capped("test", TEST_CAP) + [dict(it, group="dff_val(selection set)") for it in capped("val", TEST_CAP, groups=["dff"])] + [dict(it, group="genimage_ADM_heldout_generator") for it in ITEMS if it["split"] == "test_gen"][: (30 if SMOKE else 2500)]
LABEL = np.array([it["fake"] for it in TEST]); SETS = np.array([it["group"] for it in TEST])
P = {"old": p_fake(OLD, TEST), "new": p_fake(NEW, TEST)}
if CLIP is not None:
    P["clip"] = p_fake(CLIP, TEST, tf=CLIP_TF, clip=CLIP_HEAD)
    P["max_old_clip"] = np.maximum(P["old"], P["clip"]); P["max_new_clip"] = np.maximum(P["new"], P["clip"])
    P["avg_new_clip"] = (P["new"] + P["clip"]) / 2

res = {"history": history, "best_val_mean_bal_acc": best, "sets": {}}
for s in sorted(set(SETS)):
    ix = np.where(SETS == s)[0]
    res["sets"][s] = {m: metrics(LABEL[ix], p[ix]) for m, p in P.items()}
res["pooled_excluding_heldout_gen"] = {m: metrics(LABEL[SETS != "genimage_ADM_heldout_generator"], p[SETS != "genimage_ADM_heldout_generator"]) for m, p in P.items()}

# format-shortcut diagnostic on SUT: do fake recall / real specificity survive JPEG normalisation?
sut = [it for it in TEST if it["group"] == "sut"]
if sut:
    res["sut_format_check"] = {}
    for q in (None, 90, 60):
        for name, mdl in (("old", OLD), ("new", NEW)):
            p = p_fake(mdl, sut, jpeg_q=q)
            res["sut_format_check"][f"{name}_q{q}"] = {"fake_png_recall": float(np.mean([p[i] >= .5 for i, it in enumerate(sut) if it["fake"]])),
                                                       "real_png_spec": float(np.mean([p[i] < .5 for i, it in enumerate(sut) if not it["fake"] and it["fmt"] == "png"] or [float("nan")])),
                                                       "real_jpg_spec": float(np.mean([p[i] < .5 for i, it in enumerate(sut) if not it["fake"] and it["fmt"] != "png"] or [float("nan")]))}
# robustness: JPEG q40 on a subset of the test list
sub = take(TEST, 400 if not SMOKE else 20, 3); sl = np.array([it["fake"] for it in sub])
res["robust_jpeg_q40"] = {n: metrics(sl, p_fake(m, sub, jpeg_q=40)) for n, m in (("old", OLD), ("new", NEW))}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
with open(f"{OUT}/scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "set", "fake"] + list(P))
    for i, it in enumerate(TEST): w.writerow([it["path"], SETS[i], int(it["fake"])] + [f"{P[m][i]:.5f}" for m in P])
log("DONE", f"{time.time() - t0:.0f}s")
log(json.dumps({s: {m: round(v["bal_acc"], 3) for m, v in d.items()} for s, d in res["sets"].items()}, indent=1))
