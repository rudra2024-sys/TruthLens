"""Leave-one-group-out test on RWFS (real face-swap apps). RWFS has no app labels, so swaps are grouped by FILE-NAME STYLE
(a rough stand-in for 'which app / which download batch'). For each group: fine-tune the live round-4 model on DeepGuard pairs
+ the OTHER groups' RWFS swaps (+ a share of RWFS reals), then test on the held-out group (+ the reals set aside for that fold).
Fixed recipe, no checkpoint selection on test data. Compares against the untouched round-4 model on the same test items.

usage: python rwfs_group_cv.py <out_dir>     (needs scratchpad hfdata paths below; runs on MPS/CPU)
"""
import csv, hashlib, io, json, os, random, re, sys, time
import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

REPO = '/Users/maninij/TruthLens'
SC = '/private/tmp/claude-501/-Users-maninij-TruthLens/5f304c7d-5cb9-4800-bd7b-856cda11f7e8/scratchpad'
RWFS, DG = f'{SC}/hfdata/rwfs', f'{SC}/hfdata/deepguard'
INIT = f'{REPO}/backend/checkpoints/video/video_v1_round4.pt'
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{REPO}/backend')
from app.services.video.model_v1 import common
from app.services.image import face_swap as fs

EPOCHS, LR, BATCH, REPEAT, SMOOTH, DG_PAIRS = 3, 3e-5, 32, 3, 0.1, 400
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MEAN = torch.tensor(common.IMAGENET_MEAN).view(3, 1, 1); STD = torch.tensor(common.IMAGENET_STD).view(3, 1, 1)
casc = common.get_face_cascade()


def crop(path):
    """Same crop as production (Haar >= 64 px, else MTCNN); 224 BILINEAR RGB uint8, or None."""
    bgr = cv2.imread(path)
    if bgr is None: return None
    c, info = common.crop_face_or_center_with_info(bgr, casc)
    bb = info['face_bbox']
    if bb is not None and bb[2] >= fs.MIN_FACE_PX:
        return np.asarray(Image.fromarray(c).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
    f = fs._mtcnn_crop(bgr)
    if f is None: return None
    return np.asarray(Image.fromarray(f[0]).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def style(f):
    n, e = os.path.splitext(f); e = e.lower(); low = n.lower()
    if re.fullmatch(r'\d+ \(\d+\)', n): return 'A_num_paren'
    if re.fullmatch(r'[A-Za-z0-9]{8}', n): return 'B_rand8'
    if re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-.*', n): return 'D_uuid'
    if any(k in low for k in ('pica', 'vidwud', 'faceswapper', 'magic_hour', 'aifaceswap', 'ismart', 'pair_', 'face-swap', 'faceswap')): return 'C_named'
    return 'D_uuid'


# ---- crops (cached) ----
cache = f'{OUT}/crops.npz'
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True); X, Y, G, SRC = z['X'], z['Y'], z['G'], z['SRC']
else:
    X, Y, G, SRC = [], [], [], []
    rr = random.Random(7)
    for lab, k in ((1, 'fake'), (0, 'real')):
        d = f'{RWFS}/{k}/{k}'
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')): continue
            a = crop(f'{d}/{f}')
            if a is None: continue
            X.append(a); Y.append(lab); SRC.append('rwfs')
            G.append(style(f) if lab else f'real{rr.randrange(4)}')
    rows = list(csv.DictReader(open(f'{DG}/metadata.csv')))
    rows = [r for r in rows if int(hashlib.md5(r['target_face'].encode()).hexdigest(), 16) % 100 < 80]
    random.Random(1).shuffle(rows)
    for r in rows[:DG_PAIRS]:
        for lab, k in ((0, 'real_file'), (1, 'fake_file')):
            a = crop(f'{DG}/{r[k]}')
            if a is not None: X.append(a); Y.append(lab); G.append('DG'); SRC.append('dg')
    X, Y, G, SRC = np.stack(X), np.array(Y), np.array(G), np.array(SRC)
    np.savez(cache, X=X, Y=Y, G=G, SRC=SRC)
# ---- replay: frames from round 4's own video training crops, so photo fine-tuning does not make the model forget video ----
REPLAY = os.environ.get('REPLAY_DIR')
VID_PER_VIDEO = 2
if REPLAY:
    man = {r['key']: r for r in csv.DictReader(open(f'{REPLAY}/manifest_r4.csv'))}
    rx, ry = [], []
    rr2 = random.Random(11)
    for k in open(os.environ.get('REPLAY_KEYS', f'{REPLAY}/replay_keys.txt')).read().split():
        pth = f'{REPLAY}/out/crops/{k}.png'
        if not os.path.exists(pth): continue
        a = np.asarray(Image.open(pth).convert('RGB'))
        for i in rr2.sample(range(a.shape[1] // 224), VID_PER_VIDEO):
            rx.append(a[:, i * 224:(i + 1) * 224]); ry.append(int(man[k]['label']))
    if rx:
        X = np.concatenate([X, np.stack(rx)]); Y = np.concatenate([Y, np.array(ry)])
        G = np.concatenate([G, np.array(['V'] * len(rx))]); SRC = np.concatenate([SRC, np.array(['vid'] * len(rx))])
from collections import Counter
print('crops', len(X), Counter(zip(SRC, Y)), Counter(g for g, y in zip(G, Y) if y == 1 and g != 'DG'), flush=True)


def degrade(img, rnd):
    if rnd.random() < 0.5: img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rnd.random() < 0.35:
        s = rnd.uniform(0.25, 0.75); img = img.resize((max(16, int(224 * s)),) * 2, Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
    if rnd.random() < 0.25: img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.8)))
    if rnd.random() < 0.25:
        a = np.asarray(img, dtype=np.float32) + np.random.normal(0, rnd.uniform(2, 10), (224, 224, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rnd.random() < 0.6:
        b = io.BytesIO(); img.save(b, 'JPEG', quality=rnd.randint(15, 90)); b.seek(0); img = Image.open(b).convert('RGB')
    return img


def tens(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


class DS(Dataset):
    def __init__(self, idx, aug): self.idx, self.aug = idx, aug
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; im = Image.fromarray(X[i])
        if self.aug: im = degrade(im, random.Random())
        return tens(np.asarray(im)), torch.tensor(float(Y[i]))


def load(path):
    m = common.build_model(); m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']); return m.to(device).eval()


@torch.no_grad()
def scores(m, idx):
    out = []
    for x, _ in DataLoader(DS(idx, False), batch_size=64):
        out += m(x.to(device)).squeeze(-1).float().cpu().tolist()
    return np.array(out)


def auc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def report(s, y):
    rl, fk = s[y == 0], s[y == 1]
    t = np.quantile(rl, 0.95)
    return dict(auc=round(auc(s, y), 3), caught_at_5pct_false_alarm=round(float((fk >= t).mean()), 3),
                caught_at_logit_0_9=round(float((fk >= 0.9).mean()), 3), reals_flagged_at_logit_0_9=round(float((rl >= 0.9).mean()), 3))


if os.environ.get('FINAL'):
    # final model: same fixed recipe, trained on EVERYTHING (all RWFS + DeepGuard pairs). No test data is left in RWFS, so
    # judge it only on independent data. Saved to <out>/final_r2.pt (the live checkpoint is untouched).
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    m = load(INIT); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4); train = list(range(len(X))); t0 = time.time()
    for ep in range(EPOCHS):
        m.train(); idx = [i for i in train if G[i] != 'V'] * REPEAT + [i for i in train if G[i] == 'V']; random.shuffle(idx)
        for x, y in DataLoader(DS(idx, True), batch_size=BATCH, shuffle=False):
            x, y = x.to(device), y.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(m(x).squeeze(-1), y * (1 - SMOOTH) + SMOOTH / 2)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'  epoch {ep + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)', flush=True)
    torch.save({'model_state_dict': m.state_dict(), 'note': 'rwfs_group_cv.py FINAL=1'}, f'{OUT}/final_r2.pt')
    sd4, sd1 = load(INIT).state_dict(), m.state_dict()
    for a_ in (0.5, 0.75):
        torch.save({'model_state_dict': {k: (sd4[k] * (1 - a_) + sd1[k] * a_) if sd4[k].is_floating_point() else sd1[k] for k in sd4}}, f'{OUT}/final_r2_blend{int(a_ * 100)}.pt')
    print('DONE final'); sys.exit(0)
r4 = load(INIT); results = {}
folds = sorted({g for g in G if g[0] in 'ABCD' and g[1] == '_'})
for fold_i, hold in enumerate(folds):
    real_hold = f'real{fold_i}'
    test = [i for i in range(len(X)) if (G[i] == hold and Y[i] == 1) or G[i] == real_hold]
    train = [i for i in range(len(X)) if i not in set(test)]
    ytest = Y[test]
    print(f'\n=== hold out {hold}: test {int((ytest == 1).sum())} swaps + {int((ytest == 0).sum())} reals; train {len(train)} ===', flush=True)
    base = report(scores(r4, test), ytest)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    m = load(INIT); m.train(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train(); idx = [i for i in train if G[i] != 'V'] * REPEAT + [i for i in train if G[i] == 'V']; random.shuffle(idx)
        for x, y in DataLoader(DS(idx, True), batch_size=BATCH, shuffle=False):
            x, y = x.to(device), y.to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(m(x).squeeze(-1), y * (1 - SMOOTH) + SMOOTH / 2)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f'  epoch {ep + 1} loss {loss.item():.3f} ({time.time() - t0:.0f}s)', flush=True)
    m.eval(); new = report(scores(m, test), ytest)
    results[hold] = {'n_swaps': int((ytest == 1).sum()), 'n_reals': int((ytest == 0).sum()), 'round4': base, 'fine_tuned': new}
    # weight blend of round 4 and the fine-tuned model (alpha = share of the fine-tuned weights); keeps video behaviour closer to round 4
    sd4, sd1 = r4.state_dict(), {k: v.clone() for k, v in m.state_dict().items()}
    for a_ in (0.5, 0.75):
        mb = load(INIT)
        mb.load_state_dict({k: (sd4[k] * (1 - a_) + sd1[k] * a_) if sd4[k].is_floating_point() else sd1[k] for k in sd4}); mb.eval()
        results[hold][f'blend_{a_}'] = report(scores(mb, test), ytest); print(f'  blend {a_}', results[hold][f'blend_{a_}'], flush=True)
    print('  round4    ', base, '\n  fine-tuned', new, flush=True)
    json.dump(results, open(f'{OUT}/results.json', 'w'), indent=1)
print('\nDONE')
