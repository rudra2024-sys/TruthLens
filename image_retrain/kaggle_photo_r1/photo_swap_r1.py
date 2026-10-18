# %% [markdown]
# # TruthLens photo face-swap detector, round 1 (Colab / Kaggle)
#
# Fine-tunes the Video Model v1 round-4 face classifier for SINGLE PHOTOS. Photo data = the DeepGuard dataset (downloaded, not
# generated here: 5,352 real photos, each paired with its inswapper_128 face-swap), split so no face is shared between train, val
# and test; plus round 4's video-frame crops. The model is evaluated per photo with the production crop (Haar, MTCNN fallback),
# and finally on RWFS (swaps from 8 real online face-swap apps) which is NEVER used for training or model selection.
# Writes a SEPARATE checkpoint (`run_photo_r1/best.pt`); the deployed video model is never touched. Nothing in this notebook
# creates face swaps. Plan + acceptance criteria: `image_retrain/PHOTO_SWAP_PLAN.md`.

# %% 1. Paths, previous outputs, app code
import os, sys, glob, shutil, subprocess, csv, hashlib, json, random, time, io
import numpy as np

SMOKE = bool(os.environ.get('SMOKE'))
ON_KAGGLE = os.path.isdir('/kaggle/input')
ON_COLAB = (not ON_KAGGLE) and os.path.isdir('/content') and 'LOCAL_WORK' not in os.environ
REMOTE = ON_KAGGLE or ON_COLAB                       # Kaggle or Colab: same pipeline, inputs fetched differently
INPUT_ROOT = '/kaggle/input' if ON_KAGGLE else '/content/kdata'
WORK = '/kaggle/working' if ON_KAGGLE else '/content/work' if ON_COLAB else os.environ['LOCAL_WORK']
TEMP = '/kaggle/temp' if ON_KAGGLE else '/content/temp' if ON_COLAB else os.environ['LOCAL_TEMP']
DRIVE = '/content/drive/MyDrive/TruthLens_photo_r1' if ON_COLAB and os.path.isdir('/content/drive/MyDrive') else None
os.makedirs(WORK, exist_ok=True); os.makedirs(TEMP, exist_ok=True)


def find(name, root=None, max_depth=8, want_dir=False, avoid=('frames',)):
    root = root or INPUT_ROOT
    # shallow search (the attached datasets hold 100k+ files); match BEFORE pruning (see the round-4 notebook)
    hits = []
    base = root.rstrip('/').count('/')
    for d, dirs, files in os.walk(root):
        depth = d.count('/') - base
        if (name in dirs) if want_dir else (name in files):
            hits.append(os.path.join(d, name))
        dirs[:] = [x for x in dirs if x not in avoid] if depth < max_depth else []
    return sorted(hits, key=len)


RUN_NAME = 'run_photo_r1'
RUN = f'{WORK}/{RUN_NAME}'
os.makedirs(RUN, exist_ok=True)


def drive_put(path):
    """Colab only: mirror a file into Google Drive so a disconnected session can resume (no-op elsewhere)."""
    if DRIVE and os.path.exists(path):
        os.makedirs(f'{DRIVE}/{RUN_NAME}', exist_ok=True)
        shutil.copy(path, f'{DRIVE}/{RUN_NAME}/' + os.path.basename(path))


if DRIVE and os.path.exists(f'{DRIVE}/{RUN_NAME}/latest.pt') and not os.path.exists(f'{RUN}/latest.pt'):
    for f in os.listdir(f'{DRIVE}/{RUN_NAME}'):
        shutil.copy(f'{DRIVE}/{RUN_NAME}/{f}', f'{RUN}/{f}')
    print('restored training state from Drive')
if ON_COLAB:
    # Colab has no attached inputs: fetch them with the Kaggle CLI (needs KAGGLE_API_TOKEN or ~/.kaggle credentials; set by
    # the Colab notebook's first cell). Same layout as Kaggle's /kaggle/input afterwards, so everything below is shared.
    def kaggle(*a):
        r = subprocess.run(['kaggle', *a], capture_output=True, text=True)
        print('kaggle', ' '.join(a)[:110], '->', 'ok' if r.returncode == 0 else 'FAILED\n' + (r.stdout + r.stderr)[-600:], flush=True)
        assert r.returncode == 0
    K = INPUT_ROOT; os.makedirs(K, exist_ok=True)
    if not glob.glob(f'{K}/inputs/manifest_r4.csv'):
        kaggle('datasets', 'download', '-d', 'maninijaiswal/truthlens-video-ft-inputs', '-p', f'{K}/inputs', '--unzip')
    # previous video notebook's saved output: the 16-frame face-crop strips + the round-4 checkpoint (matched by file NAME,
    # so 'best.pt' brings every run's checkpoint; run_r4's is picked below). Re-running resumes: finished files are skipped.
    kaggle('kernels', 'output', 'maninijaiswal/truthlens-video-finetune-v2', '-p', f'{K}/prev', '--file-pattern', r'\.png$|best\.pt$')
if REMOTE:
    # Photo data from Hugging Face. DeepGuard is public; RWFS is gated (accept its terms on huggingface.co, then give this
    # notebook a read token as the Colab secret / env var HF_TOKEN). Without a token RWFS is skipped and only that final test is lost.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
    from huggingface_hub import snapshot_download
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN and ON_COLAB:
        try:
            from google.colab import userdata
            HF_TOKEN = userdata.get('HF_TOKEN')
        except Exception:
            HF_TOKEN = None
    if not HF_TOKEN and ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            HF_TOKEN = None
    DEEPGUARD = f'{WORK}/hf/deepguard'
    if not os.path.exists(f'{DEEPGUARD}/metadata.csv'):
        snapshot_download('Sowaiba01/Deepfake', repo_type='dataset', local_dir=DEEPGUARD, max_workers=8)
    RWFS = f'{WORK}/hf/rwfs'
    if HF_TOKEN and not glob.glob(f'{RWFS}/fake/**/*.*', recursive=True):
        try:
            snapshot_download('Scam-AI/RWFS', repo_type='dataset', local_dir=RWFS, max_workers=8, token=HF_TOKEN)
        except Exception as e:
            print('RWFS download failed (no access yet? accept the terms on huggingface.co):', str(e)[:200])
    RWFS = RWFS if glob.glob(f'{RWFS}/fake/**/*.*', recursive=True) else None
    print('DeepGuard', DEEPGUARD, '| RWFS', RWFS or 'not available (final real-app test will be skipped)')
    for d, dirs, _ in os.walk(INPUT_ROOT):
        depth = d.count('/') - INPUT_ROOT.count('/')
        if depth <= 4:
            print('  ' * depth + os.path.basename(d) + '/', sorted(dirs)[:6])
        if depth >= 4:
            dirs[:] = []
    crop_dirs = [h for h in find('crops', max_depth=4, want_dir=True) if os.path.isdir(h)]
    assert crop_dirs, 'previous video-finetune output (with crops/) is not attached: add it as a notebook input'
    PREV = os.path.dirname(crop_dirs[0]); CROPS = crop_dirs[0]
    MANIFEST = find('manifest_r4.csv', max_depth=4)[0]
    INIT = [h for h in find('best.pt', max_depth=5) if '/run_r4/' in h][0]
    # resume: bring back this notebook's earlier output (photocrops/, run_photo_r1/)
    prev_runs = [os.path.dirname(h) for h in find('latest.pt', max_depth=5) if f'/{RUN_NAME}/' in h]
    for pr in prev_runs:
        base = os.path.dirname(pr)
        for sub in ('photocrops', RUN_NAME):
            if os.path.isdir(f'{base}/{sub}'):
                subprocess.run(['cp', '-rn', f'{base}/{sub}/.', f'{WORK}/{sub}/'] if os.path.isdir(f'{WORK}/{sub}') else ['cp', '-rn', f'{base}/{sub}', f'{WORK}/'], check=False)
    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout
          if shutil.which('nvidia-smi') else 'WARNING: no NVIDIA GPU visible - training would run on CPU')
    # App code (video model definition + crop code): pinned GitHub commit, else nothing (no bundled fallback needed here:
    # the previous notebook proved the clone works; if offline, attach the inputs dataset's bundled copy as PKG manually)
    PKG = f'{TEMP}/TruthLens/backend'
    r = subprocess.run(['git', 'clone', '-q', 'https://github.com/rudra2024-sys/TruthLens.git', f'{TEMP}/TruthLens'])
    assert r.returncode == 0, 'could not clone the app repository (enable internet for this notebook)'
    subprocess.run(['git', '-C', f'{TEMP}/TruthLens', 'checkout', '-q', 'd360475'], check=True)
else:
    CROPS = os.environ['LOCAL_CROPS']; MANIFEST = os.environ['LOCAL_MANIFEST']; INIT = os.environ['LOCAL_INIT']
    DEEPGUARD = os.environ['LOCAL_DEEPGUARD']; RWFS = os.environ.get('LOCAL_RWFS') or None; PKG = os.environ['LOCAL_PKG']
sys.path.insert(0, PKG)
from app.services.video.model_v1 import common, optimized
ROWS = list(csv.DictReader(open(MANIFEST)))
print('init checkpoint', INIT, hashlib.sha256(open(INIT, 'rb').read()).hexdigest()[:16])
print(len(ROWS), 'manifest rows; crops at', CROPS)

# %% 2. Photo pairs from DeepGuard, split so that no face file is shared between splits
import cv2
meta_rows = list(csv.DictReader(open(f'{DEEPGUARD}/metadata.csv')))
parent = {}


def root(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


for r in meta_rows:                       # a face used in several pairs (as source or target) ties those pairs together
    parent[root(r['source_identity'])] = root(r['target_face'])


def split_of(key):                        # deterministic 80/10/10 by connected group
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return 'train' if h < 80 else 'val' if h < 90 else 'test'


PAIRS = {'train': [], 'val': [], 'test': []}
for r in meta_rows:
    rp, fp = f"{DEEPGUARD}/{r['real_file']}", f"{DEEPGUARD}/{r['fake_file']}"
    if os.path.exists(rp) and os.path.exists(fp):
        PAIRS[split_of(root(r['target_face']))].append((r['id'], rp, fp))
if SMOKE:
    PAIRS = {k: v[:n] for (k, v), n in zip(PAIRS.items(), (12, 6, 6))}
print({k: len(v) for k, v in PAIRS.items()}, 'real/swapped pairs per split')
assert not ({root(r['target_face']) for r in meta_rows if split_of(root(r['target_face'])) == 'train'} &
            {root(r['target_face']) for r in meta_rows if split_of(root(r['target_face'])) == 'test'}), 'split leak'

RWFS_ITEMS = []                           # (path, label): app-made swaps (fake) and real photos; TEST ONLY
if RWFS:
    for lab, k in ((0, 'real'), (1, 'fake')):
        fl = sorted(f for f in glob.glob(f'{RWFS}/{k}/**/*.*', recursive=True) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
        RWFS_ITEMS += [(f, lab) for f in (fl[:20] if SMOKE else fl)]
print(len(RWFS_ITEMS), 'RWFS test photos')

# %% 3. Production crop for every photo (Haar first, MTCNN fallback) -> 224 px PNGs
from PIL import Image
PC = f'{WORK}/photocrops'
FACE_PAD = common.FACE_PADDING_FRACTION
MIN_FACE_PX = 64
try:
    if REMOTE:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'facenet-pytorch==2.6.0'], check=True)
    from facenet_pytorch import MTCNN
    _mt = MTCNN(keep_all=True, device='cpu', min_face_size=40, post_process=False)
except Exception as e:
    print('MTCNN unavailable, Haar only:', e); _mt = None
_casc = common.get_face_cascade()


def crop_photo(path):
    """Mirrors backend/app/services/image/face_swap.py: Haar frontal face >= 64 px, else MTCNN (prob >= 0.9, >= 64 px);
    20% padding, BILINEAR resize to 224. None if neither finds a usable face (production would skip the check too)."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    crop, info = common.crop_face_or_center_with_info(bgr, _casc)
    bb = info['face_bbox']
    if bb is not None and bb[2] >= MIN_FACE_PX:
        return np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8), int(bb[2])
    if _mt is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = _mt.detect(Image.fromarray(rgb))
    if boxes is None:
        return None
    c = [b for b, p in zip(boxes, probs) if p is not None and p >= 0.9 and (b[2] - b[0]) >= MIN_FACE_PX]
    if not c:
        return None
    x0, y0, x1, y1 = max(c, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    h, w = rgb.shape[:2]; bw, bh = x1 - x0, y1 - y0
    a, b = max(0, int(x0 - bw * FACE_PAD)), max(0, int(y0 - bh * FACE_PAD))
    c2, d = min(w, int(x1 + bw * FACE_PAD)), min(h, int(y1 + bh * FACE_PAD))
    cr = rgb[b:d, a:c2]
    if cr.size == 0:
        return None
    return np.asarray(Image.fromarray(cr).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8), int(bw)


PHOTOS = {'train': [], 'val': [], 'test': [], 'rwfs': []}       # (png path, label, face_px, id)
dropped = 0


def add_photo(split, src, lab, name):
    global dropped
    out = f'{PC}/{split}/{name}.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.exists(out + '.px'):
        PHOTOS[split].append((out, lab, int(open(out + '.px').read()), name)); return
    res = crop_photo(src)
    if res is None:
        dropped += 1; return
    Image.fromarray(res[0]).save(out); open(out + '.px', 'w').write(str(res[1]))
    PHOTOS[split].append((out, lab, res[1], name))


for s_ in ('train', 'val', 'test'):
    for pid, rp, fp in PAIRS[s_]:
        add_photo(s_, rp, 0, f'real/{pid}'); add_photo(s_, fp, 1, f'fake/{pid}')
for i, (path, lab) in enumerate(RWFS_ITEMS):
    add_photo('rwfs', path, lab, f'{"fake" if lab else "real"}/{i:05d}')
print({s_: len(v) for s_, v in PHOTOS.items()}, 'photos with a usable face; dropped (no face found):', dropped)

# %% 4. Video-frame items from round 4's crops (same rows, same splits)
def items(split):
    out = []
    for r in ROWS:
        p = f"{CROPS}/{r['key']}.png"
        if r['split'] == split and os.path.exists(p):
            out.append((p, int(r['label']), r['dataset']))
    return out


TRAIN_V, VAL_V, TEST_V = items('train'), items('val'), items('test')
from collections import Counter
print(len(TRAIN_V), 'train /', len(VAL_V), 'val /', len(TEST_V), 'test videos')
print('train per source/label:', sorted(Counter((d, y) for _, y, d in TRAIN_V).items()))

# %% 5. Training
import torch, torch.nn as nn
from PIL import ImageFilter
from torch.utils.data import DataLoader, Dataset

EPOCHS, LR, BATCH, FRAMES_PER_VIDEO, SAVE_EVERY = 4, 3e-5, 64, 8, 400
FRAMES_PER_DS = {'phone': 16, 'phoneswap': 16}
PHOTO_REPEAT = 3           # each photo (real or fake) is seen this many times per epoch with fresh augmentation
LABEL_SMOOTHING = 0.1
CAP_PER_CLASS = 400
EVAL_FRAMES = (1, 5, 9, 13)
THR_NOW = 0.74             # logit of the current production FAKE line (calibrated p = 0.7)
if SMOKE:
    EPOCHS, BATCH, CAP_PER_CLASS, SAVE_EVERY, PHOTO_REPEAT = 1, 8, 4, 5, 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor(common.IMAGENET_MEAN).view(3, 1, 1); STD = torch.tensor(common.IMAGENET_STD).view(3, 1, 1)
N = common.NUM_FRAMES


def load_strip(p):
    a = np.asarray(Image.open(p).convert('RGB'))
    return [a[:, i * 224:(i + 1) * 224] for i in range(N)]


def degrade(img, rnd):
    # identical for real and fake, so "looks degraded" can't become a label shortcut (same recipe as round 4)
    if rnd.random() < 0.5: img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rnd.random() < 0.35:
        s = rnd.uniform(0.25, 0.75); small = img.resize((max(16, int(224 * s)),) * 2, Image.Resampling.BILINEAR)
        img = small.resize((224, 224), Image.Resampling.BILINEAR)
    if rnd.random() < 0.25: img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.3, 1.8)))
    if rnd.random() < 0.25:
        a = np.asarray(img, dtype=np.float32) + np.random.normal(0, rnd.uniform(2, 10), (224, 224, 3))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rnd.random() < 0.6:
        buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rnd.randint(15, 90)); buf.seek(0); img = Image.open(buf).convert('RGB')
    return img


def to_tensor(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


def get_crop(kind, p, i):
    return load_strip(p)[i] if kind == 'v' else np.asarray(Image.open(p).convert('RGB'))


class Mixed(Dataset):
    def __init__(self, its): self.its = its
    def __len__(self): return len(self.its)
    def __getitem__(self, k):
        kind, p, i, y = self.its[k]
        a = np.asarray(degrade(Image.fromarray(get_crop(kind, p, i)), random.Random()))
        return to_tensor(a), torch.tensor(float(y))


TRAIN_P = [(p, y) for p, y, _, _ in PHOTOS['train']]


def epoch_items(ep):
    rnd = random.Random(1000 + ep)                        # deterministic per epoch -> exact mid-epoch resume
    out = []
    for ds in sorted({d for *_, d in TRAIN_V}):
        for lab in (0, 1):
            pool = [t for t in TRAIN_V if t[2] == ds and t[1] == lab]
            for p, y, d in rnd.sample(pool, min(CAP_PER_CLASS, len(pool))):
                out += [('v', p, i, y) for i in rnd.sample(range(N), FRAMES_PER_DS.get(d, FRAMES_PER_VIDEO))]
    for _ in range(PHOTO_REPEAT):
        out += [('p', p, None, y) for p, y in TRAIN_P]
    rnd.shuffle(out)
    return out


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


@torch.no_grad()
def score_items(model, vids, photos):
    """-> rows (kind, dataset, label, logit, face_px, key) with kind 'video-frame' | 'photo'."""
    model.eval(); rows = []
    for p, y, ds in vids:
        x = torch.stack([to_tensor(load_strip(p)[i]) for i in EVAL_FRAMES]).to(device)
        for z in model(x).squeeze(-1).cpu().tolist():
            rows.append(('video-frame', ds, y, z, 0, p))
    for k in range(0, len(photos), 64):
        chunk = photos[k:k + 64]
        x = torch.stack([to_tensor(np.asarray(Image.open(p).convert('RGB'))) for p, *_ in chunk]).to(device)
        for (p, y, px, pid), z in zip(chunk, model(x).squeeze(-1).cpu().tolist()):
            rows.append(('photo', 'photo-swap', y, z, px, p))
    return rows


def summarize(rows):
    res = {}
    for ds in sorted({r[1] for r in rows if r[0] == 'video-frame'}):
        g = [r for r in rows if r[0] == 'video-frame' and r[1] == ds]
        if len({r[2] for r in g}) == 2: res[ds] = auc([r[3] for r in g], [r[2] for r in g])
    # 'phone' is real-only: scored together with the phone face-swaps, as in round 4
    ph = [r for r in rows if r[0] == 'video-frame' and r[1] in ('phone', 'phoneswap')]
    res.pop('phone', None)
    if len({r[2] for r in ph}) == 2: res['phone+swap'] = auc([r[3] for r in ph], [r[2] for r in ph])
    res['video_mean'] = float(np.nanmean(list(res.values()))) if res else float('nan')
    pr = [r for r in rows if r[0] == 'photo']
    res['photo_auc'] = auc([r[3] for r in pr], [r[2] for r in pr])
    reals = [r[3] for r in pr if r[2] == 0]
    res['photo_real_flagged@now'] = float(np.mean([z >= THR_NOW for z in reals])) if reals else float('nan')
    res['photo_swap_caught@now'] = float(np.mean([r[3] >= THR_NOW for r in pr if r[2] == 1])) if pr else float('nan')
    # selection score: half video, half photo (both AUCs), val only
    res['select'] = 0.5 * res['video_mean'] + 0.5 * res['photo_auc'] if pr else res['video_mean']
    return res


model = common.build_model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = (len(epoch_items(1)) + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps_per_epoch * EPOCHS, pct_start=0.15)
scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')


def save_latest(ep, step):
    tmp = f'{RUN}/latest.pt.tmp'
    torch.save({'model_state_dict': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'scaler': scaler.state_dict(), 'epoch': ep, 'step': step, 'history': history, 'best': best}, tmp)
    os.replace(tmp, f'{RUN}/latest.pt'); drive_put(f'{RUN}/latest.pt'); drive_put(f'{RUN}/best.pt')


if os.path.exists(f'{RUN}/latest.pt'):
    st = torch.load(f'{RUN}/latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(st['model_state_dict']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    scaler.load_state_dict(st['scaler']); start_ep, start_step, history, best = st['epoch'], st['step'], st['history'], st['best']
    print(f'resumed at epoch {start_ep} step {start_step}, best val {best:.4f}')
else:
    model.load_state_dict(torch.load(INIT, map_location='cpu', weights_only=False)['model_state_dict']); model.to(device)
    base = summarize(score_items(model, VAL_V, PHOTOS['val']))
    history = [{'epoch': 0, **base}]; best = base['select']
    torch.save({'model_state_dict': model.state_dict(), 'epoch': 0, 'val': base}, f'{RUN}/best.pt')
    start_ep, start_step = 1, 0
    print('epoch 0 (round-4 model) val', base)

loss_fn = nn.BCEWithLogitsLoss()
for ep in range(start_ep, EPOCHS + 1):
    its = epoch_items(ep)[start_step * BATCH:]
    dl = DataLoader(Mixed(its), batch_size=BATCH, shuffle=False, num_workers=min(os.cpu_count(), 8 if not SMOKE else 0), pin_memory=True)
    model.train(); t0 = time.time(); tot, n = 0.0, 0
    for k, (x, y) in enumerate(dl, start=start_step):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            loss = loss_fn(model(x).squeeze(-1), y * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / 2)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item(); n += 1
        if k % 100 == 0:
            print(f'ep{ep} step {k}/{steps_per_epoch} loss {tot / n:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if (k + 1) % SAVE_EVERY == 0:
            save_latest(ep, k + 1)
    res = summarize(score_items(model, VAL_V, PHOTOS['val']))
    history.append({'epoch': ep, 'train_loss': tot / max(n, 1), **res})
    print(f'== epoch {ep} val {res}', flush=True)
    if res['select'] > best:
        best = res['select']
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val': res}, f'{RUN}/best.pt')
        print(f'   new best {best:.4f} -> best.pt', flush=True); drive_put(f'{RUN}/best.pt')
    start_step = 0
    save_latest(ep + 1, 0)
    json.dump(history, open(f'{RUN}/history.json', 'w'), indent=1); drive_put(f'{RUN}/history.json')
print('training finished; best val select score', best)

# %% 6. Report: round-4 model vs best, on val and TEST; per-item scores for the Mac-side check and calibration
r4 = common.build_model(); r4.load_state_dict(torch.load(INIT, map_location='cpu', weights_only=False)['model_state_dict']); r4.to(device)
new = common.build_model(); new.load_state_dict(torch.load(f'{RUN}/best.pt', map_location='cpu', weights_only=False)['model_state_dict']); new.to(device)
out = {}
for split, V, P in (('val', VAL_V, PHOTOS['val']), ('test', TEST_V, PHOTOS['test'])):
    ra, rb = score_items(r4, V, P), score_items(new, V, P)
    out[split] = {'round4': summarize(ra), 'photo_r1': summarize(rb)}
    print(split, 'round4  ', out[split]['round4']); print(split, 'photo_r1', out[split]['photo_r1'], flush=True)
    with open(f'{RUN}/{split}_scores.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['kind', 'dataset', 'label', 'r4_logit', 'new_logit', 'face_px', 'path'])
        for a, b in zip(ra, rb):
            w.writerow([a[0], a[1], a[2], f'{a[3]:.6f}', f'{b[3]:.6f}', a[4], a[5].replace(WORK, '').replace(CROPS, '')])


def at_fpr(rows_val, rows_test, fpr=0.05):
    """Threshold = (1-fpr) quantile of VAL real logits (photo + video pooled); report TEST recall / false-FAKE at it."""
    thr = float(np.quantile([r[3] for r in rows_val if r[2] == 0], 1 - fpr))
    res = {'threshold_logit': thr}
    for kind in ('photo', 'video-frame'):
        f = [r[3] for r in rows_test if r[0] == kind and r[2] == 1]; rl = [r[3] for r in rows_test if r[0] == kind and r[2] == 0]
        res[kind] = {'fakes_caught': float(np.mean([z >= thr for z in f])) if f else None,
                     'reals_flagged': float(np.mean([z >= thr for z in rl])) if rl else None}
    small = [r for r in rows_test if r[0] == 'photo' and r[4] < 110]
    res['photo_small_faces(<110px)'] = {'n': len(small),
        'reals_flagged': float(np.mean([r[3] >= thr for r in small if r[2] == 0])) if any(r[2] == 0 for r in small) else None}
    return res


for name, m in (('round4', r4), ('photo_r1', new)):
    rv, rt = score_items(m, VAL_V, PHOTOS['val']), score_items(m, TEST_V, PHOTOS['test'])
    out.setdefault('at_5pct_false_alarm', {})[name] = at_fpr(rv, rt)
    print(name, 'at 5% val false-alarm threshold:', json.dumps(out['at_5pct_false_alarm'][name]), flush=True)
if PHOTOS['rwfs']:
    # real-app test: threshold = the one that gives 5% false alarms on VALIDATION real photos/frames (never tuned on RWFS)
    out['rwfs_test'] = {}
    with open(f'{RUN}/rwfs_scores.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['label', 'r4_logit', 'new_logit', 'face_px', 'path'])
        ra, rb = score_items(r4, [], PHOTOS['rwfs']), score_items(new, [], PHOTOS['rwfs'])
        for a, b in zip(ra, rb):
            w.writerow([a[2], f'{a[3]:.6f}', f'{b[3]:.6f}', a[4], a[5].replace(WORK, '')])
    for name, rows in (('round4', ra), ('photo_r1', rb)):
        thr = out['at_5pct_false_alarm'][name]['threshold_logit']
        f = [r[3] for r in rows if r[2] == 1]; rl = [r[3] for r in rows if r[2] == 0]
        out['rwfs_test'][name] = {'n_fake': len(f), 'n_real': len(rl), 'auc': auc([r[3] for r in rows], [r[2] for r in rows]),
                                  'swaps_flagged@now': float(np.mean([z >= THR_NOW for z in f])), 'reals_flagged@now': float(np.mean([z >= THR_NOW for z in rl])),
                                  'swaps_flagged@val5%': float(np.mean([z >= thr for z in f])), 'reals_flagged@val5%': float(np.mean([z >= thr for z in rl]))}
        print('RWFS (real face-swap apps)', name, json.dumps(out['rwfs_test'][name]), flush=True)
json.dump(out, open(f'{RUN}/final_report.json', 'w'), indent=1)
for f in ('final_report.json', 'val_scores.csv', 'test_scores.csv', 'rwfs_scores.csv', 'best.pt', 'history.json'):
    drive_put(f'{RUN}/{f}')
print('wrote final_report.json, val_scores.csv, test_scores.csv, best.pt, history.json' + (f' (also copied to {DRIVE}/{RUN_NAME})' if DRIVE else ''))
