"""Builds the video fine-tuning notebook for two targets from one source (run: python build_notebook.py):
  TruthLens_Video_Finetune_v2.ipynb         Colab: datasets downloaded with the Kaggle API, state on Google Drive
  TruthLens_Video_Finetune_v2_kaggle.ipynb  Kaggle: datasets attached as inputs, state in /kaggle/working
Kept as a script so the notebook's code is reviewable as plain Python in git."""
import json, sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "colab"
CELLS = []
ONLY = None   # when set, cells are only emitted for that target


def md(s):
    if ONLY in (None, TARGET):
        CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    if ONLY in (None, TARGET):
        CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


if TARGET == "kaggle":
    md(r"""
# TruthLens Video Model, round 4: continue from round 3 with DFD + real phone footage + phone face-swaps (Kaggle)

Round 3's five sources (RTFS, FakeAVCeleb, Celeb-DF, FaceForensics++ c23, DFDC; same rows and splits) plus
Google's DFD (split by actor), ~180 real selfie/liveness videos filmed on people's own phones (split by person), and
face swaps generated here from those phone videos with inswapper (same people, same encoding), so "phone look"
can't become a shortcut for "real". Starts from round 3's best.pt (attached via this notebook's previous output).
Private inputs: `maninijaiswal/truthlens-video-ft-inputs` (manifest + deployed checkpoint). Outputs land in `/kaggle/working`:
`crops/`, `run/latest.pt`, `run/best.pt`, `run/history.json`. To resume after a timeout, attach this notebook's
previous output as an input and run again: crops and training state are copied back in and finished work is skipped.
""")
    code(r"""
# 1. Paths, resume from a previous version's output if attached, app code at a pinned commit
import os, sys, glob, shutil, subprocess, csv, hashlib

def find(name, root='/kaggle/input', max_depth=8, want_dir=False, avoid=('frames',)):
    # shallow search: the attached datasets hold 100k+ files, a recursive glob would crawl all of them.
    # Match BEFORE pruning (a v1 run missed FakeAVCeleb because the match sat exactly at the depth limit).
    hits = []
    base = root.rstrip('/').count('/')
    for d, dirs, files in os.walk(root):
        depth = d.count('/') - base
        if (name in dirs) if want_dir else (name in files):
            hits.append(os.path.join(d, name))
        dirs[:] = [x for x in dirs if x not in avoid] if depth < max_depth else []
    return sorted(hits, key=len)

# Layout of the attached inputs, for the log (Kaggle has mounted datasets at different depths over time)
for d, dirs, _ in os.walk('/kaggle/input'):
    depth = d.count('/') - 2
    if depth <= 4:
        print('  ' * depth + os.path.basename(d) + '/', sorted(dirs)[:6])
    if depth >= 4:
        dirs[:] = []

WORK = '/kaggle/working'
for d in ('crops', 'run'):
    os.makedirs(f'{WORK}/{d}', exist_ok=True)
RUN_NAME = 'run_r4'      # round 4 (r1 'run', r2 'run_r2', r3 'run_r3'); a previous version's crops are reused, its run dirs kept as is
os.makedirs(f'{WORK}/{RUN_NAME}', exist_ok=True)
prev = sorted({os.path.dirname(c) for c in find('crops', max_depth=4, want_dir=True)})
for pv in prev:
    print('reusing previous output:', pv)
    subprocess.run(['cp', '-rn', f'{pv}/crops/.', f'{WORK}/crops/'], check=True)
    if os.path.isdir(f'{pv}/phoneswap'):
        subprocess.run(['cp', '-rn', f'{pv}/phoneswap', f'{WORK}/'], check=True)
    if os.path.exists(f'{pv}/run_r3/best.pt') and not os.path.exists(f'{WORK}/run_r3/best.pt'):   # keep round 3 in the output
        subprocess.run(['cp', '-rn', f'{pv}/run_r3', f'{WORK}/'], check=True)
    if os.path.exists(f'{pv}/{RUN_NAME}/latest.pt') and not os.path.exists(f'{WORK}/{RUN_NAME}/latest.pt'):
        subprocess.run(['cp', '-rn', f'{pv}/{RUN_NAME}/.', f'{WORK}/{RUN_NAME}/'], check=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout if shutil.which('nvidia-smi') else 'WARNING: no NVIDIA GPU visible - training would run on CPU')

INP = os.path.dirname(find('manifest_r4.csv', max_depth=4)[0])

# App code: GitHub at a pinned commit; if the session has no internet, the identical files bundled in the inputs dataset
REPO_COMMIT = 'd360475'
PKG = '/kaggle/temp/TruthLens/backend'
r = subprocess.run(['git', 'clone', '-q', 'https://github.com/rudra2024-sys/TruthLens.git', '/kaggle/temp/TruthLens'])
if r.returncode == 0:
    subprocess.run(['git', '-C', '/kaggle/temp/TruthLens', 'checkout', '-q', REPO_COMMIT], check=True)
    print('app code: GitHub @', REPO_COMMIT)
else:
    PKG = '/kaggle/temp/pkg'
    mdir = f'{PKG}/app/services/video/model_v1'
    os.makedirs(mdir, exist_ok=True)
    for d in (f'{PKG}/app', f'{PKG}/app/services', f'{PKG}/app/services/video', mdir):
        open(f'{d}/__init__.py', 'a').close()
    for f in ('common.py', 'optimized.py'):
        shutil.copy(f'{INP}/model_v1_{f}', f'{mdir}/{f}')
    print('app code: bundled copy from the inputs dataset (no internet)')
sys.path.insert(0, PKG)
from app.services.video.model_v1 import common, optimized

CKPT = f'{INP}/epoch_11_model_only.pt'
EXPECTED = '91eefd295e14a527cddcd3ac51e4c134341d95b9c249ccc721cb26691ffae94d'   # models/CHECKSUMS.sha256
assert hashlib.sha256(open(CKPT, 'rb').read()).hexdigest() == EXPECTED, 'checkpoint checksum mismatch'
ROWS = list(csv.DictReader(open(f'{INP}/manifest_r4.csv')))
# round 3's best.pt (the starting point) from this notebook's previous output
INIT = ([h for h in find('best.pt', max_depth=5) if '/run_r3/' in h] + [f'{INP}/r3_best.pt'])[0]   # fallback: bundled copy
assert hashlib.sha256(open(INIT, 'rb').read()).hexdigest() == '9d5d95543f8f622146f0bf66cf9caddacac044d1304bd16e6fa798ef49f04706', 'r3 checkpoint mismatch'
print('init from', INIT)

def crop_path(r):
    return f"{WORK}/crops/{r['key']}.png"

missing = [r for r in ROWS if not os.path.exists(crop_path(r))]
print(len(ROWS), 'manifest rows,', len(missing), 'still need crops; checkpoint OK')

def dataset_root(name):
    if name == 'fakeavceleb':   # the video tree sits next to meta_data.csv (a 'frames/' copy has the same folder names)
        hits = [os.path.dirname(h) for h in find('meta_data.csv')
                if os.path.isdir(os.path.join(os.path.dirname(h), 'RealVideo-RealAudio'))]
    elif name == 'ffpp':        # manifest paths are relative to FaceForensics++_C23/ (original/000.mp4, ...)
        hits = find('FaceForensics++_C23', want_dir=True)
    else:
        marker = {'rtfs': 'original_videos', 'celebdf': 'Celeb-real', 'dfdc': 'dfdc_train_part_00',
                  'dfd': 'DFD_original sequences'}[name]
        hits = [os.path.dirname(h) for h in find(marker, want_dir=True)]
    assert hits, f'dataset for {name} not found under /kaggle/input (see the layout printed above)'
    return hits[0]

ROOTS = {d: dataset_root(d) for d in ('rtfs', 'fakeavceleb', 'celebdf', 'ffpp', 'dfdc', 'dfd')}
ROOTS['phoneswap'] = f'{WORK}/phoneswap'          # generated below
_slug_root = {}
def slug_root(ref):                                 # phone rows: path = "<owner>/<slug>:<file inside dataset>"
    if ref not in _slug_root:
        hits = find(ref.split('/')[1], want_dir=True, max_depth=4)
        assert hits, f'{ref} not attached'
        _slug_root[ref] = hits[0]
    return _slug_root[ref]

def src_path(r):
    if r['dataset'] == 'phone':
        ref, f = r['path'].split(':', 1)
        return os.path.join(slug_root(ref), f)
    return os.path.join(ROOTS[r['dataset']], r['path'])

print(ROOTS)
for d in ('rtfs', 'fakeavceleb', 'celebdf', 'ffpp', 'dfdc', 'dfd', 'phone'):   # fail fast if a manifest path doesn't resolve
    bad = [r['path'] for r in ROWS if r['dataset'] == d and not os.path.exists(src_path(r))]
    print(d, 'missing files:', len(bad), bad[:3])
    assert len(bad) <= 0.05 * sum(r['dataset'] == d for r in ROWS), f'{d}: too many manifest paths not found'
""")
ONLY = 'colab'   # the next three cells are Colab-only
md(r"""
# TruthLens Video Model v2: fine-tune Video Model v1 on RTFS + FakeAVCeleb

Fine-tunes the deployed **Video Model v1** (EfficientNet-B0, `epoch_11_model_only.pt`) on production-identical face
crops, with real-world degradation augmentation (JPEG re-compression, downscaling, blur, noise).

**Fully resumable.** Everything that matters lives on Google Drive under `MyDrive/TruthLens_video_ft/`:
- `crops/`: the 16 face crops per video, extracted once with the app's own code (cloned from GitHub at a pinned commit)
- `run/latest.pt`: model + optimiser + scheduler + position, saved every 400 steps and at every epoch end
- `run/best.pt`: best epoch by validation AUC (this is the file to bring back to the Mac)
- `run/history.json`: per-epoch metrics

If Colab disconnects or the GPU quota runs out, reconnect and **Run all** again. Finished work is skipped and training
continues from the last save.

**Before the first run**, put these in `MyDrive/TruthLens_video_ft/inputs/`:
1. `manifest.csv` (from `video_retrain/manifest.csv` in the repo)
2. `epoch_11_model_only.pt` (the deployed video checkpoint, `backend/checkpoints/video/`)
3. `kaggle_token.txt`: your Kaggle API token (Kaggle -> Settings -> API -> Create New Token), single line

**Data splits** (`manifest.csv`, no source clip or person in more than one split):
- train: RTFS 1,516 real + 1,516 fake (inswapper/uniface), FakeAVCeleb 395 real + 395 fake (FSGAN/Faceswap/Wav2Lip mixes)
- val: 159 videos (RTFS + FakeAVCeleb), used only to pick the best epoch
- test: 221 videos (RTFS + FakeAVCeleb + 60 Celeb-DF), reported for information only. The authoritative before/after
  test is re-run on the Mac with the production pipeline.
""")

code(r"""
# 1. Drive + GPU
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess, shutil
WORK = '/content/drive/MyDrive/TruthLens_video_ft'
for d in ('inputs', 'crops', 'run'):
    os.makedirs(f'{WORK}/{d}', exist_ok=True)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout if shutil.which('nvidia-smi') else 'WARNING: no NVIDIA GPU visible - training would run on CPU')
""")

code(r"""
# 2. App code at a pinned commit (the exact production frame sampling + Haar face crop), and inputs check
REPO_COMMIT = 'd360475'
if not os.path.isdir('/content/TruthLens'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/rudra2024-sys/TruthLens.git', '/content/TruthLens'], check=True)
subprocess.run(['git', '-C', '/content/TruthLens', 'checkout', '-q', REPO_COMMIT], check=True)
import sys
sys.path.insert(0, '/content/TruthLens/backend')
from app.services.video.model_v1 import common, optimized

import hashlib
CKPT = f'{WORK}/inputs/epoch_11_model_only.pt'
EXPECTED = '91eefd295e14a527cddcd3ac51e4c134341d95b9c249ccc721cb26691ffae94d'   # models/CHECKSUMS.sha256
assert os.path.isfile(f'{WORK}/inputs/manifest_r3.csv'), 'upload manifest_r3.csv to inputs/'
RUN_NAME = 'run_r3'
assert os.path.isfile(CKPT), 'upload epoch_11_model_only.pt to inputs/'
assert hashlib.sha256(open(CKPT, 'rb').read()).hexdigest() == EXPECTED, 'checkpoint checksum mismatch'
import csv
ROWS = list(csv.DictReader(open(f'{WORK}/inputs/manifest_r3.csv')))
print(len(ROWS), 'manifest rows; checkpoint OK')
""")

code(r"""
# 3. Kaggle datasets: only downloaded if some crops are still missing (a fresh VM loses /content/data)
def crop_path(r):
    return f"{WORK}/crops/{r['key']}.png"

missing = [r for r in ROWS if not os.path.exists(crop_path(r))]
print(len(missing), 'videos still need crops')
if missing:
    os.environ['KAGGLE_API_TOKEN'] = open(f'{WORK}/inputs/kaggle_token.txt').read().strip()
    subprocess.run(['pip', 'install', '-q', '-U', 'kaggle'], check=True)
    need = {r['dataset'] for r in missing}
    if 'rtfs' in need and not os.path.isdir('/content/data/rtfs/original_videos'):
        subprocess.run(['kaggle', 'datasets', 'download', 'stplusplus/rtfs-10k', '-p', '/content/data/rtfs', '--unzip', '-q'], check=True)
    if 'fakeavceleb' in need and not os.path.isdir('/content/data/fakeavceleb'):
        subprocess.run(['kaggle', 'datasets', 'download', 'shreyaty08/fakeavceleb', '-p', '/content/data/fakeavceleb', '--unzip', '-q'], check=True)
    if 'celebdf' in need:   # only 60 test files: fetch individually instead of the whole 10 GB dataset
        import time
        for r in [r for r in missing if r['dataset'] == 'celebdf']:
            dst = f"/content/data/celebdf/{r['path']}"
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for attempt in range(6):
                subprocess.run(['kaggle', 'datasets', 'download', 'reubensuju/celeb-df-v2', '-f', r['path'], '-p', os.path.dirname(dst), '-q'])
                z = dst + '.zip'
                if os.path.exists(z):
                    subprocess.run(['unzip', '-o', '-q', z, '-d', os.path.dirname(dst)]); os.remove(z)
                if os.path.exists(dst):
                    break
                time.sleep(5 * (attempt + 1) ** 2)   # Kaggle 429 back-off
    subprocess.run(['df', '-h', '/content'])
""")
ONLY = None

if TARGET == "kaggle":
    code(r"""
# 3b. Phone face-swaps (round 4). For each phone video: its 16 production-sampled frames, each written twice, as a
# 32-frame H.264 video (the production sampler then picks exactly those 16 frames back), once unchanged (label real)
# and twice face-swapped with inswapper onto two other people of the SAME split (label fake). Real and fake go through
# the identical encode, so encoding can't be the cue. A fake is dropped if fewer than 8 of its frames got a swap.
import json as _json
PS = f'{WORK}/phoneswap'
todo = [r for r in ROWS if r['dataset'] == 'phoneswap' and not os.path.exists(f"{PS}/{r['path']}")]
print(len(todo), 'phoneswap videos to generate')
# Runs in a SEPARATE process: round 4's first try loaded onnxruntime in this notebook process, which (a) silently fell
# back to CPU (3.7 h for 358 swaps) and (b) left its thread pools in the process the crop step then forks from.
# The child imports torch first so onnxruntime-gpu finds the CUDA/cuDNN libraries torch ships, and prints the
# provider actually in use.
GEN = r'''
import json, os, random, subprocess, sys, time
job = json.load(open(sys.argv[1]))
sys.path.insert(0, job['PKG'])
import torch, numpy as np
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception as e:
    print('preload_dlls:', e)
print('onnxruntime', ort.__version__, 'providers available:', ort.get_available_providers(), flush=True)
import insightface
from insightface.app import FaceAnalysis
from app.services.video.model_v1 import optimized
prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']
fa = FaceAnalysis(name='buffalo_l', root='/kaggle/temp/insightface', providers=prov)
fa.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(job['SW'], providers=prov)
print('swapper running on', swapper.session.get_providers(), flush=True)
PS = job['PS']

def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None

def write(frames, out):
    h, w = frames[0].shape[:2]; h -= h % 2; w -= w % 2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
                          '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                          '-pix_fmt', 'yuv420p', out + '.tmp.mp4'], stdin=subprocess.PIPE)
    for f in frames:
        b = np.ascontiguousarray(f[:h, :w]).tobytes(); p.stdin.write(b); p.stdin.write(b)
    p.stdin.close(); assert p.wait() == 0, out
    os.replace(out + '.tmp.mp4', out)

src_rows = {}
for r in job['rows']:
    src_rows.setdefault(r['source'], []).append(r)
split_of = {src: rs[0]['split'] for src, rs in src_rows.items()}
person_of = {src: rs[0]['group'] for src, rs in src_rows.items()}
frames_of, donor, log = {}, {}, {}
t0 = time.time()
for k, src in enumerate(sorted(src_rows)):          # decode + one donor face per person (middle frame)
    try:
        fr = optimized._decode_selected_frames(job['files'][src])
    except Exception as e:
        print('decode failed', src, e, flush=True); continue
    frames_of[src] = fr
    if person_of[src] not in donor:
        for i in (8, 4, 12, 0, 15):
            f = biggest(fa.get(fr[i]))
            if f is not None:
                donor[person_of[src]] = (split_of[src], f); break
    if k % 50 == 0:
        print('decoded', k, '/', len(src_rows), f'{time.time() - t0:.0f}s', flush=True)
done = 0
for src, rs in sorted(src_rows.items()):
    if src not in frames_of:
        continue
    fr = frames_of[src]
    rnd = random.Random(src)
    pool = sorted(p for p, (sp, _) in donor.items() if sp == split_of[src] and p != person_of[src])
    dons = rnd.sample(pool, min(2, len(pool)))
    for r in rs:
        out = f"{PS}/{r['path']}"
        if os.path.exists(out):
            continue
        if r['label'] == '0':
            write(fr, out); continue
        j = int(r['kind'][-1])
        if j >= len(dons):
            continue
        dface = donor[dons[j]][1]; swapped, n_sw = [], 0
        for f in fr:
            t = biggest(fa.get(f))
            if t is None:
                swapped.append(f)
            else:
                swapped.append(swapper.get(f.copy(), t, dface, paste_back=True)); n_sw += 1
        log[r['key']] = {'donor': dons[j], 'frames_swapped': n_sw}
        if n_sw >= 8:
            write(swapped, out)
        done += 1
        if done % 25 == 0:
            print('swapped', done, f'{time.time() - t0:.0f}s', flush=True)
json.dump(log, open(f'{PS}/swap_log.json', 'w'), indent=1)
print('generator finished', f'{time.time() - t0:.0f}s', flush=True)
'''
if todo:
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'insightface==0.7.3', 'onnxruntime-gpu'], check=True)
        import urllib.request
        SW = '/kaggle/temp/inswapper_128.onnx'
        if not os.path.exists(SW):
            for url in ('https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
                        'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx'):
                try:
                    urllib.request.urlretrieve(url, SW); break
                except Exception as e:
                    print('download failed', url, e)
        print('inswapper sha256', hashlib.sha256(open(SW, 'rb').read()).hexdigest(), os.path.getsize(SW))
        srcs = sorted({r['source'] for r in ROWS if r['dataset'] == 'phoneswap'})
        job = {'PKG': PKG, 'SW': SW, 'PS': PS,
               'rows': [r for r in ROWS if r['dataset'] == 'phoneswap'],
               'files': {s_: os.path.join(slug_root(s_.split(':', 1)[0]), s_.split(':', 1)[1]) for s_ in srcs}}
        _json.dump(job, open('/kaggle/temp/phoneswap_job.json', 'w'))
        open('/kaggle/temp/gen_phoneswap.py', 'w').write(GEN)
        r_ = subprocess.run([sys.executable, '/kaggle/temp/gen_phoneswap.py', '/kaggle/temp/phoneswap_job.json'], timeout=3 * 3600)
        print('generator exit code', r_.returncode)
    except Exception:
        import traceback; traceback.print_exc()
        print('WARNING: phone face-swap generation failed; training continues without (some) phoneswap videos')
made = sum(os.path.exists(f"{PS}/{r['path']}") for r in ROWS if r['dataset'] == 'phoneswap')
print('phoneswap videos present:', made, 'of', sum(r['dataset'] == 'phoneswap' for r in ROWS), flush=True)
missing = [r for r in ROWS if not os.path.exists(crop_path(r))]
""")

code(r"""
# 4. Production-identical face crops -> Drive (resumable; skips anything already extracted)
import glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

if 'dataset_root' not in globals():
    def dataset_root(name):
        marker = {'rtfs': 'original_videos', 'fakeavceleb': 'RealVideo-RealAudio', 'celebdf': 'Celeb-real'}[name]
        hits = glob.glob(f'/content/data/{name}/**/{marker}', recursive=True)
        return os.path.dirname(hits[0]) if hits else f'/content/data/{name}'

if 'ROOTS' not in globals():
    ROOTS = {d: dataset_root(d) for d in {r['dataset'] for r in missing}} if missing else {}
print(ROOTS)

import cv2

def _worker_init():
    cv2.setNumThreads(1)          # avoid OpenCV thread-pool deadlocks in forked workers

def extract(r):
    out = crop_path(r)
    if os.path.exists(out):
        return 'skip'
    src = src_path(r) if 'src_path' in globals() else os.path.join(ROOTS[r['dataset']], r['path'])
    if not os.path.exists(src):
        return 'ERR missing source'
    try:
        cascade = common.get_face_cascade()
        frames = optimized._decode_selected_frames(src)
        crops, faces = [], 0
        for f in frames:
            crop_rgb, info = common.crop_face_or_center_with_info(f, cascade)
            crops.append(np.asarray(Image.fromarray(crop_rgb).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8))
            faces += info['face_bbox'] is not None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + '.tmp.png'
        Image.fromarray(np.concatenate(crops, axis=1)).save(tmp)
        os.replace(tmp, out)            # atomic: a half-written file is never mistaken for a finished one
        return f'ok faces={faces}'
    except Exception as e:
        return f'ERR {type(e).__name__}: {e}'

if missing:
    from collections import Counter
    stats = Counter()
    import multiprocessing as _mp
    # watchdog: round 4's first try hung here for 7 h with no output. If no worker returns anything for 20 min the pool
    # is abandoned and whatever is still missing is cropped in this process instead (slower, but it can't hang silently)
    with Pool(os.cpu_count(), initializer=_worker_init) as pool:
        it = pool.imap_unordered(extract, missing, chunksize=1)   # chunksize>1 returns a plain generator (no .next(timeout))
        i = 0
        while True:
            try:
                res = it.next(timeout=1200)
            except StopIteration:
                break
            except _mp.TimeoutError:
                print('WATCHDOG: no crop result for 20 min, finishing sequentially', flush=True)
                pool.terminate(); break
            stats[res.split(' ')[0]] += 1
            if i % 200 == 0:
                print(i, '/', len(missing), dict(stats), flush=True)
            i += 1
    for r in [r for r in missing if not os.path.exists(crop_path(r))]:
        if stats.get('_seq', 0) == 0:
            print('sequential pass over', sum(not os.path.exists(crop_path(x)) for x in missing), 'videos', flush=True)
        stats['_seq'] = stats.get('_seq', 0) + 1
        stats[extract(r).split(' ')[0]] += 1
    print('done', dict(stats))
errs = [r['key'] for r in ROWS if not os.path.exists(crop_path(r))]
print(len(errs), 'videos without crops (unreadable/too short; excluded from training/eval)')
""")

if TARGET == "kaggle":
    code(r"""
# 5. Crops are already on local disk
LOCAL = f'{WORK}/crops'
""")
else:
    code(r"""
# 5. Copy crops to local disk for fast training reads
LOCAL = '/content/crops_local'
subprocess.run(['rsync', '-a', f'{WORK}/crops/', f'{LOCAL}/'], check=True)
print(subprocess.run(['du', '-sh', LOCAL], capture_output=True, text=True).stdout)
""")

code(r"""
# 6. Resumable fine-tuning
import io, json, random, time
import torch, torch.nn as nn
from PIL import ImageFilter
from torch.utils.data import DataLoader, Dataset

# Round 4. Round 3 (five sources) separated deepfakes well on held-out benchmarks but left real phone videos in the
# uncertain middle, so round 4 continues from round 3's best.pt with three phone-oriented sources added (DFD, real
# phone/selfie videos, phone face-swaps generated above). The small phone sources use all 16 frames per video per
# epoch (others 8), so they carry weight next to the large benchmark sets. Same recipe otherwise.
EPOCHS, LR, BATCH, FRAMES_PER_VIDEO, SAVE_EVERY = 4, 4e-5, 64, 8, 400
FRAMES_PER_DS = {'phone': 16, 'phoneswap': 16}
LABEL_SMOOTHING = 0.1      # targets 0.05 / 0.95
CAP_PER_CLASS = 600        # per dataset per label per epoch, so no single source dominates
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = torch.tensor(common.IMAGENET_MEAN).view(3, 1, 1); STD = torch.tensor(common.IMAGENET_STD).view(3, 1, 1)
N = common.NUM_FRAMES

def items(split):
    out = []
    for r in ROWS:
        p = f"{LOCAL}/{r['key']}.png"
        if r['split'] == split and os.path.exists(p):
            out.append((p, int(r['label']), r['dataset']))
    return out

TRAIN, VAL, TEST = items('train'), items('val'), items('test')

# Note: DFDC also labels a few audio-only swaps FAKE (picture = the original). A crop-diff filter was tried on the Mac
# and dropped: plain re-encoding moves crops as much as a real swap does. Label smoothing absorbs that small noise.
from collections import Counter
print(len(TRAIN), 'train /', len(VAL), 'val /', len(TEST), 'test videos')
print('train per source/label:', sorted(Counter((d, y) for _, y, d in TRAIN).items()))
print('val   per source/label:', sorted(Counter((d, y) for _, y, d in VAL).items()))

def load_strip(p):
    a = np.asarray(Image.open(p).convert('RGB'))
    return [a[:, i * 224:(i + 1) * 224] for i in range(N)]

def degrade(img, rnd):
    # applied identically to real and fake, so "looks degraded" can't become a label shortcut
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

class Frames(Dataset):
    def __init__(self, frames): self.frames = frames
    def __len__(self): return len(self.frames)
    def __getitem__(self, k):
        p, i, y = self.frames[k]
        a = load_strip(p)[i]
        a = np.asarray(degrade(Image.fromarray(a), random.Random()))
        return to_tensor(a), torch.tensor(float(y))

def epoch_frames(ep):
    rnd = random.Random(1000 + ep)                     # deterministic per epoch -> exact mid-epoch resume
    vids = []
    for ds in sorted({d for *_, d in TRAIN}):
        for lab in (0, 1):
            pool = [t for t in TRAIN if t[2] == ds and t[1] == lab]
            vids += rnd.sample(pool, min(CAP_PER_CLASS, len(pool)))
    fr = [(p, i, y) for p, y, d in vids for i in rnd.sample(range(N), FRAMES_PER_DS.get(d, FRAMES_PER_VIDEO))]
    rnd.shuffle(fr)
    return fr

def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); pos, neg = s[y == 1], s[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))

@torch.no_grad()
def evaluate(model, its):
    model.eval(); sc = []
    for p, y, ds in its:
        x = torch.stack([to_tensor(a) for a in load_strip(p)]).to(device)
        sc.append((float(model(x).squeeze(-1).mean()), y, ds))      # video score = mean of 16 frame logits (production)
    # 'phone' is real-only, so it is scored together with the phone face-swaps ('phone+swap': real phone videos and
    # re-encoded ones vs swapped ones); 'phone_real_mean_logit' tracks how real the real phone videos look (lower = more real)
    groups = {ds: [t for t in sc if t[2] == ds] for ds in sorted({d for *_, d in sc}) if ds != 'phone'}
    if any(d == 'phone' for *_, d in sc):
        groups['phone+swap'] = [t for t in sc if t[2] in ('phone', 'phoneswap')]
    res = {k: auc([z for z, _, _ in v], [y for _, y, _ in v]) for k, v in groups.items() if len({y for _, y, _ in v}) == 2}
    res['mean'] = float(np.mean(list(res.values())))
    ph = [z for z, _, d in sc if d == 'phone']
    if ph:
        res['phone_real_mean_logit'] = float(np.mean(ph))
    return res

model = common.build_model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = (len(epoch_frames(1)) + BATCH - 1) // BATCH
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps_per_epoch * EPOCHS, pct_start=0.15)
scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
RUN = f'{WORK}/{RUN_NAME}'
os.makedirs(RUN, exist_ok=True)

if os.path.exists(f'{RUN}/latest.pt'):
    st = torch.load(f'{RUN}/latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(st['model_state_dict']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
    scaler.load_state_dict(st['scaler']); start_ep, start_step, history, best = st['epoch'], st['step'], st['history'], st['best']
    print(f'resumed at epoch {start_ep} step {start_step}, best val {best:.4f}')
else:
    model.load_state_dict(torch.load(INIT, map_location='cpu', weights_only=False)['model_state_dict']); model.to(device)
    base = evaluate(model, VAL); history = [{'epoch': 0, **base}]; best = base['mean']
    torch.save({'model_state_dict': model.state_dict(), 'epoch': 0, 'val': base}, f'{RUN}/best.pt')
    start_ep, start_step = 1, 0
    print('epoch 0 (round-3 model) val AUC', base)

def save_latest(ep, step):
    tmp = f'{RUN}/latest.pt.tmp'
    torch.save({'model_state_dict': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'scaler': scaler.state_dict(), 'epoch': ep, 'step': step, 'history': history, 'best': best}, tmp)
    os.replace(tmp, f'{RUN}/latest.pt')

loss_fn = nn.BCEWithLogitsLoss()
for ep in range(start_ep, EPOCHS + 1):
    frames = epoch_frames(ep)[start_step * BATCH:]
    dl = DataLoader(Frames(frames), batch_size=BATCH, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
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
    res = evaluate(model, VAL)
    history.append({'epoch': ep, 'train_loss': tot / max(n, 1), **res})
    print(f'== epoch {ep} val AUC {res}', flush=True)
    if res['mean'] > best:
        best = res['mean']
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep, 'val': res}, f'{RUN}/best.pt')
        print(f'   new best {best:.4f} -> best.pt', flush=True)
    start_step = 0
    save_latest(ep + 1, 0)
    json.dump(history, open(f'{RUN}/history.json', 'w'), indent=1)
print('training finished; best mean val AUC', best)
""")

code(r"""
# 7. Report (informational): deployed vs best on val and test. The authoritative test is re-run on the Mac.
deployed = common.build_model(); common.load_checkpoint(deployed, CKPT); deployed.to(device)
r3 = common.build_model(); r3.load_state_dict(torch.load(INIT, map_location='cpu', weights_only=False)['model_state_dict']); r3.to(device)
tuned = common.build_model(); tuned.load_state_dict(torch.load(f'{RUN}/best.pt', map_location='cpu', weights_only=False)['model_state_dict']); tuned.to(device)
for name, its in (('val', VAL), ('test', TEST)):
    for mname, m in (('deployed', deployed), ('round3', r3), ('round4', tuned)):
        print(name, f'{mname:9s}', evaluate(m, its), flush=True)
print(json.dumps(history, indent=1))

# per-video scores of best.pt on val (used on the Mac to fit the verdict calibration) and test, with resolution
@torch.no_grad()
def dump_scores(its, out):
    for m in (deployed, r3, tuned): m.eval()
    key_of = {f"{LOCAL}/{r['key']}.png": r for r in ROWS}
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dataset', 'label', 'key', 'mean_logit', 'short_side', 'deployed_logit', 'r3_logit', 'kind'])
        for p, y, ds in its:
            r = key_of[p]
            cap = cv2.VideoCapture(src_path(r))
            side = int(min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))); cap.release()
            x = torch.stack([to_tensor(a) for a in load_strip(p)]).to(device)
            z = [f"{float(m(x).squeeze(-1).mean()):.6f}" for m in (tuned, deployed, r3)]
            w.writerow([ds, y, r['key'], z[0], side, z[1], z[2], r['kind']])
dump_scores(VAL, f'{RUN}/val_scores.csv'); dump_scores(TEST, f'{RUN}/test_scores.csv')
print('wrote val_scores.csv / test_scores.csv')
""")

md(r"""
## When it's done

Download `MyDrive/TruthLens_video_ft/run/best.pt` and `history.json` and give them to Claude on the Mac. They get
re-tested there against the deployed model with the exact production pipeline before anything changes in the app.
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
name = "TruthLens_Video_Finetune_v2" + ("_kaggle" if TARGET == "kaggle" else "") + ".ipynb"
json.dump(nb, open(name, "w"), indent=1)
print("wrote", name, "with", len(CELLS), "cells")
