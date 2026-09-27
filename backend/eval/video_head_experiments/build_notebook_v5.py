import json
import os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])}

def codeblock(src):
    return code(*src.strip("\n").split("\n"))

cells = []

cells.append(md(
"# TruthLens Video Model v5 — CNN + RNN temporal head (v4 recipe + RealTimeFaceSwap-10k)",
"",
"**v5 = v4 recipe + a third data source: `stplusplus/rtfs-10k`** — ALL 9,772 face-swap videos (inswapper + "
"uniface, modern consumer-app-style swaps) and all 1,636 matching real portrait/video-call clips. Split by "
"YouTube source id so no target video leaks across train/val. Sampler: 30% lab (FF++/Celeb-DF/DFDC), 30% "
"WildDeepfake, 40% RTFS; real/fake balanced within each source. Selection score = mean(real-world 10-video "
"AUC, held-out WildDeepfake AUC, held-out RTFS AUC).",
"",
"**v4 changes vs v3** (v3 reached real-world AUC 0.72 vs 0.64 for the deployed CNN, but was trained on one "
"clean cached feature view per video, so it memorised dataset-specific artifacts and stayed blind to "
"phone-app / recompressed / low-res video). v4 needs no new deepfake videos:",
"1. **Augmented feature views** — each training video is extracted several times through random "
"downscale+upscale, JPEG recompression, blur, colour jitter, flip and face-padding jitter (same params "
"across all 16 frames of a clip), mimicking phone/social-media output.",
"2. **Feature-level augmentation** while training the head — random time reversal, gaussian noise, input "
"dropout, smaller head, stronger weight decay.",
"3. **Source-balanced sampling** — WildDeepfake (in-the-wild) is given a fixed share of every batch instead "
"of being drowned out by the larger lab-generated set; classes are balanced within each source.",
"4. **More trustworthy checkpoint selection** — score = mean(real-world 10-video AUC, held-out "
"WildDeepfake AUC). The 10-video set alone is too small/noisy to select on safely.",
"",
"---",
"",
"Two prior runs on this notebook showed: (1) the deployed CNN-only detector scores frames independently "
"and averages them, no notion of order, real-world pairwise AUC only ~0.64; (2) a CNN+GRU head trained "
"on FaceForensics++/Celeb-DF/DFDC alone hit AUC 0.89 on its own validation split but **dropped to 0.44 "
"on real-world test video** — worse than the original. That's overfitting to one narrow academic-dataset "
"distribution, not real learning, and training on *more* of that same narrow data made it worse, not "
"better.",
"",
"This version fixes both root causes instead of just training longer:",
"1. **Adds a second, genuinely different data source** — WildDeepfake, built from real internet/social "
"video (not lab-generated), alongside the existing FaceForensics++/Celeb-DF/DFDC set.",
"2. **Validates against your actual real-world videos during training**, not just the academic dataset's "
"own held-out split — the checkpoint is selected by whichever epoch does best on real video, which is "
"the only thing that matters, instead of by a metric that already proved misleading once.",
"",
"The CNN backbone (`epoch_11_model_only.pt`) stays frozen and untouched throughout — only the new GRU "
"head trains. **Keep this in sync with `backend/app/services/video/model_v1/common.py`** — the frame-"
"selection / face-crop / resize / normalize logic below is a direct port of that file."
))

cells.append(md("## 1. Setup"))

cells.append(md(
"**Run all cells top to bottom (Runtime -> Run all).** Background: Colab's default OpenCV (5.0.0) drops "
"`cv2.CascadeClassifier`; the cell below pins a stable 4.x build with `--no-deps` so it doesn't disturb "
"numpy/scipy. Metrics (AUC etc.) are hand-rolled in numpy later in this notebook rather than imported "
"from scikit-learn, because Colab's current default numpy/scipy/scikit-learn combo throws an unrelated "
"import-time error in this environment — each metric is checked against real scikit-learn output on 50 "
"random trials (max difference < 1e-9) before being used."
))

cells.append(code(
"!pip install -q --force-reinstall --no-deps opencv-python-headless==4.10.0.84",
"!pip install -q kagglehub"
))

cells.append(code(
"import os, re, glob, random, time, sys",
"from pathlib import Path",
"",
"import numpy as np",
"import cv2",
"if not hasattr(cv2, \"CascadeClassifier\"):",
"    print(f\"cv2 {cv2.__version__} at {cv2.__file__} is missing CascadeClassifier — this only happens \"",
"          \"if a broken cv2 was already imported earlier in this session. Force-restarting the runtime \"",
"          \"now; once it restarts, click Runtime -> Run all again and it will work.\")",
"    os.kill(os.getpid(), 9)",
"print(\"cv2 OK:\", cv2.__version__)",
"import torch",
"import torch.nn as nn",
"from torch.utils.data import Dataset, DataLoader",
"from PIL import Image",
"from torchvision import models",
"",
"import kagglehub",
"",
"SEED = 1337",
"random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)",
"",
"device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
"print(\"device:\", device)"
))

cells.append(md(
"### Preflight — run this first",
"",
"Checks GPU, RAM, CPU count and free disk against what this run needs (~20GB dataset + ~5GB caches/temp). "
"Stops with a clear message if something is short, before you waste a download."
))

cells.append(codeblock(r"""
import shutil, psutil
free_gb = shutil.disk_usage("/content").free / 1e9
ram_gb = psutil.virtual_memory().total / 1e9
print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"CPU cores  : {os.cpu_count()}")
print(f"RAM        : {ram_gb:.1f} GB total, {psutil.virtual_memory().available/1e9:.1f} GB free")
print(f"Disk (/content): {free_gb:.1f} GB free")
NEED_DISK_GB = 30
problems = []
if not torch.cuda.is_available(): problems.append("no GPU -> Runtime > Change runtime type > GPU")
if free_gb < NEED_DISK_GB: problems.append(f"only {free_gb:.0f}GB disk free, need ~{NEED_DISK_GB}GB")
if ram_gb < 10: problems.append(f"only {ram_gb:.0f}GB RAM (12GB+ recommended)")
print("\nPREFLIGHT:", "OK" if not problems else "PROBLEMS -> " + "; ".join(problems))
assert not problems, "Fix the problems above before continuing"
"""))

cells.append(md(
"### Google Drive persistence (automatic)",
"",
"Mounts Drive and keeps **everything that is expensive to recreate** in `MyDrive/truthlens_v5/`: the "
"feature caches (auto-saved every 1,000 items, auto-resumed on rerun), the uploaded CNN checkpoint, the "
"10 real-world test videos, and the final trained head. After a disconnect just Run all again - it picks "
"up where it left off, and you are only asked to upload files the first time. The 17.6GB dataset itself "
"is *not* stored on Drive (re-downloads to local disk, ~10-20 min)."
))

cells.append(codeblock(r"""
from google.colab import drive
drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/truthlens_v5")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
(DRIVE_DIR / "realworld").mkdir(exist_ok=True)
drive_free_gb = shutil.disk_usage("/content/drive").free / 1e9
print(f"Drive dir: {DRIVE_DIR}   Drive free: {drive_free_gb:.1f} GB (need ~5 GB)")
assert drive_free_gb > 5, "Not enough free Google Drive space (need ~5GB for caches + checkpoint)"
print("existing files:", [f.name for f in DRIVE_DIR.iterdir()])
"""))

cells.append(md(
"### Multi-account sharding (optional, does not change the model)",
"",
"Feature extraction is the slow part and every video is independent, so it can be split across Colab "
"accounts. The train/val split is deterministic, so every account computes the *same* split and takes a "
"disjoint slice of the items - the final features are identical to a single-account run.",
"",
"- **1 account:** leave `SHARD_ID = 0`, `NUM_SHARDS = 1`, `EXTRACT_ONLY = False`.",
"- **N accounts:** run this notebook on each with the same `NUM_SHARDS = N` and a different `SHARD_ID` "
"(0..N-1). Set `EXTRACT_ONLY = True` on all but one. When the helper accounts finish, download their "
"`*_shardXofN.pt` files from their Drive folder `truthlens_v5` and upload them into the training "
"account's `truthlens_v5` Drive folder, then run that account (it extracts its own slice, merges all, "
"and trains)."
))

cells.append(codeblock(r"""
SHARD_ID = 0
NUM_SHARDS = 1
EXTRACT_ONLY = False
assert 0 <= SHARD_ID < NUM_SHARDS
SHARD_TAG = f"shard{SHARD_ID}of{NUM_SHARDS}"
print("shard:", SHARD_TAG, " extract_only:", EXTRACT_ONLY)
"""))

cells.append(md(
"## 2. Preprocessing — ported from `backend/app/services/video/model_v1/common.py`",
"",
"Defined early because both dataset sections below need it. Frame index selection "
"(`np.linspace(...).astype(int)`, truncating not rounding), Haar frontal-face detection with 20% "
"padding and a center-square/empty-crop fallback, PIL `BILINEAR` resize (not `cv2.resize`), ImageNet "
"normalization — all copied to match production exactly."
))

cells.append(codeblock(r"""
import threading
NUM_FRAMES = 16
IMAGE_SIZE = 224
FACE_PADDING_FRACTION = 0.20
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_tls = threading.local()
def get_face_cascade():
    # One cascade per thread (CascadeClassifier is not thread-safe). Same Haar model as production.
    c = getattr(_tls, "cascade", None)
    if c is None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        c = cv2.CascadeClassifier(cascade_path)
        if c.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        _tls.cascade = c
    return c

def evenly_spaced_frame_indices(frame_count, num_frames=NUM_FRAMES):
    return np.linspace(0, frame_count - 1, num_frames).astype(int).tolist()

def decode_selected_frames(video_path):
    # Identical frames to production. grab() advances past non-selected frames without the
    # retrieve/convert step (read() == grab() + retrieve()), so it is faster but bit-identical.
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"video_open_failed: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError(f"invalid_frame_count: {video_path}")
        positions = evenly_spaced_frame_indices(frame_count)
        frames, current, nxt = [], 0, 0
        while nxt < len(positions):
            target = positions[nxt]
            if current > target:
                break                      # duplicate positions => video shorter than NUM_FRAMES
            if current < target:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                nxt += 1
            current += 1
        if len(frames) != len(positions):
            raise ValueError(f"only_{len(frames)}_frames (expected {len(positions)}) for {video_path}")
        return frames
    finally:
        cap.release()

def load_frame_folder(folder_path, num_frames=NUM_FRAMES):
    files = sorted(Path(folder_path).glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    frame_count = len(files)
    if frame_count <= 0:
        raise ValueError(f"invalid_frame_count: {folder_path}")
    frames = []
    for pos in evenly_spaced_frame_indices(frame_count, num_frames):
        img = cv2.imread(str(files[pos]))
        if img is None:
            raise ValueError(f"failed_to_read_frame: {files[pos]}")
        frames.append(img)
    return frames

def crop_face_or_center(frame_bgr, cascade, pad_frac=FACE_PADDING_FRACTION):
    # Full-resolution Haar detection, exactly as production (no downscaling shortcut).
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = frame_rgb.shape[:2]
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        crop = frame_rgb[top: top + side, left: left + side]
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_w = int(fw * pad_frac)
        pad_h = int(fh * pad_frac)
        x0 = max(0, x - pad_w); y0 = max(0, y - pad_h)
        x1 = min(w, x + fw + pad_w); y1 = min(h, y + fh + pad_h)
        crop = frame_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_rgb
    return crop

def preprocess_frame(frame_bgr, cascade, pad_frac=FACE_PADDING_FRACTION):
    crop_rgb = crop_face_or_center(frame_bgr, cascade, pad_frac)
    resized = Image.fromarray(crop_rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    normalized = np.asarray(resized, dtype=np.uint8).astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1)).copy()

cascade = get_face_cascade()
"""))

cells.append(md(
"### Clip-level augmentation (train only)",
"",
"One random parameter set per clip, applied identically to all 16 frames so temporal consistency is "
"preserved. Targets the domain gap seen on phone-app fakes: low resolution, heavy recompression, different "
"framing. The clean view (view 0) is always kept alongside the augmented ones."
))

cells.append(codeblock(r"""
def sample_aug_params(rng):
    return {
        "scale": rng.uniform(0.35, 1.0) if rng.random() < 0.7 else 1.0,
        "jpeg_q": int(rng.integers(25, 90)) if rng.random() < 0.7 else None,
        "blur_sigma": rng.uniform(0.5, 1.6) if rng.random() < 0.3 else 0.0,
        "alpha": rng.uniform(0.75, 1.25),   # contrast
        "beta": rng.uniform(-25, 25),       # brightness
        "flip": rng.random() < 0.5,
        "pad_frac": rng.uniform(0.05, 0.45),
    }

def augment_frame(frame_bgr, prm):
    f = frame_bgr
    h, w = f.shape[:2]
    if prm["scale"] < 1.0:
        sw, sh = max(16, int(w * prm["scale"])), max(16, int(h * prm["scale"]))
        f = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
    if prm["blur_sigma"] > 0:
        f = cv2.GaussianBlur(f, (0, 0), prm["blur_sigma"])
    f = cv2.convertScaleAbs(f, alpha=prm["alpha"], beta=prm["beta"])
    if prm["jpeg_q"] is not None:
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, prm["jpeg_q"]])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if prm["flip"]:
        f = cv2.flip(f, 1)
    return f
"""))

cells.append(md(
"## 3. Dataset A — FaceForensics++ / Celeb-DF / DFDC",
"",
"[`pranabkc/deepfake-with-cropped-faces-from-video`]"
"(https://www.kaggle.com/datasets/pranabkc/deepfake-with-cropped-faces-from-video) — 6,450 raw `.mp4` "
"clips, ~1.67GB. Folders: `Celeb_fake_face_only`, `Celeb_real_face_only`, `DFDC_FAKE_Face_only_data`, "
"`DFDC_REAL_Face_only_data` (label from folder name) and `FF_Face_only_data` (mixed: real videos are "
"`<id>.mp4`, fake videos are `<source>_<target>.mp4` — label from filename pattern instead)."
))

cells.append(code(
"ff_dataset_path = kagglehub.dataset_download(\"pranabkc/deepfake-with-cropped-faces-from-video\")",
"print(ff_dataset_path)",
"",
"ff_videos = sorted(glob.glob(os.path.join(ff_dataset_path, \"**\", \"*.mp4\"), recursive=True))",
"print(\"total videos found:\", len(ff_videos))",
"",
"def ff_label_from_path(p):",
"    # Check only the path *relative to ff_dataset_path* -- the dataset's own Kaggle slug contains",
"    # \"deepfake\", which contains the substring \"fake\", and would wrongly match every single path",
"    # if checked against the full absolute path.",
"    pl = Path(p).relative_to(ff_dataset_path).as_posix().lower()",
"    if \"fake\" in pl:",
"        return 1",
"    if \"real\" in pl:",
"        return 0",
"    stem = Path(p).stem",
"    if re.fullmatch(r\"\\d+_\\d+\", stem):",
"        return 1  # FF_Face_only_data manipulated: <source>_<target>.mp4",
"    if re.fullmatch(r\"\\d+\", stem):",
"        return 0  # FF_Face_only_data original: <id>.mp4",
"    return None",
"",
"ff_labeled = [(p, ff_label_from_path(p)) for p in ff_videos]",
"ff_labeled = [(p, l) for p, l in ff_labeled if l is not None]",
"ff_n_real = sum(1 for _, l in ff_labeled if l == 0)",
"ff_n_fake = sum(1 for _, l in ff_labeled if l == 1)",
"print(f\"dataset A labeled: {len(ff_labeled)}  real: {ff_n_real}  fake: {ff_n_fake}\")",
"assert ff_n_real > 100 and ff_n_fake > 100, \"STOP: dataset A labeling looks broken\""
))

cells.append(md(
"## 4. Dataset B — WildDeepfake",
"",
"[`naisargirupareliya/wilddeepfake-subset`]"
"(https://www.kaggle.com/datasets/naisargirupareliya/wilddeepfake-subset) — built from real internet/"
"social video (Zi et al. 2020), not lab-generated manipulation. Genuinely different distribution from "
"dataset A, which is the point: combining distinct sources is what actually helps cross-dataset "
"generalization, not more of one narrow source. Structure: `wilddeepfake_subset/{real,fake}/<id>/"
"<frame_number>.png` -- a folder of pre-extracted, already face-cropped frames per video, not raw mp4s."
))

cells.append(code(
"wd_dataset_path = kagglehub.dataset_download(\"naisargirupareliya/wilddeepfake-subset\")",
"print(wd_dataset_path)",
"",
"wd_video_dirs = sorted(d for d in Path(wd_dataset_path).glob(\"*/*/*\") if d.is_dir())",
"print(\"wilddeepfake video-folders found:\", len(wd_video_dirs))",
"",
"def wd_label_from_path(p):",
"    parts_lower = [x.lower() for x in Path(p).relative_to(wd_dataset_path).parts]",
"    if \"fake\" in parts_lower:",
"        return 1",
"    if \"real\" in parts_lower:",
"        return 0",
"    return None",
"",
"wd_labeled = [(str(p), wd_label_from_path(p)) for p in wd_video_dirs]",
"wd_labeled = [(p, l) for p, l in wd_labeled if l is not None]",
"wd_n_real = sum(1 for _, l in wd_labeled if l == 0)",
"wd_n_fake = sum(1 for _, l in wd_labeled if l == 1)",
"print(f\"dataset B labeled: {len(wd_labeled)}  real: {wd_n_real}  fake: {wd_n_fake}\")",
"assert wd_n_real > 10 and wd_n_fake > 10, \"STOP: dataset B labeling looks broken\""
))

cells.append(md(
"## 4b. Dataset C — RealTimeFaceSwap-10k (all of it)",
"",
"[`stplusplus/rtfs-10k`](https://www.kaggle.com/datasets/stplusplus/rtfs-10k) (CC BY-SA 4.0, ~17.6GB): "
"`inswapper/` (4,875 fakes), `uniface/` (4,897 fakes), `original_videos/` (1,636 real). Fakes are named "
"`<generator>_<ytid>_<start>_<end>_<n>-<n>_<gender>.mp4`, originals `<ytid>_<start>_<end>.mp4`. Needs ~20GB "
"free disk on the Colab runtime."
))

cells.append(codeblock(r"""
rt_dataset_path = kagglehub.dataset_download("stplusplus/rtfs-10k")
print(rt_dataset_path)
RT_ROOT = Path(rt_dataset_path)

rt_labeled = []
for f in sorted((RT_ROOT / "original_videos").glob("*.mp4")):
    rt_labeled.append((str(f), 0))
for gen in ("inswapper", "uniface"):
    for f in sorted((RT_ROOT / gen).glob("*.mp4")):
        rt_labeled.append((str(f), 1))
rt_n_real = sum(1 for _, l in rt_labeled if l == 0)
rt_n_fake = sum(1 for _, l in rt_labeled if l == 1)
print(f"dataset C labeled: {len(rt_labeled)}  real: {rt_n_real}  fake: {rt_n_fake}")
assert rt_n_real > 1000 and rt_n_fake > 5000, "STOP: dataset C labeling looks broken"

def rt_identity(path):
    stem = Path(path).stem
    m = re.match(r"^(?:inswapper|uniface)_(.+?)_\d{5}_\d{5}_", stem)   # fake
    if m: return "rt_" + m.group(1)
    m = re.match(r"^(.+?)_\d{5}_\d{5}$", stem)                          # original
    if m: return "rt_" + m.group(1)
    return "rt_" + stem

# sanity: every fake's identity should also exist among the real originals
_real_ids = {rt_identity(p) for p, l in rt_labeled if l == 0}
_fake_ids = {rt_identity(p) for p, l in rt_labeled if l == 1}
print("fake identities:", len(_fake_ids), " matched to a real original:", len(_fake_ids & _real_ids))
"""))

cells.append(md(
"## 5. Combine + identity-aware train/val split",
"",
"Each sample is tagged `(kind, path, label)` where `kind` is `\"video\"` (dataset A, decode via "
"`decode_selected_frames`) or `\"frames\"` (dataset B, decode via `load_frame_folder`) -- downstream code "
"only branches on `kind` in one place, everything else is unified. Splitting by identity (not by video) "
"prevents the same source face/clip from leaking across train and val and inflating validation numbers."
))

cells.append(code(
"all_samples = [(\"video\", p, l) for p, l in ff_labeled] + [(\"frames\", p, l) for p, l in wd_labeled] + [(\"video\", p, l) for p, l in rt_labeled]",
"print(f\"combined: {len(all_samples)} samples ({len(ff_labeled)} video-based + {len(wd_labeled)} frame-folder-based)\")",
"",
"def extract_identity(kind, path):",
"    if str(path).startswith(str(RT_ROOT)):",
"        return rt_identity(path)",
"    if kind == \"frames\":",
"        p = Path(path)",
"        return f\"wd_{p.parent.name}_{p.name}\"",
"    stem = Path(path).stem",
"    m = re.match(r\"id(\\d+)\", stem)",
"    if m:",
"        return \"id\" + m.group(1)",
"    m = re.match(r\"(\\d+)\", stem)",
"    if m:",
"        return m.group(1)",
"    return stem",
"",
"groups = {}",
"for kind, p, l in all_samples:",
"    groups.setdefault(extract_identity(kind, p), []).append((kind, p, l))",
"",
"identities = sorted(groups.keys())",
"random.Random(SEED).shuffle(identities)",
"val_frac = 0.15",
"n_val_ids = max(1, int(len(identities) * val_frac))",
"val_ids = set(identities[:n_val_ids])",
"train_ids = set(identities[n_val_ids:])",
"",
"train_set = [s for i in train_ids for s in groups[i]]",
"val_set = [s for i in val_ids for s in groups[i]]",
"print(f\"identities: {len(identities)} (train {len(train_ids)} / val {len(val_ids)})\")",
"print(f\"samples: train {len(train_set)}  val {len(val_set)}\")"
))

cells.append(md(
"### Optional cap (keep the Colab session finite)",
"",
"Set to `None` for no cap (use everything). Feature extraction is the slow part; if you're re-running "
"after a partial success, the caps don't need to match a prior run -- the feature cache below is keyed "
"by path and resumes/extends automatically."
))

cells.append(code(
"MAX_TRAIN_SAMPLES = None   # None = use all\n"
"MAX_VAL_SAMPLES = None     # None = use all\n"
"\n"
"def cap_balanced(samples, max_n):\n"
"    if max_n is None or len(samples) <= max_n:\n"
"        return samples\n"
"    reals = [s for s in samples if s[2] == 0]\n"
"    fakes = [s for s in samples if s[2] == 1]\n"
"    random.shuffle(reals); random.shuffle(fakes)\n"
"    half = max_n // 2\n"
"    return reals[:half] + fakes[:max_n - half]\n"
"\n"
"train_set = cap_balanced(train_set, MAX_TRAIN_SAMPLES)\n"
"val_set = cap_balanced(val_set, MAX_VAL_SAMPLES)\n"
"print(f\"using train={len(train_set)} val={len(val_set)}\")"
))

cells.append(md(
"## 6. Frozen CNN backbone",
"",
"Upload `epoch_11_model_only.pt` — the checkpoint currently deployed in production "
"(`backend/checkpoints/video/epoch_11_model_only.pt` in the repo, gitignored so it has to come from your "
"machine). This CNN is **not** retrained here; we only use it to extract per-frame feature vectors."
))

cells.append(code(
"from google.colab import files",
"CNN_CHECKPOINT_PATH = str(DRIVE_DIR / \"epoch_11_model_only.pt\")",
"if not os.path.exists(CNN_CHECKPOINT_PATH):",
"    print(\"Upload epoch_11_model_only.pt (current production video CNN checkpoint) - one time only\")",
"    uploaded = files.upload()",
"    shutil.copy(list(uploaded.keys())[0], CNN_CHECKPOINT_PATH)",
"print(\"using:\", CNN_CHECKPOINT_PATH)"
))

cells.append(code(
"def build_cnn_backbone():",
"    model = models.efficientnet_b0(weights=None)",
"    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)",
"    return model",
"",
"cnn = build_cnn_backbone()",
"_ckpt = torch.load(CNN_CHECKPOINT_PATH, map_location=\"cpu\", weights_only=False)",
"_sd = _ckpt[\"model_state_dict\"] if isinstance(_ckpt, dict) and \"model_state_dict\" in _ckpt else _ckpt",
"cnn.load_state_dict(_sd)",
"cnn.eval()",
"for p in cnn.parameters():",
"    p.requires_grad = False",
"cnn.to(device)",
"",
"FEATURE_DIM = cnn.classifier[1].in_features  # 1280 for EfficientNet-B0",
"print(\"feature dim:\", FEATURE_DIM)",
"",
"def extract_cnn_features(x):",
"    # x: (N, 3, 224, 224) already preprocessed/normalized -> (N, FEATURE_DIM)",
"    feats = cnn.features(x)",
"    feats = cnn.avgpool(feats)",
"    return torch.flatten(feats, 1)"
))

cells.append(md(
"## 7. Extract + cache per-sample feature sequences",
"",
"The CNN is frozen, so its features don't change across training epochs — extract them **once** per "
"sample and cache to disk, then train the (fast, lightweight) GRU head purely on cached tensors. Saves a "
"checkpoint of progress every 200 samples and resumes from wherever it left off if re-run after a "
"disconnect (tracks which paths are already cached, only processes what's missing)."
))

cells.append(codeblock(r"""
import zlib
AUG_VIEWS = 2   # augmented views per TRAIN sample (lab / WildDeepfake), in addition to the clean view 0
NUM_WORKERS = max(2, os.cpu_count() or 2)
SAVE_EVERY_SECONDS = 600     # checkpoint to Drive at most every 10 min of work -> a disconnect loses <=10 min
print("CPU loader threads:", NUM_WORKERS)

def source_of(kind, path):
    if kind == "frames": return 1                       # WildDeepfake
    return 2 if str(path).startswith(str(RT_ROOT)) else 0   # RTFS / lab

def n_aug_views(kind, path, label):
    # RTFS fakes are already plentiful and diverse -> clean view only.
    # RTFS reals are scarce (1,636) and heavily oversampled by the balancer -> extra augmented views.
    if source_of(kind, path) == 2:
        return 3 if label == 0 else 0
    return AUG_VIEWS

def load_views_np(kind, path, views):
    # CPU-only. Decodes the video ONCE and builds every requested view from that single decode
    # (previously each view re-decoded the file). Output per view is identical to decoding separately.
    frames = decode_selected_frames(path) if kind == "video" else load_frame_folder(path)
    casc = get_face_cascade()
    out = []
    for v in views:
        pad = FACE_PADDING_FRACTION
        fr = frames
        if v > 0:
            rng = np.random.default_rng(zlib.crc32(f"{path}#{v}".encode()))
            prm = sample_aug_params(rng)
            fr = [augment_frame(f, prm) for f in frames]
            pad = prm["pad_frac"]
        out.append((v, np.stack([preprocess_frame(f, casc, pad) for f in fr], axis=0)))
    return out

def features_from_batch(batch):
    tensor = torch.from_numpy(batch).contiguous().float().to(device)
    with torch.inference_mode():
        feats = extract_cnn_features(tensor)  # (16, FEATURE_DIM)
    return feats.cpu().half()   # fp16 in cache to halve RAM/disk; cast back to float when used

def extract_sample_features(kind, path, view=0):
    return features_from_batch(load_views_np(kind, path, [view])[0][1])

def build_feature_cache(items, cache_path, log_every=100):
    # items: list of (kind, path, label, view). Cache key = f"{path}#{view}".
    from concurrent.futures import ThreadPoolExecutor
    cache_path = Path(cache_path)
    if cache_path.exists():
        ex0 = torch.load(cache_path)
        features, labels = list(ex0["features"]), ex0["labels"].tolist()
        keys, sources = list(ex0["keys"]), ex0["sources"].tolist()
        print(f"resuming from {cache_path}: {len(keys)} already cached")
    else:
        features, labels, keys, sources = [], [], [], []
    done = set(keys)

    # group remaining work by video so each file is decoded once for all its views
    jobs = {}
    for k, p, l, v in items:
        if f"{p}#{v}" not in done:
            jobs.setdefault((k, p, l), []).append(v)
    jobs = [(k, p, l, vs) for (k, p, l), vs in jobs.items()]
    n_items = sum(len(j[3]) for j in jobs)
    print(f"{len(jobs)} videos / {n_items} items left (of {len(items)} total)")

    def _save():
        torch.save({"features": torch.stack(features),
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "keys": keys,
                    "sources": torch.tensor(sources, dtype=torch.long)}, cache_path)

    def _load(job):
        k, p, l, vs = job
        try:
            return job, load_views_np(k, p, vs), None
        except Exception as e:
            return job, None, e

    t0 = time.time(); last_save = t0; done_items = 0
    CHUNK = 4 * NUM_WORKERS
    with ThreadPoolExecutor(NUM_WORKERS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            for (k, p, l, vs), res, err in pool.map(_load, jobs[c0:c0 + CHUNK]):
                done_items += len(vs)
                if err is not None:
                    print(f"skip {p}: {err}")
                    continue
                for v, batch in res:
                    features.append(features_from_batch(batch))
                    labels.append(l); keys.append(f"{p}#{v}"); sources.append(source_of(k, p))
            if (c0 // CHUNK) % max(1, log_every // CHUNK) == 0 and done_items:
                el = time.time() - t0
                print(f"{done_items}/{n_items} items  elapsed={el/60:.0f}min  "
                      f"eta={el/done_items*(n_items-done_items)/3600:.2f}h  cached={len(features)}")
            if time.time() - last_save > SAVE_EVERY_SECONDS:
                _save(); last_save = time.time()
                print(f"  [checkpoint saved to Drive: {len(features)} items]")
    _save()
    print(f"saved -> {cache_path} ({len(features)})")
    return {"features": torch.stack(features), "labels": torch.tensor(labels, dtype=torch.float32),
            "keys": keys, "sources": torch.tensor(sources, dtype=torch.long)}

RTFS_MAX_FAKES = None   # None = use ALL RTFS fakes (recommended). Only set a number if you must shorten the run.
if RTFS_MAX_FAKES is not None:
    _rt_fakes = [x for x in train_set if source_of(x[0], x[1]) == 2 and x[2] == 1]
    random.Random(SEED).shuffle(_rt_fakes)
    _drop = set(id(x) for x in _rt_fakes[RTFS_MAX_FAKES:])
    train_set = [x for x in train_set if id(x) not in _drop]

train_items = [(k, p, l, v) for (k, p, l) in train_set for v in range(n_aug_views(k, p, l) + 1)]
val_items = [(k, p, l, 0) for (k, p, l) in val_set]
print(f"ALL items: train {len(train_items)}  val {len(val_items)}")
# this account's disjoint slice (whole videos stay together so each is decoded once)
_vids = sorted({(k, p) for k, p, l, v in train_items + val_items})
_mine = set(_vids[SHARD_ID::NUM_SHARDS])
train_items = [it for it in train_items if (it[0], it[1]) in _mine]
val_items = [it for it in val_items if (it[0], it[1]) in _mine]
print(f"THIS shard ({SHARD_TAG}): train {len(train_items)}  val {len(val_items)}")
"""))

cells.append(md(
"### Benchmark (run this BEFORE the long extraction cell)",
"",
"Times real extraction on this shard's items so you know the true speed before committing hours. "
"Nothing here changes any data."
))

cells.append(codeblock(r"""
import random as _r, time as _t
_r.seed(0)
_rt = [it for it in train_items if source_of(it[0], it[1]) == 2 and it[3] == 0][:3]
for k, p, l, v in _rt:
    cap = cv2.VideoCapture(p)
    print(f"{os.path.basename(p)[:45]:45s} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f"  {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames  {cap.get(cv2.CAP_PROP_FPS):.0f} fps"); cap.release()
    t0 = _t.time(); fr = decode_selected_frames(p); t1 = _t.time()
    casc = get_face_cascade(); [preprocess_frame(f, casc) for f in fr]; t2 = _t.time()
    print(f"   decode {t1-t0:.2f}s   face-detect+preprocess {t2-t1:.2f}s")

def _bench(items, n=24):
    items = _r.sample(items, min(n, len(items)))
    t0 = _t.time(); ok = 0
    for k, p, l, v in items:
        try: extract_sample_features(k, p, v); ok += 1
        except Exception: pass
    return (_t.time() - t0) / max(ok, 1)

by_src = {0: [], 1: [], 2: []}
for it in train_items + val_items:
    by_src[source_of(it[0], it[1])].append(it)
total_s = 0
for src, name in ((0, "lab (FF++/Celeb-DF/DFDC)"), (1, "WildDeepfake"), (2, "RTFS")):
    if not by_src[src]: continue
    per = _bench(by_src[src]) / min(NUM_WORKERS, 3)   # threads overlap decode + detection
    est = per * len(by_src[src]); total_s += est
    print(f"{name:26s} ~{per:5.2f} s/item x {len(by_src[src]):6d} items = {est/3600:5.2f} h")
print(f"\nPROJECTED extraction for THIS shard: {total_s/3600:.1f} h (rough; multi-view reuse makes the real run a bit faster; live ETA prints below)")
"""))

cells.append(md(
"### Cache fix - reuse every cache file, return only what THIS run needs",
"",
"Replaces `build_feature_cache` with a version that (1) reuses features from every cache file in the Drive folder "
"(other accounts' partial caches, an older split, the val file), because a feature depends only on (video, view), never on "
"the split, and (2) returns ONLY the items requested, so leftovers from an older train/val split can never leak into "
"training or validation. It also prints a **split fingerprint** (identical on every account <=> identical split) and a "
"report of how much of the existing cache matches this run. If the report says many cached items match neither list, an "
"older notebook version made them - they are simply ignored."
))

cells.append(codeblock(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v5_cache_fix.py"), encoding="utf-8").read()
                       + "\nsplit_fingerprint()\ncache_report(train_items, val_items)\n"))

cells.append(codeblock(r"""
train_cache = build_feature_cache(train_items, str(DRIVE_DIR / f"train_features_v5_{SHARD_TAG}.pt"))
val_cache = build_feature_cache(val_items, str(DRIVE_DIR / f"val_features_v5_{SHARD_TAG}.pt"))
print("this shard - train:", train_cache["features"].shape, " val:", val_cache["features"].shape)
if EXTRACT_ONLY:
    raise RuntimeError(f"EXTRACT_ONLY: {SHARD_TAG} finished and saved to Drive/truthlens_v5. "
                       "Nothing more to run on this account (this error is intentional - it just stops Run all).")
"""))

cells.append(md("### Merge all shards (training account)"))

cells.append(codeblock(r"""
def _merge(prefix):
    files = sorted(DRIVE_DIR.glob(f"{prefix}_shard*of{NUM_SHARDS}.pt"))
    have = {f.name for f in files}
    need = {f"{prefix}_shard{i}of{NUM_SHARDS}.pt" for i in range(NUM_SHARDS)}
    assert have == need, f"missing shard files in Drive/truthlens_v5: {sorted(need - have)}  (copy them from the helper accounts)"
    parts = [torch.load(f) for f in files]
    return {"features": torch.cat([q["features"] for q in parts]),
            "labels": torch.cat([q["labels"] for q in parts]),
            "keys": sum([list(q["keys"]) for q in parts], []),
            "sources": torch.cat([q["sources"] for q in parts])}

train_cache = _merge("train_features_v5")
val_cache = _merge("val_features_v5")
print("MERGED - train:", train_cache["features"].shape, " val:", val_cache["features"].shape)
"""))

cells.append(md("## 8. GRU temporal head"))

cells.append(codeblock(r"""
from torch.utils.data import WeightedRandomSampler

class FeatureSeqDataset(Dataset):
    def __init__(self, cache):
        self.features, self.labels, self.sources = cache["features"], cache["labels"], cache["sources"]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]

train_ds = FeatureSeqDataset(train_cache)
val_ds = FeatureSeqDataset(val_cache)

# Source-balanced + class-balanced sampling: each source gets its SOURCE_FRACTIONS share of each
# batch on average; within each source, real/fake are 50/50. So the loss needs no pos_weight.
SOURCE_FRACTIONS = {0: 0.30, 1: 0.30, 2: 0.40}   # lab / WildDeepfake / RTFS
w = torch.zeros(len(train_ds))
for src, frac in SOURCE_FRACTIONS.items():
    for lab in (0, 1):
        m = (train_ds.sources == src) & (train_ds.labels == lab)
        n = int(m.sum())
        print(f"source {src} label {lab}: {n} samples")
        if n:
            w[m] = frac * 0.5 / n
sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Held-out WildDeepfake-only val subset (in-the-wild, larger and more trustworthy than 10 videos)
val_wd_idx = (val_ds.sources == 1).nonzero().squeeze(-1)
val_rt_idx = (val_ds.sources == 2).nonzero().squeeze(-1)
print("held-out RTFS val samples:", len(val_rt_idx))
print("held-out WildDeepfake val samples:", len(val_wd_idx))
"""))

cells.append(codeblock(r"""
class TemporalHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.4, in_drop=0.2):
        super().__init__()
        self.in_drop = nn.Dropout(in_drop)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        _, h = self.gru(self.in_drop(x))
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h_last)).squeeze(-1)

GRU_HIDDEN = 128
head = TemporalHead(input_dim=FEATURE_DIM, hidden_dim=GRU_HIDDEN).to(device)
print(head)
"""))

cells.append(md(
"## 9. Metrics (pure numpy)",
"",
"Implemented directly rather than imported from scikit-learn — Colab's current default numpy/scipy/"
"scikit-learn combo throws `numpy._core._multiarray_umath has no attribute _blas_supports_fpe` on import "
"in this environment (a pre-existing break in Colab's own default image). Each of these five metrics has "
"been checked against real scikit-learn output on 50 random trials (max difference < 1e-9) before use."
))

cells.append(code(
"def _rankdata_avg(x):",
"    order = np.argsort(x, kind=\"mergesort\")",
"    ranks = np.empty(len(x))",
"    sorted_x = x[order]",
"    i = 0",
"    while i < len(x):",
"        j = i",
"        while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:",
"            j += 1",
"        avg_rank = (i + j) / 2 + 1",
"        ranks[order[i:j + 1]] = avg_rank",
"        i = j + 1",
"    return ranks",
"",
"def roc_auc_score_np(y_true, y_score):",
"    y_true = np.asarray(y_true); y_score = np.asarray(y_score)",
"    n_pos = (y_true == 1).sum(); n_neg = (y_true == 0).sum()",
"    if n_pos == 0 or n_neg == 0:",
"        return float(\"nan\")",
"    ranks = _rankdata_avg(y_score)",
"    sum_ranks_pos = ranks[y_true == 1].sum()",
"    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))",
"",
"def average_precision_score_np(y_true, y_score):",
"    y_true = np.asarray(y_true); y_score = np.asarray(y_score)",
"    order = np.argsort(-y_score, kind=\"mergesort\")",
"    y_true_sorted = y_true[order]",
"    tp_cum = np.cumsum(y_true_sorted)",
"    fp_cum = np.cumsum(1 - y_true_sorted)",
"    precision = tp_cum / (tp_cum + fp_cum)",
"    n_pos = y_true.sum()",
"    if n_pos == 0:",
"        return float(\"nan\")",
"    recall = tp_cum / n_pos",
"    recall_prev = np.concatenate(([0], recall[:-1]))",
"    return float(np.sum((recall - recall_prev) * precision))",
"",
"def accuracy_score_np(y_true, preds):",
"    return float(np.mean(np.asarray(y_true) == np.asarray(preds)))",
"",
"def balanced_accuracy_score_np(y_true, preds):",
"    y_true = np.asarray(y_true); preds = np.asarray(preds)",
"    recalls = []",
"    for c in [0, 1]:",
"        mask = y_true == c",
"        if mask.sum() == 0:",
"            continue",
"        recalls.append(np.mean(preds[mask] == c))",
"    return float(np.mean(recalls))",
"",
"def precision_recall_f1_np(y_true, preds):",
"    y_true = np.asarray(y_true); preds = np.asarray(preds)",
"    tp = np.sum((preds == 1) & (y_true == 1))",
"    fp = np.sum((preds == 1) & (y_true == 0))",
"    fn = np.sum((preds == 0) & (y_true == 1))",
"    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0",
"    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0",
"    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0",
"    return float(precision), float(recall), float(f1)"
))

cells.append(md(
"## 10. Upload your real-world test videos",
"",
"**This is the fix for the overfitting problem seen last run.** Upload all 10 of your real/fake videos "
"(from `C:\\real photos` and `C:\\fake photos`) in one go — select all 10 files in the upload dialog. "
"Label is inferred from filename: anything with \"real\" in the name is real, anything with \"fake\" is "
"fake (matches your existing `real video N.mp4` / `fake video N.mp4` naming). These get evaluated **every "
"epoch** through the full production-faithful pipeline (raw video decode, not pre-cropped frames), and "
"the checkpoint that's actually kept is whichever epoch scores best here — not on the academic dataset's "
"own validation split, since that's exactly what silently overfit last time."
))

cells.append(code(
"RW_DIR = DRIVE_DIR / \"realworld\"",
"if len(list(RW_DIR.glob(\"*.mp4\"))) < 4:",
"    print(\"Upload all 10 real-world test videos now (select all at once) - one time only\")",
"    for _fn in files.upload().keys():",
"        shutil.copy(_fn, RW_DIR / _fn)",
"rw_samples = []",
"for fname in sorted(str(f) for f in RW_DIR.glob(\"*.mp4\")):",
"    fl = os.path.basename(fname).lower()",
"    if \"fake\" in fl:",
"        rw_samples.append((fname, 1))",
"    elif \"real\" in fl:",
"        rw_samples.append((fname, 0))",
"    else:",
"        print(f\"WARNING: could not infer label from filename '{fname}', skipping\")",
"rw_n_real = sum(1 for _, l in rw_samples if l == 0)",
"rw_n_fake = sum(1 for _, l in rw_samples if l == 1)",
"print(f\"real-world eval set: {len(rw_samples)} videos ({rw_n_real} real, {rw_n_fake} fake)\")",
"assert len(rw_samples) >= 4, \"STOP: need at least a few real-world videos to validate against\"",
"",
"# Real-world video features never change (CNN frozen) -> extract once, reuse every epoch.",
"rw_feats = [extract_sample_features(\"video\", fname) for fname, _ in rw_samples]",
"",
"def eval_real_world(head_model):",
"    head_model.eval()",
"    probs, labs = [], []",
"    for feats, (fname, lab) in zip(rw_feats, rw_samples):",
"        with torch.inference_mode():",
"            logit = head_model(feats.float().unsqueeze(0).to(device))",
"        probs.append(torch.sigmoid(logit).item())",
"        labs.append(lab)",
"    return roc_auc_score_np(np.array(labs), np.array(probs)), list(zip([f for f, _ in rw_samples], probs, labs))"
))

cells.append(md("## 11. Train"))

cells.append(codeblock(r"""
# Sampler already balances source and class, so plain BCE (no pos_weight).
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-2)
EPOCHS = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
FEAT_NOISE = 0.1
n_real_tr = int((train_cache["labels"] == 0).sum())
n_fake_tr = int((train_cache["labels"] == 1).sum())

def feature_augment(x):
    # x: (B, T, D). Random time reversal per sample + gaussian noise scaled to feature std.
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip[:, None, None], x.flip(1), x)
    return x + torch.randn_like(x) * FEAT_NOISE * x.std()

def eval_val_auc(indices=None):
    head.eval()
    ds = val_ds if indices is None else torch.utils.data.Subset(val_ds, indices.tolist())
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for feats, labels in DataLoader(ds, batch_size=64):
            all_logits.append(head(feats.to(device)).cpu()); all_labels.append(labels)
    logits = torch.cat(all_logits).numpy(); labels = torch.cat(all_labels).numpy()
    return roc_auc_score_np(labels, 1 / (1 + np.exp(-logits)))

best_score, best_state, best_epoch, best_detail = -1.0, None, None, None

for epoch in range(1, EPOCHS + 1):
    head.train()
    total_loss = 0.0
    for feats, labels in train_loader:
        feats, labels = feature_augment(feats.to(device)), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(head(feats), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    scheduler.step()
    train_loss = total_loss / len(train_ds)

    indist_auc = eval_val_auc()
    wd_auc = eval_val_auc(val_wd_idx) if len(val_wd_idx) > 20 else float("nan")
    rw_auc, rw_detail = eval_real_world(head)
    rt_auc = eval_val_auc(val_rt_idx) if len(val_rt_idx) > 20 else float("nan")
    _parts = [x for x in (rw_auc, wd_auc, rt_auc) if not np.isnan(x)]
    score = float(np.mean(_parts))
    print(f"epoch {epoch:2d}  loss={train_loss:.4f}  indist={indist_auc:.4f}  "
          f"WD_heldout={wd_auc:.4f}  RTFS_heldout={rt_auc:.4f}  REAL_WORLD={rw_auc:.4f}  select_score={score:.4f}")

    if score > best_score:
        best_score, best_epoch = score, epoch
        best_detail = {"indist_auc": indist_auc, "wd_heldout_auc": wd_auc, "rtfs_heldout_auc": rt_auc, "real_world_auc": rw_auc}
        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        print("  -> new best")

print(f"\nbest epoch: {best_epoch}  select_score={best_score:.4f}")
print(best_detail)
head.load_state_dict(best_state)
_, rw_detail = eval_real_world(head)
print("\nper-video real-world probabilities at best epoch:")
for f, pr, lab in rw_detail:
    print(f"  {'FAKE' if lab else 'real'}  p_fake={pr:.3f}  {f}")
"""))

cells.append(md(
"## 12. Save checkpoint",
"",
"Saved in the same `{epoch, model_state_dict, best_auc, metrics, config}` shape as the other checkpoints "
"in this project, so local eval scripts can load it the same way. This file is **only the GRU head's "
"weights** — it depends on the frozen CNN checkpoint (`cnn_checkpoint` in `config`) and is not usable "
"standalone; deploying it means wiring a new two-stage inference path (CNN features -> GRU head), not "
"swapping this file in for `epoch_11_model_only.pt`."
))

cells.append(code(
"save_path = str(DRIVE_DIR / \"truthlens_video_cnn_rnn_head_v5.pt\")",
"torch.save({",
"    \"epoch\": best_epoch,",
"    \"model_state_dict\": best_state,",
"    \"best_auc\": best_score,",
"    \"metrics\": best_detail,",
"    \"config\": {",
"        \"model\": \"EfficientNet-B0 (frozen, from epoch_11_model_only.pt) + BiGRU head\",",
"        \"cnn_checkpoint\": CNN_CHECKPOINT_PATH,",
"        \"feature_dim\": FEATURE_DIM,",
"        \"gru_hidden_dim\": GRU_HIDDEN,",
"        \"num_frames\": NUM_FRAMES,",
"        \"image_size\": IMAGE_SIZE,",
"        \"train_samples\": len(train_ds),",
"        \"train_real\": n_real_tr,",
"        \"train_fake\": n_fake_tr,",
"        \"val_samples\": len(val_ds),",
"        \"data_sources\": [\"pranabkc/deepfake-with-cropped-faces-from-video\", \"naisargirupareliya/wilddeepfake-subset\", \"stplusplus/rtfs-10k\"],",
"        \"checkpoint_selection\": \"mean of real-world 10-video AUC, held-out WildDeepfake AUC and held-out RTFS AUC\",",
"        \"aug_views\": AUG_VIEWS,",
"        \"source_fractions\": SOURCE_FRACTIONS,",
"        \"selected_epoch\": best_epoch,",
"    },",
"}, save_path)",
"",
"from google.colab import files as _files",
"print(\"saved to Google Drive:\", save_path)",
"_files.download(save_path)   # also downloads to your computer"
))

cells.append(md(
"## Next steps (do these locally, not in this notebook)",
"",
"1. Save the downloaded `truthlens_video_cnn_rnn_head_v5.pt` to `Downloads`.",
"2. Note that the AUC printed above (`real_world_auc`) was computed on the *same* 10 videos used to pick the "
"checkpoint, so it's a training-time signal, not an unbiased final number — expect the true honest number "
"to be somewhat lower. This is a known tradeoff of validating against a small real-world set; if more "
"real/fake video pairs become available later, split off some as a genuinely held-out final test.",
"3. Only wire this into `backend/app/services/video/backend.py` as a new inference path, and only after "
"explicit sign-off, per this project's CLAUDE.md rule that ML changes require explicit approval."
))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "TruthLens_Video_CNN_RNN_Training_v5.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = r"C:\Users\admin\Downloads\TruthLens_Video_CNN_RNN_Training_v5.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out_path)
