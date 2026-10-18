"""Sample a reduced training set for the Mac run (round 1, "local"): reads each Kaggle zip over HTTP range requests
(no full downloads, ~12 GB free disk) and writes image_retrain/work/local_data/** + local_manifest.csv.

Normalisation: files larger than 1024 px on the long side are downscaled (LANCZOS); PNGs (and ALL of SUT, both classes)
are re-encoded as JPEG q92 so file format is not a class cue; small JPEGs are copied byte-for-byte.
usage: python fetch_local_data.py <urls.env> <index_dir>   (indexes made by remote_zip.py)"""
import csv, io, os, random, sys, threading, time, urllib.request, zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
URLS = dict(l.rstrip("\n").split("=", 1) for l in open(sys.argv[1]))
IDX = sys.argv[2]
OUT = os.path.join(HERE, "work", "local_data")
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(2026)


def idx(name):
    return [r["name"] for r in csv.DictReader(open(os.path.join(IDX, name + ".csv")))]


def take(lst, n):
    lst = sorted(lst); rnd.shuffle(lst); return lst[:n]


class R(io.RawIOBase):
    def __init__(s, url):
        s.url = url; h = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})).headers
        s.size = int(h["Content-Range"].split("/")[1]); s.pos = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = o if w == 0 else s.pos + o if w == 1 else s.size + o; return s.pos
    def readinto(s, b):
        n = min(len(b), s.size - s.pos)
        if n <= 0: return 0
        for a in range(4):
            try:
                d = urllib.request.urlopen(urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{s.pos + n - 1}"}), timeout=60).read(); break
            except Exception:
                time.sleep(1 + a)
        else:
            raise IOError("range read failed")
        b[:len(d)] = d; s.pos += len(d); return len(d)


_tls = threading.local()


def zf(ds):
    z = getattr(_tls, "z", None) or {}
    if ds not in z:
        z[ds] = zipfile.ZipFile(io.BufferedReader(R(URLS[ds]), 128 * 1024)); _tls.z = z
    return z[ds]


def save(ds, member, dest, force_jpeg=False):
    data = zf(ds).read(member)
    ext = member.rsplit(".", 1)[-1].lower()
    im = Image.open(io.BytesIO(data)); im.load()
    big = max(im.size) > 1024
    if ext in ("jpg", "jpeg") and not big and not force_jpeg:
        open(dest, "wb").write(data); return
    im = im.convert("RGB")
    if big:
        s = 1024 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dest, "JPEG", quality=92)


JOBS = []   # (ds, member, relpath, force_jpeg)
ROWS = []   # manifest rows


def want(ds, member, group, sub, split, fake, force_jpeg=False):
    rel = f"{group}/{split}/{'fake' if fake else 'real'}_{len(JOBS):06d}.jpg"
    JOBS.append((ds, member, rel, force_jpeg))
    ROWS.append({"path": rel, "fake": int(fake), "group": group, "sub": sub, "split": split,
                 "fmt": member.rsplit(".", 1)[-1].lower()})


# ---- primary sources from manifest_r1 (group-aware splits already decided) ----
man = list(csv.DictReader(open(os.path.join(HERE, "manifest_r1.csv"))))
bucket = defaultdict(list)
for r in man:
    bucket[(r["dataset"], r["split"], r["label"], r["source"])].append(r["path"])


def pick(ds, zipname, split_to_count, force=False, key_fn=None):
    """split_to_count: {(split, fake): [(source or None, n)]}"""
    for (split, fake), plan in split_to_count.items():
        for src, n in plan:
            pool = []
            for (d, sp, lab, s), paths in bucket.items():
                if d == ds and sp == split and int(lab) == fake and (src is None or s == src):
                    pool += [(p, s) for p in paths]
            chosen = take(pool, n)
            for p, s in chosen:
                want(zipname, p, ds, s, split if split != "test_gen" else "test_gen", bool(fake), force)


GEN6 = ["BigGAN", "Midjourney", "VQDM", "glide", "stable_diffusion_v_1_5", "wukong"]
pick("genimage", "genimage", {("train", 1): [(g, 700) for g in GEN6], ("train", 0): [(None, 2500)],
                              ("val", 1): [(None, 150)], ("val", 0): [(None, 150)],
                              ("test", 1): [(None, 400)], ("test", 0): [(None, 400)], ("test_gen", 1): [(None, 600)]})
pick("sut", "sut", {("train", 1): [(None, 1800)], ("train", 0): [(None, 1800)], ("val", 1): [(None, 150)], ("val", 0): [(None, 150)],
                    ("test", 1): [(None, 300)], ("test", 0): [(None, 300)]}, force=True)
FF = ["ffpp_deepfakes", "ffpp_face2face", "ffpp_faceshifter", "ffpp_faceswap", "ffpp_neuraltextures"]
dff_plan = {("train", 1): [(f, 400) for f in FF] + [("cdfv2", 800)], ("train", 0): [("ffpp_original", 1200), ("cdfv2", 1200)]}
pick("dff", "dff", dff_plan)
# dff has no true test split (frame leakage, see README): val rows -> 150/class for selection, 300/class disjoint "test"
dv = {fk: take([(p, s) for (d, sp, lab, s), ps in bucket.items() if d == "dff" and sp == "val" and int(lab) == fk for p in ps], 450) for fk in (0, 1)}
for fk in (0, 1):
    for i, (p, s) in enumerate(dv[fk]):
        want("dff", p, "dff", s, "val" if i < 150 else "test", bool(fk))

# ---- rehearsal sources ----
def rehearsal(ds, pools, tr, va, te):
    for fake, (trp, hop) in pools.items():
        for p in take(trp, tr): want(ds, p, ds, "fake" if fake else "real", "train", fake)
        ho = take(hop, va + te)
        for i, p in enumerate(ho): want(ds, p, ds, "fake" if fake else "real", "val" if i < va else "test", fake)


c = idx("cifake")
rehearsal("cifake", {True: ([n for n in c if n.startswith("train/FAKE/")], [n for n in c if n.startswith("test/FAKE/")]),
                     False: ([n for n in c if n.startswith("train/REAL/")], [n for n in c if n.startswith("test/REAL/")])}, 800, 150, 150)
f = idx("faces140k"); B = "real_vs_fake/real-vs-fake/"
rehearsal("faces140k", {True: ([n for n in f if n.startswith(B + "train/fake/")], [n for n in f if n.startswith(B + "test/fake/")]),
                        False: ([n for n in f if n.startswith(B + "train/real/")], [n for n in f if n.startswith(B + "test/real/")])}, 800, 150, 150)
d = idx("deepdetect")
rehearsal("deepdetect", {True: ([n for n in d if n.startswith("ddata/train/fake/")], [n for n in d if n.startswith("ddata/test/fake/")]),
                         False: ([n for n in d if n.startswith("ddata/train/real/")], [n for n in d if n.startswith("ddata/test/real/")])}, 600, 150, 150)

# ai-vs-human: csv label direction is resolved with the deployed v2 model (it scored ~99% on this source)
import torch, torch.nn as nn
from torchvision.models import convnext_tiny
from torchvision import transforms
avh_csv = list(csv.DictReader(io.StringIO(zf("avh").read("train.csv").decode())))
by = defaultdict(list)
for r in avh_csv: by[int(r["label"])].append(r["file_name"])
m = convnext_tiny(weights=None); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
ck = torch.load(os.path.join(HERE, "..", "models", "checkpoints", "image", "convnext_tiny_diversified_v2.pth"), map_location="cpu")
m.load_state_dict(ck["model_state_dict"]); m.eval()
tf = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
means = {}
for lab in (0, 1):
    ps = []
    for fn in take(by[lab], 40):
        im = Image.open(io.BytesIO(zf("avh").read(fn))).convert("RGB")
        with torch.inference_mode(): ps.append(torch.softmax(m(tf(im)[None]), 1)[0, 0].item())
    means[lab] = sum(ps) / len(ps)
print("avh label check: mean P(fake) by csv label", means)
assert abs(means[1] - means[0]) > 0.5, "avh label direction ambiguous"
fk_label = 1 if means[1] > means[0] else 0
for fake, lab in ((True, fk_label), (False, 1 - fk_label)):
    pool = take(by[lab], 800 + 300)
    for i, p in enumerate(pool):
        want("avh", p, "avh", "fake" if fake else "real", "train" if i < 800 else "val" if i < 950 else "test", fake)

print(len(JOBS), "images to fetch")
done = [0]; t0 = time.time(); errs = []


def run(j):
    ds, member, rel, force = j
    dest = os.path.join(OUT, rel); os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest): return True
    try:
        save(ds, member, dest, force); return True
    except Exception as e:
        errs.append((member, repr(e))); return False


with ThreadPoolExecutor(14) as ex:
    ok = []
    for i, r in enumerate(ex.map(run, JOBS)):
        ok.append(r)
        if (i + 1) % 1000 == 0: print(f"  {i + 1}/{len(JOBS)} {time.time() - t0:.0f}s errors={len(errs)}", flush=True)
rows = [r for r, o in zip(ROWS, ok) if o]
with open(os.path.join(HERE, "local_manifest.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, ["path", "fake", "group", "sub", "split", "fmt"]); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows;", len(errs), "failed", errs[:3], f"{time.time() - t0:.0f}s")
