"""Consumer-style face swaps on CelebA STILL photos (inswapper_128), run as a SEPARATE process (round 4 learned that loading
onnxruntime in the notebook process silently fell back to CPU and left thread pools behind).

job.json: {"sw": path to inswapper_128.onnx, "out": output dir, "python_path": [...], "seed": int, "gpu": bool,
           "splits": {"train": [{"id","path","identity"}...], "val": [...], "test": [...]}, "targets": {"train": N, ...},
           "upsample_prob": 0.5}
For each target photo: the untouched original (label real) and the same photo with its largest face replaced by a donor face
from a DIFFERENT identity of the SAME split (label fake). Both are written with the same random JPEG quality (88-98), and
with the same optional 1.5-2.5x bicubic upscale, so neither size nor encoding is a cue. Resumable: finished pairs are skipped.
Output: <out>/<split>/{real,fake}/<id>.jpg and <out>/meta.csv (split,id,identity,donor_identity,upscale,jpeg_q).
Mirrors image_retrain/gen_photo_swaps.py (the local version used for the first consumer-swap test)."""
import csv, json, os, random, sys, time

job = json.load(open(sys.argv[1]))
for p in job.get("python_path", []):
    sys.path.insert(0, p)
import cv2
import numpy as np
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception as e:
    print("preload_dlls:", e)
import insightface
from insightface.app import FaceAnalysis

prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if job.get("gpu") else ["CPUExecutionProvider"]
print("onnxruntime", ort.__version__, "available:", ort.get_available_providers(), flush=True)
fa = FaceAnalysis(name="buffalo_l", root=job.get("insightface_root", os.path.expanduser("~/.insightface")), providers=prov)
fa.prepare(ctx_id=0 if job.get("gpu") else -1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(job["sw"], providers=prov)
print("swapper providers:", swapper.session.get_providers(), flush=True)

OUT = job["out"]
big = lambda fs: max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if fs else None
meta_path = f"{OUT}/meta.csv"
os.makedirs(OUT, exist_ok=True)
new_meta = not os.path.exists(meta_path)
meta = open(meta_path, "a", newline="")
mw = csv.writer(meta)
if new_meta:
    mw.writerow(["split", "id", "identity", "donor_identity", "upscale", "jpeg_q"])

t0 = time.time()
for split, items in job["splits"].items():
    for d in ("real", "fake"):
        os.makedirs(f"{OUT}/{split}/{d}", exist_ok=True)
    want = job["targets"][split]
    rnd = random.Random(f"{job['seed']}-{split}")
    items = list(items); rnd.shuffle(items)
    # donor pool: faces from the first photos of the split (one embedding is enough; the swap only uses the identity vector)
    pool = []
    for it in items[: max(200, want // 4)]:
        im = cv2.imread(it["path"])
        f = big(fa.get(im)) if im is not None else None
        if f is not None:
            pool.append((it["identity"], f))
    print(split, "donor pool", len(pool), "identities", len({i for i, _ in pool}), flush=True)
    made = 0
    for it in items:
        if made >= want:
            break
        base = it["id"]
        if os.path.exists(f"{OUT}/{split}/fake/{base}.jpg") and os.path.exists(f"{OUT}/{split}/real/{base}.jpg"):
            made += 1
            continue
        im = cv2.imread(it["path"])
        if im is None:
            continue
        up = 1.0
        if rnd.random() < job.get("upsample_prob", 0.5):
            up = rnd.uniform(1.5, 2.5)
            im = cv2.resize(im, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        tf = big(fa.get(im))
        if tf is None:
            continue
        cands = [(i, f) for i, f in pool if i != it["identity"]]
        if not cands:
            continue
        did, dface = rnd.choice(cands)
        sw = swapper.get(im.copy(), tf, dface, paste_back=True)
        q = rnd.randint(88, 98)
        # atomic writes (temp name + rename): a backup tar taken while this runs can never contain a half-written jpg, and the
        # pair counts as done only when BOTH files exist (real is written last)
        for kind, arr in (("fake", sw), ("real", im)):
            tmp = f"{OUT}/{split}/{kind}/{base}.tmp.jpg"
            cv2.imwrite(tmp, arr, [cv2.IMWRITE_JPEG_QUALITY, q])
            os.replace(tmp, f"{OUT}/{split}/{kind}/{base}.jpg")
        mw.writerow([split, base, it["identity"], did, f"{up:.2f}", q]); meta.flush()
        made += 1
        if made % 100 == 0:
            print(split, "pairs", made, "/", want, f"{time.time() - t0:.0f}s", flush=True)
    print(split, "done:", made, "pairs", flush=True)
print("generator finished", f"{time.time() - t0:.0f}s", flush=True)
