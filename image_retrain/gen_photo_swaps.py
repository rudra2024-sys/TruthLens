"""Consumer-style face swaps on STILL photos, made with inswapper_128 (the engine behind roop/FaceFusion-style apps and
many Telegram/web face-swap bots). For each target photo: its untouched original (label real) and the same photo with
its largest face replaced by a donor face from a DIFFERENT photo (label fake), both written as JPEG q95 so encoding is
not a cue. Purpose: the missing test set for 'photos from face-swap apps' (CLAUDE.md section 28) and, later, training data.
Run in an isolated env (insightface==0.7.3, onnxruntime, opencv, numpy<2), not the backend venv:
  python gen_photo_swaps.py <inswapper_128.onnx> <out_dir> <n> <src_dir> [<src_dir> ...]"""
import os, random, sys
import cv2, numpy as np
import insightface
from insightface.app import FaceAnalysis

sw_path, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); srcs = sys.argv[4:]
for d in ("real", "fake"):
    os.makedirs(f"{out}/{d}", exist_ok=True)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); fa.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(sw_path, providers=["CPUExecutionProvider"])
files = []
for s in srcs:
    fl = sorted(f for f in os.listdir(s) if f.lower().endswith((".jpg", ".jpeg", ".png")) and f.startswith("real"))
    random.Random(1).shuffle(fl); files += [os.path.join(s, f) for f in fl[: n]]
random.Random(2).shuffle(files)
big = lambda fs: max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if fs else None
faces = {}
for p in files:
    im = cv2.imread(p); f = big(fa.get(im)) if im is not None else None
    if f is not None:
        faces[p] = (im, f)
paths = sorted(faces); made = 0
for i, p in enumerate(paths):
    if made >= n:
        break
    donor_p = paths[(i + len(paths) // 2) % len(paths)]
    if donor_p == p:
        continue
    im, tf = faces[p]; _, df = faces[donor_p]
    sw = swapper.get(im.copy(), tf, df, paste_back=True)
    base = f"{made:04d}_{os.path.basename(os.path.dirname(p))}_{os.path.splitext(os.path.basename(p))[0]}"
    cv2.imwrite(f"{out}/fake/{base}.jpg", sw, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{out}/real/{base}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    made += 1
    if made % 20 == 0:
        print("made", made, flush=True)
print("done", made)
