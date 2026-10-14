"""Score the round-3 final-test videos with a checkpoint through production predict (resumable: skips files already
in out.csv). None of these videos, nor any person in them, is in round 3's training or validation data.
  faceforensics  work/ff_test           (60; identities held out of training)
  celebdf        work/celebdf_test_r3   (60; persons held out of training - round 2's Celeb-DF test was not)
  dfdc           work/dfdc_test         (DFDC part 9; never attached for training)
  rtfs/fakeavceleb  step-1 test split   (161)
  my_videos      the user's videos + collected fakes (minus the two Celeb-DF clips of a training identity)
usage (from backend/): ../.venv/bin/python ../video_retrain/score_r3.py <checkpoint> <out.csv>"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from app.services.video.model_v1 import optimized  # noqa: E402
import av  # noqa: E402

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
ck, out = sys.argv[1], sys.argv[2]
EXCLUDE = {"faceswap_celebdf_id0_id16_0001.mp4", "faceswap_celebdf_id0_id1_0002.mp4"}   # id0 is a training identity
items = []
for lab in ("real", "fake"):
    for f in sorted(os.listdir(f"{W}/ff_test/{lab}")):
        items.append(("faceforensics", int(lab == "fake"), f"ff_test/{lab}/{f}"))
    for f in sorted(os.listdir(f"{W}/my_videos/{lab}")):
        if not f.startswith(".") and f not in EXCLUDE:
            items.append(("my_videos", int(lab == "fake"), f"my_videos/{lab}/{f}"))
for r in csv.DictReader(open(f"{W}/celebdf_test_r3_list.csv")):
    items.append(("celebdf", int(r["label"]), f"celebdf_test_r3/videos/{r['file'].replace('/', '_')}"))
for r in csv.DictReader(open(f"{W}/dfdc_test_list.csv")):
    items.append(("dfdc", int(r["label"]), f"dfdc_test/videos/{os.path.basename(r['file'])}"))
    if r["original"]:   # the originals of the test fakes are real part-9 videos too
        items.append(("dfdc", 0, f"dfdc_test/videos/{os.path.basename(r['original'])}"))
for r in csv.DictReader(open(f"{W}/step1_calibration/manifest.csv")):
    if r["split"] == "test":
        items.append((r["dataset"], int(r["label"]), f"step1_calibration/videos/{r['local']}"))
missing = [rel for _, _, rel in items if not os.path.isfile(os.path.join(W, rel))]
print(len(items), "videos,", len(missing), "missing", missing[:3], flush=True)

done = {r["file"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["dataset", "label", "file", "mean_logit", "prob", "short_side"])
    for ds, y, rel in items:
        if rel in done or rel in missing:
            continue
        p = os.path.join(W, rel)
        try:
            c = av.open(p); s = c.streams.video[0]; side = min(s.codec_context.width, s.codec_context.height); c.close()
            r = optimized.predict(p, checkpoint_path=ck)
        except Exception as e:   # e.g. a clip shorter than 16 frames: production rejects it too
            print("SKIP", rel, type(e).__name__, e, flush=True); continue
        w.writerow([ds, y, rel, f"{r.logit:.6f}", f"{r.probability:.6f}", side]); fh.flush()
        print(ds, y, f"{r.probability:.4f}", rel, flush=True)
