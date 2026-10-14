"""Score the step-1 FIT split (or my_videos) with a given checkpoint through production predict; writes mean logits.
usage (from backend/): ../.venv/bin/python ../video_retrain/score_split.py <checkpoint> <out.csv> [fit|my_videos]"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from app.services.video.model_v1 import optimized  # noqa: E402
import av  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.join(HERE, "work")
ck, out, which = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "fit")
if which == "fit":
    items = [(r["dataset"], int(r["label"]), os.path.join(WORK, "step1_calibration", "videos", r["local"]))
             for r in csv.DictReader(open(os.path.join(WORK, "step1_calibration", "manifest.csv"))) if r["split"] == "fit"]
else:
    root = os.path.join(WORK, "my_videos")
    items = [("my_videos", int(lab == "fake"), os.path.join(root, lab, f)) for lab in ("real", "fake")
             for f in sorted(os.listdir(os.path.join(root, lab))) if not f.startswith(".")]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "label", "file", "mean_logit", "prob", "short_side"])
    for ds, y, p in items:
        c = av.open(p); s = c.streams.video[0]; side = min(s.codec_context.width, s.codec_context.height); c.close()
        r = optimized.predict(p, checkpoint_path=ck)
        w.writerow([ds, y, os.path.relpath(p, WORK), f"{r.logit:.6f}", f"{r.probability:.6f}", side]); fh.flush()
        print(ds, y, f"{r.probability:.4f}", os.path.basename(p), flush=True)
