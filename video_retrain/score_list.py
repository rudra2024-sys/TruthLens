"""Score the round-2 final-test videos (FaceForensics++, user's my_videos minus training-identity clips) with a
checkpoint through production predict. Appends/skips by file so it can resume.
usage (from backend/): ../.venv/bin/python ../video_retrain/score_list.py <checkpoint> <out.csv>"""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from app.services.video.model_v1 import optimized  # noqa: E402
import av  # noqa: E402

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
ck, out = sys.argv[1], sys.argv[2]
EXCLUDE = {"faceswap_celebdf_id0_id16_0001.mp4", "faceswap_celebdf_id0_id1_0002.mp4"}   # identity id0 is in r2 training
items = []
for lab in ("real", "fake"):
    for f in sorted(os.listdir(f"{W}/ff_test/{lab}")):
        items.append(("faceforensics", int(lab == "fake"), f"ff_test/{lab}/{f}"))
    for f in sorted(os.listdir(f"{W}/my_videos/{lab}")):
        if not f.startswith(".") and f not in EXCLUDE:
            items.append(("my_videos", int(lab == "fake"), f"my_videos/{lab}/{f}"))
done = {r["file"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
new = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new:
        w.writerow(["dataset", "label", "file", "mean_logit", "prob", "short_side"])
    for ds, y, rel in items:
        if rel in done:
            continue
        p = os.path.join(W, rel)
        c = av.open(p); s = c.streams.video[0]; side = min(s.codec_context.width, s.codec_context.height); c.close()
        r = optimized.predict(p, checkpoint_path=ck)
        w.writerow([ds, y, rel, f"{r.logit:.6f}", f"{r.probability:.6f}", side]); fh.flush()
        print(ds, y, f"{r.probability:.4f}", rel, flush=True)
