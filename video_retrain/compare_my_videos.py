"""Score every video in video_retrain/work/my_videos/{real,fake}/ with the deployed and the fine-tuned video model
(production predict, same face crops) and print them side by side. Read-only: the app is not changed.

usage (from backend/): ../.venv/bin/python ../video_retrain/compare_my_videos.py [candidate.pt]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from app.services.video.model_v1 import optimized  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "work", "my_videos")
DEPLOYED = os.path.join(HERE, "..", "backend", "checkpoints", "video", "epoch_11_model_only.pt")
CAND = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "work", "kaggle_output_v2", "run", "best.pt")
EXT = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")

items = [(lab, os.path.join(ROOT, lab, f)) for lab in ("real", "fake")
         for f in sorted(os.listdir(os.path.join(ROOT, lab))) if f.lower().endswith(EXT)]
if not items:
    sys.exit(f"no videos found under {ROOT}/real or /fake")
scores = {}
for name, ck in (("current", DEPLOYED), ("new", CAND)):
    optimized.clear_model_cache()
    for lab, p in items:
        try:
            scores[(name, p)] = optimized.predict(p, checkpoint_path=ck).probability
        except Exception as e:  # noqa: BLE001
            scores[(name, p)] = None; print(f"could not score {os.path.basename(p)}: {e}")
print(f"\n{'truth':5s}  {'current':>8s}  {'new':>8s}  file   (P(fake) at the raw 0.5 cut: >=0.5 means 'fake')")
right = {"current": 0, "new": 0}
for lab, p in items:
    row = []
    for name in ("current", "new"):
        s = scores[(name, p)]
        row.append("   error" if s is None else f"{s:8.3f}")
        if s is not None and (s >= 0.5) == (lab == "fake"):
            right[name] += 1
    print(f"{lab:5s}  {row[0]}  {row[1]}  {os.path.basename(p)}")
print(f"\ncorrect at 0.5: current {right['current']}/{len(items)}, new {right['new']}/{len(items)}")
