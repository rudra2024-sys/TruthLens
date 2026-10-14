"""Round-3 verdict-level comparison, deployed vs round-3 model, on the held-out test videos (score_r3.py output).

Each model is judged with its own Platt calibration and the production rule (band 0.3/0.7 around 0.5 on the
calibrated scale, always UNCERTAIN below 360 px). Deployed: the live calibration (calibration.json). Round 3: Platt
fitted ONLY on run_r3/val_scores.csv (validation videos, short side >= 360) - no test video is used for fitting.
usage: python compare_r3.py   (prints tables, writes work/r3_platt.json)"""
import csv, json, os

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
MIN_SIDE = 360

val = [r for r in csv.DictReader(open(f"{W}/kaggle_output_v5/run_r3/val_scores.csv")) if int(r["short_side"]) >= MIN_SIDE]
m = LogisticRegression(C=1e6, max_iter=10000).fit([[float(r["mean_logit"])] for r in val], [int(r["label"]) for r in val])
a, b = float(m.coef_[0][0]), float(m.intercept_[0])
json.dump({"method": "platt", "a": round(a, 4), "b": round(b, 4), "n": len(val), "fit_on": "round-3 val, short side >= 360"},
          open(f"{W}/r3_platt.json", "w"))
live = json.load(open(os.path.join(HERE, "..", "backend", "app", "core", "calibration.json")))["video"]
P = {"deployed": (live["a"], live["b"]), "round3": (a, b)}
print(f"round-3 Platt a={a:.4f} b={b:.4f} fitted on {len(val)} validation videos (deployed: a={live['a']} b={live['b']})\n")


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); p, n = s[y == 1], s[y == 0]
    return float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :]))) if len(p) and len(n) else float("nan")


def verdict(r, ab):
    if int(r["short_side"]) < MIN_SIDE:
        return "UNCERTAIN"
    c = 1 / (1 + np.exp(-(ab[0] * float(r["mean_logit"]) + ab[1])))
    return "FAKE" if c >= 0.7 else "REAL" if c <= 0.3 else "UNCERTAIN"


rows = {name: {r["file"]: r for r in csv.DictReader(open(f"{W}/{f}"))}
        for name, f in (("deployed", "r3_test_deployed.csv"), ("round3", "r3_test_candidate.csv"))}
common = sorted(set(rows["deployed"]) & set(rows["round3"]))
print(f"{len(common)} videos scored by both models\n")
print("| source | model | AUC | fakes caught (FAKE / UNC / REAL) | real videos: REAL / UNC / **wrongly FAKE** |")
print("|---|---|---|---|---|")
for ds in ("faceforensics", "celebdf", "dfdc", "rtfs", "fakeavceleb", "my_videos", "ALL"):
    for name in ("deployed", "round3"):
        sub = [rows[name][f] for f in common if ds == "ALL" or rows[name][f]["dataset"] == ds]
        if not sub:
            continue
        cnt = lambda lab, v: sum(1 for r in sub if int(r["label"]) == lab and verdict(r, P[name]) == v)
        nf, nr = sum(int(r["label"]) for r in sub), sum(1 - int(r["label"]) for r in sub)
        a_ = auc([float(r["mean_logit"]) for r in sub], [int(r["label"]) for r in sub])
        print(f"| {ds} | {name} | {a_:.3f} | {cnt(1,'FAKE')} / {cnt(1,'UNCERTAIN')} / {cnt(1,'REAL')} (of {nf}) | "
              f"{cnt(0,'REAL')} / {cnt(0,'UNCERTAIN')} / **{cnt(0,'FAKE')}** (of {nr}) |")
print("\nmy_videos detail:")
for f in common:
    if rows["round3"][f]["dataset"] == "my_videos":
        d, c = rows["deployed"][f], rows["round3"][f]
        print(f"  {'fake' if d['label'] == '1' else 'real'} {d['short_side']:>5}px  now {verdict(d, P['deployed']):9s} "
              f"round3 {verdict(c, P['round3']):9s} {os.path.basename(f)}")
