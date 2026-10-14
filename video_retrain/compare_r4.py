"""Round-4 verdict-level comparison: deployed vs round 3 vs round 4, each with its own Platt calibration and the
production rule (FAKE >= 0.7, REAL <= 0.3 on the calibrated scale, always UNCERTAIN below 360 px).
  Platt: deployed = live calibration.json; round 3 = work/r3_platt.json; round 4 = fitted here ONLY on
         run_r4/val_scores.csv (validation videos, short side >= 360) - no test video is used for fitting.
  Test sets: (a) round 3's local held-out videos + my_videos, scored through production predict (score_r3.py output);
             (b) round 4's Kaggle test split (DFD held-out actors, phone held-out persons + their face swaps), whose
                 scores for all three models come from the notebook's test_scores.csv.
usage: python compare_r4.py <kaggle run_r4 dir> <r4 local scores csv>   (prints tables, writes work/r4_platt.json)"""
import csv, json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
MIN_SIDE = 360
run, local_r4 = sys.argv[1], sys.argv[2]

val = [r for r in csv.DictReader(open(f"{run}/val_scores.csv")) if int(r["short_side"]) >= MIN_SIDE]
m = LogisticRegression(C=1e6, max_iter=10000).fit([[float(r["mean_logit"])] for r in val], [int(r["label"]) for r in val])
a, b = float(m.coef_[0][0]), float(m.intercept_[0])
json.dump({"method": "platt", "a": round(a, 4), "b": round(b, 4), "n": len(val), "fit_on": "round-4 val, short side >= 360"},
          open(f"{W}/r4_platt.json", "w"))
live = json.load(open(os.path.join(HERE, "..", "backend", "app", "core", "calibration.json")))["video"]
r3p = json.load(open(f"{W}/r3_platt.json"))
P = {"deployed": (live["a"], live["b"]), "round3": (r3p["a"], r3p["b"]), "round4": (a, b)}
print(f"round-4 Platt a={a:.4f} b={b:.4f} on {len(val)} val videos; round 3 a={r3p['a']} b={r3p['b']}; "
      f"deployed a={live['a']} b={live['b']}\n")


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y); p, n = s[y == 1], s[y == 0]
    return float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :]))) if len(p) and len(n) else float("nan")


def verdict(z, side, ab):
    if side < MIN_SIDE:
        return "UNCERTAIN"
    c = 1 / (1 + np.exp(-(ab[0] * z + ab[1])))
    return "FAKE" if c >= 0.7 else "REAL" if c <= 0.3 else "UNCERTAIN"


def table(title, groups):
    """groups: {source: [(label, side, {model: logit})]}"""
    print(f"### {title}\n")
    print("| source | model | AUC | fakes: FAKE / UNC / REAL | reals: REAL / UNC / **wrongly FAKE** |")
    print("|---|---|---|---|---|")
    for ds, items in groups.items():
        for name in P:
            if not all(name in t[2] for t in items):
                continue
            v = [(y, verdict(t[name], s, P[name])) for y, s, t in items]
            cnt = lambda lab, x: sum(1 for y, vv in v if y == lab and vv == x)
            nf = sum(y for y, _ in v); nr = len(v) - nf
            au = auc([t[name] for _, _, t in items], [y for y, _, _ in items])
            print(f"| {ds} | {name} | {au:.3f} | {cnt(1,'FAKE')} / {cnt(1,'UNCERTAIN')} / {cnt(1,'REAL')} (of {nf}) | "
                  f"{cnt(0,'REAL')} / {cnt(0,'UNCERTAIN')} / **{cnt(0,'FAKE')}** (of {nr}) |")
    print()


# (a) local held-out set
loc = {name: {r["file"]: r for r in csv.DictReader(open(f))}
       for name, f in (("deployed", f"{W}/r3_test_deployed.csv"), ("round3", f"{W}/r3_test_candidate.csv"), ("round4", local_r4))}
common = sorted(set(loc["deployed"]) & set(loc["round3"]) & set(loc["round4"]))
groups = {}
for f in common:
    r = loc["round4"][f]
    t = (int(r["label"]), int(r["short_side"]), {n: float(loc[n][f]["mean_logit"]) for n in loc})
    groups.setdefault(r["dataset"], []).append(t)
    groups.setdefault("ALL (local)", []).append(t)
table(f"Local held-out videos ({len(common)})", groups)

# (b) Kaggle round-4 test split
groups = {}
for r in csv.DictReader(open(f"{run}/test_scores.csv")):
    t = (int(r["label"]), int(r["short_side"]),
         {"deployed": float(r["deployed_logit"]), "round3": float(r["r3_logit"]), "round4": float(r["mean_logit"])})
    ds = r["dataset"]
    if ds in ("phone", "phoneswap"):
        ds = "phone+swap"
    groups.setdefault(ds, []).append(t)
    if ds in ("dfd", "phone+swap"):
        groups.setdefault("NEW (dfd+phone)", []).append(t)
table("Round-4 Kaggle test split (held-out actors/persons)", groups)

print("my_videos detail (calibrated P(fake), verdict):")
for f in common:
    if loc["round4"][f]["dataset"] == "my_videos":
        side = int(loc["round4"][f]["short_side"]); out = []
        for n in P:
            z = float(loc[n][f]["mean_logit"]); c = 1 / (1 + np.exp(-(P[n][0] * z + P[n][1])))
            out.append(f"{n} {c:.2f} {verdict(z, side, P[n]):9s}")
        print(f"  {'fake' if loc['round4'][f]['label'] == '1' else 'real'} {side:>5}px  " + "  ".join(out) + f"  {os.path.basename(f)}")
