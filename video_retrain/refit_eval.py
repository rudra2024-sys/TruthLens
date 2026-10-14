"""Refit the Platt verdict scale for a fine-tuned checkpoint on the FIT split and report banded verdicts
(FAKE >= 0.7, REAL <= 0.3 on the calibrated scale) on held-out data, with and without the 360 px guard.

Inputs (all produced with production predict): work/v2_fit_scores.csv (fit split), work/checkpoint_eval_v2.csv
(test split + Celeb-DF, candidate rows), work/v2_my_videos_scores.csv (the user's own videos).
usage: python refit_eval.py   (prints tables, writes work/v2_platt.json)"""
import csv, json, os

import av
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work")
MIN_SIDE = 360

fit = list(csv.DictReader(open(f"{W}/v2_fit_scores.csv")))
m = LogisticRegression(C=1e6, max_iter=10000).fit([[float(r["mean_logit"])] for r in fit], [int(r["label"]) for r in fit])
a, b = float(m.coef_[0][0]), float(m.intercept_[0])
json.dump({"method": "platt", "a": round(a, 4), "b": round(b, 4), "n": len(fit)}, open(f"{W}/v2_platt.json", "w"))
print(f"Platt for the fine-tuned model, fitted on {len(fit)} fit-split videos: a={a:.4f} b={b:.4f}\n")

res = {r["local"]: min(int(r["w"]), int(r["h"])) for r in csv.DictReader(open(f"{W}/step1_calibration/scores_res.csv"))}


def side_of(rel):
    key = rel.split("step1_calibration/videos/")[-1]
    if key in res:
        return res[key]
    c = av.open(os.path.join(W, rel)); s = c.streams.video[0]; v = min(s.codec_context.width, s.codec_context.height); c.close()
    return v


test = [dict(dataset=r["dataset"], label=int(r["label"]), z=float(r["mean_logit"]), side=side_of(r["file"]), file=r["file"])
        for r in csv.DictReader(open(f"{W}/checkpoint_eval_v2.csv")) if r["model"] == "candidate"]
mine = [dict(dataset="my_videos", label=int(r["label"]), z=float(r["mean_logit"]), side=int(r["short_side"]), file=r["file"])
        for r in csv.DictReader(open(f"{W}/v2_my_videos_scores.csv"))]


def verdict(r, guard):
    if guard and r["side"] < MIN_SIDE:
        return "UNCERTAIN"
    cal = 1 / (1 + np.exp(-(a * r["z"] + b)))
    return "FAKE" if cal >= 0.7 else "REAL" if cal <= 0.3 else "UNCERTAIN"


for guard in (True, False):
    print(f"### {'with' if guard else 'WITHOUT'} the {MIN_SIDE} px guard")
    print("| source | fakes -> FAKE / UNCERTAIN / REAL | reals -> REAL / UNCERTAIN / FAKE (false alarm) |")
    print("|---|---|---|")
    for ds in ("rtfs", "fakeavceleb", "celebdf", "my_videos"):
        sub = [r for r in test + mine if r["dataset"] == ds]
        cnt = lambda lab, v: sum(1 for r in sub if r["label"] == lab and verdict(r, guard) == v)
        nf, nr = sum(r["label"] for r in sub), sum(1 - r["label"] for r in sub)
        print(f"| {ds} | {cnt(1,'FAKE')} / {cnt(1,'UNCERTAIN')} / {cnt(1,'REAL')}  (of {nf}) | "
              f"{cnt(0,'REAL')} / {cnt(0,'UNCERTAIN')} / **{cnt(0,'FAKE')}**  (of {nr}) |")
    print()
print("my_videos detail (calibrated P(fake), verdict with guard):")
for r in mine:
    cal = 1 / (1 + np.exp(-(a * r["z"] + b)))
    print(f"  {'fake' if r['label'] else 'real'}  {cal:6.3f}  {verdict(r, True):9s}  {r['side']:>4d}px  {os.path.basename(r['file'])}")
