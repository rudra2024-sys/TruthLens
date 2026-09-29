"""Turn a predictions CSV (from run_predictions.py) into report tables + figures.

  python -m eval.make_report --media image --pred eval/data/pred_image.csv --out eval/results/image
  python -m eval.make_report --media video --pred eval/data/pred_video.csv --out eval/results/video

Writes <out>/report.md, <out>/summary.csv, <out>/per_source.csv and PNG figures.
Everything is computed from the stored scores -- nothing is re-run through a model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval import metrics as M

BAND = 0.2   # production UNCERTAIN margin around the threshold (image/video/audio detectors)

COLORS = {"convnext": "#1f77b4", "clip": "#2ca02c", "average": "#9467bd",
          "max": "#d62728", "video_cnn": "#d62728"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("predictions file is empty")
    return rows


def systems_for(media, rows):
    """name -> score array. For images: the two models + both ensemble rules."""
    if media == "image":
        cn = np.array([float(r["convnext_fake"]) for r in rows])
        cl = np.array([float(r["clip_fake"]) for r in rows])
        return {"convnext": cn, "clip": cl, "average": (cn + cl) / 2, "max": np.maximum(cn, cl)}
    return {"video_cnn": np.array([float(r["video_fake"]) for r in rows])}


def fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", choices=["image", "video"], required=True)
    ap.add_argument("--pred", required=True, nargs="+",
                    help="one or more predictions CSVs (concatenated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold (default 0.5 image, 0.525 video = production)")
    ap.add_argument("--prod", default=None,
                    help="system treated as the deployed one (default: max for image, video_cnn for video)")
    args = ap.parse_args()

    thr = args.threshold if args.threshold is not None else (0.5 if args.media == "image" else 0.525)
    prod = args.prod or ("max" if args.media == "image" else "video_cnn")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pth in args.pred:
        rows += load(pth)
    y = np.array([int(r["label"]) for r in rows])
    src = np.array([r["source"] for r in rows])
    trained = {r["source"]: r["trained_on"] for r in rows}
    S = systems_for(args.media, rows)
    sources = sorted(set(src))
    both = M.has_both_classes(y)

    # ---------------- overall table ----------------
    summ = {name: M.summary(y, s, thr) for name, s in S.items()}
    ci = {}
    if both:
        for name, s in S.items():
            ci[name] = {"auc": M.bootstrap_ci(M.roc_auc, y, s),
                        "acc": M.bootstrap_ci(lambda a, b: M.summary(a, b, thr)["accuracy"], y, s)}

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["n", "n_real", "n_fake", "accuracy", "balanced_accuracy", "fake_recall",
                "real_specificity", "precision", "f1", "auc", "ap", "tp", "tn", "fp", "fn"]
        w.writerow(["system"] + keys)
        for name, s in summ.items():
            w.writerow([name] + [s[k] for k in keys])

    # ---------------- per-source table ----------------
    per_src_rows = []
    for name, s in S.items():
        for sc in sources:
            m = src == sc
            r = M.summary(y[m], s[m], thr)
            per_src_rows.append((name, sc, trained[sc], r))
    with open(out / "per_source.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "source", "trained_on", "n_real", "n_fake", "accuracy",
                    "fake_recall", "real_specificity", "auc"])
        for name, sc, tr, r in per_src_rows:
            w.writerow([name, sc, tr, r["n_real"], r["n_fake"], r["accuracy"],
                        r["fake_recall"], r["real_specificity"], r["auc"]])

    # ---------------- figures ----------------
    figs = []

    if both:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        for name, s in S.items():
            fpr, tpr = M.roc_curve(y, s)
            ax[0].plot(fpr, tpr, color=COLORS[name], label=f"{name} (AUC {summ[name]['auc']:.3f})")
            rec, pre = M.pr_curve(y, s)
            ax[1].plot(rec, pre, color=COLORS[name], label=f"{name} (AP {summ[name]['ap']:.3f})")
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        ax[1].set(xlabel="Recall (fake)", ylabel="Precision", title="Precision-Recall", ylim=(0, 1.02))
        for a in ax:
            a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=8)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "roc_pr.png", dpi=150)
        plt.close(fig)
        figs.append(("roc_pr.png", "ROC and precision-recall curves, all sources pooled."))

    # confusion matrix of the deployed system incl. the UNCERTAIN band
    sp = S[prod]
    lo, hi = thr - BAND, thr + BAND
    band_idx = np.where(sp >= hi, 2, np.where(sp <= lo, 0, 1))   # 0 REAL, 1 UNCERTAIN, 2 FAKE
    cm = np.zeros((2, 3), dtype=int)
    for t, b in zip(y, band_idx):
        cm[t, b] += 1
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), ["REAL", "UNCERTAIN", "FAKE"])
    ax.set_yticks(range(2), ["true REAL", "true FAKE"])
    for i in range(2):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(f"Deployed verdicts ({prod}, band +-{BAND})")
    fig.tight_layout()
    fig.savefig(out / "confusion_verdicts.png", dpi=150)
    plt.close(fig)
    figs.append(("confusion_verdicts.png",
                 f"Verdict confusion matrix for the deployed system, including the UNCERTAIN band "
                 f"({lo:.2f}-{hi:.2f})."))

    # score distributions
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(sp[y == 0], bins=bins, alpha=0.6, label="real", color="#2ca02c")
    ax.hist(sp[y == 1], bins=bins, alpha=0.6, label="fake", color="#d62728")
    ax.axvline(thr, color="k", ls="--", lw=0.8)
    ax.set(xlabel=f"P(fake), {prod}", ylabel="count", title="Score distribution by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "score_hist.png", dpi=150)
    plt.close(fig)
    figs.append(("score_hist.png", "How separated the deployed system's scores are for real vs fake."))

    # reliability diagram
    ece = float("nan")
    if both:
        ece, conf, acc, cnt = M.expected_calibration_error(y, sp)
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
        ok = cnt > 0
        ax.plot(conf[ok], acc[ok], "o-", color=COLORS.get(prod, "#d62728"), label=f"{prod} (ECE {ece:.3f})")
        ax.set(xlabel="mean predicted P(fake)", ylabel="observed fraction fake",
               title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "reliability.png", dpi=150)
        plt.close(fig)
        figs.append(("reliability.png", "Calibration: does 90% confidence mean 90% correct? "
                     "Points below the diagonal are over-confident."))

    # per-source accuracy bars
    if len(sources) > 1 or args.media == "image":
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(sources) + 2), 4.2))
        names = list(S.keys())
        w = 0.8 / len(names)
        for k, name in enumerate(names):
            vals = [next(r["accuracy"] for n2, sc2, _, r in per_src_rows if n2 == name and sc2 == sc)
                    for sc in sources]
            ax.bar(np.arange(len(sources)) + k * w, vals, w, label=name, color=COLORS[name])
        ax.set_xticks(np.arange(len(sources)) + 0.4 - w / 2, sources, rotation=25, ha="right", fontsize=8)
        ax.set(ylabel=f"accuracy @ {thr}", ylim=(0, 1.22), title="Accuracy per source")
        ax.legend(fontsize=8, ncol=len(names), loc="upper center")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "per_source_accuracy.png", dpi=150)
        plt.close(fig)
        figs.append(("per_source_accuracy.png", "Accuracy per data source and system."))

    # threshold sweep
    if both:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ts = np.linspace(0.05, 0.95, 91)
        for name, s in S.items():
            ax.plot(ts, [M.summary(y, s, t)["balanced_accuracy"] for t in ts],
                    color=COLORS[name], label=name)
        ax.axvline(thr, color="k", ls="--", lw=0.8)
        ax.set(xlabel="decision threshold", ylabel="balanced accuracy", title="Threshold sensitivity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "threshold_sweep.png", dpi=150)
        plt.close(fig)
        figs.append(("threshold_sweep.png", "Balanced accuracy as the decision threshold moves "
                     "(dashed = deployed threshold). Descriptive only - no threshold is changed."))

    # ---------------- markdown ----------------
    L = []
    L.append(f"# TruthLens {args.media} evaluation\n")
    L.append(f"- Predictions: `{', '.join(Path(q).name for q in args.pred)}`  |  files: **{len(rows)}** "
             f"({int((y == 0).sum())} real, {int((y == 1).sum())} fake)  |  sources: {len(sources)}")
    L.append(f"- Decision threshold: **{thr}**  |  deployed system: **{prod}**  |  UNCERTAIN band: +-{BAND}")
    L.append("- Labels: 0 = real, 1 = fake. All numbers computed from stored model scores "
             "(`run_predictions.py`) using the exact production pipeline classes.\n")

    L.append("## Overall (all sources pooled)\n")
    hdr = ["system", "n", "accuracy", "balanced acc", "fake recall", "real specificity",
           "precision", "F1", "AUC", "AP"]
    body = []
    for name, s in summ.items():
        auc_txt = fmt(s["auc"])
        if name in ci:
            auc_txt += f" [{fmt(ci[name]['auc'][0])}, {fmt(ci[name]['auc'][1])}]"
        body.append([("**" + name + "**") if name == prod else name, str(s["n"]), fmt(s["accuracy"], True),
                     fmt(s["balanced_accuracy"], True), fmt(s["fake_recall"], True),
                     fmt(s["real_specificity"], True), fmt(s["precision"], True),
                     fmt(s["f1"]), auc_txt, fmt(s["ap"])])
    L.append(md_table(hdr, body))
    if both:
        L.append("\nAUC brackets are 95% bootstrap confidence intervals (1000 resamples).")
        L.append(f"\nCalibration of the deployed system: ECE = **{fmt(ece)}** (10 bins; lower is better).")
    else:
        L.append("\n_Only one class present in these predictions, so AUC/precision/calibration are "
                 "undefined; accuracy here is recall on that class._")

    statuses = sorted(set(trained.values()))
    if len(statuses) > 1:
        L.append("\n## By training status (the honest comparison)\n")
        L.append("`yes` = the model saw this data source in training (measures fit). `no` = never seen "
                 "(measures generalisation). Pooled numbers above mix the two and depend on how many files "
                 "of each were sampled, so read this table instead.\n")
        hdr = ["training status", "system", "files", "real", "fake", "accuracy", "balanced acc",
               "fake recall", "real specificity", "AUC"]
        body = []
        for st in statuses:
            m = np.array([trained[x] == st for x in src])
            for name, sc_ in S.items():
                r = M.summary(y[m], sc_[m], thr)
                body.append([st, ("**" + name + "**") if name == prod else name, str(r["n"]),
                             str(r["n_real"]), str(r["n_fake"]), fmt(r["accuracy"], True),
                             fmt(r["balanced_accuracy"], True), fmt(r["fake_recall"], True),
                             fmt(r["real_specificity"], True), fmt(r["auc"])])
        L.append(md_table(hdr, body))

    L.append("\n## Per source\n")
    hdr = ["source", "trained on?", "real", "fake"] + [f"{n} acc" for n in S]
    body = []
    for sc in sources:
        r0 = next(r for n2, sc2, _, r in per_src_rows if sc2 == sc)
        body.append([sc, trained[sc], str(r0["n_real"]), str(r0["n_fake"])] +
                    [fmt(next(r["accuracy"] for n2, sc2, _, r in per_src_rows if n2 == n and sc2 == sc), True)
                     for n in S])
    L.append(md_table(hdr, body))
    fake_only = [sc for sc in sources if not (y[src == sc] == 0).any()]
    if fake_only and (y == 0).any():
        L.append("\n### Fake-only sources: AUC against ALL real images\n")
        L.append("Each fake-only source (typically one generator) is scored against every real image in the "
                 "evaluation set, so a per-generator ranking quality is visible even though the source has "
                 "no real images of its own.\n")
        hdr = ["source (generator)", "trained on?", "fakes"] + [f"{n} AUC" for n in S]
        body = []
        real_mask = y == 0
        for sc in fake_only:
            m = (src == sc) | real_mask
            body.append([sc, trained[sc], str(int((src == sc).sum()))] +
                        [fmt(M.roc_auc(y[m], S[n][m])) for n in S])
        L.append(md_table(hdr, body))
    L.append("\n`trained on?` records whether the deployed model saw that source in training. "
             "`yes` sources measure fit, not generalisation; only `no` sources are honest held-out tests. "
             "Fake-only or real-only sources report recall on that single class.")

    L.append("\n## Figures\n")
    for fn, cap in figs:
        L.append(f"![{fn}]({fn})\n\n*{cap}*\n")

    if args.media == "image":
        L.append("## Ablation: single models vs ensemble rules\n")
        L.append("`convnext` and `clip` are the two sub-models alone; `average` and `max` are the two "
                 "combination rules. Production uses `max` (see CLAUDE.md section 3 for why). "
                 "The table above is the ablation.\n")

    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")
    for name, s in summ.items():
        print(f"  {name:10s} acc={fmt(s['accuracy'], True):>7s}  auc={fmt(s['auc']):>6s}  "
              f"recall={fmt(s['fake_recall'], True):>7s}  spec={fmt(s['real_specificity'], True):>7s}")


if __name__ == "__main__":
    main()
