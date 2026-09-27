"""Tables + charts from eval/data/robust_pred.csv (produced by run_robustness.py).

  python -m eval.make_robustness_report [--pred eval/data/robust_pred.csv] [--out eval/results/robustness]

Reports, for the deployed ensemble (max of ConvNeXt-Tiny and CLIP) and each sub-model, how detection quality changes
under each degradation, separately for images from sources the models trained on ("seen") and never saw ("unseen").
Quality = ROC AUC (threshold-free, so a shift in score calibration is not mistaken for lost ability) plus fake recall
and real specificity at the deployed 0.5 threshold (which show *how* it fails: missing fakes or flagging real photos).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import matplotlib                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

from eval import metrics as M                      # noqa: E402
from eval import perturbations as P                # noqa: E402

SYSTEMS = ["convnext", "clip", "average", "max"]
COLORS = {"convnext": "#1f77b4", "clip": "#2ca02c", "average": "#9467bd", "max": "#d62728"}
GROUPS = {"seen": "Seen sources (trained on)", "unseen": "Unseen sources (never seen)", "all": "All"}
THRESHOLD = 0.5


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("no rows in predictions file")
    return rows


def scores(rows):
    cn = np.array([float(r["convnext_fake"]) for r in rows])
    cl = np.array([float(r["clip_fake"]) for r in rows])
    return {"convnext": cn, "clip": cl, "average": (cn + cl) / 2, "max": np.maximum(cn, cl)}


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def f3(x):
    return "n/a" if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(BACKEND / "eval" / "data" / "robust_pred.csv"))
    ap.add_argument("--out", default=str(BACKEND / "eval" / "results" / "robustness"))
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load(args.pred)
    # keep only images that have every setting (an interrupted run leaves a partial tail)
    n_settings = len(P.settings())
    counts = {}
    for r in rows:
        counts[r["path"]] = counts.get(r["path"], 0) + 1
    complete = {p for p, c in counts.items() if c >= n_settings}
    rows = [r for r in rows if r["path"] in complete]
    n_img = len(complete)
    print(f"{n_img} complete images, {len(rows)} rows")

    def key(r):
        lvl = r["level"]
        return (r["family"], lvl)

    by_setting = {}
    for r in rows:
        by_setting.setdefault(key(r), []).append(r)

    # ---------------------------------------------------------------- metrics table
    table = {}                                   # (group, family, level_str, system) -> dict
    for (fam, lvl), rs in by_setting.items():
        for group in GROUPS:
            sub = rs if group == "all" else [r for r in rs if r["status"] == group]
            if not sub:
                continue
            y = np.array([int(r["label"]) for r in sub])
            sc = scores(sub)
            for sysname in SYSTEMS:
                s = sc[sysname]
                sm = M.summary(y, s, THRESHOLD)
                lo, hi = M.bootstrap_ci(M.roc_auc, y, s, n_boot=args.boot, seed=1) if args.boot else (float("nan"),) * 2
                table[(group, fam, lvl, sysname)] = {**sm, "auc_lo": lo, "auc_hi": hi}

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "family", "level", "system", "n", "auc", "auc_lo", "auc_hi", "accuracy", "balanced_accuracy",
                    "fake_recall", "real_specificity"])
        for (group, fam, lvl, sysname), v in sorted(table.items()):
            w.writerow([group, fam, lvl, sysname, v["n"], v["auc"], v["auc_lo"], v["auc_hi"], v["accuracy"],
                        v["balanced_accuracy"], v["fake_recall"], v["real_specificity"]])

    def get(group, fam, lvl, sysname="max"):
        return table.get((group, fam, str(lvl), sysname))

    settings = P.settings()

    # ---------------------------------------------------------------- figures
    fam_list = list(P.FAMILIES)

    def line_fig(fname, title, ylabel, fn, ylim=(0.35, 1.02), systems=("max",), groups=("seen", "unseen")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey=True)
        for ax, fam in zip(axes.ravel(), fam_list):
            levels = P.FAMILIES[fam]["levels"]
            xs = list(range(len(levels)))
            for group, ls in zip(groups, ("-", "--")):
                for sysname in systems:
                    base = get(group, "clean", 0, sysname)
                    ys = [fn(get(group, fam, lv, sysname)) for lv in levels]
                    label = f"{sysname} / {group}" if len(systems) > 1 else f"{group}"
                    ax.plot(xs, ys, ls, marker="o", color=COLORS[sysname] if len(systems) > 1 else
                            {"seen": "#1f77b4", "unseen": "#d62728"}[group], label=label)
                    if base is not None:
                        ax.axhline(fn(base), color=COLORS[sysname] if len(systems) > 1 else
                                   {"seen": "#1f77b4", "unseen": "#d62728"}[group], lw=0.6, ls=":", alpha=0.7)
            ax.set_xticks(xs, [str(l) for l in levels], fontsize=8)
            ax.set(title=P.FAMILIES[fam]["title"], xlabel=P.FAMILIES[fam]["xlabel"], ylim=ylim)
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel(ylabel)
        axes[1, 0].set_ylabel(ylabel)
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(title + "   (dotted = untouched baseline)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=140)
        plt.close(fig)

    line_fig("auc_by_family.png", "Deployed ensemble: ROC AUC under degradation", "ROC AUC", lambda v: v["auc"] if v else np.nan)
    line_fig("recall_by_family.png", "Deployed ensemble: share of AI images caught (fake recall @ 0.5)", "fake recall",
             lambda v: v["fake_recall"] if v else np.nan, ylim=(0, 1.02))
    line_fig("specificity_by_family.png", "Deployed ensemble: share of real photos kept real (specificity @ 0.5)", "real specificity",
             lambda v: v["real_specificity"] if v else np.nan, ylim=(0, 1.02))
    line_fig("models_seen_auc.png", "Which model is more fragile? ROC AUC on SEEN sources", "ROC AUC",
             lambda v: v["auc"] if v else np.nan, systems=("convnext", "clip", "max"), groups=("seen",), ylim=(0.35, 1.02))
    line_fig("models_unseen_auc.png", "Which model is more fragile? ROC AUC on UNSEEN sources", "ROC AUC",
             lambda v: v["auc"] if v else np.nan, systems=("convnext", "clip", "max"), groups=("unseen",), ylim=(0.35, 1.02))

    # heatmap: rows = settings, cols = seen/unseen x (AUC, recall, specificity)
    cols = [("seen", "auc"), ("unseen", "auc"), ("seen", "fake_recall"), ("unseen", "fake_recall"),
            ("seen", "real_specificity"), ("unseen", "real_specificity")]
    labels = [f"{g}\n{m.replace('_', ' ')}" for g, m in cols]
    mat = np.array([[(get(g, fam, lvl)[m] if get(g, fam, lvl) else np.nan) for g, m in cols] for fam, lvl in settings])
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(settings) + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)), labels, fontsize=8)
    ax.set_yticks(range(len(settings)), [f"{f} {l}" if f != "clean" else "clean (baseline)" for f, l in settings], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == mat[i, j]:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Deployed ensemble, every setting")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=140)
    plt.close(fig)

    # ---------------------------------------------------------------- markdown
    L = ["# Robustness of the deployed image detectors\n",
         f"- **{n_img} images** (label-balanced; 3 seen sources = AI-vs-Human, 140k Faces, DeepDetect; 2 unseen = OpenFake, "
         f"GenImage) x **{len(settings)} settings**, scored by the production classes (`ImagePipeline`, `ClipPipeline`).",
         "- **Deployed system** = max(ConvNeXt-Tiny, CLIP). Quality = ROC AUC (threshold-free) plus fake recall / real "
         "specificity at the deployed 0.5 threshold, which show *how* it fails.",
         "- 95 % AUC intervals are bootstrap (300 resamples); with 90-120 images per group they are wide - read differences "
         "smaller than the interval as noise.",
         "- Every degradation is deterministic (`eval/perturbations.py`); the untouched baseline reproduces the earlier "
         "evaluation exactly.\n"]

    def block(group):
        L.append(f"## {GROUPS[group]}\n")
        L.append("| setting | AUC (deployed) | Δ AUC | fake recall | real specificity | AUC convnext | AUC clip |")
        L.append("|---|---|---|---|---|---|---|")
        base = get(group, "clean", 0)
        for fam, lvl in settings:
            v = get(group, fam, lvl)
            if not v:
                continue
            d = v["auc"] - base["auc"] if base else float("nan")
            name = "clean (baseline)" if fam == "clean" else f"{fam} {lvl}"
            cn, cl = get(group, fam, lvl, "convnext"), get(group, fam, lvl, "clip")
            ci = f" [{f3(v['auc_lo'])}, {f3(v['auc_hi'])}]"
            L.append(f"| {name} | {f3(v['auc'])}{ci} | {'' if fam == 'clean' else f'{d:+.3f}'} | {pct(v['fake_recall'])} | "
                     f"{pct(v['real_specificity'])} | {f3(cn['auc'])} | {f3(cl['auc'])} |")
        L.append("")

    block("seen")
    block("unseen")

    # most damaging settings
    L.append("## Most damaging degradations (deployed AUC drop vs clean)\n")
    L.append("| rank | setting | seen ΔAUC | unseen ΔAUC |")
    L.append("|---|---|---|---|")
    drops = []
    for fam, lvl in settings:
        if fam == "clean":
            continue
        s, u = get("seen", fam, lvl), get("unseen", fam, lvl)
        bs, bu = get("seen", "clean", 0), get("unseen", "clean", 0)
        if s and u and bs and bu:
            drops.append((f"{fam} {lvl}", s["auc"] - bs["auc"], u["auc"] - bu["auc"]))
    for i, (name, ds, du) in enumerate(sorted(drops, key=lambda t: t[1] + t[2])[:8], 1):
        L.append(f"| {i} | {name} | {ds:+.3f} | {du:+.3f} |")

    L.append("\n## Figures\n")
    for fn, cap in [("auc_by_family.png", "Deployed ROC AUC per degradation family (solid = seen, dashed = unseen, dotted = clean baseline)."),
                    ("recall_by_family.png", "Share of AI images caught: does degradation make the detector miss fakes?"),
                    ("specificity_by_family.png", "Share of real photos kept real: does degradation make it cry wolf?"),
                    ("models_seen_auc.png", "ConvNeXt vs CLIP vs the max ensemble on seen sources."),
                    ("models_unseen_auc.png", "The same on unseen sources."),
                    ("heatmap.png", "Every setting at a glance.")]:
        L.append(f"![{fn}]({fn})\n\n*{cap}*\n")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", out / "report.md")
    b_s, b_u = get("seen", "clean", 0), get("unseen", "clean", 0)
    print(f"baseline deployed AUC: seen {f3(b_s['auc'])}  unseen {f3(b_u['auc'])}")


if __name__ == "__main__":
    main()
