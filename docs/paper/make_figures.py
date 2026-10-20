"""Builds the figures for the TruthLens IEEE paper from the repo's measured eval results.

Run from the repo root: .venv/Scripts/python.exe docs/paper/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "hatch.linewidth": 0.6,
})


def architecture():
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#f0efec"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=fc, ec=MUTED, lw=0.7))
        ax.text(x + w / 2, y + h / 2 + (2.2 if sub else 0), title, ha="center", va="center",
                fontsize=7, weight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 3.2, sub, ha="center", va="center", fontsize=5.8, color=MUTED)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, mutation_scale=7))

    box(25, 89, 50, 9, "React client", "upload, results, history, reports")
    box(25, 74, 50, 9, "FastAPI gateway", "JWT auth, magic-byte check, job queue")
    arrow(50, 89, 50, 83.5)
    box(2, 52, 30, 14, "Image", "ConvNeXt-T + CLIP\nmax ensemble", fc="#dbe8f8")
    box(35, 52, 30, 14, "Video", "Haar face crop\nEfficientNet-B0, 16 fr.", fc="#fbe1d6")
    box(68, 52, 30, 14, "Audio", "wav2vec2-base\n4 s windows", fc="#d4f0e5")
    for x in (17, 50, 83):
        arrow(50, 74, x, 66.6)
    box(2, 27, 96, 16, "Evidence layer (all modalities)",
        "provenance: C2PA, EXIF/XMP, SD watermark, ELA\nexplanation: Grad-CAM, per-frame scores, audio saliency")
    for x in (17, 50, 83):
        arrow(x, 52, x, 43.6)
    box(15, 7, 70, 12, "Verdict + evidence report", "banded verdict, calibrated confidence, PDF")
    arrow(50, 27, 50, 19.6)
    ax.text(50, 0.5, "SQL store (SQLAlchemy): uploads, results, feedback, audit log",
            ha="center", fontsize=5.8, color=MUTED, style="italic")
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


def image_robustness():
    rows = list(csv.DictReader(open(ROOT / "backend/eval/results/robustness/summary.csv")))
    settings = [("clean", "0", "Clean"), ("jpeg", "90", "JPEG q90"), ("jpeg", "10", "JPEG q10"),
                ("noise", "5", "Noise σ=5"), ("blur", "3", "Blur σ=3"), ("resize", "0.25", "Resize 25%"),
                ("social", "whatsapp", "WhatsApp")]
    systems = [("convnext", "ConvNeXt-T", BLUE, ""), ("clip", "CLIP", ORANGE, "////"), ("max", "Max ensemble", AQUA, "....")]
    look = {(r["group"], r["family"], r["level"], r["system"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    w = 0.26
    for i, (key, label, color, hatch) in enumerate(systems):
        vals = [100 * float(look[("seen", f, l, key)]["real_specificity"]) for f, l, _ in settings]
        xs = [j + (i - 1) * (w + 0.02) for j in range(len(settings))]
        ax.bar(xs, vals, w, color=color, hatch=hatch, edgecolor="white", lw=0.4, label=label)
    ax.set_xticks(range(len(settings)), [s[2] for s in settings], rotation=30, ha="right")
    ax.set_ylabel("Real photos kept REAL (%)")
    ax.set_ylim(50, 100)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(OUT / "fig2_image_robustness.png")
    plt.close(fig)


def video_robustness():
    data = [("Clean", 0.665), ("JPEG q90", 0.745), ("JPEG q50", 0.625), ("JPEG q10", 0.585),
            ("Noise 10", 0.685), ("Resize 50%", 0.570), ("WhatsApp", 0.610), ("WebP", 0.545),
            ("Blur σ=1", 0.540), ("Resize 25%", 0.465), ("Blur σ=3", 0.410)]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    xs = range(len(data))
    ax.bar(xs, [d[1] for d in data], 0.6, color=ORANGE, edgecolor="white", lw=0.4)
    ax.axhline(0.5, color=INK, lw=0.7, ls="--")
    ax.text(len(data) - 0.6, 0.505, "chance", fontsize=6, color=INK, ha="right", va="bottom")
    ax.set_xticks(list(xs), [d[0] for d in data], rotation=35, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.3, 0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig3_video_robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    architecture(); image_robustness(); video_robustness()
    print("figures written to", OUT)
