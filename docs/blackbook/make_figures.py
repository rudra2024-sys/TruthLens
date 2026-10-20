"""Diagrams and charts for the TruthLens final project report (blackbook).

Every number plotted here is copied from backend/eval/results/, video_retrain/ results or CLAUDE.md
(sections cited next to each chart). Run: python make_figures.py  ->  figures/*.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
RES = ROOT / "backend" / "eval" / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

INK = "#2b2b2b"
C = {  # muted, print-friendly fills
    "grey": "#eeeeec", "blue": "#dbe7f6", "red": "#f9e0d9", "green": "#dcefe4",
    "gold": "#f6ead2", "violet": "#e8e3f3", "teal": "#d9eef0",
}
S = {"blue": "#3b6fb6", "red": "#c4553f", "green": "#3f8f5f", "gold": "#b8862b", "grey": "#7a7a7a", "violet": "#6b5aa6"}


# ----------------------------------------------------------------------------------- diagram helpers
def canvas(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, fill="grey", fs=11, bold=True, round_=True, lw=1.4):
    style = "round,pad=0.02,rounding_size=1.2" if round_ else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=C.get(fill, fill), ec=INK, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 2, color="#444")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK)
    return (x, y, w, h)


def arrow(ax, p, q, label=None, style="-|>", ls="-", color=INK, fs=8.5, rad=0.0, lpos=0.5, loff=(0, 1.2)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, lw=1.2, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        mx = p[0] + (q[0] - p[0]) * lpos + loff[0]
        my = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=fs, color="#333",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))


def bottom(b): return (b[0] + b[2] / 2, b[1])
def top(b): return (b[0] + b[2] / 2, b[1] + b[3])
def left(b): return (b[0], b[1] + b[3] / 2)
def right(b): return (b[0] + b[2], b[1] + b[3] / 2)


def save(fig, name):
    for ax in fig.axes:
        if not ax.axison and not getattr(ax, "_no_fit", False):  # diagrams: fit the canvas to what was drawn
            ax.set_aspect("equal")
            ax.relim()
            ax.autoscale(True)
            ax.margins(0.015)
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------- diagrams
def fig_architecture():
    fig, ax = canvas(11, 8.6)
    ax.add_patch(Rectangle((1, 1), 98, 76, fc="none", ec="#999", ls="--", lw=1))
    ax.text(2.5, 75, "Backend process (FastAPI, one uvicorn worker)", fontsize=9.5, color="#666", style="italic")
    ui = box(ax, 18, 81, 64, 8, "Presentation layer — React 18 + Vite SPA",
             "Home · Verify · History · Reports · Dashboard · ProctorLens · Admin", "violet")
    ng = box(ax, 30, 70.5, 40, 6, "nginx: static files, /api proxy, CSP, rate limits", fill="grey", fs=9.5, bold=False)
    arrow(ax, bottom(ui), top(ng), "HTTPS / JSON + JWT")
    api = box(ax, 4, 57, 92, 9, "API layer — routers under /api/v1",
              "auth · upload · detect · jobs · report · dashboard · feedback · identity · exams · admin", "gold")
    arrow(ax, bottom(ng), top(api))
    mw = ax.text(50, 54.2, "magic-byte sniffing · owner checks (404 not 403) · request IDs · audit log · start-up guard",
                 ha="center", fontsize=8.5, color="#555")
    im = box(ax, 4, 36, 21, 14, "Image detector", "ConvNeXt-Tiny\n+ CLIP ViT-B/16\n+ face-swap check", "blue", fs=10)
    vi = box(ax, 27, 36, 21, 14, "Video detector", "16 frames · Haar crop\nEfficientNet-B0 r4\nPlatt + 360 px guard", "red", fs=10)
    au = box(ax, 50, 36, 21, 14, "Audio detector", "16 kHz · 4 s windows\nwav2vec 2.0 base\n(AASIST fallback)", "green", fs=10)
    pr = box(ax, 73, 36, 23, 14, "ProctorLens", "MTCNN + FaceNet ID\nYOLOv8n · VAD · pose\nexam enforcement", "teal", fs=10)
    for b in (im, vi, au, pr):
        arrow(ax, (b[0] + b[2] / 2, 57), top(b))
    ev = box(ax, 4, 20, 67, 10, "Evidence layer",
             "Grad-CAM / per-frame / audio saliency  ·  C2PA · EXIF/XMP · SD watermark · ELA  ·  calibration", "grey")
    for b in (im, vi, au):
        arrow(ax, bottom(b), (b[0] + b[2] / 2, 30))
    jb = box(ax, 73, 20, 23, 10, "Job managers", "video scans + monitoring\nchecks (thread pool)", "grey", fs=10)
    arrow(ax, bottom(pr), top(jb))
    rp = box(ax, 4, 4, 30, 10, "PDF report generator", "ReportLab", "gold", fs=10)
    db = box(ax, 38, 4, 28, 10, "SQLite (async SQLAlchemy)", "users · uploads · results · exams", "grey", fs=9.5)
    fs_ = box(ax, 70, 4, 26, 10, "File store", "uploads · reports · checkpoints", "grey", fs=10)
    arrow(ax, (19, 20), top(rp))
    arrow(ax, (52, 20), top(db))
    arrow(ax, bottom(jb), top(fs_))
    save(fig, "arch")


def fig_usecase():
    fig, ax = canvas(11, 7.6)

    def actor(x, y, name):
        ax.add_patch(Ellipse((x, y + 7), 3.2, 3.2, fc="white", ec=INK, lw=1.3))
        ax.plot([x, x], [y + 5.4, y + 1.5], color=INK, lw=1.3)
        ax.plot([x - 2.6, x + 2.6], [y + 4.2, y + 4.2], color=INK, lw=1.3)
        ax.plot([x, x - 2], [y + 1.5, y - 1.5], color=INK, lw=1.3)
        ax.plot([x, x + 2], [y + 1.5, y - 1.5], color=INK, lw=1.3)
        ax.text(x, y - 4, name, ha="center", fontsize=10, fontweight="bold")
        return (x, y + 3)

    ax.add_patch(Rectangle((17, 2), 70, 72, fc="#fbfbfa", ec=INK, lw=1.2))
    ax.text(50, 71, "TruthLens system boundary", ha="center", fontsize=10, style="italic")
    u = actor(7, 50, "User / Student")
    o = actor(96, 50, "Organisation\n(admin)")
    s = actor(7, 12, "Scheduler /\nmodels")
    cases_l = ["Sign up / log in", "Upload image / video / audio", "View verdict + confidence", "Open explanation (Grad-CAM)",
               "Check provenance (C2PA)", "Download PDF report", "Give feedback / withdraw", "Join organisation (code)",
               "Verify ID + take proctored exam", "Delete account and data"]
    ys = [66, 60, 54, 48, 42, 36, 30, 24, 18, 12]
    for t, y in zip(cases_l, ys):
        ax.add_patch(Ellipse((40, y), 34, 5, fc=C["blue"], ec=INK, lw=1))
        ax.text(40, y, t, ha="center", va="center", fontsize=9)
        ax.plot([u[0] + 2.5, 23], [u[1], y], color="#777", lw=0.8)
    cases_r = ["Create exams + MCQs", "Share / regenerate join code", "Review attempts + flags", "View session report",
               "Manage roster"]
    for t, y in zip(cases_r, [60, 50, 40, 30, 20]):
        ax.add_patch(Ellipse((66, y), 26, 5, fc=C["teal"], ec=INK, lw=1))
        ax.text(66, y, t, ha="center", va="center", fontsize=9)
        ax.plot([79, o[0] - 2.5], [y, o[1]], color="#777", lw=0.8)
    ax.plot([s[0] + 2.5, 21], [s[1], 18], color="#777", lw=0.8, ls="--")
    ax.text(14, 22, "periodic\nchecks", fontsize=8, color="#555", ha="center")
    save(fig, "usecase")


def fig_dfd0():
    fig, ax = canvas(10, 5.4)
    u = box(ax, 2, 20, 18, 12, "User", round_=False, fill="violet")
    o = box(ax, 80, 20, 18, 12, "Organisation", round_=False, fill="teal")
    ax.add_patch(Ellipse((50, 26), 34, 26, fc=C["gold"], ec=INK, lw=1.5))
    ax.text(50, 28, "0.0\nTruthLens\nPlatform", ha="center", va="center", fontsize=12, fontweight="bold")
    arrow(ax, (20, 30), (33.5, 30), "media file, credentials", loff=(0, 0.6))
    arrow(ax, (33.5, 22), (20, 22), "verdict, evidence, PDF", loff=(0, -3.6))
    arrow(ax, (80, 30), (66.5, 30), "exams, questions", loff=(0, 0.6))
    arrow(ax, (66.5, 22), (80, 22), "attempts, flags, reports", loff=(0, -3.6))
    ax.plot([40, 60], [6, 6], color=INK, lw=1.2); ax.plot([40, 60], [1.5, 1.5], color=INK, lw=1.2)
    ax.text(50, 3.6, "D1  Database + file store", ha="center", va="center", fontsize=10)
    arrow(ax, (47, 13.2), (47, 6.2), style="<|-|>")
    save(fig, "dfd0")


def fig_dfd1():
    fig, ax = canvas(11, 7.4)
    u = box(ax, 1, 32, 13, 10, "User", round_=False, fill="violet")

    def proc(pid, name, x, y):
        ax.add_patch(Ellipse((x, y), 17, 11, fc=C["gold"], ec=INK, lw=1.2))
        ax.text(x, y, pid + "\n" + name, ha="center", va="center", fontsize=9)
        return (x - 8.5, y - 5.5, 17, 11)

    def store(name, x, y, w=20):
        ax.plot([x, x + w], [y + 2.6, y + 2.6], color=INK, lw=1.2)
        ax.plot([x, x + w], [y - 2.6, y - 2.6], color=INK, lw=1.2)
        ax.plot([x, x], [y - 2.6, y + 2.6], color=INK, lw=1.2)
        ax.text(x + w / 2, y, name, ha="center", va="center", fontsize=9)
        return (x, y - 2.6, w, 5.2)

    p1 = proc("1.0", "Authenticate", 30, 70)
    p7 = proc("7.0", "Proctor exam", 56, 70)
    p2 = proc("2.0", "Upload &\nvalidate", 30, 37)
    p3 = proc("3.0", "Detect\n(image/video/audio)", 56, 37)
    p4 = proc("4.0", "Explain &\nprovenance", 80, 50)
    p5 = proc("5.0", "Generate\nPDF report", 80, 24)
    p6 = proc("6.0", "History &\nfeedback", 56, 9)
    d1 = store("D1 users", 86, 82, 16)
    d4 = store("D4 exams + monitoring", 76, 64, 26)
    d2 = store("D2 uploads + files", 33, 21, 20)
    d3 = store("D3 results + analyses", 76, 9, 26)
    arrow(ax, (10, 42), (24, 66), "credentials", lpos=0.4, loff=(-4, 0))
    arrow(ax, top(p1), (86, 83), "user row, JWT", rad=-0.18, lpos=0.5, loff=(0, 2))
    arrow(ax, (14, 41), (47.8, 67.5), "webcam frames, answers", lpos=0.62, loff=(-3, 0))
    arrow(ax, right(p7), (76, 64), "checks, attempts", lpos=0.45, loff=(0, 0.6))
    arrow(ax, right(u), left(p2), "media file", loff=(0, 0.6))
    arrow(ax, (34, 31.8), (40, 23.6), "row + file", lpos=0.5, loff=(5, -1))
    arrow(ax, right(p2), left(p3), "upload id", loff=(0, 0.6))
    arrow(ax, (50, 23.6), (54, 31.6), "file", lpos=0.5, loff=(2.5, -1))
    arrow(ax, (62, 41), (73, 47), "result", loff=(-2, 0.4))
    arrow(ax, (62, 33), (73, 27), "result", loff=(-2, -3))
    arrow(ax, (61, 32.5), (76, 10), "scores", lpos=0.45, loff=(-3.5, -1))
    arrow(ax, bottom(p3), top(p6), "verdict", lpos=0.62, loff=(-4.5, 0))
    arrow(ax, (75, 19.5), (63, 12), "PDF", loff=(1, 0.4))
    arrow(ax, left(p6), (7.5, 32), "history, PDF", rad=-0.25, lpos=0.5, loff=(0, -3))
    save(fig, "dfd1")


def fig_pipe_image():
    fig, ax = canvas(11, 6.6)
    a = box(ax, 1, 54, 15, 9, "Upload", "JPEG/PNG/WebP/HEIC", "violet", fs=10)
    b = box(ax, 20, 54, 17, 9, "Magic-byte sniff", "≤ 100 MB, PIL decode", "grey", fs=10)
    arrow(ax, right(a), left(b))
    c1 = box(ax, 43, 59, 24, 7.5, "ConvNeXt-Tiny", "resize 224 bicubic · ImageNet norm", "blue", fs=10)
    c2 = box(ax, 43, 48.5, 24, 7.5, "CLIP ViT-B/16 (frozen)", "512-d embedding → MLP head", "blue", fs=10)
    c3 = box(ax, 43, 38, 24, 7.5, "Face-swap check", "Haar → MTCNN, face ≥ 64 px", "red", fs=10)
    for c in (c1, c2, c3):
        arrow(ax, right(b), left(c))
    c3b = box(ax, 43, 28, 24, 7.5, "EfficientNet-B0 photo r2", "single face crop → Platt", "red", fs=10)
    arrow(ax, bottom(c3), top(c3b))
    m = box(ax, 73, 44, 12, 9, "max(·)", "p_C, p_L, p_F", "gold", fs=11)
    arrow(ax, right(c1), (73, 51))
    arrow(ax, right(c2), left(m))
    arrow(ax, right(c3b), (73, 46))
    v = box(ax, 88.5, 44, 11, 9, "Band", "FAKE ≥ 0.7\nREAL ≤ 0.3", "gold", fs=10)
    arrow(ax, right(m), left(v))
    t = box(ax, 69.5, 27, 17, 10, "Calibrated conf.", "T = 4.45 (ensemble)\nface: Platt", "grey", fs=10)
    arrow(ax, bottom(m), (79, 37.2))
    g = box(ax, 1, 68, 36, 8, "Grad-CAM on ConvNeXt last stage", "intensity scaled by ConvNeXt P(fake)", "grey", fs=10)
    arrow(ax, left(c1), right(g), ls="--")
    p = box(ax, 1, 28, 36, 9, "Provenance (independent)", "C2PA · EXIF/XMP · SD watermark · ELA", "grey", fs=10)
    arrow(ax, bottom(b), top(p), ls="--")
    r = box(ax, 43, 12, 56.5, 10, "Stored result → API / React card / PDF",
            "verdict · per-model probabilities · face px · conflict note if provenance says AI", "violet", fs=10)
    arrow(ax, bottom(v), (94, 22.2))
    arrow(ax, right(p), left(r))
    ax.text(1, 3, "Rule: p = max(p_ConvNeXt, p_CLIP, p_face); a failure inside the face check only means 'not applied'.",
            fontsize=9, style="italic", color="#444")
    save(fig, "pipe_image")


def fig_pipe_video():
    fig, ax = canvas(11, 6.2)
    steps = [("Video file", "MP4/MOV/WebM", "violet"),
             ("Sample 16 frames", "np.linspace indices\ngrab()-skip decode", "grey"),
             ("Haar frontal face", "largest box + 20 % pad\nelse centre square", "red"),
             ("Crop → 224 px", "PIL bilinear\nImageNet norm", "red"),
             ("EfficientNet-B0", "round-4 weights\none logit / frame", "red")]
    xs = [1, 21, 41, 61, 81]
    bs = []
    for (t, s, f), x in zip(steps, xs):
        bs.append(box(ax, x, 44, 18, 13, t, s, f, fs=10))
    for i in range(4):
        arrow(ax, right(bs[i]), left(bs[i + 1]))
    m = box(ax, 81, 24, 18, 12, "Mean logit z", "z̄ = (1/16) Σ z_i", "gold", fs=10)
    arrow(ax, bottom(bs[4]), top(m))
    pl = box(ax, 59, 24, 18, 12, "Platt scaling", "σ(1.8237 z̄ − 0.0946)", "gold", fs=10)
    arrow(ax, left(m), right(pl))
    gd = box(ax, 37, 24, 18, 12, "Resolution guard", "short side < 360 px\n→ UNCERTAIN", "gold", fs=10)
    arrow(ax, left(pl), right(gd))
    bd = box(ax, 15, 24, 18, 12, "Verdict band", "FAKE ≥ 0.7 · REAL ≤ 0.3", "gold", fs=10)
    arrow(ax, left(gd), right(bd))
    ex = box(ax, 1, 4, 47, 12, "Explanation", "per-frame P(fake) bars · exact crops\nGrad-CAM on 3 most suspicious frames", "grey", fs=10)
    tr = box(ax, 52, 4, 47, 12, "Transparency + jobs", "frames-with-face count, low-face warning\nbackground job: decode 0–70 %, crop 70–90 %, score", "grey", fs=10)
    arrow(ax, bottom(bd), (20, 16.2), ls="--")
    arrow(ax, bottom(gd), (60, 16.2), ls="--")
    save(fig, "pipe_video")


def fig_pipe_audio():
    fig, ax = canvas(11, 5.0)
    steps = [("Audio file", "WAV/MP3/M4A/\nOGG/Opus/WebM", "violet"), ("PyAV decode", "16 kHz mono", "green"),
             ("RMS normalise", "whole clip, −20 dBFS", "green"), ("Windowing", "≤ 6 × 4.0 s (64 000)\nlinspace offsets, tile short", "green"),
             ("wav2vec 2.0 base", "frozen CNN extractor\n12 transformer layers", "green")]
    xs = [1, 21, 41, 61, 81]
    bs = [box(ax, x, 32, 18, 12, t, s, f, fs=10) for (t, s, f), x in zip(steps, xs)]
    for i in range(4):
        arrow(ax, right(bs[i]), left(bs[i + 1]))
    h = box(ax, 81, 12, 18, 12, "Head", "mean-pool 768-d\n→ 256 → 2, softmax", "green", fs=10)
    arrow(ax, bottom(bs[4]), top(h))
    m = box(ax, 57, 12, 20, 12, "Clip score", "mean of window P(fake)\n(+ std reported)", "gold", fs=10)
    arrow(ax, left(h), right(m))
    v = box(ax, 33, 12, 20, 12, "Verdict band", "t = 0.5 ± 0.2", "gold", fs=10)
    arrow(ax, left(m), right(v))
    e = box(ax, 1, 12, 28, 12, "Explanation", "input-gradient saliency,\n80 bins on spectrogram", "grey", fs=10)
    arrow(ax, left(v), right(e), ls="--")
    save(fig, "pipe_audio")


def fig_band():
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    ax._no_fit = True
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.6, 1.2)
    for a, b, col, lab in [(0, 0.3, S["green"], "REAL"), (0.3, 0.7, S["gold"], "UNCERTAIN"), (0.7, 1, S["red"], "FAKE")]:
        ax.add_patch(Rectangle((a, 0.2), b - a, 0.45, fc=col, alpha=0.25, ec=col))
        ax.text((a + b) / 2, 0.42, lab, ha="center", va="center", fontsize=12, fontweight="bold", color=col)
    for x in (0, 0.3, 0.5, 0.7, 1):
        ax.plot([x, x], [0.12, 0.2], color=INK); ax.text(x, -0.05, f"{x:g}", ha="center", fontsize=10)
    ax.text(0.5, 0.85, "threshold t = 0.5, margin ±0.2 (video: applied to the Platt-calibrated score)", ha="center", fontsize=10)
    ax.text(0.5, -0.45, "fake probability p", ha="center", fontsize=10, style="italic")
    save(fig, "band")


def fig_provenance():
    fig, ax = canvas(11, 5.6)
    f = box(ax, 1, 22, 14, 12, "Uploaded\nfile", None, "violet", fs=10)
    sig = [("C2PA Content Credentials", "signature · unchanged-since-signing\ntrust list · IPTC AI source type", 44),
           ("Embedded metadata", "EXIF camera · software · GPS presence\nSD/ComfyUI prompt, seed, sampler", 31),
           ("SD / SDXL watermark", "Haar-DWT decode, 48/48 bit match\n(8 frames for video)", 18),
           ("ELA (JPEG only)", "visual aid, never scored", 5)]
    bs = []
    for t, s, y in sig:
        b = box(ax, 24, y, 34, 10.5, t, s, "grey", fs=10)
        bs.append(b)
        arrow(ax, right(f), left(b))
    a = box(ax, 66, 24, 15, 12, "_assess()", "declared_ai /\nnone / weak", "gold", fs=10)
    for b in bs[:3]:
        arrow(ax, right(b), left(a))
    c = box(ax, 86, 30, 13, 12, "Conflict\nnote", "models REAL\nbut file says AI", "red", fs=9)
    d = box(ax, 86, 12, 13, 12, "Panel +\nPDF", "never changes\nthe verdict", "grey", fs=9)
    arrow(ax, right(a), left(c)); arrow(ax, right(a), left(d))
    save(fig, "provenance")


def fig_seq_scan():
    fig, ax = canvas(11, 6.4)
    parts = ["Browser", "FastAPI", "Detector", "Database", "Evidence svc"]
    xs = [8, 30, 52, 74, 93]
    for p, x in zip(parts, xs):
        box(ax, x - 7, 56, 14, 6, p, None, "violet" if p == "Browser" else "grey", fs=10)
        ax.plot([x, x], [2, 56], color="#888", ls="--", lw=0.9)
    msgs = [(0, 1, "POST /upload (multipart)"), (1, 3, "store row + file"), (1, 0, "upload_id"),
            (0, 1, "POST /detect/{id}"), (1, 2, "run detector (thread pool)"), (2, 3, "result + analysis rows"),
            (1, 0, "verdict, confidence, sub-scores"), (0, 1, "GET /explain · /provenance"), (1, 4, "compute on demand"),
            (4, 0, "heatmaps / signals"), (0, 1, "GET /report/{id}"), (1, 0, "PDF stream")]
    y = 52
    for a, b, t in msgs:
        arrow(ax, (xs[a], y), (xs[b], y), t, ls="--" if b < a else "-", fs=8.5, loff=(0, 0.3))
        y -= 4.2
    save(fig, "seq_scan")


def fig_seq_job():
    fig, ax = canvas(11, 5.6)
    parts = ["Browser", "API", "Job manager", "Worker thread", "Database"]
    xs = [8, 29, 50, 71, 92]
    for p, x in zip(parts, xs):
        box(ax, x - 7.5, 48, 15, 6, p, None, "violet" if p == "Browser" else "grey", fs=10)
        ax.plot([x, x], [2, 48], color="#888", ls="--", lw=0.9)
    msgs = [(0, 1, "POST /detect/{id}/jobs"), (1, 2, "submit (idempotent per upload)"), (1, 0, "202 queued + job_id"),
            (2, 3, "run when semaphore free"), (3, 2, "progress: frame k/16, stage"), (0, 1, "GET /jobs/{job} (poll)"),
            (1, 0, "state, progress, queue position"), (3, 4, "store result"), (0, 1, "GET /detect/{id}/result")]
    y = 44
    for a, b, t in msgs:
        arrow(ax, (xs[a], y), (xs[b], y), t, ls="--" if b < a else "-", fs=8.5, loff=(0, 0.3))
        y -= 4.6
    ax.text(50, 0.5, "DELETE /jobs/{job} cancels: the progress callback raises and nothing is stored", ha="center",
            fontsize=9, style="italic", color="#444")
    save(fig, "seq_job")


def fig_proctor_flow():
    fig, ax = canvas(11, 6.8)
    st = [("Join organisation", "private join code", "teal"), ("Open exam", "MCQ, pass mark, max_flags", "teal"),
          ("Government-ID check", "one face, blur gate\nembedding used transiently", "teal"),
          ("Live selfie match", "FaceNet cosine ≥ 0.44\n(±0.1 band)", "teal")]
    xs = [1, 26, 51, 76]
    bs = [box(ax, x, 52, 23, 11, t, s, f, fs=10) for (t, s, f), x in zip(st, xs)]
    for i in range(3):
        arrow(ax, right(bs[i]), left(bs[i + 1]))
    at = box(ax, 76, 32, 23, 11, "Attempt starts", "MonitoringSession 1:1", "gold", fs=10)
    arrow(ax, bottom(bs[3]), top(at))
    ck = box(ax, 40, 30, 30, 15, "Periodic check (every 5–20 s)", "webcam frame (+ short audio clip)\n→ background check job", "gold", fs=10)
    arrow(ax, left(at), right(ck))
    ev = box(ax, 1, 30, 33, 15, "Browser events", "tab / focus loss · paste\ndevtools heuristic · camera off", "grey", fs=10)
    en = box(ax, 40, 8, 30, 14, "enforce_flag_threshold()", "flagged checks ≥ max_flags\n→ terminated_for_flags", "red", fs=10)
    arrow(ax, bottom(ck), top(en))
    fin = box(ax, 76, 8, 23, 14, "Finish + score", "answers vs key\nsession PDF report", "violet", fs=10)
    arrow(ax, right(en), left(fin), "else")
    rv = box(ax, 1, 8, 33, 14, "Organisation review", "attempts list, flag timeline,\nsnapshots, PDF", "teal", fs=10)
    arrow(ax, bottom(ev), top(rv), ls="--")
    arrow(ax, left(en), right(rv), ls="--")
    save(fig, "proctor_flow")


def fig_monitor_signals():
    fig, ax = canvas(11, 6.0)
    fr = box(ax, 36, 25.5, 28, 9, "Webcam frame (one check)", "+ optional short audio clip", "violet", fs=10)
    sig = [("MTCNN detect", "0 / 1 / >1 faces", "no_face, multiple_faces"),
           ("FaceNet embedding", "cosine vs reference", "identity_mismatch / uncertain"),
           ("ConvNeXt + CLIP", "max ensemble on frame", "deepfake_signal"),
           ("YOLOv8n (ONNX)", "COCO phone, book ≥ 0.4", "object_detected"),
           ("Energy VAD", "speech ratio ≥ 0.4", "talking_detected"),
           ("Head pose (solvePnP)", "yaw 25° / pitch 20° vs baseline", "looking_away"),
           ("Landmark motion", "≥ 3 still checks", "static_image_suspected"),
           ("User-Agent", "differs from 1st check", "device_mismatch_suspected")]
    w = 23.5
    for i, (t, s, flag) in enumerate(sig):
        x = 1 + (i % 4) * 25
        y = 41 if i < 4 else 5
        b = box(ax, x, y, w, 14, t, s + "\n→ " + flag, "teal", fs=9.5)
        if i < 4:
            arrow(ax, top(fr), bottom(b), color="#777")
        else:
            arrow(ax, bottom(fr), top(b), color="#777")
    save(fig, "monitor_signals")


def fig_er():
    fig, ax = canvas(11, 7.4)

    def ent(x, y, name, fields, fill="grey", w=21):
        h = 4.6 + 2.6 * len(fields)
        ax.add_patch(Rectangle((x, y - h), w, h, fc="white", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((x, y - 3.4), w, 3.4, fc=C[fill], ec=INK, lw=1.2))
        ax.text(x + w / 2, y - 1.7, name, ha="center", va="center", fontsize=9.5, fontweight="bold")
        for i, f in enumerate(fields):
            ax.text(x + 1, y - 5.3 - 2.6 * i, f, fontsize=8, va="center", family="monospace")
        return (x, y - h, w, h)

    us = ent(1, 72, "users", ["PK user_id", "email (unique)", "password_hash", "role", "join_code / org_id"], "violet")
    up = ent(28, 72, "uploads", ["PK upload_id", "FK user_id", "media_type", "storage_url", "file_size_kb"], "gold")
    dr = ent(55, 72, "detection_results", ["PK result_id", "FK upload_id (1:1)", "verdict", "confidence_score",
                                           "calibrated_confidence", "model_used"], "gold", w=23)
    ia = ent(82, 72, "image_analysis", ["FK result_id", "convnext_p, clip_p", "face_swap_p, face_px"], "blue", w=17)
    va = ent(82, 55, "video_analysis", ["FK result_id", "raw / calibrated p", "frames_with_face", "low_resolution"], "red", w=17)
    aa = ent(82, 36, "audio_analysis", ["FK result_id", "mean / std P(fake)"], "green", w=17)
    fb = ent(28, 48, "feedback", ["PK feedback_id", "FK upload_id, user_id", "agrees, true_label", "allow_reuse"], "grey")
    al = ent(1, 48, "audit_log", ["PK id", "user_id (no FK)", "action, ts"], "grey")
    ex = ent(1, 30, "exams", ["PK exam_id", "FK org user_id", "title, pass_threshold", "max_flags"], "teal")
    eq = ent(1, 10.5, "exam_questions", ["PK question_id", "FK exam_id", "options, answer"], "teal")
    ea = ent(28, 26, "exam_attempts", ["PK attempt_id", "FK exam_id, user_id", "FK session_id (1:1)", "status, score"], "teal")
    ms = ent(55, 40, "monitoring_sessions", ["PK session_id", "FK user_id", "started / ended"], "teal", w=23)
    mc = ent(55, 23, "monitoring_checks", ["PK check_id", "FK session_id", "similarity, yaw/pitch", "flagged, reasons"], "teal", w=23)
    fr = ent(28, 7.5, "face_references", ["FK user_id", "embedding_json"], "teal")
    links = [(right(us), left(up)), (right(up), left(dr)), (right(dr), left(ia)), (right(dr), left(va)), (right(dr), left(aa)),
             (bottom(up), top(fb)), (bottom(ex), top(eq)), (right(ex), left(ea)), (right(ea), left(ms)), (bottom(ms), top(mc)),
             (bottom(ea), top(fr))]
    for a, b in links:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#555", lw=1)
    save(fig, "er")


def fig_deploy():
    fig, ax = canvas(11, 5.4)
    ax.add_patch(Rectangle((22, 3), 76, 48, fc="#fafaf9", ec=INK, lw=1.2, ls="--"))
    ax.text(24, 48, "Docker host (docker-compose.prod.yml)", fontsize=10, style="italic")
    cl = box(ax, 1, 22, 16, 12, "Browser", "HTTPS", "violet", fs=10)
    tl = box(ax, 25, 22, 16, 12, "TLS proxy", "optional, 127.0.0.1", "grey", fs=10)
    fe = box(ax, 45, 26, 22, 18, "frontend container", "nginx.prod.conf\nCSP, security headers\nauth 10 r/min, upload 20 r/min", "violet", fs=9.5)
    be = box(ax, 72, 26, 24, 18, "backend container", "Dockerfile.prod, non-root\nuvicorn, DEBUG=false\nsecret-key guard", "gold", fs=9.5)
    vol = box(ax, 45, 6, 22, 12, "truthlens_data:/data", "SQLite, uploads, reports", "grey", fs=9.5)
    ck = box(ax, 72, 6, 24, 12, "checkpoints (read-only)", "SHA-256 verified fetch", "grey", fs=9.5)
    arrow(ax, right(cl), left(tl)); arrow(ax, right(tl), left(fe)); arrow(ax, right(fe), left(be), "/api")
    arrow(ax, (80, 26), top(vol), rad=0.2); arrow(ax, bottom(be), top(ck))
    save(fig, "deploy")


def fig_gantt():
    # first and last commit date of each phase (git log); work before the first commit is not dated, so not drawn
    phases = [("Review 2 build: upload, auth, PDF, history, owner scoping", "2026-09-03", "2026-09-10"),
              ("CLIP second opinion + portrait-app source", "2026-09-11", "2026-09-16"),
              ("Evaluation, explainability, provenance, jobs, CI, deployment", "2026-09-23", "2026-09-27"),
              ("Audio Model v1 (wav2vec 2.0) integration", "2026-09-29", "2026-09-29"),
              ("Video experiments, robustness, calibration, hardening", "2026-10-02", "2026-10-03"),
              ("ProctorLens: identity, monitoring signals, session reports", "2026-10-09", "2026-10-10"),
              ("Government ID, proctored MCQ exams, ProctorLens reorganisation", "2026-10-14", "2026-10-16"),
              ("Video calibration + round 4, image face-swap check", "2026-10-10", "2026-10-18"),
              ("UI polish, HEIC support, final report", "2026-10-19", "2026-10-20")]
    import datetime as dt
    d = lambda s: dt.date.fromisoformat(s)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (n, a, b) in enumerate(phases):
        ax.barh(i, max(1, (d(b) - d(a)).days + 1), left=d(a).toordinal(), color=list(S.values())[i % 6], alpha=0.75)
    ax.set_yticks(range(len(phases)), [p[0] for p in phases], fontsize=9.5)
    ax.invert_yaxis()
    ticks = [dt.date(2026, 9, 1), dt.date(2026, 9, 8), dt.date(2026, 9, 15), dt.date(2026, 9, 22),
             dt.date(2026, 9, 29), dt.date(2026, 10, 6), dt.date(2026, 10, 13), dt.date(2026, 10, 20)]
    ax.set_xticks([t.toordinal() for t in ticks], [t.strftime("%d %b") for t in ticks])
    ax.grid(axis="y", alpha=0)
    ax.set_title("Implementation phases by commit date (earlier, undated work not shown)", fontsize=11)
    save(fig, "gantt")


# ----------------------------------------------------------------------------------- charts
def bars(ax, groups, series, values, colors, ylabel, ylim=(0, 100), fmt="{:.1f}", fs=8):
    import numpy as np
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, col) in enumerate(zip(series, colors)):
        vals = [v[i] for v in values]
        bb = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=s, color=col, alpha=0.85)
        for r, v in zip(bb, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.01, fmt.format(v), ha="center", va="bottom", fontsize=fs)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0)


def chart_image_seen_unseen():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    systems = ["ConvNeXt", "CLIP", "Mean", "Max (deployed)"]
    seen = [[99.3, 99.5, 99.2], [97.6, 97.8, 97.5], [99.6, 99.8, 99.4], [98.4, 100.0, 96.8]]
    unseen = [[50.2, 12.0, 98.7], [49.7, 20.6, 86.7], [48.5, 10.9, 96.3], [53.2, 27.3, 86.0]]
    for ax, data, t in [(axes[0], seen, "Seen sources (4,000 images)"), (axes[1], unseen, "Unseen generators (2,517 images)")]:
        bars(ax, systems, ["Accuracy", "Fake recall", "Real specificity"], data, [S["blue"], S["red"], S["green"]], "%", (0, 110), "{:.0f}", 7.5)
        ax.set_title(t, fontsize=11)
    axes[1].set_ylabel("")
    axes[0].legend(fontsize=8.5, loc="lower left", ncol=3)
    save(fig, "c_image_seen_unseen")


def chart_confusion():
    """Counts from backend/eval/results/{image,video}/summary.csv (tp/tn/fp/fn at the threshold) and the deployed
    three-way verdict tables behind backend/eval/results/{image,video}/confusion_verdicts.png."""
    import numpy as np
    mats = [("Image ensemble, threshold 0.5", [[2889, 220], [1026, 2397]], ["REAL", "FAKE"]),
            ("Image ensemble, deployed bands", [[2826, 105, 178], [941, 166, 2316]], ["REAL", "UNCERTAIN", "FAKE"]),
            ("Video epoch 11, threshold 0.525", [[131, 24], [40, 115]], ["REAL", "FAKE"]),
            ("Video epoch 11, raw ±0.2 band", [[6, 149, 0], [0, 155, 0]], ["REAL", "UNCERTAIN", "FAKE"])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, (title, m, cols) in zip(axes.flat, mats):
        m = np.array(m)
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(m):
            ax.text(c, r, f"{v:,}\n({share[r, c]:.0%})", ha="center", va="center", fontsize=10,
                    color="white" if share[r, c] > 0.55 else INK)
        ax.set_xticks(range(len(cols)), cols, fontsize=9)
        ax.set_yticks([0, 1], ["true REAL", "true FAKE"], fontsize=9)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "c_confusion")


def chart_unseen_sources():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    names = ["GenImage SD", "GenImage MJ", "GenImage BigGAN", "OpenFake", "ChatGPT (8)", "Portrait app (15)"]
    vals = [30.7, 20.0, 26.0, 30.8, 0.0, 93.3]
    cols = [S["red"]] * 5 + [S["green"]]
    bb = ax.bar(names, vals, color=cols, alpha=0.85)
    for r, v in zip(bb, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("AI images caught (%)")
    ax.grid(axis="x", alpha=0)
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "c_unseen_sources")


def chart_video_r4():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    axes[0].bar(["epoch 11", "round 4"], [0.765, 0.961], color=[S["grey"], S["red"]]); axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("ROC-AUC, 356 held-out videos", fontsize=10)
    axes[1].bar(["epoch 11", "round 4"], [21, 8], color=[S["grey"], S["green"]]); axes[1].set_title("Real videos wrongly FAKE (of 185)", fontsize=10)
    axes[2].bar(["epoch 11", "round 4"], [82, 114], color=[S["grey"], S["blue"]]); axes[2].set_title("Fakes caught (of 171)", fontsize=10)
    for ax, vals, f in [(axes[0], [0.765, 0.961], "{:.3f}"), (axes[1], [21, 8], "{}"), (axes[2], [82, 114], "{}")]:
        for i, v in enumerate(vals):
            ax.text(i, v, f.format(v), ha="center", va="bottom", fontsize=9)
        ax.grid(axis="x", alpha=0)
    save(fig, "c_video_r4")


def chart_video_variants():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    groups = ["Deployed\n(Haar, 20 % pad)", "+ rotation\nfallback", "Two-cascade\nconsensus", "Wider 35 %\npadding"]
    vals = [[79.3, 74.7, 84.0], [79.3, 74.7, 84.0], [79.0, 74.0, 84.0], [76.3, 74.0, 78.7]]
    bars(ax, groups, ["Accuracy", "Fake recall", "Real specificity"], vals, [S["blue"], S["red"], S["green"]], "%", (60, 90))
    ax.legend(fontsize=8.5, ncol=3, loc="upper right")
    ax.set_title("300 unseen RTFS clips, epoch-11 weights (AUC 0.895 / 0.895 / 0.890 / 0.849)", fontsize=10)
    save(fig, "c_video_variants")


def chart_video_res():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    px = [1080, 720, 480, 360, 240]
    auc = [0.923, 0.919, 0.885, 0.865, 0.729]
    ax.plot(px, auc, "o-", color=S["red"])
    for x, y in zip(px, auc):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.axvline(360, ls="--", color=S["gold"]); ax.text(372, 0.75, "guard: < 360 px\n→ UNCERTAIN", fontsize=9, color=S["gold"])
    ax.set_xlabel("short side of the video (px)"); ax.set_ylabel("ROC-AUC (RTFS test split)")
    ax.invert_xaxis(); ax.set_ylim(0.7, 0.95)
    save(fig, "c_video_res")


def chart_img_robust():
    rows = list(csv.DictReader(open(RES / "robustness" / "summary.csv", encoding="utf-8")))
    settings = [("clean", "0", "clean"), ("jpeg", "90", "JPEG q90"), ("jpeg", "50", "JPEG q50"), ("jpeg", "10", "JPEG q10"),
                ("noise", "5", "noise σ5"), ("noise", "20", "noise σ20"), ("blur", "3", "blur σ3"), ("resize", "0.25", "resize 25 %"),
                ("social", "screenshot", "screenshot"), ("social", "whatsapp", "WhatsApp"), ("social", "webp", "WebP")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in [(axes[0], "real_specificity", "Real-photo specificity (seen sources)"),
                              (axes[1], "fake_recall", "Fake recall (seen sources)")]:
        for sysname, col in [("convnext", S["blue"]), ("clip", S["green"]), ("max", S["red"])]:
            ys = []
            for fam, lvl, _ in settings:
                r = next(r for r in rows if r["group"] == "seen" and r["family"] == fam and r["level"] == lvl and r["system"] == sysname)
                ys.append(100 * float(r[metric]))
            ax.plot(range(len(settings)), ys, "o-", ms=4, color=col, label=sysname if sysname != "max" else "max (deployed)")
        ax.set_xticks(range(len(settings)), [s[2] for s in settings], rotation=45, ha="right", fontsize=8.5)
        ax.set_ylim(40, 102); ax.set_title(title, fontsize=10.5); ax.set_ylabel("%")
    axes[0].legend(fontsize=8.5, loc="lower left")
    save(fig, "c_img_robust")


def chart_vid_robust():
    fig, ax = plt.subplots(figsize=(9, 3.3))
    names = ["clean", "JPEG q90", "JPEG q50", "JPEG q10", "resize 50 %", "resize 25 %", "blur σ1", "blur σ3", "noise σ10", "WebP", "WhatsApp"]
    auc = [0.665, 0.745, 0.625, 0.585, 0.570, 0.465, 0.540, 0.410, 0.685, 0.545, 0.610]
    rec = [70, 75, 10, 0, 60, 30, 70, 30, 30, 25, 25]
    ax.bar(range(len(names)), auc, color=[S["red"] if a < 0.5 else S["blue"] for a in auc], alpha=0.85, label="ROC-AUC")
    ax.axhline(0.5, ls="--", color="#555"); ax.text(10.4, 0.51, "chance", fontsize=8.5, ha="right")
    ax2 = ax.twinx(); ax2.plot(range(len(names)), rec, "s-", color=S["gold"], label="fake recall %"); ax2.set_ylim(0, 100)
    ax2.set_ylabel("fake recall (%)"); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8.5)
    ax.set_ylim(0, 0.85); ax.set_ylabel("ROC-AUC"); ax.grid(axis="x", alpha=0)
    ax.set_title("Video Model v1 (epoch 11) on 30 RTFS clips, every sampled frame degraded", fontsize=10)
    save(fig, "c_vid_robust")


def chart_calibration():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars(ax, ["Image (n = 6,500)\ntemperature T = 4.45", "Video (n = 300)\ntemperature T = 0.116", "Video round 4 test split\nPlatt (pooled)"],
         ["raw", "calibrated"], [[0.158, 0.056], [0.254, 0.059], [0.161, 0.078]], [S["grey"], S["green"]], "ECE (10 bins)", (0, 0.3), "{:.3f}", 8.5)
    ax.legend(fontsize=9)
    save(fig, "c_calibration")


def chart_fairness():
    rows = list(csv.DictReader(open(RES / "fairness" / "per_source.csv", encoding="utf-8")))
    groups = sorted({r["source"] for r in rows})
    lab = [g.replace("utkface_", "").replace("_", "\n") for g in groups]
    data = [[100 * float(next(r for r in rows if r["source"] == g and r["system"] == s)["real_specificity"]) for s in ("convnext", "clip", "max")] for g in groups]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bars(ax, lab, ["ConvNeXt", "CLIP", "max (deployed)"], data, [S["blue"], S["green"], S["red"]], "real kept REAL (%)", (80, 103), "{:.0f}", 6.5)
    ax.legend(fontsize=8.5, ncol=3, loc="lower right")
    ax.tick_params(axis="x", labelsize=8.5)
    save(fig, "c_fairness")


def chart_faceswap():
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    groups = ["Research video frames\n(356 held-out videos)", "inswapper photo pairs\n(Sowaiba01, 250 pairs)", "RWFS real apps\n(500 swaps + 500 real)"]
    vals = [[0.61, 0.95], [0.659, 0.859], [0.675, 0.617]]
    bars(ax, groups, ["whole-image ensemble", "face-swap check"], vals, [S["blue"], S["red"]], "ROC-AUC", (0.4, 1.05), "{:.2f}", 9)
    ax.axhline(0.5, ls="--", color="#777")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Face-swap check (round-4 weights) vs whole-image models; photo model r2 raised RWFS held-out-group AUC to 0.77–0.83", fontsize=9.5)
    save(fig, "c_faceswap")


def chart_audio():
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    names = ["VCC 2020 stress test\n(59 systems), v8", "VCC 2020 stress test\n(59 systems), v9", "Unseen LibriSpeech\ngenuine (10), v9"]
    vals = [50.8, 98.3, 100.0]
    bb = ax.bar(names, vals, color=[S["grey"], S["green"], S["green"]], alpha=0.85)
    for r, v in zip(bb, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel("correct (%)"); ax.grid(axis="x", alpha=0)
    save(fig, "c_audio")


def chart_identity_thresholds():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(["Accuracy", "False accept", "False reject"], [96.5, 0.4, 6.6], color=[S["blue"], S["red"], S["gold"]], alpha=0.85)
    for i, v in enumerate([96.5, 0.4, 6.6]):
        ax.text(i, v + 1.5, f"{v}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110); ax.set_ylabel("%"); ax.grid(axis="x", alpha=0)
    ax.set_title("FaceNet cosine threshold 0.44 on 1,000 LFW pairs (AUC 0.974)", fontsize=10)
    save(fig, "c_identity")


if __name__ == "__main__":
    for f in [fig_architecture, fig_usecase, fig_dfd0, fig_dfd1, fig_pipe_image, fig_pipe_video, fig_pipe_audio, fig_band,
              fig_provenance, fig_seq_scan, fig_seq_job, fig_proctor_flow, fig_monitor_signals, fig_er, fig_deploy, fig_gantt,
              chart_image_seen_unseen, chart_confusion, chart_unseen_sources, chart_video_r4, chart_video_variants, chart_video_res,
              chart_img_robust, chart_vid_robust, chart_calibration, chart_fairness, chart_faceswap, chart_audio,
              chart_identity_thresholds]:
        f()
