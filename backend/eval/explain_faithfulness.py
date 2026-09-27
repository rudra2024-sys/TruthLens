"""Deletion test: do Grad-CAM heatmaps point at regions the model really uses?

For fake images that ConvNeXt-Tiny already scores FAKE, mask the top-K% most important pixels according to
Grad-CAM (replace with the dataset mean colour) and record how far the FAKE probability falls. Compare with
masking the same area at random (mean of several random masks). If the heatmaps are faithful, the Grad-CAM
mask should hurt the FAKE score clearly more than a random mask of equal size.

  python -m eval.explain_faithfulness --pred eval/data/pred_b1.csv eval/data/pred_b2.csv ... --n 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("IMAGE_MODEL_DEVICE", "cpu")

from app.pipelines.image.inference import ImagePipeline                      # noqa: E402
from app.pipelines.image.preprocessing import IMAGE_TRANSFORM                 # noqa: E402
from app.services.explain import core                                         # noqa: E402
from app.services.explain.image_explainer import gradcam_for_tensor           # noqa: E402

CKPT = REPO / "models" / "checkpoints" / "image" / "convnext_tiny_diversified_v2.pth"


def fake_prob(model, x):
    with torch.inference_mode():
        return float(torch.softmax(model(x), dim=1)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.20, help="fraction of pixels to mask")
    ap.add_argument("--random-masks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(BACKEND / "eval" / "results" / "explain_faithfulness.md"))
    args = ap.parse_args()

    rows = []
    for p in args.pred:
        with open(p, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f)]
    # fakes ConvNeXt itself already calls FAKE (nothing to explain otherwise), spread over sources
    cand = [r for r in rows if r["label"] == "1" and float(r["convnext_fake"]) >= 0.6]
    rng = np.random.default_rng(args.seed)
    by_src = {}
    for r in cand:
        by_src.setdefault(r["source"], []).append(r)
    picked = []
    per = max(1, args.n // max(len(by_src), 1))
    for s, lst in sorted(by_src.items()):
        idx = rng.permutation(len(lst))[:per]
        picked += [lst[i] for i in idx]
    print(f"{len(cand)} candidate fakes over {len(by_src)} sources; testing {len(picked)}")

    pipe = ImagePipeline(checkpoint_path=str(CKPT))
    model, device = pipe.model, pipe.device

    res = []
    for r in picked:
        with Image.open(r["path"]) as im:
            x = IMAGE_TRANSFORM(im.convert("RGB")).unsqueeze(0).to(device)
        base = fake_prob(model, x)
        cam, _ = gradcam_for_tensor(model, x)
        heat = core.upsample_cam(cam, (224, 224))
        k = int(args.frac * 224 * 224)
        order = np.argsort(-heat.reshape(-1))
        m_cam = np.zeros(224 * 224, dtype=bool); m_cam[order[:k]] = True
        m_cam = torch.from_numpy(m_cam.reshape(1, 1, 224, 224)).to(device)
        x_cam = torch.where(m_cam, torch.zeros_like(x), x)          # 0 in normalised space = mean colour
        drop_cam = base - fake_prob(model, x_cam)

        drops_rand = []
        for _ in range(args.random_masks):
            m = np.zeros(224 * 224, dtype=bool); m[rng.permutation(224 * 224)[:k]] = True
            # random *blobs*, not salt-and-pepper: mask a shuffled set of 16x16 cells with equal total area
            cells = rng.permutation(14 * 14)[: int(round(args.frac * 14 * 14))]
            mc = np.zeros((14, 14), dtype=bool); mc.reshape(-1)[cells] = True
            mc = np.kron(mc, np.ones((16, 16), dtype=bool))
            mt = torch.from_numpy(mc.reshape(1, 1, 224, 224)).to(device)
            drops_rand.append(base - fake_prob(model, torch.where(mt, torch.zeros_like(x), x)))
        res.append((r["source"], base, drop_cam, float(np.mean(drops_rand))))

    a = np.array([[b, dc, dr] for _, b, dc, dr in res])
    win = float((a[:, 1] > a[:, 2]).mean())
    lines = [
        "# Grad-CAM faithfulness (deletion test)\n",
        f"- Model: ConvNeXt-Tiny v2 (the model the heatmap explains). Images: {len(res)} fakes it scores FAKE (>= 0.6), "
        f"sampled across {len(by_src)} sources.",
        f"- Mask: top {args.frac:.0%} of pixels by Grad-CAM vs. the same area as random 16x16 blocks "
        f"(mean of {args.random_masks} draws), filled with the dataset mean colour.\n",
        "| | mean FAKE probability |",
        "|---|---|",
        f"| original | {a[:, 0].mean():.3f} |",
        f"| after masking Grad-CAM region | {(a[:, 0] - a[:, 1]).mean():.3f} |",
        f"| after masking random region | {(a[:, 0] - a[:, 2]).mean():.3f} |\n",
        f"- Mean drop: Grad-CAM **{a[:, 1].mean():.3f}** vs random **{a[:, 2].mean():.3f}**.",
        f"- Grad-CAM mask hurt the FAKE score more than the random mask on **{win:.0%}** of images.\n",
        "| source | n | mean drop (Grad-CAM) | mean drop (random) | Grad-CAM wins |",
        "|---|---|---|---|---|",
    ]
    for s in sorted({r[0] for r in res}):
        sub = np.array([[dc, dr] for src, _, dc, dr in res if src == s])
        lines.append(f"| {s} | {len(sub)} | {sub[:, 0].mean():.3f} | {sub[:, 1].mean():.3f} | "
                     f"{(sub[:, 0] > sub[:, 1]).mean():.0%} |")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
