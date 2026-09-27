"""Error Level Analysis (ELA): a visual aid, NOT a detector.

Re-saves a JPEG at a fixed quality and shows where the recompression error is larger or smaller. Regions with a
different compression history (e.g. pasted in, or re-saved separately) can stand out. It is unreliable in general:
resizing, screenshots, social-media recompression and plain image content (edges, texture) all change the map, and
it says nothing about images from AI generators that never went through the same JPEG history. So it is shown as
a labelled visualisation with summary numbers, never as a verdict. `eval/ela_usefulness.py` measures how well the
summary numbers actually separate real from fake on our test sets.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops

from app.services.explain import core

QUALITY = 90
DISPLAY_MAX_SIDE = 512
BLOCK = 16


@dataclass
class ElaResult:
    applicable: bool
    reason: str | None = None
    quality: int = QUALITY
    mean_error: float | None = None          # mean recompression error, 0..255
    p95_error: float | None = None
    max_error: float | None = None
    block_cv: float | None = None            # spread of per-16px-block mean error (std / mean): higher = more uneven
    heatmap_jpeg: bytes | None = None


def error_level_analysis(path: str) -> ElaResult:
    try:
        with Image.open(path) as im:
            fmt = im.format
            rgb = im.convert("RGB")
    except Exception as e:
        return ElaResult(applicable=False, reason=f"could not read image ({type(e).__name__})")

    if fmt != "JPEG":
        return ElaResult(
            applicable=False,
            reason=f"ELA is only meaningful for JPEG files; this file is {fmt or 'unknown'} (already lossless/"
                   "differently compressed), so the result would not be interpretable.",
        )

    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=QUALITY)
    buf.seek(0)
    with Image.open(buf) as re_saved:
        diff = ImageChops.difference(rgb, re_saved.convert("RGB"))
    err = np.asarray(diff, dtype=np.float32).mean(axis=2)               # (H, W), 0..255
    peak = float(err.max())

    h, w = err.shape
    hb, wb = h // BLOCK, w // BLOCK
    block_cv = None
    if hb >= 2 and wb >= 2:
        blocks = err[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
        m = float(blocks.mean())
        block_cv = float(blocks.std() / m) if m > 1e-6 else 0.0

    scaled = np.clip(err / max(peak, 1.0), 0.0, 1.0) ** 0.6              # gamma lifts faint structure
    heat = Image.fromarray(core.colormap(scaled))
    heat = core.fit_max_side(heat, DISPLAY_MAX_SIDE)
    return ElaResult(
        applicable=True,
        mean_error=float(err.mean()),
        p95_error=float(np.percentile(err, 95)),
        max_error=peak,
        block_cv=block_cv,
        heatmap_jpeg=core.encode(heat),
    )
