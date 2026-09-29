"""Image degradations that mimic what happens to a picture on its way through the internet.

Each function takes an RGB PIL image and returns a degraded RGB PIL image of a plausible size. Everything is
deterministic (noise is seeded from the image name + setting), so any number in a report can be reproduced.

Families and levels
  jpeg    quality 90/70/50/30/10 (re-compression, the most common alteration)
  resize  downscale to 75/50/25 % then back up (a thumbnail that was enlarged again / a re-share at lower resolution)
  blur    Gaussian blur, sigma 0.5/1/2/3 px (soft focus, motion, upscaling)
  noise   Gaussian pixel noise, std 5/10/20 on a 0-255 scale (sensor noise, night photos, grain)
  crop    centre crop keeping 80 % / 60 % of each side (framing changes)
  social  "screenshot" (rescaled by 0.8-1.25x, lossless, metadata gone), "whatsapp" (max side 1600 px + JPEG q65) and
          "webp" (WebP q75)
"""

from __future__ import annotations

import io
import zlib

import numpy as np
from PIL import Image, ImageFilter

FAMILIES: dict[str, dict] = {
    "jpeg": {"title": "JPEG re-compression", "xlabel": "JPEG quality (lower = worse)", "levels": [90, 70, 50, 30, 10]},
    "resize": {"title": "Downscale then restore", "xlabel": "scale kept", "levels": [0.75, 0.5, 0.25]},
    "blur": {"title": "Gaussian blur", "xlabel": "sigma (px)", "levels": [0.5, 1, 2, 3]},
    "noise": {"title": "Gaussian noise", "xlabel": "noise std (0-255)", "levels": [5, 10, 20]},
    "crop": {"title": "Centre crop", "xlabel": "share of each side kept", "levels": [0.8, 0.6]},
    "social": {"title": "Sharing pipelines", "xlabel": "pipeline", "levels": ["screenshot", "whatsapp", "webp"]},
}


def settings(include_clean: bool = True) -> list[tuple[str, object]]:
    """Every (family, level) in a stable order; ("clean", 0) is the untouched baseline."""
    out: list[tuple[str, object]] = [("clean", 0)] if include_clean else []
    for fam, spec in FAMILIES.items():
        out += [(fam, lvl) for lvl in spec["levels"]]
    return out


def _seed(key: str, family: str, level) -> int:
    return zlib.crc32(f"{key}|{family}|{level}".encode())


def _roundtrip(img: Image.Image, fmt: str, **kw) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    return _roundtrip(img, "JPEG", quality=int(quality))


def resize(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.Resampling.BICUBIC)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def noise(img: Image.Image, std: float, seed: int) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng(seed)
    return Image.fromarray(np.clip(arr + rng.normal(0.0, float(std), arr.shape), 0, 255).astype(np.uint8))


def crop(img: Image.Image, keep: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(8, round(w * keep)), max(8, round(h * keep))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def social(img: Image.Image, kind: str, seed: int) -> Image.Image:
    if kind == "screenshot":
        factor = 0.8 + 0.45 * (seed % 1000) / 999.0                       # 0.80 .. 1.25, deterministic per image
        w, h = img.size
        shot = img.resize((max(8, round(w * factor)), max(8, round(h * factor))), Image.Resampling.BICUBIC)
        return _roundtrip(shot, "PNG")                                      # lossless, no metadata
    if kind == "whatsapp":
        w, h = img.size
        scale = min(1.0, 1600 / max(w, h))
        small = img.resize((max(8, round(w * scale)), max(8, round(h * scale))), Image.Resampling.BICUBIC) if scale < 1 else img
        return _roundtrip(small, "JPEG", quality=65)
    if kind == "webp":
        return _roundtrip(img, "WEBP", quality=75)
    raise ValueError(f"unknown social pipeline: {kind}")


def apply(family: str, level, img: Image.Image, key: str = "") -> Image.Image:
    """Degrade `img`. `key` (e.g. the file name) seeds the noise / screenshot scale so results are reproducible."""
    img = img.convert("RGB")
    if family == "clean":
        return img
    seed = _seed(key, family, level)
    if family == "jpeg":
        return jpeg(img, level)
    if family == "resize":
        return resize(img, level)
    if family == "blur":
        return blur(img, level)
    if family == "noise":
        return noise(img, level, seed)
    if family == "crop":
        return crop(img, level)
    if family == "social":
        return social(img, level, seed)
    raise ValueError(f"unknown perturbation family: {family}")
