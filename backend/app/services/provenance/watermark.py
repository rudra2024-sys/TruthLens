"""Detects the fixed "invisible watermark" that diffusers' default Stable Diffusion / SDXL pipeline embeds
(huggingface/diffusers `pipelines/stable_diffusion_xl/watermark.py`, method name "dwtDct", from the
`invisible-watermark` project: https://github.com/ShieldMnt/invisible-watermark).

This is read-only, additive evidence, same as the rest of `provenance/` - it never changes a stored result,
threshold or model score.

`detect_sd_watermark()` checks a single image. `detect_sd_watermark_video()` (added 2026-10-02) runs the exact
same per-frame decode on a handful of evenly-sampled video frames - a video is just a sequence of frames the
watermark could be embedded into, same as an image, and some generators/editors re-encode frame-by-frame rather
than applying one watermark to the whole container, so checking several frames catches a watermark that survived
on only some of them. `present` is True if ANY sampled frame matches exactly.

What this catches (and does not):
  * diffusers embeds the SAME fixed 48-bit message in every image by default, unless the caller disabled the
    watermarker - which many popular front-ends (AUTOMATIC1111, ComfyUI, most hosted SDXL APIs) do. So a miss
    proves nothing: it could be a real photo, an SD/SDXL image from a pipeline that skipped the watermark, or
    an image from a different generator entirely (Midjourney, DALL-E, Imagen, ...) that never used this scheme.
  * the watermark is destroyed by resizing, cropping, or noticeable recompression, so a miss on a re-uploaded
    or re-encoded copy also proves nothing - see EVALUATION_SUMMARY.md's robustness findings for how much
    typical re-sharing (screenshots, WhatsApp-style recompression) degrades pixel-level signals.
  * an exact 48/48 bit match is a strong signal when it happens: the chance of an unrelated image matching a
    fixed 48-bit pattern by chance is negligible, so unlike EXIF this is treated as strong "ai" evidence, not
    a weak one - but it is still just one signal, reported next to (never fused into) the verdict.

Algorithm (verified bit-for-bit against the real `invisible-watermark` package's `EmbedMaxDct` class in a scratch
venv before being wired in - see `backend/tests/test_watermark.py`): convert BGR -> YUV, take a single-level
Haar DWT of the U channel
(diffusers/imwatermark's default scales embed only into channel index 1 of YUV), split the approximation
sub-band into non-overlapping 4x4 blocks, and for each block read one bit from the block's largest-magnitude
non-DC coefficient (`(abs(coef) % 36) > 18` => 1). Blocks are assigned to the 48 bit positions round-robin
and each position's bit is the majority vote across all blocks assigned to it. No actual DCT is computed -
"dwtDct" is the upstream library's method name, not a literal description of this step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# The fixed message diffusers' StableDiffusionXLWatermarker / StableDiffusionSafetyChecker embed by default.
_WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
_WATERMARK_BITS = np.array([int(b) for b in bin(_WATERMARK_MESSAGE)[2:]], dtype=np.uint8)
_WM_LEN = len(_WATERMARK_BITS)  # 48

_BLOCK = 4
_SCALE = 36.0            # imwatermark's default scale for the channel it actually uses
_MIN_PIXELS = 256 * 256  # imwatermark itself refuses to encode/decode below this

_cv2 = None
_pywt = None
_import_failed = False


@dataclass
class WatermarkResult:
    applicable: bool
    reason: str | None = None
    present: bool = False           # exact 48/48 bit match against the known diffusers message
    bit_match: float | None = None  # fraction of the 48 bits that matched, 0..1 (informational, not a verdict)
    decoded_hex: str | None = None


@dataclass
class VideoWatermarkResult:
    """Same check as WatermarkResult, applied to a handful of sampled frames instead of one image.

    Frame-by-frame rather than a single pass: recompression can destroy the watermark on some frames of a video
    and not others (see the module docstring's note that resizing/recompression kills it), so `present` is True
    if ANY sampled frame matches, not all of them - a miss on most frames with a hit on one is still a real hit.
    """
    applicable: bool
    reason: str | None = None
    present: bool = False
    frames_checked: int = 0
    frames_matched: int = 0
    best_bit_match: float | None = None   # highest per-frame bit_match seen, 0..1
    decoded_hex: str | None = None        # decoded bits from the best-matching frame


def _load_libs():
    global _cv2, _pywt, _import_failed
    if _cv2 is not None or _import_failed:
        return _cv2, _pywt
    try:
        import cv2
        import pywt

        _cv2, _pywt = cv2, pywt
    except Exception:
        logger.exception("cv2/pywt unavailable; SD watermark check disabled")
        _import_failed = True
    return _cv2, _pywt


def _infer_bit(block: np.ndarray, scale: float) -> int:
    flat = np.abs(block.flatten()[1:])
    pos = int(np.argmax(flat)) + 1
    i, j = divmod(pos, block.shape[1])
    val = abs(float(block[i, j]))
    return 1 if (val % scale) > 0.5 * scale else 0


def _decode_bits(bgr: np.ndarray) -> np.ndarray | None:
    cv2, pywt = _load_libs()
    if cv2 is None:
        return None
    rows, cols = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    channel = yuv[: rows // 4 * 4, : cols // 4 * 4, 1]
    ca1, _ = pywt.dwt2(channel, "haar")
    br, bc = ca1.shape[0] // _BLOCK, ca1.shape[1] // _BLOCK
    if br * bc < _WM_LEN:
        return None
    scores: list[list[int]] = [[] for _ in range(_WM_LEN)]
    num = 0
    for i in range(br):
        for j in range(bc):
            block = ca1[i * _BLOCK:(i + 1) * _BLOCK, j * _BLOCK:(j + 1) * _BLOCK]
            scores[num % _WM_LEN].append(_infer_bit(block, _SCALE))
            num += 1
    avg = np.array([np.mean(s) for s in scores])
    return (avg * 255 > 127).astype(np.uint8)  # matches imwatermark's exact threshold, not the same as avg > 0.5


def _check_bgr_frame(bgr: np.ndarray) -> tuple[float, str] | None:
    """Runs the watermark decode on one already-decoded BGR frame/image. Returns (bit_match, decoded_hex),
    or None if the frame is too small or too low-resolution to read all 48 bits from."""
    rows, cols = bgr.shape[:2]
    if rows * cols < _MIN_PIXELS:
        return None
    bits = _decode_bits(bgr)
    if bits is None:
        return None
    match = float((bits == _WATERMARK_BITS).mean())
    decoded_hex = "%012x" % int("".join(str(b) for b in bits.tolist()), 2)
    return match, decoded_hex


def detect_sd_watermark(path: str) -> WatermarkResult:
    cv2, _pywt_mod = _load_libs()
    if cv2 is None:
        return WatermarkResult(False, reason="watermark decoder libraries are not installed")
    try:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception as e:
        return WatermarkResult(False, reason=f"could not read image ({type(e).__name__})")
    if bgr is None:
        return WatermarkResult(False, reason="could not decode image file")
    checked = _check_bgr_frame(bgr)
    if checked is None:
        return WatermarkResult(False, reason="image is smaller than 256x256, too small for this watermark's block grid")
    match, decoded_hex = checked
    return WatermarkResult(
        applicable=True,
        present=match == 1.0,
        bit_match=match,
        decoded_hex=decoded_hex,
    )


def detect_sd_watermark_video(path: str, max_frames: int = 8) -> VideoWatermarkResult:
    """Samples up to `max_frames` evenly-spaced frames from the video and runs the same per-frame check as
    detect_sd_watermark() on each. Independent of the video detector's own 16-frame sampling (video/model_v1) -
    this is a provenance signal, not a model input, so it degrades gracefully on short clips instead of raising
    ShortVideoError: fewer frames than requested is still a valid (partial) check, not a failure.
    """
    cv2, _pywt_mod = _load_libs()
    if cv2 is None:
        return VideoWatermarkResult(False, reason="watermark decoder libraries are not installed")
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return VideoWatermarkResult(False, reason="could not open video file")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return VideoWatermarkResult(False, reason="video reports no frames")

        n = max(1, min(max_frames, frame_count))
        positions = sorted(set(np.linspace(0, frame_count - 1, n).astype(int).tolist()))

        checked = 0
        matched = 0
        best_match: float | None = None
        best_hex: str | None = None
        current = 0
        for target in positions:
            while current < target:
                if not cap.grab():
                    break
                current += 1
            ok, frame = cap.read()
            current += 1
            if not ok:
                continue
            result = _check_bgr_frame(frame)
            if result is None:
                continue
            match, decoded_hex = result
            checked += 1
            if match == 1.0:
                matched += 1
            if best_match is None or match > best_match:
                best_match, best_hex = match, decoded_hex

        if checked == 0:
            return VideoWatermarkResult(
                False, reason="no sampled frame was large enough to check (all below 256x256)"
            )
        return VideoWatermarkResult(
            applicable=True,
            present=matched > 0,
            frames_checked=checked,
            frames_matched=matched,
            best_bit_match=best_match,
            decoded_hex=best_hex,
        )
    finally:
        cap.release()
