"""SD/SDXL fixed invisible-watermark detector (app/services/provenance/watermark.py).

The decode algorithm was hand-verified bit-for-bit against the real `invisible-watermark` package
(ShieldMnt/invisible-watermark, the library diffusers uses) in a scratch venv before being wired in - see the
module docstring. These tests don't depend on that package (not a project dependency); the positive-path test
embeds the watermark itself using the same DWT-block math, verified in that same scratch-venv session to
round-trip through our decoder identically to a real diffusers-encoded image.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pywt = pytest.importorskip("pywt")

from app.services.provenance.watermark import _WATERMARK_BITS, detect_sd_watermark, detect_sd_watermark_video
from app.services.provenance.service import build_provenance
import types


def _random_image(h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def _embed_known_watermark(bgr, scale=36.0, block=4):
    """Mirrors imwatermark's EmbedMaxDct.encode/diffuse_dct_matrix exactly (dwtDct method, default scale)."""
    bits = _WATERMARK_BITS.tolist()
    rows, cols = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV).astype(np.float64)
    ch = yuv[: rows // 4 * 4, : cols // 4 * 4, 1].copy()
    ca1, (h1, v1, d1) = pywt.dwt2(ch, "haar")
    br, bc = ca1.shape[0] // block, ca1.shape[1] // block
    num = 0
    for i in range(br):
        for j in range(bc):
            blk = ca1[i * block:(i + 1) * block, j * block:(j + 1) * block]
            bit = bits[num % len(bits)]
            pos = int(np.argmax(np.abs(blk.flatten()[1:]))) + 1
            bi, bj = divmod(pos, block)
            val = blk[bi, bj]
            if val >= 0:
                blk[bi, bj] = (val // scale + 0.25 + 0.5 * bit) * scale
            else:
                blk[bi, bj] = -1.0 * (abs(val) // scale + 0.25 + 0.5 * bit) * scale
            ca1[i * block:(i + 1) * block, j * block:(j + 1) * block] = blk
            num += 1
    rec = pywt.idwt2((ca1, (h1, v1, d1)), "haar")
    yuv[: rows // 4 * 4, : cols // 4 * 4, 1] = rec[: rows // 4 * 4, : cols // 4 * 4]
    return cv2.cvtColor(np.clip(yuv, 0, 255).astype(np.uint8), cv2.COLOR_YUV2BGR)


def up(path):
    return types.SimpleNamespace(media_type="image", storage_url=str(path), upload_id="t")


def test_watermarked_image_is_detected_exactly(tmp_path):
    wm = _embed_known_watermark(_random_image(seed=1))
    p = tmp_path / "wm.png"
    cv2.imwrite(str(p), wm)
    r = detect_sd_watermark(str(p))
    assert r.applicable and r.present is True and r.bit_match == 1.0


def test_unwatermarked_random_image_is_not_detected(tmp_path):
    for seed in range(8):
        p = tmp_path / f"clean_{seed}.png"
        cv2.imwrite(str(p), _random_image(seed=seed))
        r = detect_sd_watermark(str(p))
        assert r.applicable
        assert r.present is False, "false positive on plain random image"


def test_watermark_does_not_survive_jpeg_recompression(tmp_path):
    """Documents real fragility (per the module docstring) rather than a bug: this is expected behaviour."""
    wm = _embed_known_watermark(_random_image(seed=2))
    p = tmp_path / "wm.jpg"
    cv2.imwrite(str(p), wm, [cv2.IMWRITE_JPEG_QUALITY, 90])
    r = detect_sd_watermark(str(p))
    assert r.applicable and r.present is False


def test_image_smaller_than_256px_is_not_applicable(tmp_path):
    p = tmp_path / "small.png"
    cv2.imwrite(str(p), _random_image(h=128, w=128, seed=3))
    r = detect_sd_watermark(str(p))
    assert r.applicable is False and "256" in r.reason


def test_bad_or_missing_file_never_raises(tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    assert detect_sd_watermark(str(junk)).applicable is False
    assert detect_sd_watermark(str(tmp_path / "missing.png")).applicable is False


def test_present_watermark_feeds_the_provenance_assessment(tmp_path):
    wm = _embed_known_watermark(_random_image(seed=4))
    p = tmp_path / "wm.png"
    cv2.imwrite(str(p), wm)
    pv = build_provenance(up(p))
    assert pv.watermark.present is True
    assert pv.assessment.level == "declared_ai"
    assert any(s.kind == "ai" and s.strength == "strong" for s in pv.signals)
    # the top-level "nothing found" signal must not also fire alongside a real positive match
    assert not any(s.kind == "absent" for s in pv.signals)


def test_absent_watermark_does_not_change_the_none_assessment(tmp_path):
    p = tmp_path / "clean.png"
    cv2.imwrite(str(p), _random_image(seed=5))
    pv = build_provenance(up(p))
    assert pv.watermark.applicable and pv.watermark.present is False
    assert pv.assessment.level == "none"


def _write_lossless_video(path, frames):
    """FFV1 (lossless) so pixel values - and therefore the watermark - survive encoding exactly, same requirement
    documented in the module docstring (any real lossy compression destroys the watermark). Verified round-trip
    byte-identical in this environment before being used here."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 5, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return True


def up_video(path):
    return types.SimpleNamespace(media_type="video", storage_url=str(path), upload_id="t")


def test_video_with_a_watermarked_frame_is_detected(tmp_path):
    clean = [_random_image(h=320, w=320, seed=s) for s in range(6)]
    frames = clean[:3] + [_embed_known_watermark(clean[3])] + clean[4:]
    p = tmp_path / "wm.avi"
    if not _write_lossless_video(p, frames):
        pytest.skip("this OpenCV build cannot write an uncompressed AVI")
    r = detect_sd_watermark_video(str(p), max_frames=6)
    assert r.applicable and r.present is True
    assert r.frames_matched >= 1
    assert r.frames_checked >= r.frames_matched


def test_video_with_no_watermarked_frames_is_not_detected(tmp_path):
    frames = [_random_image(h=320, w=320, seed=s) for s in range(6)]
    p = tmp_path / "clean.avi"
    if not _write_lossless_video(p, frames):
        pytest.skip("this OpenCV build cannot write an uncompressed AVI")
    r = detect_sd_watermark_video(str(p), max_frames=6)
    assert r.applicable and r.present is False and r.frames_matched == 0


def test_video_bad_or_missing_file_never_raises(tmp_path):
    junk = tmp_path / "junk.avi"
    junk.write_bytes(b"not a video")
    assert detect_sd_watermark_video(str(junk)).applicable is False
    assert detect_sd_watermark_video(str(tmp_path / "missing.avi")).applicable is False


def test_watermarked_video_feeds_the_provenance_assessment(tmp_path):
    clean = [_random_image(h=320, w=320, seed=s) for s in range(6)]
    frames = clean[:3] + [_embed_known_watermark(clean[3])] + clean[4:]
    p = tmp_path / "wm.avi"
    if not _write_lossless_video(p, frames):
        pytest.skip("this OpenCV build cannot write an uncompressed AVI")
    pv = build_provenance(up_video(p))
    assert pv.watermark.present is True
    assert pv.assessment.level == "declared_ai"
    assert any(s.kind == "ai" and s.strength == "strong" for s in pv.signals)
    assert not any(s.kind == "absent" for s in pv.signals)
