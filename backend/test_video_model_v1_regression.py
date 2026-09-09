"""Regression test for TruthLens Video Model v1 -- ORIGINAL-EVALUATOR
PARITY, not merely baseline-vs-optimized self-agreement.

Why this distinction matters: baseline.py and optimized.py both call the
same shared functions in common.py, so if common.py itself drifted from
the recovered original Celeb-DF-v2 evaluator, baseline and optimized
would still agree with each other while both being wrong. That happened
once already (round()-vs-truncate frame indices, cv2 vs PIL resize) and
went undetected by a baseline-vs-optimized-only test. This file therefore
also carries a literal, minimal, single-video port of the original
evaluator's read_video_fast() (REFERENCE_* below, transcribed directly
from the recovered script -- no threading/chunking/multi-video batching,
since those don't affect per-video numerics) and checks common.py /
baseline.py / optimized.py against THAT independently-derived ground
truth, not just against each other.

Not a pytest suite (this repo has none -- see CLAUDE.md Sec 7); a
standalone script matching the convention of test_image_pipeline_smoke.py
/ test_audio_pipeline_smoke.py. Run with:
    ../.venv/Scripts/python.exe test_video_model_v1_regression.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from app.services.video.model_v1 import baseline, common, optimized

TMP_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test_fixtures", "video_v1")
os.makedirs(TMP_DIR, exist_ok=True)

PROB_TOL = 1e-4


def _logit_tol(reference_logit: float) -> float:
    """Batch-size-dependent CPU floating-point reduction order (single-
    frame loop vs. batched forward pass) causes a real but bounded
    relative logit deviation -- confirmed ~2.7e-7 absolute on real,
    in-distribution face content, but larger in absolute terms on these
    synthetic random-noise fixtures because noise pushes the network
    into a poorly-conditioned, unstable regime with abnormally large
    logit magnitudes (50-150) that amplify the same relative effect. A
    fixed absolute tolerance would either be too loose for real content
    or too tight for noise; scale with magnitude instead."""
    return max(1e-3, 2e-3 * abs(reference_logit))


# ======================================================================
# REFERENCE_* -- a literal, minimal port of the recovered original
# evaluator's single-video path (read_video_fast + the per-batch
# normalize/model/pool/sigmoid block from infer_chunk), transcribed
# directly, not reimplemented from a description. This is independent
# ground truth: it does not import or call common.py/baseline.py/
# optimized.py's frame-selection, cropping, or resize code at all.
# ======================================================================

def REFERENCE_read_video_fast(path: str, cascade: cv2.CascadeClassifier, num_frames: int = 16):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "video_open_failed"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, "invalid_frame_count"

    positions = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    current = 0
    next_position_index = 0

    while next_position_index < len(positions):
        ret, frame = cap.read()
        if not ret:
            break

        if current == positions[next_position_index]:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            h, w = frame_rgb.shape[:2]

            if len(faces) > 0:
                x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                pad_x = int(fw * 0.20)
                pad_y = int(fh * 0.20)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)
                crop = frame_rgb[y1:y2, x1:x2]
            else:
                side = min(h, w)
                x1 = (w - side) // 2
                y1 = (h - side) // 2
                crop = frame_rgb[y1 : y1 + side, x1 : x1 + side]

            if crop.size == 0:
                crop = frame_rgb

            image = Image.fromarray(crop)
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
            next_position_index += 1

        current += 1

    cap.release()

    if len(frames) != num_frames:
        return None, f"only_{len(frames)}_frames"

    return np.stack(frames), ""


def REFERENCE_predict(path: str, model, cascade) -> tuple[float, float, str] | None:
    frames, error = REFERENCE_read_video_fast(path, cascade)
    if frames is None:
        return None

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    x = x / 255.0
    x = (x - mean) / std

    with torch.inference_mode():
        frame_logits = model(x).squeeze(1)
        video_logit = frame_logits.mean().item()
        probability = torch.sigmoid(torch.tensor(video_logit)).item()

    verdict = "FAKE" if probability >= 0.525 else "REAL"
    return video_logit, probability, verdict


# ======================================================================
# fixtures
# ======================================================================

def _make_video(path: str, num_frames: int, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    for i in range(num_frames):
        frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _full_sequential_decode(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    return frames


# ======================================================================
# tests
# ======================================================================

def test_exact_frame_index_parity() -> bool:
    ok = True
    for frame_count in (16, 17, 30, 90, 150, 229, 300, 1000):
        mine = common.evenly_spaced_frame_indices(frame_count)
        ref = np.linspace(0, frame_count - 1, 16).astype(int).tolist()
        passed = mine == ref
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] frame_count={frame_count}: {mine}")
    return ok


def test_short_video_raises_like_original() -> bool:
    """frame_count < 16 must be dropped (ShortVideoError), matching the
    original's under-selection/only_N_frames error path -- not silently
    clamped/repeated (the bug from before)."""
    ok = True
    for num_frames in (1, 5, 8, 15):
        path = os.path.join(TMP_DIR, f"short_{num_frames}.mp4")
        _make_video(path, num_frames, 160, 120, seed=num_frames)

        raised_baseline = raised_optimized = False
        try:
            baseline.predict(path)
        except common.ShortVideoError:
            raised_baseline = True
        try:
            optimized.clear_model_cache()
            optimized.predict(path)
        except common.ShortVideoError:
            raised_optimized = True

        passed = raised_baseline and raised_optimized
        ok &= passed
        print(
            f"  [{'OK' if passed else 'FAIL'}] {num_frames}-frame video: "
            f"baseline_raised={raised_baseline} optimized_raised={raised_optimized}"
        )
    return ok


def test_exact_resize_parity() -> bool:
    """common.preprocess_frame's PIL-BILINEAR path must match the
    original's Image.fromarray(crop).resize(..., BILINEAR) bit-for-bit,
    given the identical crop."""
    ok = True
    cascade = common.get_face_cascade()
    rng = np.random.default_rng(7)
    for shape in [(300, 400, 3), (180, 140, 3), (500, 500, 3)]:
        frame_bgr = rng.integers(0, 255, size=shape, dtype=np.uint8)

        mine = common.preprocess_frame(frame_bgr, cascade)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        h, w = frame_rgb.shape[:2]
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            pad_x, pad_y = int(fw * 0.20), int(fh * 0.20)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
            crop = frame_rgb[y1:y2, x1:x2]
        else:
            side = min(h, w)
            x1, y1 = (w - side) // 2, (h - side) // 2
            crop = frame_rgb[y1 : y1 + side, x1 : x1 + side]
        if crop.size == 0:
            crop = frame_rgb
        ref_img = Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR)
        ref_arr = np.asarray(ref_img, dtype=np.uint8).astype(np.float32) / 255.0
        ref_arr = (ref_arr - common.IMAGENET_MEAN) / common.IMAGENET_STD
        ref_arr = np.transpose(ref_arr, (2, 0, 1))

        identical = np.array_equal(mine, ref_arr)
        ok &= identical
        print(f"  [{'OK' if identical else 'FAIL'}] shape={shape}: bit-identical={identical}")
    return ok


def test_empty_crop_fallback() -> bool:
    """A face bbox that reduces to a zero-size crop must fall back to the
    full frame (crop.size == 0 -> crop = frame_rgb), not raise."""

    class _ZeroSizeFaceCascade:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[50, 50, 0, 0]])  # zero-width/height "face"

    frame_bgr = np.random.default_rng(1).integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    crop, info = common.crop_face_or_center_with_info(frame_bgr, _ZeroSizeFaceCascade())

    fallback_used = crop.shape[:2] == (100, 100)
    print(f"  [{'OK' if fallback_used else 'FAIL'}] zero-size face bbox falls back to full frame")
    return fallback_used


def test_grab_skip_decode_matches_full_decode() -> bool:
    ok = True
    for seed, (name, (num_frames, w, h)) in enumerate({
        "typical": (90, 320, 240),
        "exact16": (16, 160, 120),
        "boundary17": (17, 160, 120),
    }.items()):
        path = os.path.join(TMP_DIR, f"decode_{name}.mp4")
        _make_video(path, num_frames, w, h, seed=seed)

        full = common.decode_selected_frames(path, use_grab_skip=False)
        skip = common.decode_selected_frames(path, use_grab_skip=True)

        identical = len(full) == len(skip) and all(np.array_equal(a, b) for a, b in zip(full, skip))
        ok &= identical
        print(f"  [{'OK' if identical else 'FAIL'}] {name}: grab-skip decode == full decode")
    return ok


def test_against_reference_reimplementation() -> bool:
    """The real test: common.py/baseline.py/optimized.py vs. an
    independently-transcribed port of the recovered original evaluator."""
    ok = True
    optimized.clear_model_cache()
    model = common.build_model()
    common.load_checkpoint(model)
    model.eval()
    cascade = common.get_face_cascade()

    max_abs_logit_diff = 0.0
    max_abs_prob_diff = 0.0

    for seed, (name, (num_frames, w, h)) in enumerate({
        "typical_90_640x480": (90, 640, 480),
        "exact_16_480x360": (16, 480, 360),
        "boundary_17_400x300": (17, 400, 300),
        "large_150_800x600": (150, 800, 600),
    }.items()):
        path = os.path.join(TMP_DIR, f"refparity_{name}.mp4")
        _make_video(path, num_frames, w, h, seed=seed)

        ref = REFERENCE_predict(path, model, cascade)
        assert ref is not None, f"reference reimplementation errored on {name}"
        ref_logit, ref_prob, ref_verdict = ref

        b = baseline.predict(path)
        o = optimized.predict(path)

        b_diff_logit = abs(b.logit - ref_logit)
        o_diff_logit = abs(o.logit - ref_logit)
        b_diff_prob = abs(b.probability - ref_prob)
        o_diff_prob = abs(o.probability - ref_prob)
        max_abs_logit_diff = max(max_abs_logit_diff, b_diff_logit, o_diff_logit)
        max_abs_prob_diff = max(max_abs_prob_diff, b_diff_prob, o_diff_prob)

        tol = _logit_tol(ref_logit)
        passed = (
            b.verdict == ref_verdict == o.verdict
            and b_diff_logit <= tol
            and o_diff_logit <= tol
            and b_diff_prob <= PROB_TOL
            and o_diff_prob <= PROB_TOL
        )
        ok &= passed

        print(
            f"  [{'OK' if passed else 'FAIL'}] {name}: "
            f"ref={ref_verdict}/{ref_prob:.6f} baseline={b.verdict}/{b.probability:.6f} "
            f"optimized={o.verdict}/{o.probability:.6f} "
            f"|d(base,ref)|={b_diff_logit:.8f} |d(opt,ref)|={o_diff_logit:.8f}"
        )

    print(f"  max |logit diff| vs reference: {max_abs_logit_diff:.8f}")
    print(f"  max |prob diff| vs reference : {max_abs_prob_diff:.8f}")
    return ok


def test_baseline_matches_optimized() -> bool:
    ok = True
    optimized.clear_model_cache()

    videos = {}
    for seed, (name, (num_frames, w, h)) in enumerate({
        "typical_90_640x480": (90, 640, 480),
        "exact_16_480x360": (16, 480, 360),
        "boundary_17_400x300": (17, 400, 300),
        "large_150_800x600": (150, 800, 600),
    }.items()):
        path = os.path.join(TMP_DIR, f"parity_{name}.mp4")
        _make_video(path, num_frames, w, h, seed=seed)
        videos[name] = path

    for name, path in videos.items():
        b = baseline.predict(path)
        o = optimized.predict(path)

        prob_diff = abs(b.probability - o.probability)
        logit_diff = abs(b.logit - o.logit)
        verdict_match = b.verdict == o.verdict
        passed = verdict_match and prob_diff <= PROB_TOL and logit_diff <= _logit_tol(b.logit)
        ok &= passed

        print(
            f"  [{'OK' if passed else 'FAIL'}] {name}: "
            f"baseline={b.verdict}/{b.probability:.6f} optimized={o.verdict}/{o.probability:.6f} "
            f"|dprob|={prob_diff:.6f} |dlogit|={logit_diff:.6f}"
        )
    return ok


def test_optimized_model_cache_is_reused() -> bool:
    optimized.clear_model_cache()
    path = os.path.join(TMP_DIR, "cache_check.mp4")
    _make_video(path, 20, 160, 120, seed=99)

    first = optimized.predict(path)
    second = optimized.predict(path)

    passed = first.timings["model_load_s"] > 0.0 and second.timings["model_load_s"] == 0.0
    print(
        f"  [{'OK' if passed else 'FAIL'}] first call loads model "
        f"({first.timings['model_load_s']:.4f}s), second call reuses cache "
        f"({second.timings['model_load_s']:.4f}s)"
    )
    return passed


def main() -> None:
    print("=" * 70)
    print("VIDEO MODEL V1 -- ORIGINAL-EVALUATOR PARITY REGRESSION TEST")
    print("=" * 70)

    print("\n-- exact frame-index parity (np.linspace(...).astype(int)) --")
    r1 = test_exact_frame_index_parity()

    print("\n-- short videos raise ShortVideoError, like the original --")
    r2 = test_short_video_raises_like_original()

    print("\n-- exact resize/preprocessing parity (PIL BILINEAR) --")
    r3 = test_exact_resize_parity()

    print("\n-- empty-crop safety fallback --")
    r4 = test_empty_crop_fallback()

    print("\n-- grab()-skip decode vs full sequential decode (pixel-identical) --")
    r5 = test_grab_skip_decode_matches_full_decode()

    print("\n-- common/baseline/optimized vs. independent reference reimplementation --")
    r6 = test_against_reference_reimplementation()

    print("\n-- baseline vs optimized prediction parity --")
    r7 = test_baseline_matches_optimized()

    print("\n-- optimized model cache is loaded once, reused after --")
    r8 = test_optimized_model_cache_is_reused()

    ok = r1 and r2 and r3 and r4 and r5 and r6 and r7 and r8
    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASSED" if ok else "SOME CASES FAILED")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
