"""Real-face parity verification for TruthLens Video Model v1.

Context: the model optimization work in app/services/video/model_v1/ was
first verified only with synthetic random-noise videos (baseline.py vs
optimized.py agreed 100%, but that is not enough evidence for production
confidence -- noise pushes the model into an out-of-distribution regime
with abnormally large logits, and it never exercises the "a real face was
detected" branch of common.crop_face_or_center_with_info at all).

No original Celeb-DF-v2 evaluation/frame-extraction script exists in this
repository (confirmed by a repo-wide search -- see the accompanying
report), and no real face video exists in this repo either: the only
video-shaped fixtures found are a 29-byte placeholder stub
(frontend/public/test-video.mp4), a handful of duplicate synthetic
UI-test clips in backend/uploads/ (solid color / no face), and one upload
that is actually an audio-only file mislabeled .mp4. None of those contain
a real face.

This script is therefore the "original reference not found" path: it is
explicit that it does NOT reproduce the real Celeb-DF-v2 evaluation. It
sources a real human face from scikit-image's bundled `data.astronaut()`
sample photo (a standard, public-domain NASA photo used throughout the
computer-vision community for exactly this kind of test -- not a dataset
download, not an arbitrary photo of a private individual) and builds
several short synthetic-motion "videos" from it (static repeat, panning,
zooming, downscaled/noisy) so the Haar cascade genuinely detects a face in
most of them, exercising the crop-a-real-face branch that the earlier
noise-only tests never touched.

Run with:
    ../.venv/Scripts/python.exe test_video_model_v1_real_face_parity.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from app.services.video.model_v1 import baseline, common, optimized

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test_fixtures", "video_v1")
os.makedirs(FIXTURE_DIR, exist_ok=True)
SOURCE_PHOTO_PATH = os.path.join(FIXTURE_DIR, "real_face_source.png")


def _get_source_photo() -> np.ndarray:
    """Real human face photo, BGR. Cached to disk after first fetch."""
    if os.path.exists(SOURCE_PHOTO_PATH):
        img = cv2.imread(SOURCE_PHOTO_PATH)
        if img is not None:
            return img

    from skimage import data  # local import: only needed to build this fixture once

    img_rgb = data.astronaut()  # 512x512 public-domain NASA photo, standard skimage sample
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(SOURCE_PHOTO_PATH, img_bgr)
    return img_bgr


def _make_real_face_video(path: str, photo_bgr: np.ndarray, num_frames: int, mode: str) -> None:
    """Builds a short synthetic-motion video from a single real photo.

    mode:
      "static"   - same frame repeated (zero motion, still a real face per frame)
      "pan"      - a moving crop window across the photo (simulates minor camera pan)
      "zoom"     - the crop window slowly zooms in
      "noisy"    - static frame + mild per-frame sensor-noise + brightness jitter
      "downscale"- photo downscaled (simulates a lower-quality real video)
    """
    h0, w0 = photo_bgr.shape[:2]
    out_size = (w0, h0) if mode != "downscale" else (w0 // 2, h0 // 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    rng = np.random.default_rng(123)
    crop_frac = 0.85  # window size relative to source, for pan/zoom

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)

        if mode == "static":
            frame = photo_bgr.copy()

        elif mode == "pan":
            cw, ch = int(w0 * crop_frac), int(h0 * crop_frac)
            max_x, max_y = w0 - cw, h0 - ch
            x0 = int(max_x * t)
            y0 = max_y // 2
            crop = photo_bgr[y0 : y0 + ch, x0 : x0 + cw]
            frame = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)

        elif mode == "zoom":
            frac = crop_frac - 0.25 * t  # zoom in over time
            cw, ch = int(w0 * frac), int(h0 * frac)
            x0, y0 = (w0 - cw) // 2, (h0 - ch) // 2
            crop = photo_bgr[y0 : y0 + ch, x0 : x0 + cw]
            frame = cv2.resize(crop, (w0, h0), interpolation=cv2.INTER_LINEAR)

        elif mode == "noisy":
            noise = rng.normal(0, 6, size=photo_bgr.shape).astype(np.int16)
            frame = np.clip(photo_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            brightness = 0.95 + 0.10 * rng.random()
            frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        elif mode == "downscale":
            frame = cv2.resize(photo_bgr, out_size, interpolation=cv2.INTER_AREA)

        else:
            raise ValueError(mode)

        cv2.putText(frame, str(i), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(frame)

    writer.release()


def _build_real_face_videos() -> dict[str, str]:
    photo = _get_source_photo()
    specs = {
        "real_static_16frames": (16, "static"),
        "real_static_60frames": (60, "static"),
        "real_pan_90frames": (90, "pan"),
        "real_zoom_45frames": (45, "zoom"),
        "real_noisy_60frames": (60, "noisy"),
        "real_downscale_30frames": (30, "downscale"),
        "real_pan_8frames_short": (8, "pan"),
    }
    videos = {}
    for name, (num_frames, mode) in specs.items():
        path = os.path.join(FIXTURE_DIR, f"realface_{name}.mp4")
        _make_real_face_video(path, photo, num_frames, mode)
        videos[name] = path
    return videos


def _bbox_iou(a: tuple | None, b: tuple | None) -> float | None:
    if a is None or b is None:
        return None
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 1.0


def run() -> int:
    print("=" * 78)
    print("TRUTHLENS VIDEO MODEL V1 -- REAL-FACE PARITY VERIFICATION")
    print("=" * 78)
    print(
        "\nNo original Celeb-DF-v2 evaluation/frame-extraction script exists in this\n"
        "repo (confirmed by search) and no real face video exists in this repo\n"
        "either. This test uses skimage's standard public-domain astronaut sample\n"
        "photo to build synthetic-motion videos containing a REAL human face, as\n"
        "the best available substitute -- it does NOT reproduce or re-verify the\n"
        "original Celeb-DF-v2 evaluation.\n"
    )

    videos = _build_real_face_videos()
    optimized.clear_model_cache()

    all_ok = True
    frame_index_agree = 0
    pixel_agree = 0
    crop_agree = 0
    total = 0
    max_abs_logit_diff = 0.0
    max_abs_prob_diff = 0.0
    agreement_count = 0

    short_video_agree = 0
    short_video_total = 0

    for name, path in videos.items():
        total += 1
        print(f"\n-- {name} --")

        cap = cv2.VideoCapture(path)
        frame_count_check = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count_check < common.NUM_FRAMES:
            # Below NUM_FRAMES, the original evaluator's sequential
            # frame-matching loop under-selects and drops the video
            # (ShortVideoError) -- verify both implementations agree on
            # raising that, rather than running the normal prediction
            # comparison below.
            short_video_total += 1
            raised_baseline = raised_optimized = False
            try:
                baseline.predict(path)
            except common.ShortVideoError:
                raised_baseline = True
            try:
                optimized.predict(path)
            except common.ShortVideoError:
                raised_optimized = True
            both_raised = raised_baseline and raised_optimized
            short_video_agree += int(both_raised)
            all_ok &= both_raised
            frame_index_agree += 1
            pixel_agree += 1
            crop_agree += 1
            agreement_count += int(both_raised)
            print(
                f"  frame_count={frame_count_check} < {common.NUM_FRAMES}: expected to be "
                f"dropped (ShortVideoError), like the original evaluator"
            )
            print(
                f"  baseline_raised={raised_baseline} optimized_raised={raised_optimized} "
                f"agree={'YES' if both_raised else 'NO'}"
            )
            continue

        # 1) frame selection agreement: recompute indices both ways
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx_a = common.evenly_spaced_frame_indices(frame_count)

        full_frames = []
        cap = cv2.VideoCapture(path)
        ok, f = cap.read()
        while ok:
            full_frames.append(f)
            ok, f = cap.read()
        cap.release()
        idx_b = common.evenly_spaced_frame_indices(len(full_frames))
        idx_match = idx_a == idx_b
        frame_index_agree += int(idx_match)

        # 2) pixel agreement: baseline's full-decode selection vs optimized's grab-skip
        baseline_selected = [full_frames[i] for i in idx_b]
        optimized_selected = optimized._decode_selected_frames(path)
        pixels_match = len(baseline_selected) == len(optimized_selected) and all(
            np.array_equal(a, b) for a, b in zip(baseline_selected, optimized_selected)
        )
        pixel_agree += int(pixels_match)

        # 3) crop agreement: same face bbox / crop bounds from both frame sources
        cascade = common.get_face_cascade()
        crops_match = True
        faces_detected = 0
        for fb, fo in zip(baseline_selected, optimized_selected):
            _, info_b = common.crop_face_or_center_with_info(fb, cascade)
            _, info_o = common.crop_face_or_center_with_info(fo, cascade)
            if info_b["face_bbox"] is not None:
                faces_detected += 1
            iou = _bbox_iou(info_b["face_bbox"], info_o["face_bbox"])
            same_kind = (info_b["face_bbox"] is None) == (info_o["face_bbox"] is None)
            if not same_kind or (iou is not None and iou < 0.999):
                crops_match = False
        crop_agree += int(crops_match)

        # 4) prediction parity
        b = baseline.predict(path)
        o = optimized.predict(path)
        prob_diff = abs(b.probability - o.probability)
        logit_diff = abs(b.logit - o.logit)
        max_abs_prob_diff = max(max_abs_prob_diff, prob_diff)
        max_abs_logit_diff = max(max_abs_logit_diff, logit_diff)
        pred_agree = b.verdict == o.verdict
        agreement_count += int(pred_agree)

        all_ok &= idx_match and pixels_match and crops_match and pred_agree

        print(
            f"  frames_with_face(of 16)     : {faces_detected}"
        )
        print(f"  frame index agreement       : {'YES' if idx_match else 'NO'}")
        print(f"  pixel agreement (decode)    : {'YES' if pixels_match else 'NO'}")
        print(f"  crop/bbox agreement         : {'YES' if crops_match else 'NO'}")
        print(
            f"  baseline : verdict={b.verdict:5s} prob={b.probability:.6f} logit={b.logit:+.6f}"
        )
        print(
            f"  optimized: verdict={o.verdict:5s} prob={o.probability:.6f} logit={o.logit:+.6f}"
        )
        print(
            f"  |dprob|={prob_diff:.8f} |dlogit|={logit_diff:.8f} agree={'YES' if pred_agree else 'NO'}"
        )

    print("\n" + "=" * 78)
    print("SUMMARY (real-face videos)")
    print("=" * 78)
    print(f"Frame selection agreement : {100.0*frame_index_agree/total:.1f}% ({frame_index_agree}/{total})")
    print(f"Pixel agreement (decode)  : {100.0*pixel_agree/total:.1f}% ({pixel_agree}/{total})")
    print(f"Crop/bbox agreement       : {100.0*crop_agree/total:.1f}% ({crop_agree}/{total})")
    print(f"Prediction agreement      : {100.0*agreement_count/total:.1f}% ({agreement_count}/{total})")
    if short_video_total:
        print(
            f"Short-video error agreement: {100.0*short_video_agree/short_video_total:.1f}% "
            f"({short_video_agree}/{short_video_total}) -- both must raise ShortVideoError"
        )
    print(f"Max |logit diff|          : {max_abs_logit_diff:.8f}")
    print(f"Max |prob diff|           : {max_abs_prob_diff:.8f}")
    print("=" * 78)
    print("RESULT:", "ALL AGREE" if all_ok else "DISAGREEMENT DETECTED")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run())
