"""Baseline-vs-optimized benchmark + correctness comparison for TruthLens
Video Model v1. Standalone script (this repo has no pytest suite -- see
CLAUDE.md Sec 7), matching the convention of the other test_*_smoke.py
scripts.

No real, non-trivial sample videos exist in this repo (frontend's
test-video.mp4 is a 29-byte placeholder stub, not decodable). This script
synthesizes a handful of representative videos on the fly -- varying frame
count, resolution, and whether a detectable face is present -- covering
the "16 evenly spaced frames," "fewer than 16 frames," "face found," and
"no face -> center-square fallback" code paths. This cannot reproduce the
real Celeb-DF-v2 evaluation (correctly out of scope here); it only proves
baseline.py and optimized.py agree with each other and measures their
relative speed.

Run with:
    ../../../../../.venv/Scripts/python.exe -m app.services.video.model_v1.benchmark
from backend/, or:
    backend/../.venv/Scripts/python.exe backend/app/services/video/model_v1/benchmark.py
"""
from __future__ import annotations

import statistics
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import baseline, optimized


def _make_video(
    path: str,
    num_frames: int,
    width: int,
    height: int,
    draw_face: bool,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    for i in range(num_frames):
        frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        if draw_face:
            cx, cy = width // 2, height // 2
            radius = min(width, height) // 4
            color = (200, 180, 160)
            cv2.circle(frame, (cx, cy), radius, color, thickness=-1)
            eye_r = max(2, radius // 6)
            cv2.circle(frame, (cx - radius // 2, cy - radius // 4), eye_r, (30, 30, 30), -1)
            cv2.circle(frame, (cx + radius // 2, cy - radius // 4), eye_r, (30, 30, 30), -1)
            cv2.ellipse(
                frame, (cx, cy + radius // 3), (radius // 2, radius // 4), 0, 0, 180, (60, 40, 40), 2
            )
        # per-frame index baked into pixels so decode-order bugs are visible
        cv2.putText(frame, str(i), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _build_sample_videos(tmp_dir: Path) -> dict[str, Path]:
    videos = {}

    spec = tmp_dir / "typical_90frames_640x480.mp4"
    _make_video(str(spec), num_frames=90, width=640, height=480, draw_face=False, seed=1)
    videos["typical_90frames_640x480"] = spec

    spec = tmp_dir / "exact_16frames_480x360.mp4"
    _make_video(str(spec), num_frames=16, width=480, height=360, draw_face=False, seed=2)
    videos["exact_16frames_480x360"] = spec

    spec = tmp_dir / "short_20frames_320x240.mp4"
    _make_video(str(spec), num_frames=20, width=320, height=240, draw_face=False, seed=3)
    videos["short_20frames_320x240"] = spec

    spec = tmp_dir / "with_synthetic_face_120frames_720x480.mp4"
    _make_video(str(spec), num_frames=120, width=720, height=480, draw_face=True, seed=4)
    videos["with_synthetic_face_120frames_720x480"] = spec

    spec = tmp_dir / "large_300frames_1280x720.mp4"
    _make_video(str(spec), num_frames=300, width=1280, height=720, draw_face=False, seed=5)
    videos["large_300frames_1280x720"] = spec

    return videos


def run_benchmark(checkpoint_path: str | None = None, repeats: int = 3) -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="tl_video_v1_bench_"))
    print(f"Synthesizing sample videos in {tmp_dir} ...")
    videos = _build_sample_videos(tmp_dir)

    print("=" * 78)
    print("TRUTHLENS VIDEO MODEL V1 -- BASELINE VS OPTIMIZED BENCHMARK")
    print("=" * 78)

    optimized.clear_model_cache()

    all_ok = True
    agreement_count = 0
    total_count = 0
    max_abs_prob_diff = 0.0
    max_abs_logit_diff = 0.0
    abs_prob_diffs: list[float] = []

    baseline_totals: list[float] = []
    optimized_cold_totals: list[float] = []
    optimized_warm_totals: list[float] = []
    baseline_stage_totals: dict[str, list[float]] = {
        "model_load_s": [], "preprocess_s": [], "inference_s": []
    }
    optimized_warm_stage_totals: dict[str, list[float]] = {
        "model_load_s": [], "preprocess_s": [], "inference_s": []
    }

    for name, path in videos.items():
        print(f"\n-- {name} --")

        b = baseline.predict(str(path), checkpoint_path)
        baseline_totals.append(b.timings["total_s"])
        for k in baseline_stage_totals:
            baseline_stage_totals[k].append(b.timings[k])

        optimized.clear_model_cache()
        o_cold = optimized.predict(str(path), checkpoint_path)
        optimized_cold_totals.append(o_cold.timings["total_s"])

        warm_times = []
        warm_stage_samples: list[dict[str, float]] = []
        o_warm = o_cold
        for _ in range(repeats):
            o_warm = optimized.predict(str(path), checkpoint_path)
            warm_times.append(o_warm.timings["total_s"])
            warm_stage_samples.append(o_warm.timings)
        optimized_warm_totals.append(statistics.mean(warm_times))
        for k in optimized_warm_stage_totals:
            optimized_warm_stage_totals[k].append(
                statistics.mean(s[k] for s in warm_stage_samples)
            )

        prob_diff = abs(b.probability - o_warm.probability)
        logit_diff = abs(b.logit - o_warm.logit)
        max_abs_prob_diff = max(max_abs_prob_diff, prob_diff)
        max_abs_logit_diff = max(max_abs_logit_diff, logit_diff)
        abs_prob_diffs.append(prob_diff)

        total_count += 1
        agree = b.verdict == o_warm.verdict
        agreement_count += int(agree)
        all_ok &= agree

        print(
            f"  baseline : verdict={b.verdict:5s} prob={b.probability:.6f} "
            f"logit={b.logit:+.6f} total={b.timings['total_s']*1000:.1f}ms"
        )
        print(
            f"  optimized: verdict={o_warm.verdict:5s} prob={o_warm.probability:.6f} "
            f"logit={o_warm.logit:+.6f} warm_total={statistics.mean(warm_times)*1000:.1f}ms "
            f"cold_total={o_cold.timings['total_s']*1000:.1f}ms"
        )
        print(
            f"  diff     : |dprob|={prob_diff:.8f} |dlogit|={logit_diff:.8f} "
            f"agree={'YES' if agree else 'NO'}"
        )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    agreement_pct = 100.0 * agreement_count / total_count
    mean_abs_prob_diff = statistics.mean(abs_prob_diffs)
    print(f"Prediction agreement       : {agreement_pct:.1f}% ({agreement_count}/{total_count})")
    print(f"Max |prob diff|             : {max_abs_prob_diff:.8f}")
    print(f"Max |logit diff|             : {max_abs_logit_diff:.8f}")
    print(f"Mean |prob diff|             : {mean_abs_prob_diff:.8f}")
    print(f"Mean baseline total (s)      : {statistics.mean(baseline_totals):.4f}")
    print(f"Mean optimized cold total (s): {statistics.mean(optimized_cold_totals):.4f}")
    print(f"Mean optimized warm total (s): {statistics.mean(optimized_warm_totals):.4f}")
    print("-- stage breakdown (mean across videos) --")
    for k in ("model_load_s", "preprocess_s", "inference_s"):
        print(
            f"  {k:14s} baseline={statistics.mean(baseline_stage_totals[k]):.4f}s "
            f"optimized_warm={statistics.mean(optimized_warm_stage_totals[k]):.4f}s"
        )
    speedup_cold = statistics.mean(baseline_totals) / statistics.mean(optimized_cold_totals)
    speedup_warm = statistics.mean(baseline_totals) / statistics.mean(optimized_warm_totals)
    print(f"Speedup (cold optimized)      : {speedup_cold:.2f}x")
    print(f"Speedup (warm optimized)      : {speedup_warm:.2f}x")
    print("=" * 78)
    print("RESULT:", "ALL AGREE" if all_ok else "DISAGREEMENT DETECTED")
    print("=" * 78)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
