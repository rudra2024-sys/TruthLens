"""Practical smoke test for the video detection pipeline, post Video Model
v1 integration.

Not a pytest suite (this repo has none) -- a standalone script matching
the existing convention (see test_image_pipeline_smoke.py /
test_audio_pipeline_smoke.py). Run with:
    ../.venv/Scripts/python.exe test_video_pipeline_smoke.py

Covers: model loading via the real get_video_backend() seam, a real-face
video (should succeed end-to-end through the actual production code path
run_video_detection -> video/backend.py -> model_v1/optimized.py, using
the real epoch_11 checkpoint), a too-short video (should map to a clean
422 via UnprocessableMediaError, not a raw 500), and confirms the
heuristic backend is still reachable as a fallback via
VIDEO_MODEL_BACKEND=heuristic.
"""
import asyncio
import os
import sys

import cv2
import numpy as np
from sqlalchemy import select

sys.path.insert(0, os.path.dirname(__file__))

from app.core.database import AsyncSessionLocal, create_tables
from app.models.models import Upload, DetectionResult, VideoAnalysis
from app.services.detection_errors import UnprocessableMediaError
from app.services.video.detector import run_video_detection

TMP_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test_fixtures", "video_v1")
os.makedirs(TMP_DIR, exist_ok=True)


def _get_real_face_photo() -> np.ndarray:
    source_path = os.path.join(TMP_DIR, "real_face_source.png")
    if os.path.exists(source_path):
        img = cv2.imread(source_path)
        if img is not None:
            return img
    from skimage import data
    img_rgb = data.astronaut()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(source_path, img_bgr)
    return img_bgr


def _make_video(path: str, photo_bgr: np.ndarray, num_frames: int) -> None:
    h, w = photo_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 24.0, (w, h))
    for i in range(num_frames):
        frame = photo_bgr.copy()
        cv2.putText(frame, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def _make_fixtures() -> dict[str, str]:
    photo = _get_real_face_photo()
    paths = {}

    p = os.path.join(TMP_DIR, "smoke_real_face_45frames.mp4")
    _make_video(p, photo, 45)
    paths["real_face"] = p

    p = os.path.join(TMP_DIR, "smoke_too_short_5frames.mp4")
    _make_video(p, photo, 5)
    paths["too_short"] = p

    return paths


async def _run_success_case(db, name: str, path: str) -> bool:
    upload = Upload(
        file_name=os.path.basename(path),
        media_type="video",
        storage_url=path,
        file_size_kb=os.path.getsize(path) / 1024,
    )
    db.add(upload)
    await db.flush()

    try:
        result_id = await run_video_detection(upload, db)
    except Exception as e:
        print(f"  [FAIL] {name}: raised unexpected {type(e).__name__}: {e}")
        return False

    r = await db.execute(select(DetectionResult).where(DetectionResult.result_id == result_id))
    detection = r.scalar_one()
    r2 = await db.execute(select(VideoAnalysis).where(VideoAnalysis.result_id == result_id))
    analysis = r2.scalar_one()

    checks = [
        ("model_used mentions Video Model v1", "Video Model v1" in detection.model_used),
        ("verdict is REAL or FAKE", detection.verdict in ("REAL", "FAKE")),
        ("confidence_score in [0,1]", 0.0 <= detection.confidence_score <= 1.0),
        ("frames_analyzed == 16", analysis.frames_analyzed == 16),
        ("xception_score in [0,1]", 0.0 <= analysis.xception_score <= 1.0),
        ("face_voice_sync is None (model_v1 doesn't compute this)", analysis.face_voice_sync is None),
        ("processing_time_ms > 0", detection.processing_time_ms > 0),
    ]
    ok = True
    for label, passed in checks:
        ok &= passed
        print(f"  [{'OK' if passed else 'FAIL'}] {name}: {label}")

    print(
        f"  -> verdict={detection.verdict} confidence={detection.confidence_score:.6f} "
        f"model_used={detection.model_used!r} frames_analyzed={analysis.frames_analyzed} "
        f"time={detection.processing_time_ms:.1f}ms"
    )
    return ok


async def _run_expect_422_case(db, name: str, path: str) -> bool:
    upload = Upload(
        file_name=os.path.basename(path),
        media_type="video",
        storage_url=path,
        file_size_kb=os.path.getsize(path) / 1024,
    )
    db.add(upload)
    await db.flush()

    try:
        await run_video_detection(upload, db)
    except UnprocessableMediaError as e:
        print(f"  [OK]   {name}: raised UnprocessableMediaError as expected: {e}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: raised unexpected {type(e).__name__}: {e}")
        return False

    print(f"  [FAIL] {name}: expected UnprocessableMediaError but got a result")
    return False


async def main():
    await create_tables()
    fixtures = _make_fixtures()

    print("=" * 70)
    print("VIDEO PIPELINE SMOKE TEST (post Video Model v1 integration)")
    print("=" * 70)

    print("\n-- backend selection --")
    from app.services.video.backend import get_video_backend
    backend = get_video_backend()
    print(f"  [OK] default backend: {backend.metadata.name}")
    default_is_model_v1 = "EfficientNet-B0" in backend.metadata.name
    print(f"  [{'OK' if default_is_model_v1 else 'FAIL'}] default backend is Video Model v1")

    print("\n-- real-face video (should succeed via the real production path) --")
    ok = True
    async with AsyncSessionLocal() as db:
        ok &= await _run_success_case(db, "real_face_45frames", fixtures["real_face"])
        await db.commit()

    print("\n-- too-short video (should fail cleanly with 422, not crash) --")
    async with AsyncSessionLocal() as db:
        ok &= await _run_expect_422_case(db, "too_short_5frames", fixtures["too_short"])
        await db.commit()

    print("\n-- heuristic fallback still reachable via VIDEO_MODEL_BACKEND=heuristic --")
    os.environ["VIDEO_MODEL_BACKEND"] = "heuristic"
    try:
        from app.services.video import backend as backend_module
        fallback = backend_module.get_video_backend()
        fallback_ok = fallback.metadata.name == "Video forensic heuristics v1"
        ok &= fallback_ok
        print(f"  [{'OK' if fallback_ok else 'FAIL'}] fallback backend: {fallback.metadata.name}")
    finally:
        del os.environ["VIDEO_MODEL_BACKEND"]

    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASSED" if ok else "SOME CASES FAILED")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
