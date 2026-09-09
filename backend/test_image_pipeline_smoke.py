"""Practical smoke test for the image detection pipeline.

Not a pytest suite (this repo has none) -- a standalone script matching the
existing convention (see test_image_detection.py). Run with:
    ../.venv/Scripts/python.exe test_image_pipeline_smoke.py

Covers: model loading, a real RGB photo, grayscale, RGBA, a corrupt file, and
an invalid (non-image) file -- run through the actual production code path
(run_image_detection), not a reimplementation of it.
"""
import asyncio
import os
import sys

from PIL import Image
from sqlalchemy import select

from app.core.database import AsyncSessionLocal, create_tables
from app.models.models import Upload, DetectionResult
from app.services.detection_errors import UnprocessableMediaError
from app.services.image.detector import run_image_detection

TMP_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test_fixtures")
os.makedirs(TMP_DIR, exist_ok=True)


def _make_fixtures() -> dict[str, str]:
    paths = {}

    p = os.path.join(TMP_DIR, "rgb.jpg")
    Image.new("RGB", (200, 200), color=(120, 90, 60)).save(p)
    paths["real_rgb"] = p

    p = os.path.join(TMP_DIR, "grayscale.png")
    Image.new("L", (100, 100), color=128).save(p)
    paths["grayscale"] = p

    p = os.path.join(TMP_DIR, "rgba.png")
    Image.new("RGBA", (100, 100), color=(10, 20, 30, 128)).save(p)
    paths["rgba"] = p

    p = os.path.join(TMP_DIR, "corrupt.jpg")
    with open(p, "wb") as f:
        f.write(b"\xff\xd8\xff\xe0" + os.urandom(200))
    paths["corrupt"] = p

    p = os.path.join(TMP_DIR, "not_an_image.jpg")
    with open(p, "w") as f:
        f.write("not an image")
    paths["invalid"] = p

    return paths


async def _run_case(db, name: str, path: str, expect_error: bool) -> bool:
    upload = Upload(
        file_name=os.path.basename(path),
        media_type="image",
        storage_url=path,
        file_size_kb=os.path.getsize(path) / 1024,
    )
    db.add(upload)
    await db.flush()

    try:
        result_id = await run_image_detection(upload, db)
    except UnprocessableMediaError as e:
        if expect_error:
            print(f"  [OK]   {name}: raised UnprocessableMediaError as expected: {e}")
            return True
        print(f"  [FAIL] {name}: raised UnprocessableMediaError unexpectedly: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] {name}: raised unexpected {type(e).__name__}: {e}")
        return False

    if expect_error:
        print(f"  [FAIL] {name}: expected an error but got a result ({result_id})")
        return False

    r = await db.execute(select(DetectionResult).where(DetectionResult.result_id == result_id))
    detection = r.scalar_one()
    print(
        f"  [OK]   {name}: verdict={detection.verdict} "
        f"confidence={detection.confidence_score:.4f} model={detection.model_used} "
        f"time={detection.processing_time_ms:.1f}ms"
    )
    return True


async def main():
    await create_tables()
    fixtures = _make_fixtures()

    print("=" * 70)
    print("IMAGE PIPELINE SMOKE TEST")
    print("=" * 70)

    print("\n-- model loading --")
    from app.services.image.detector import get_image_pipeline
    pipeline = get_image_pipeline()
    print(f"  [OK]   ConvNeXt-Tiny loaded on device={pipeline.device}")

    print("\n-- valid inputs (should succeed) --")
    ok = True
    async with AsyncSessionLocal() as db:
        for name in ("real_rgb", "grayscale", "rgba"):
            ok &= await _run_case(db, name, fixtures[name], expect_error=False)
        await db.commit()

    print("\n-- invalid inputs (should fail cleanly, not crash) --")
    async with AsyncSessionLocal() as db:
        for name in ("corrupt", "invalid"):
            ok &= await _run_case(db, name, fixtures[name], expect_error=True)
        await db.commit()

    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASSED" if ok else "SOME CASES FAILED")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
