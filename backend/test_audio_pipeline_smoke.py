"""Practical smoke test for the audio detection pipeline.

Not a pytest suite (this repo has none) -- a standalone script matching the
existing convention (see test_image_detection.py). Run with:
    ../.venv/Scripts/python.exe test_audio_pipeline_smoke.py

Covers: model loading, a valid WAV, a valid MP3, short audio, silence, a
corrupt file, and an invalid (non-audio) file -- run through the actual
production code path (run_audio_detection). Also verifies the dead
models:8001 network call is no longer attempted by default.
"""
import asyncio
import math
import os
import struct
import sys
import time
import wave

from sqlalchemy import select

from app.core.database import AsyncSessionLocal, create_tables
from app.models.models import Upload, DetectionResult
from app.services.detection_errors import UnprocessableMediaError
from app.services.audio.detector import run_audio_detection

TMP_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test_fixtures")
os.makedirs(TMP_DIR, exist_ok=True)


def _make_wav(path: str, sr: int, duration_s: float, freq: float | None, amplitude: int = 3000) -> None:
    n = int(sr * duration_s)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        if freq is None:
            frames = b"\x00\x00" * n  # silence
        else:
            frames = b"".join(
                struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / sr)))
                for i in range(n)
            )
        w.writeframes(frames)


def _make_fixtures() -> dict[str, str]:
    paths = {}

    p = os.path.join(TMP_DIR, "valid.wav")
    _make_wav(p, 16000, 2.0, 440.0)
    paths["valid_wav"] = p

    p = os.path.join(TMP_DIR, "short.wav")
    _make_wav(p, 16000, 0.1, 440.0)
    paths["short"] = p

    p = os.path.join(TMP_DIR, "silence.wav")
    _make_wav(p, 16000, 2.0, None)
    paths["silence"] = p

    p = os.path.join(TMP_DIR, "corrupt.wav")
    with open(p, "wb") as f:
        f.write(b"RIFF" + (1000).to_bytes(4, "little") + b"WAVEfmt " + os.urandom(200))
    paths["corrupt"] = p

    p = os.path.join(TMP_DIR, "not_audio.wav")
    with open(p, "w") as f:
        f.write("not an audio file")
    paths["invalid"] = p

    # Valid MP3, encoded via PyAV from the same tone -- confirms the claimed
    # multi-format support actually works, not just WAV.
    try:
        import av
        import numpy as np

        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        tone = (0.3 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        mp3_path = os.path.join(TMP_DIR, "valid.mp3")
        container = av.open(mp3_path, mode="w")
        stream = container.add_stream("mp3", rate=sr)
        frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="fltp", layout="mono")
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        paths["valid_mp3"] = mp3_path
    except Exception as e:
        print(f"  [SKIP] could not generate MP3 fixture ({e}); skipping MP3 case")

    return paths


async def _run_case(db, name: str, path: str, expect_error: bool) -> bool:
    upload = Upload(
        file_name=os.path.basename(path),
        media_type="audio",
        storage_url=path,
        file_size_kb=os.path.getsize(path) / 1024,
    )
    db.add(upload)
    await db.flush()

    try:
        result_id = await run_audio_detection(upload, db)
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
    print("AUDIO PIPELINE SMOKE TEST")
    print("=" * 70)

    print("\n-- dead network call check --")
    from app.services.models import model_client
    if model_client.MODEL_SERVICE_URL:
        print(f"  [WARN] MODEL_SERVICE_URL is set to {model_client.MODEL_SERVICE_URL!r} "
              f"-- the models/ microservice call will be attempted. If it isn't actually "
              f"deployed, this will reintroduce the latency bug.")
    else:
        print("  [OK]   MODEL_SERVICE_URL is unset -- local AASIST path used directly, no dead call.")

    print("\n-- model loading + latency --")
    t0 = time.perf_counter()
    from app.services.models.audio_model import score_audio_path
    score_audio_path(fixtures["valid_wav"])  # forces ONNX session init
    cold_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    score_audio_path(fixtures["valid_wav"])
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK]   AASIST session loaded. cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms")
    if warm_ms > 1500:
        print(f"  [WARN] warm inference took {warm_ms:.1f}ms -- expected well under 1s for one window")

    print("\n-- valid inputs (should succeed) --")
    ok = True
    case_names = ["valid_wav", "short", "silence"]
    if "valid_mp3" in fixtures:
        case_names.append("valid_mp3")
    async with AsyncSessionLocal() as db:
        for name in case_names:
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
