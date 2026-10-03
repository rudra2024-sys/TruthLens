"""Shared test setup.

The app builds its settings and database engine at import time from environment variables, so the environment
is pointed at a throw-away temp directory *before* anything imports `app`. Nothing here touches the developer's
real database, uploads or reports. Detectors are stubbed in the API tests so no model checkpoint is needed.
"""

import atexit
import io
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

_TMP = Path(tempfile.mkdtemp(prefix="truthlens_tests_"))
atexit.register(shutil.rmtree, _TMP, True)      # last-resort cleanup once every file handle is closed
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_TMP / "uploads")
os.environ["REPORT_DIR"] = str(_TMP / "reports")
os.environ["AUTH_SECRET_KEY"] = "test-only-secret"
for _k in ("IMAGE_MODEL_DEVICE", "CLIP_MODEL_DEVICE", "VIDEO_MODEL_V1_DEVICE"):
    os.environ.setdefault(_k, "cpu")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.main import app  # noqa: E402

TEST_DATA = BACKEND / "test_data" / "blind_spot_images"
PASSWORD = "correct-horse-battery"


def pytest_sessionfinish(session, exitstatus):
    # On Windows the temp SQLite file cannot be deleted while the engine still holds a connection to it.
    try:
        import asyncio

        from app.core.database import engine

        asyncio.run(engine.dispose())
    except Exception:
        pass
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:      # `with` runs the lifespan (creates the tables)
        yield c


def _signup(client, name):
    email = f"{name}-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/v1/auth/signup", json={"name": name, "email": email, "password": PASSWORD})
    assert r.status_code == 201, r.text
    return {"Authorization": "Bearer " + r.json()["access_token"]}, email


@pytest.fixture()
def user(client):
    """A fresh user: (auth headers, email)."""
    return _signup(client, "tester")


@pytest.fixture()
def auth(user):
    return user[0]


@pytest.fixture()
def other_auth(client):
    return _signup(client, "other")[0]


def png_bytes(color=(120, 90, 60), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(color=(10, 80, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def wav_bytes(seconds=2.0, sr=16000) -> bytes:
    """A real, decodable mono WAV (unlike a handful of zero bytes) - needed for anything that actually reads
    the audio container (provenance's duration/sample-rate probe, the real explainer)."""
    import numpy as np
    import wave

    buf = io.BytesIO()
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def video_bytes() -> bytes:
    """Not a decodable video (these tests stub the detector and never touch the frames) - just enough bytes
    for a valid ISO-BMFF `ftyp` box so the magic-byte upload check (app/services/file_sniff.py, added
    2026-10-03) doesn't reject it. All-zero placeholder bytes used to pass here only because upload
    validation trusted the Content-Type header alone; now it also checks the file's actual leading bytes."""
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 40


@pytest.fixture()
def upload(client):
    """upload(headers, data, name, content_type) -> upload_id."""
    def _upload(headers, data, name="sample.png", content_type="image/png"):
        r = client.post("/api/v1/upload/", files={"file": (name, data, content_type)}, headers=headers)
        assert r.status_code == 201, r.text
        return r.json()["upload_id"]
    return _upload


# ---- stub detectors: write real DB rows without loading any model ----------------------------------------------

def make_stub(kind, verdict="FAKE", confidence=0.93, model_used=None, frames_with_face=None):
    from app.models.models import AudioAnalysis, DetectionResult, ImageAnalysis, VideoAnalysis

    async def _stub(upload, db):
        r = DetectionResult(upload_id=upload.upload_id, confidence_score=confidence, verdict=verdict,
                            model_used=model_used or f"stub-{kind}", processing_time_ms=12.0)
        db.add(r)
        await db.flush()
        if kind == "image":
            db.add(ImageAnalysis(result_id=r.result_id, fake_probability=confidence, real_probability=1 - confidence,
                                 convnext_fake_probability=confidence * 0.9, clip_fake_probability=confidence))
        elif kind == "video":
            db.add(VideoAnalysis(result_id=r.result_id, xception_score=confidence, frames_analyzed=16,
                                 frames_with_face=frames_with_face))
        else:
            db.add(AudioAnalysis(result_id=r.result_id, wav2vec_score=confidence, lcnn_score=0.1))
        await db.flush()
        return r.result_id
    return _stub


@pytest.fixture()
def stub_detectors(monkeypatch):
    """Replace the three detectors with stubs; returns a setter to choose the verdict per kind."""
    def _set(kind, verdict="FAKE", confidence=0.93, model_used=None, frames_with_face=None):
        monkeypatch.setattr(f"app.api.routes.detect.run_{kind}_detection",
                            make_stub(kind, verdict, confidence, model_used, frames_with_face))
    for k in ("image", "video", "audio"):
        _set(k)
    return _set


@pytest.fixture()
def scan(client, upload, stub_detectors):
    """scan(headers, data, name, content_type) -> upload_id after a stubbed detection."""
    def _scan(headers, data=None, name="sample.png", content_type="image/png"):
        uid = upload(headers, data if data is not None else png_bytes(), name, content_type)
        r = client.post(f"/api/v1/detect/{uid}", headers=headers)
        assert r.status_code == 201, r.text
        return uid
    return _scan
