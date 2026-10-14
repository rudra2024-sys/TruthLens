"""confidence_score means "how sure the model is of the verdict it gave", max(P(FAKE), 1 - P(FAKE)), for all three
media types (fixed 2026-10-09: video and audio used to store the raw P(FAKE), so a confident REAL showed ~0%).
The raw P(FAKE) itself is still kept in the per-media analysis row. Runs the real detectors with stubbed models."""

import asyncio

import pytest
from sqlalchemy import text

from app.services.detection_interface import DetectionResult
from tests.conftest import png_bytes, video_bytes, wav_bytes

# (raw P(FAKE), expected confidence_score)
CASES = [(0.0004, 0.9996), (0.02, 0.98), (0.45, 0.55), (0.5, 0.5), (0.93, 0.93), (0.9999, 0.9999)]


class _Backend:
    def __init__(self, p_fake, verdict):
        self.p_fake, self.verdict = p_fake, verdict

    def score(self, path, progress=None):
        return DetectionResult(verdict=self.verdict, confidence=self.p_fake, model_used="stub", processing_time_ms=0.0,
                               raw_scores={"structure": self.p_fake, "mean": self.p_fake, "std": 0.0, "windows": 2})


def _verdict(p):
    return "FAKE" if p >= 0.7 else "REAL" if p <= 0.3 else "UNCERTAIN"


@pytest.mark.parametrize("p_fake, expected", CASES)
def test_audio_confidence_is_confidence_in_the_verdict(client, auth, upload, monkeypatch, p_fake, expected):
    monkeypatch.setattr("app.services.audio.detector.get_audio_backend", lambda: _Backend(p_fake, _verdict(p_fake)))
    uid = upload(auth, wav_bytes(), "clip.wav", "audio/wav")
    body = client.post(f"/api/v1/detect/{uid}", headers=auth).json()
    assert body["confidence_score"] == pytest.approx(expected)
    assert body["audio_analysis"]["spoof_probability"] == pytest.approx(p_fake)    # raw P(FAKE) unchanged


@pytest.mark.parametrize("p_fake, expected", CASES)
def test_video_confidence_is_confidence_in_the_verdict(client, auth, upload, monkeypatch, p_fake, expected):
    monkeypatch.setattr("app.services.video.detector.get_video_backend", lambda: _Backend(p_fake, _verdict(p_fake)))
    uid = upload(auth, video_bytes(), "clip.mp4", "video/mp4")
    body = client.post(f"/api/v1/detect/{uid}", headers=auth).json()
    assert body["confidence_score"] == pytest.approx(expected)
    if body["calibrated_confidence"] is not None:                                   # same two-sided convention
        assert 0.5 <= body["calibrated_confidence"] <= 1.0
        assert (body["calibrated_confidence"] - 0.5) * (expected - 0.5) >= 0


class _Pipeline:
    def __init__(self, p_fake):
        self.p_fake = p_fake

    def predict(self, path):
        return {"fake_probability": self.p_fake, "real_probability": 1 - self.p_fake, "confidence": 0.0}


@pytest.mark.parametrize("p_fake, expected", CASES)
def test_image_confidence_is_confidence_in_the_verdict(client, auth, upload, monkeypatch, p_fake, expected):
    monkeypatch.setattr("app.services.image.detector.get_image_pipeline", lambda: _Pipeline(p_fake))
    monkeypatch.setattr("app.services.image.detector.get_clip_pipeline", lambda: _Pipeline(p_fake))
    uid = upload(auth, png_bytes())
    body = client.post(f"/api/v1/detect/{uid}", headers=auth).json()
    assert body["confidence_score"] == pytest.approx(expected)


def test_migration_fixes_old_rows_and_is_idempotent(client, auth, scan):
    """Rows written before the fix stored P(FAKE) (e.g. 0.02 for a confident REAL); startup converts them once."""
    from app.core.database import _run_light_migrations, engine

    uid = scan(auth)

    async def set_old(conf, cal):
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE detection_results SET confidence_score=:c, calibrated_confidence=:k "
                                    "WHERE upload_id=:u"), {"c": conf, "k": cal, "u": uid})

    async def read():
        async with engine.begin() as conn:
            return (await conn.execute(text("SELECT confidence_score, calibrated_confidence FROM detection_results "
                                             "WHERE upload_id=:u"), {"u": uid})).one()

    async def go():
        await set_old(0.02, 0.1)
        await _run_light_migrations()
        first = await read()
        await _run_light_migrations()                    # runs on every startup: must not flip it back
        return first, await read()

    first, second = asyncio.run(go())
    assert first == pytest.approx((0.98, 0.9))
    assert second == pytest.approx((0.98, 0.9))
