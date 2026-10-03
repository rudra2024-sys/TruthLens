"""Deep readiness probe (app/core/readiness.py, GET /health/ready, added 2026-10-03). /health itself stays
untouched (fast, dependency-free liveness probe) - these tests are only for the new endpoint."""
from app.core import readiness


def test_health_is_unaffected_and_still_fast(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_ready_reports_db_ok_in_the_test_environment(client):
    r = client.get("/health/ready")
    body = r.json()
    assert body["database"]["ok"] is True
    assert "error" not in body["database"]


def test_health_ready_checkpoint_status_reflects_the_filesystem(client, tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
    present_ckpt = tmp_path / "present.pth"
    present_ckpt.write_bytes(b"x")
    monkeypatch.setenv("CLIP_MODEL_CHECKPOINT", str(present_ckpt))

    r = client.get("/health/ready")
    body = r.json()
    assert body["checkpoints"]["image_convnext"]["present"] is False
    assert body["checkpoints"]["image_clip"]["present"] is True


def test_health_ready_returns_503_when_a_checkpoint_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL_CHECKPOINT", str(tmp_path / "missing.pth"))
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False


def test_check_readiness_reports_db_failure_without_crashing():
    import asyncio

    class _BrokenSession:
        async def execute(self, *a, **k):
            raise RuntimeError("simulated DB outage")

    out = asyncio.run(readiness.check_readiness(_BrokenSession()))
    assert out["database"]["ok"] is False
    assert "simulated DB outage" in out["database"]["error"]
    assert out["ready"] is False
