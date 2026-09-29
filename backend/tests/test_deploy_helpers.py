"""Deployment safety: start-up checks, CORS parsing, checkpoint fetch/verify, and the production compose file."""

import asyncio
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from app.core import startup
from app.core.startup import DEFAULT_SECRET, enforce_startup_checks, parse_origins, production_problems

REPO = Path(__file__).resolve().parents[2]
GOOD_SECRET = "x" * 48


# ------------------------------------------------------------------ start-up checks

def test_debug_mode_enforces_nothing():
    assert production_problems(True, DEFAULT_SECRET, ["*"]) == ([], [])
    enforce_startup_checks(True, "", ["*"])                                       # must not raise


@pytest.mark.parametrize("secret", [DEFAULT_SECRET, "", "short-secret", "y" * 31])
def test_production_refuses_an_unset_default_or_short_secret(secret):
    errors, _ = production_problems(False, secret, ["https://a.example"])
    assert errors and "AUTH_SECRET_KEY" in errors[0]
    with pytest.raises(RuntimeError, match="unsafe configuration"):
        enforce_startup_checks(False, secret, ["https://a.example"])


def test_production_accepts_a_strong_secret_with_explicit_origins():
    assert production_problems(False, GOOD_SECRET, ["https://truthlens.example"]) == ([], [])


def test_wildcard_cors_in_production_is_a_warning_not_a_blocker(caplog):
    errors, warnings = production_problems(False, GOOD_SECRET, ["*"])
    assert errors == [] and len(warnings) == 1 and "CORS_ORIGINS" in warnings[0]
    with caplog.at_level("WARNING"):
        enforce_startup_checks(False, GOOD_SECRET, ["*"])                          # logs, does not raise
    assert "CORS_ORIGINS" in caplog.text


def test_the_app_itself_refuses_to_start_unsafely(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main.settings, "DEBUG", False)
    monkeypatch.setattr(main, "SECRET_KEY", DEFAULT_SECRET)

    async def enter():
        async with main.lifespan(main.app):
            pass
    with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY"):
        asyncio.run(enter())


def test_default_secret_matches_the_one_in_security_module():
    """If someone changes the fallback in core/security.py the guard would silently stop protecting it."""
    from app.core import security
    assert startup.DEFAULT_SECRET == "change-this-development-secret-key"
    if security.os.getenv("AUTH_SECRET_KEY") is None:
        assert security.SECRET_KEY == startup.DEFAULT_SECRET


@pytest.mark.parametrize("value,expected", [
    (None, ["*"]), ("", ["*"]), ("*", ["*"]),
    ("https://a.example", ["https://a.example"]),
    ("https://a.example/, https://b.example ,", ["https://a.example", "https://b.example"]),
])
def test_parse_origins(value, expected):
    assert parse_origins(value) == expected


# ------------------------------------------------------------------ checkpoint fetch + verify

def load_fetch_module():
    path = REPO / "backend" / "scripts" / "fetch_checkpoints.py"
    spec = importlib.util.spec_from_file_location("fetch_checkpoints", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_checkpoints"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fetch_downloads_verifies_and_is_idempotent(tmp_path):
    f = load_fetch_module()
    src, root = tmp_path / "hosted", tmp_path / "repo"
    src.mkdir()
    payload = b"model-weights" * 1000
    (src / "model.pth").write_bytes(payload)
    url = src.as_uri()
    assert f.fetch(url, "checkpoints/model.pth", sha(payload), root=root) == "downloaded"
    assert (root / "checkpoints" / "model.pth").read_bytes() == payload
    assert f.fetch(url, "checkpoints/model.pth", sha(payload), root=root) == "present"


def test_fetch_discards_a_corrupt_download_and_leaves_nothing_behind(tmp_path):
    f = load_fetch_module()
    src, root = tmp_path / "hosted", tmp_path / "repo"
    src.mkdir()
    (src / "model.pth").write_bytes(b"tampered bytes")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        f.fetch(src.as_uri(), "checkpoints/model.pth", sha(b"the real weights"), root=root)
    assert not (root / "checkpoints" / "model.pth").exists()
    assert not list((root / "checkpoints").glob("*.part"))


def test_fetch_replaces_a_corrupted_local_copy(tmp_path):
    f = load_fetch_module()
    src, root = tmp_path / "hosted", tmp_path / "repo"
    (root / "checkpoints").mkdir(parents=True)
    src.mkdir()
    good = b"good"
    (src / "model.pth").write_bytes(good)
    (root / "checkpoints" / "model.pth").write_bytes(b"half-written garbage")
    assert f.fetch(src.as_uri(), "checkpoints/model.pth", sha(good), root=root) == "downloaded"
    assert (root / "checkpoints" / "model.pth").read_bytes() == good


def test_the_recorded_checksum_list_is_well_formed():
    f = load_fetch_module()
    sums = f.read_checksums()
    assert len(sums) == 3
    for rel, digest in sums.items():
        assert len(digest) == 64 and int(digest, 16) >= 0 and rel.endswith((".pth", ".pt"))


# ------------------------------------------------------------------ the production compose file

yaml = pytest.importorskip("yaml")


def compose(name):
    return yaml.safe_load((REPO / name).read_text(encoding="utf-8"))


def test_production_compose_is_hardened_relative_to_the_dev_one():
    prod, dev = compose("docker-compose.prod.yml"), compose("docker-compose.yml")
    be = prod["services"]["backend"]
    assert be["environment"]["DEBUG"] == "false"
    assert be["build"]["dockerfile"] == "Dockerfile.prod"
    assert "./backend:/app" not in be["volumes"], "production must not bind-mount the source tree"
    assert "./backend:/app" in dev["services"]["backend"]["volumes"]               # (dev keeps its live-reload mount)
    assert any(v.startswith("truthlens_data:/data") for v in be["volumes"])          # db/uploads/reports persist
    assert be["environment"]["DATABASE_URL"].endswith("/data/truthlens.db")
    assert "healthcheck" in be and be["restart"] == "unless-stopped"
    assert prod["services"]["frontend"]["build"]["args"]["NGINX_CONF"] == "nginx.prod.conf"
    checkpoint_mounts = [v for v in be["volumes"] if ":ro" in v]
    assert len(checkpoint_mounts) >= 3                                                # weights are read-only


def test_production_image_runs_one_process_as_non_root_without_reload():
    lines = (REPO / "backend" / "Dockerfile.prod").read_text(encoding="utf-8").splitlines()
    text = "\n".join(l for l in lines if not l.lstrip().startswith("#"))          # instructions only, not comments
    assert "USER truthlens" in text and "--reload" not in text and "--workers" not in text
    assert "--reload" in (REPO / "backend" / "Dockerfile").read_text(encoding="utf-8")   # dev image unchanged


def test_env_example_lists_the_required_settings_and_no_real_secret():
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for name in ("AUTH_SECRET_KEY=", "CORS_ORIGINS=", "PUBLIC_PORT="):
        assert name in text
    secret_line = next(l for l in text.splitlines() if l.startswith("AUTH_SECRET_KEY="))
    assert secret_line == "AUTH_SECRET_KEY="                                          # the example ships no value


def test_prod_nginx_sets_security_headers_and_long_proxy_timeouts():
    text = (REPO / "frontend" / "nginx.prod.conf").read_text(encoding="utf-8")
    for needle in ("X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy", "proxy_read_timeout", "server_tokens off",
                   "limit_req_zone", "limit_req zone=auth_zone", "location /api/v1/auth/"):
        assert needle in text
