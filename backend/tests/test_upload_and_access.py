"""Upload validation and per-user data isolation across every authenticated route."""

import pytest

from app.core.config import settings
from tests.conftest import jpeg_bytes, png_bytes, video_bytes, wav_bytes


# ---------------------------------------------------------------- upload validation

def test_upload_image_ok(client, auth):
    r = client.post("/api/v1/upload/", files={"file": ("a.png", png_bytes(), "image/png")}, headers=auth)
    assert r.status_code == 201
    body = r.json()
    assert body["media_type"] == "image" and body["file_name"] == "a.png" and body["file_size_kb"] > 0


@pytest.mark.parametrize("ctype,expected,content", [
    pytest.param("image/jpeg", "image", jpeg_bytes(), id="image-jpeg"),
    pytest.param("video/mp4", "video", video_bytes(), id="video-mp4"),
    pytest.param("audio/wav", "audio", wav_bytes(), id="audio-wav"),
    # ISO-BMFF (MP4/MOV/M4A) magic bytes can't be told apart from plain video/mp4 without parsing which
    # track types the container declares - see file_sniff.py's module docstring - so the same ftyp-box
    # placeholder content passes the sniff check for these audio content-types too.
    pytest.param("audio/mp4", "audio", video_bytes(), id="audio-mp4"),
    pytest.param("audio/x-m4a", "audio", video_bytes(), id="audio-x-m4a"),
    pytest.param("audio/m4a", "audio", video_bytes(), id="audio-m4a"),
])
def test_upload_maps_content_type_to_media_type(client, auth, ctype, expected, content):
    r = client.post("/api/v1/upload/", files={"file": ("f.bin", content, ctype)}, headers=auth)
    assert r.status_code == 201, r.text
    assert r.json()["media_type"] == expected


@pytest.mark.parametrize("ctype", ["application/pdf", "text/plain", "application/x-msdownload", "image/gif"])
def test_upload_rejects_unsupported_types(client, auth, ctype):
    r = client.post("/api/v1/upload/", files={"file": ("f.bin", b"data", ctype)}, headers=auth)
    assert r.status_code == 415


def test_upload_rejects_empty_file(client, auth):
    r = client.post("/api/v1/upload/", files={"file": ("empty.png", b"", "image/png")}, headers=auth)
    assert r.status_code == 400


def test_upload_rejects_a_file_whose_content_does_not_match_its_declared_type(client, auth):
    """Regression guard for the magic-byte check (app/services/file_sniff.py, added 2026-10-03): a correct
    Content-Type header alone used to be enough - a renamed executable claiming to be an image would have
    been accepted. Plain text content with an image/png Content-Type must now be rejected."""
    r = client.post("/api/v1/upload/", files={"file": ("fake.png", b"this is not a real PNG file" * 3, "image/png")},
                    headers=auth)
    assert r.status_code == 415
    assert "does not look like" in r.json()["detail"]


def test_upload_accepts_a_real_file_whose_name_does_not_match_its_declared_type(client, auth):
    """The sniff check is about the BYTES, not the filename - a .bin name with a real image/png
    Content-Type and real PNG bytes must still be accepted."""
    r = client.post("/api/v1/upload/", files={"file": ("photo.bin", png_bytes(), "image/png")}, headers=auth)
    assert r.status_code == 201


def test_upload_rejects_oversize(client, auth, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)
    r = client.post("/api/v1/upload/", files={"file": ("big.png", png_bytes(), "image/png")}, headers=auth)
    assert r.status_code == 413


def test_upload_is_stored_under_a_generated_name_not_the_client_filename(client, auth):
    """A hostile filename must not influence where the file lands (path traversal)."""
    import os

    r = client.post("/api/v1/upload/", files={"file": ("../../evil.png", png_bytes(), "image/png")}, headers=auth)
    assert r.status_code == 201
    stored = [f for f in os.listdir(settings.UPLOAD_DIR) if f.startswith(r.json()["upload_id"])]
    assert len(stored) == 1
    assert not os.path.exists(os.path.join(os.path.dirname(settings.UPLOAD_DIR), "evil.png"))


# ---------------------------------------------------------------- authentication required everywhere

PROTECTED = [
    ("get", "/api/v1/auth/me"),
    ("get", "/api/v1/history"),
    ("get", "/api/v1/stats"),
    ("post", "/api/v1/upload/"),
    ("get", "/api/v1/upload/some-id"),
    ("post", "/api/v1/detect/some-id"),
    ("get", "/api/v1/detect/some-id/result"),
    ("get", "/api/v1/detect/some-id/explain"),
    ("get", "/api/v1/detect/some-id/provenance"),
    ("get", "/api/v1/report/some-id"),
]


@pytest.mark.parametrize("method,path", PROTECTED)
def test_routes_require_a_token(client, method, path):
    r = getattr(client, method)(path)
    assert r.status_code in (401, 403), f"{method.upper()} {path} answered {r.status_code} without a token"


# ---------------------------------------------------------------- one user can never reach another's data

OWNED = [
    ("get", "/api/v1/upload/{id}"),
    ("post", "/api/v1/detect/{id}"),
    ("get", "/api/v1/detect/{id}/result"),
    ("get", "/api/v1/detect/{id}/explain"),
    ("get", "/api/v1/detect/{id}/provenance"),
    ("get", "/api/v1/report/{id}"),
]


@pytest.mark.parametrize("method,template", OWNED)
def test_other_users_get_404_not_403(client, scan, auth, other_auth, method, template):
    """404 (not 403) so the status code itself does not reveal that an id belongs to someone else."""
    uid = scan(auth)
    mine = getattr(client, method)(template.format(id=uid), headers=auth)
    theirs = getattr(client, method)(template.format(id=uid), headers=other_auth)
    unknown = getattr(client, method)(template.format(id="does-not-exist"), headers=auth)
    assert mine.status_code in (200, 201), mine.text
    assert theirs.status_code == 404
    assert unknown.status_code == 404
    assert theirs.json() == unknown.json()          # indistinguishable from "does not exist"


def test_history_and_stats_only_show_own_scans(client, scan, auth, other_auth):
    scan(auth)
    scan(auth, jpeg_bytes(), "b.jpg", "image/jpeg")
    scan(other_auth)
    mine = client.get("/api/v1/history", headers=auth).json()
    theirs = client.get("/api/v1/history", headers=other_auth).json()
    assert len(mine) == 2 and len(theirs) == 1
    assert {i["upload_id"] for i in mine}.isdisjoint({i["upload_id"] for i in theirs})
    assert client.get("/api/v1/stats", headers=auth).json()["total_scans"] == 2
    assert client.get("/api/v1/stats", headers=other_auth).json()["total_scans"] == 1
