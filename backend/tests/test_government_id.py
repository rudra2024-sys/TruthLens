"""Government-ID verification tests (added 2026-10-12) -- see identity.py::
post_verify_government_id and detector.py::verify_government_id. Transient-use: the uploaded ID
photo must never survive the request, success or failure.
"""

import pytest

from tests.conftest import _signup, png_bytes, video_bytes


@pytest.fixture()
def stub_embed(monkeypatch):
    queue = []

    def _embed_stub(path):
        return queue.pop(0) if queue else [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)
    monkeypatch.setattr(
        "app.services.identity.detector._detect_faces",
        lambda path: {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)},
    )
    monkeypatch.setattr("app.services.identity.detector.assess_government_id_quality", lambda path, detection: [])

    def _queue(vector):
        queue.append(list(vector))

    return _queue


def test_verify_government_id_creates_face_reference_and_marks_it_verified(client, auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 201, r.text
    assert {"reference_id", "upload_id", "enrolled_at"} <= r.json().keys()

    me = client.get("/api/v1/identity/reference", headers=auth)
    assert me.status_code == 200
    assert me.json() is not None


def test_verify_government_id_deletes_upload_file_and_row_on_success(client, auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 201, r.text

    # The Upload row is gone -- a later owned-upload lookup (e.g. re-using the same upload_id
    # elsewhere) must 404, same convention as every other "unknown/not yours" check in this API.
    reuse = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert reuse.status_code == 404


def test_verify_government_id_deletes_upload_even_when_quality_gate_rejects(client, auth, upload, stub_embed, monkeypatch):
    monkeypatch.setattr(
        "app.services.identity.detector.assess_government_id_quality",
        lambda path, detection: ["This photo looks too blurry. Use a sharper, well-lit photo of the ID card."],
    )
    uid = upload(auth, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 422, r.text

    reuse = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert reuse.status_code == 404


def test_verify_government_id_requires_consent(client, auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": False}, headers=auth)
    assert r.status_code == 400


def test_verify_government_id_works_without_any_organization_link(client, upload, stub_embed):
    """Confirms the reordering: ID verification happens before a student joins any org."""
    headers, _email = _signup(client, "unjoined-id")
    uid = upload(headers, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text


def test_verify_government_id_rejects_non_image_upload(client, auth, upload, stub_embed):
    uid = upload(auth, video_bytes(), "id.mp4", "video/mp4")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 400


def test_other_user_cannot_use_my_upload_for_government_id(client, auth, other_auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "id.png", "image/png")
    r = client.post("/api/v1/identity/verify-government-id", json={"upload_id": uid, "consent": True}, headers=other_auth)
    assert r.status_code == 404


def test_id_quality_rejects_multiple_faces_without_touching_disk():
    """count > 1 short-circuits before the blur check's file I/O -- safe to call with a bogus
    path, same as enrollment_quality.py's own multi-face case."""
    from app.services.identity.id_quality import assess_government_id_quality

    issues = assess_government_id_quality("unused", {"count": 2, "box": None, "image_size": (100, 100)})
    assert len(issues) == 1
    assert "one face" in issues[0].lower()


def test_id_quality_skips_face_area_and_edge_checks_but_keeps_blur(tmp_path):
    """A face box tiny and right at the image edge would fail the selfie gate's face-area-
    fraction/edge-margin checks -- but this gate doesn't have those checks, so the only possible
    rejection for a solid-color (zero-variance, i.e. blurry) image is the blur check, proving it
    still runs."""
    from app.services.identity.id_quality import assess_government_id_quality

    img_path = tmp_path / "id.png"
    img_path.write_bytes(png_bytes())
    detection = {"count": 1, "box": [0.0, 0.0, 5.0, 5.0], "image_size": (1000, 1000)}
    issues = assess_government_id_quality(str(img_path), detection)
    assert issues == ["This photo looks too blurry. Use a sharper, well-lit photo of the ID card."]
