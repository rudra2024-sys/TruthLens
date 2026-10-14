"""Identity-match tests (added 2026-10-08) -- enroll a reference photo, match a live frame.

The real face-embedding extraction (app/services/identity/detector.py::_embed, which wraps
facenet-pytorch's MTCNN+InceptionResnetV1) is stubbed in every test but one, the same way
stub_detectors stubs the deepfake detectors elsewhere in this suite -- these tests never
construct the real pipeline, so they need no cached pretrained weights and run in CI like
everything else here. The one exception at the bottom is marked `models` and self-skips when
the weights aren't cached locally, same convention as test_models.py.
"""

import pytest

from tests.conftest import _signup, png_bytes, video_bytes


@pytest.fixture()
def stub_embed(monkeypatch):
    """Queue up canned embedding vectors for successive _embed() calls (enroll, then match,
    in call order). Falls back to a fixed vector if the queue is empty.

    Also stubs _detect_faces (a well-framed single face, so the enrollment quality gate -- see
    test_enrollment_quality.py for its own dedicated tests -- doesn't reject these tests' plain
    solid-color PNG fixtures) and assess_enrollment_quality itself (so these tests stay focused
    on enroll/match behaviour, not quality-gate specifics)."""
    queue = []

    def _embed_stub(path):
        if queue:
            return queue.pop(0)
        return [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)
    monkeypatch.setattr(
        "app.services.identity.detector._detect_faces",
        lambda path: {"count": 1, "landmarks": None, "box": [8.0, 8.0, 56.0, 56.0], "image_size": (64, 64)},
    )
    monkeypatch.setattr("app.services.identity.detector.assess_enrollment_quality", lambda path, detection: [])

    def _queue(vector):
        queue.append(list(vector))

    return _queue


def _enroll(client, headers, upload, stub_embed, vector=(1.0, 0.0), name="ref.png"):
    stub_embed(vector)
    uid = upload(headers, png_bytes(), name, "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json(), uid


def test_enroll_no_longer_requires_organization(client, upload, stub_embed):
    """The org-gate added 2026-10-12 on POST /identity/reference was relocated the same day to
    exam-attempt-start (see services/exams/attempts.py::start_attempt) -- a raw, unjoined signup
    here on purpose, bypassing the `auth` fixture's auto-join to the shared default test org, to
    lock in that plain enrollment itself no longer requires an organisation link."""
    headers, _email = _signup(client, "unjoined")
    stub_embed((1.0, 0.0))
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=headers)
    assert r.status_code == 201, r.text
    assert client.get("/api/v1/identity/reference", headers=headers).json() is not None


def test_enroll_returns_reference_without_leaking_embedding(client, auth, upload, stub_embed):
    body, _ = _enroll(client, auth, upload, stub_embed)
    assert {"reference_id", "upload_id", "enrolled_at"} <= body.keys()
    assert "embedding_json" not in body
    assert "embedding" not in body


def test_get_reference_reflects_current_state(client, auth, upload, stub_embed):
    r0 = client.get("/api/v1/identity/reference", headers=auth)
    assert r0.status_code == 200
    assert r0.json() is None

    body, _ = _enroll(client, auth, upload, stub_embed)
    r1 = client.get("/api/v1/identity/reference", headers=auth)
    assert r1.status_code == 200
    assert r1.json()["reference_id"] == body["reference_id"]


def test_reenroll_replaces_the_existing_reference(client, auth, upload, stub_embed):
    first, _ = _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0), name="ref1.png")
    second, uid2 = _enroll(client, auth, upload, stub_embed, vector=(0.0, 1.0), name="ref2.png")

    assert second["reference_id"] == first["reference_id"]   # same row, updated in place
    assert second["upload_id"] == uid2
    assert client.get("/api/v1/identity/reference", headers=auth).json()["upload_id"] == uid2


def test_reference_must_be_an_image(client, auth, upload, stub_embed):
    uid = upload(auth, video_bytes(), "clip.mp4", "video/mp4")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 400


def test_match_without_enrollment_404s(client, auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "live.png", "image/png")
    r = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert r.status_code == 404


@pytest.mark.parametrize(
    "live_vector, expected_verdict",
    [
        ((1.0, 0.0), "MATCH"),        # identical to the reference -> similarity 1.0
        ((0.0, 1.0), "NO_MATCH"),     # orthogonal -> similarity 0.0
        ((0.44, 0.898), "UNCERTAIN"), # similarity ~0.44, inside the default +-0.1 band around 0.44
    ],
)
def test_match_verdict_bands_by_similarity(client, auth, upload, stub_embed, live_vector, expected_verdict):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    stub_embed(live_vector)
    uid = upload(auth, png_bytes(), "live.png", "image/png")

    r = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == expected_verdict
    assert body["reference_id"]
    assert body["upload_id"] == uid


def test_match_must_be_an_image(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    uid = upload(auth, video_bytes(), "clip.mp4", "video/mp4")
    r = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert r.status_code == 400


def test_enroll_rejects_a_low_quality_photo(client, auth, upload, stub_embed, monkeypatch):
    """Route-level check that assess_enrollment_quality's issues reach the caller as a 422 with
    the issue text -- the function's own pass/fail logic is covered separately in
    test_enrollment_quality.py."""
    monkeypatch.setattr(
        "app.services.identity.detector.assess_enrollment_quality",
        lambda path, detection: ["This photo looks too blurry. Use a sharper, well-focused photo."],
    )
    uid = upload(auth, png_bytes(), "blurry.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 422
    assert "blurry" in r.json()["detail"]
    # Nothing was stored -- GET /reference still reports no enrollment.
    assert client.get("/api/v1/identity/reference", headers=auth).json() is None


def test_enroll_without_consent_is_rejected(client, auth, upload, stub_embed):
    uid = upload(auth, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": False}, headers=auth)
    assert r.status_code == 400
    assert client.get("/api/v1/identity/reference", headers=auth).json() is None


def test_enroll_records_consent_timestamp(client, auth, upload, stub_embed):
    body, _ = _enroll(client, auth, upload, stub_embed)
    assert body["consent_given_at"]

    # Re-enrolling refreshes it, same as enrolled_at.
    body2, _ = _enroll(client, auth, upload, stub_embed, vector=(0.0, 1.0), name="ref2.png")
    assert body2["consent_given_at"]


def test_delete_reference_then_match_404s_again(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)

    r = client.delete("/api/v1/identity/reference", headers=auth)
    assert r.status_code == 204
    assert client.get("/api/v1/identity/reference", headers=auth).json() is None

    uid = upload(auth, png_bytes(), "live.png", "image/png")
    r3 = client.post("/api/v1/identity/match", json={"upload_id": uid}, headers=auth)
    assert r3.status_code == 404


def test_delete_reference_is_idempotent(client, auth):
    assert client.delete("/api/v1/identity/reference", headers=auth).status_code == 204
    assert client.delete("/api/v1/identity/reference", headers=auth).status_code == 204


def test_other_user_cannot_use_my_upload_for_reference_or_match(client, auth, other_auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    my_upload_id = upload(auth, png_bytes(), "mine.png", "image/png")

    r = client.post("/api/v1/identity/reference", json={"upload_id": my_upload_id, "consent": True}, headers=other_auth)
    assert r.status_code == 404

    r2 = client.post("/api/v1/identity/match", json={"upload_id": my_upload_id}, headers=other_auth)
    assert r2.status_code == 404


def test_each_user_has_their_own_independent_reference(client, auth, other_auth, upload, stub_embed):
    mine, _ = _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    assert client.get("/api/v1/identity/reference", headers=other_auth).json() is None

    theirs, _ = _enroll(client, other_auth, upload, stub_embed, vector=(0.0, 1.0))
    assert theirs["reference_id"] != mine["reference_id"]
    assert client.get("/api/v1/identity/reference", headers=auth).json()["reference_id"] == mine["reference_id"]


# ---- real-pipeline test: the one place allowed to construct facenet-pytorch for real --------------------------

@pytest.mark.models
def test_no_face_detected_raises_422_with_the_real_pipeline(client, auth, upload):
    """Deliberately NOT using stub_embed -- exercises the real IdentityPipeline's "no face
    found" path end to end. Self-skips when the pretrained weights aren't cached locally
    (same convention as test_models.py), so it never runs in CI (no network there)."""
    from app.pipelines.identity.model import weights_present

    if not weights_present():
        pytest.skip("InceptionResnetV1 pretrained weights not cached locally")

    uid = upload(auth, png_bytes(), "faceless.png", "image/png")  # solid-color, no face
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid, "consent": True}, headers=auth)
    assert r.status_code == 422
    assert "no face" in r.json()["detail"].lower()
