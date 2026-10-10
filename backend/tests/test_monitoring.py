"""Continuous-monitoring-session tests (added 2026-10-09) -- periodic identity/deepfake/
presence checks within a MonitoringSession, each of which can flag independently.

Stubs at the same primitive level test_identity.py's stub_embed does (the ML boundary: the
identity pipeline's embed(), and the deepfake ensemble call), while letting the REAL
_compute_signals banding/reason-building logic run -- so the integration (not just each piece
in isolation) is actually exercised. Never constructs real models in CI.

Covers "multiple faces" again too -- the first attempt (Haar cascade) was dropped after failing
on real input, then redone 2026-10-09 via IdentityPipeline.detect_faces() (MTCNN detect + IoU
dedup), validated directly against real images (see monitoring.py's module docstring and
pipelines/identity/inference.py::detect_faces's docstring for the full story).

Also covers items 7/8/10 (speech/talking, phone/book objects, gaze/head-pose), added 2026-10-09.
"""

import numpy as np
import pytest

from app.pipelines.identity.inference import NoFaceDetectedError
from tests.conftest import png_bytes, video_bytes, wav_bytes

# A plausible, roughly-frontal 5-point landmark set (MTCNN order: left eye, right eye, nose,
# left mouth corner, right mouth corner) -- only used as a stand-in input to the real
# estimate_yaw_pitch/mouth_width functions, which are NOT stubbed (the point is to exercise the
# real geometry code, same "stub only the ML boundary" principle as everywhere else here).
_FRONTAL_LANDMARKS = np.array([
    [292.44, 141.61], [362.04, 139.58], [326.10, 170.51], [299.42, 213.41], [357.15, 212.87],
])


@pytest.fixture()
def stub_embed(monkeypatch):
    """Same technique as test_identity.py's fixture of the same name -- queues canned
    embeddings for app.services.identity.detector._embed, used here only to enroll a reference
    with a known vector."""
    queue = []

    def _embed_stub(path):
        if queue:
            return queue.pop(0)
        return [1.0, 0.0]

    monkeypatch.setattr("app.services.identity.detector._embed", _embed_stub)

    def _queue(vector):
        queue.append(list(vector))

    return _queue


@pytest.fixture()
def stub_monitoring(monkeypatch):
    """Controls face count/landmarks, presence (via embed()/NoFaceDetectedError, the real
    production mechanism), the deepfake signal, object detections, and speech_ratio
    independently, while leaving app.services.identity.monitoring._compute_signals's own
    banding/reason-building logic (and the real head-pose geometry) real."""
    state = {
        "face_count": 1,
        "landmarks": _FRONTAL_LANDMARKS,
        "embedding": [1.0, 0.0],
        "no_face": False,
        "fake_probability": 0.1,
        "deepfake_verdict": "REAL",
        "object_detections": [],
        "speech_ratio": None,
    }

    class _FakePipeline:
        def detect_faces(self, path):
            return {
                "count": state["face_count"],
                "landmarks": state["landmarks"] if state["face_count"] == 1 else None,
                "box": [250.0, 130.0, 400.0, 300.0] if state["face_count"] == 1 else None,
                "image_size": (480, 640),
            }

        def embed(self, path):
            if state["no_face"]:
                raise NoFaceDetectedError("stub: no face")
            return state["embedding"]

    monkeypatch.setattr("app.services.identity.monitoring.get_identity_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(
        "app.services.identity.monitoring._run_deepfake_check",
        lambda path: (state["fake_probability"], state["deepfake_verdict"]),
    )
    monkeypatch.setattr(
        "app.services.identity.monitoring.get_object_pipeline",
        lambda: type("_FakeObjectPipeline", (), {"detect": staticmethod(lambda path, conf_thresh=0.4: state["object_detections"])})(),
    )
    monkeypatch.setattr(
        "app.services.identity.audio_vad.speech_ratio",
        lambda path: state["speech_ratio"],
    )

    def _set(
        face_count=1,
        embedding=(1.0, 0.0),
        no_face=False,
        fake_probability=0.1,
        deepfake_verdict="REAL",
        landmarks=_FRONTAL_LANDMARKS,
        object_detections=(),
        speech_ratio=None,
    ):
        state.update(
            face_count=face_count,
            embedding=list(embedding) if embedding is not None else None,
            no_face=no_face,
            fake_probability=fake_probability,
            deepfake_verdict=deepfake_verdict,
            landmarks=landmarks,
            object_detections=list(object_detections),
            speech_ratio=speech_ratio,
        )

    return _set


def _enroll(client, headers, upload, stub_embed, vector=(1.0, 0.0)):
    stub_embed(vector)
    uid = upload(headers, png_bytes(), "ref.png", "image/png")
    r = client.post("/api/v1/identity/reference", json={"upload_id": uid}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _start_session(client, headers):
    r = client.post("/api/v1/identity/sessions", headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def _submit_check(client, headers, upload, session_id):
    uid = upload(headers, png_bytes(), "frame.png", "image/png")
    return client.post(f"/api/v1/identity/sessions/{session_id}/checks", json={"upload_id": uid}, headers=headers)


def test_session_requires_enrollment(client, auth):
    r = client.post("/api/v1/identity/sessions", headers=auth)
    assert r.status_code == 404


def test_unflagged_check_when_everything_normal(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["flagged"] is False
    assert body["flag_reasons"] == []
    assert body["identity_verdict"] == "MATCH"
    assert body["deepfake_verdict"] == "REAL"
    assert body["face_count"] == 1


def test_check_includes_face_overlay_fields_when_one_face_found(client, auth, upload, stub_embed, stub_monitoring):
    """Added 2026-10-10 (follow-up item 3, live face-overlay visualization) -- face_box/
    landmarks/image dimensions should pass through to the API response whenever exactly one
    face was found, in the same pixel space as image_size."""
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_box"] == [250.0, 130.0, 400.0, 300.0]
    assert body["landmarks"] is not None and len(body["landmarks"]) == 5
    assert body["image_width"] == 480
    assert body["image_height"] == 640


def test_check_omits_face_overlay_fields_when_no_face_found(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(face_count=0)

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_box"] is None
    assert body["landmarks"] is None
    # image_size is still reported regardless of face presence (same as the rest of the frame).
    assert body["image_width"] == 480
    assert body["image_height"] == 640


def test_check_flags_identity_mismatch_even_when_deepfake_is_clean(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(0.0, 1.0), fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["flagged"] is True
    assert "identity_mismatch" in body["flag_reasons"]
    assert "deepfake_signal" not in body["flag_reasons"]
    assert body["identity_verdict"] == "NO_MATCH"


def test_check_flags_deepfake_signal_even_when_identity_matches(client, auth, upload, stub_embed, stub_monitoring):
    """The literal "even if they have matched with the identity proof" case."""
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.95, deepfake_verdict="FAKE")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["identity_verdict"] == "MATCH"
    assert body["flagged"] is True
    assert "deepfake_signal" in body["flag_reasons"]
    assert "identity_mismatch" not in body["flag_reasons"]


def test_check_flags_no_face_and_skips_identity_subcheck(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(no_face=True, fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_count"] == 0
    assert body["identity_verdict"] is None
    assert body["similarity_score"] is None
    assert body["flagged"] is True
    assert body["flag_reasons"] == ["no_face"]


def test_check_flags_multiple_faces_and_skips_identity_subcheck(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(face_count=2, fake_probability=0.05, deepfake_verdict="REAL")

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["face_count"] == 2
    assert body["identity_verdict"] is None
    assert body["similarity_score"] is None
    assert body["flagged"] is True
    assert body["flag_reasons"] == ["multiple_faces"]


def test_end_session_is_idempotent(client, auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    r1 = client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)
    assert r1.status_code == 200
    assert r1.json()["ended_at"] is not None

    r2 = client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)
    assert r2.status_code == 200
    assert r2.json()["ended_at"] == r1.json()["ended_at"]


def test_check_after_session_ended_400s(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=auth)

    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.status_code == 400


def test_get_session_returns_timeline(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed, vector=(1.0, 0.0))
    session = _start_session(client, auth)
    stub_monitoring(embedding=(1.0, 0.0), fake_probability=0.05, deepfake_verdict="REAL")
    _submit_check(client, auth, upload, session["session_id"])
    stub_monitoring(no_face=True, fake_probability=0.05, deepfake_verdict="REAL")
    _submit_check(client, auth, upload, session["session_id"])

    r = client.get(f"/api/v1/identity/sessions/{session['session_id']}", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == session["session_id"]
    assert len(body["checks"]) == 2


def test_ownership_isolation(client, auth, other_auth, upload, stub_embed):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    assert client.get(f"/api/v1/identity/sessions/{session['session_id']}", headers=other_auth).status_code == 404
    assert client.post(f"/api/v1/identity/sessions/{session['session_id']}/end", headers=other_auth).status_code == 404

    uid = upload(other_auth, png_bytes(), "frame.png", "image/png")
    r = client.post(
        f"/api/v1/identity/sessions/{session['session_id']}/checks",
        json={"upload_id": uid},
        headers=other_auth,
    )
    assert r.status_code == 404


# ---- items 7/8/10 (speech/talking, phone/book objects, gaze/head-pose) -----------------------

_TURNED_LANDMARKS = np.array([
    [212.44, 141.61], [242.04, 139.58], [246.10, 170.51], [219.42, 213.41], [237.15, 212.87],
])


def test_first_good_check_sets_baseline_and_is_not_flagged_looking_away(client, auth, upload, stub_embed, stub_monitoring):
    """The very first face-detected check becomes the session's own baseline -- it can't be
    "looking away" relative to a baseline that doesn't exist yet."""
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    stub_monitoring(landmarks=_FRONTAL_LANDMARKS)

    r = _submit_check(client, auth, upload, session["session_id"])
    body = r.json()
    assert body["yaw_deg"] is not None
    assert "looking_away" not in body["flag_reasons"]


def test_looking_away_requires_two_consecutive_off_baseline_checks(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    stub_monitoring(landmarks=_FRONTAL_LANDMARKS)
    _submit_check(client, auth, upload, session["session_id"])  # sets the baseline

    stub_monitoring(landmarks=_TURNED_LANDMARKS)
    r1 = _submit_check(client, auth, upload, session["session_id"])
    assert "looking_away" not in r1.json()["flag_reasons"], "a single off-baseline check should not flag yet"

    r2 = _submit_check(client, auth, upload, session["session_id"])
    assert "looking_away" in r2.json()["flag_reasons"], "two consecutive off-baseline checks should flag"


def test_looking_away_resets_after_returning_to_baseline(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    stub_monitoring(landmarks=_FRONTAL_LANDMARKS)
    _submit_check(client, auth, upload, session["session_id"])  # baseline
    stub_monitoring(landmarks=_TURNED_LANDMARKS)
    _submit_check(client, auth, upload, session["session_id"])  # off-baseline #1, not yet flagged
    stub_monitoring(landmarks=_FRONTAL_LANDMARKS)
    back_to_baseline = _submit_check(client, auth, upload, session["session_id"])
    assert "looking_away" not in back_to_baseline.json()["flag_reasons"]

    stub_monitoring(landmarks=_TURNED_LANDMARKS)
    r = _submit_check(client, auth, upload, session["session_id"])
    assert "looking_away" not in r.json()["flag_reasons"], "the immediately-preceding check was on-baseline, so this is only off-baseline #1 again"


def _submit_check_with_audio(client, headers, upload, session_id):
    uid = upload(headers, png_bytes(), "frame.png", "image/png")
    audio_uid = upload(headers, wav_bytes(), "clip.wav", "audio/wav")
    return client.post(
        f"/api/v1/identity/sessions/{session_id}/checks",
        json={"upload_id": uid, "audio_upload_id": audio_uid},
        headers=headers,
    )


def test_talking_detected_bands_around_threshold(client, auth, upload, stub_embed, stub_monitoring):
    """No audio_upload_id at all -> speech_ratio stays None (never computed, nothing to band)."""
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)
    r = _submit_check(client, auth, upload, session["session_id"])
    assert r.json()["speech_ratio"] is None
    assert "talking_detected" not in r.json()["flag_reasons"]


def test_talking_detected_with_audio_clip_bands_around_threshold(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    stub_monitoring(speech_ratio=0.1)
    quiet = _submit_check_with_audio(client, auth, upload, session["session_id"])
    assert "talking_detected" not in quiet.json()["flag_reasons"]
    assert quiet.json()["speech_ratio"] == 0.1

    stub_monitoring(speech_ratio=0.8)
    loud = _submit_check_with_audio(client, auth, upload, session["session_id"])
    assert "talking_detected" in loud.json()["flag_reasons"]
    assert loud.json()["speech_ratio"] == 0.8


def test_object_detected_when_phone_found(client, auth, upload, stub_embed, stub_monitoring):
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    stub_monitoring(object_detections=())
    clean = _submit_check(client, auth, upload, session["session_id"])
    assert "object_detected" not in clean.json()["flag_reasons"]
    assert clean.json()["object_detections"] == []

    stub_monitoring(object_detections=("cell phone",))
    r = _submit_check(client, auth, upload, session["session_id"])
    assert "object_detected" in r.json()["flag_reasons"]
    assert r.json()["object_detections"] == ["cell phone"]


def test_object_detection_runs_even_with_no_face_in_frame(client, auth, upload, stub_embed, stub_monitoring):
    """A phone in frame doesn't require a face to also be in frame -- same reasoning as the
    deepfake check already running independent of face_count."""
    _enroll(client, auth, upload, stub_embed)
    session = _start_session(client, auth)

    stub_monitoring(no_face=True, object_detections=("book",))
    r = _submit_check(client, auth, upload, session["session_id"])
    body = r.json()
    assert body["face_count"] == 0
    assert "no_face" in body["flag_reasons"]
    assert "object_detected" in body["flag_reasons"]
    assert body["object_detections"] == ["book"]
