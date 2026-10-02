"""Tests that need the REAL model checkpoints. Marked `models`; each one skips itself when its checkpoint is absent
(CI has none: they are gitignored), so `pytest` is green everywhere and `pytest -m models` exercises them locally.

They pin the explanation code to the deployed models (a heatmap must describe the same computation as the verdict)
and check the deployed detectors' documented behaviour on the repository's own sample images.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import BACKEND, TEST_DATA

REPO = BACKEND.parent
pytestmark = pytest.mark.models

IMAGE_CKPT = Path(os.getenv("IMAGE_MODEL_CHECKPOINT") or REPO / "models" / "checkpoints" / "image" / "convnext_tiny_diversified_v2.pth")
CLIP_CKPT = Path(os.getenv("CLIP_MODEL_CHECKPOINT") or REPO / "models" / "checkpoints" / "image_clip" / "clip_head_round3_portrait_app.pth")
VIDEO_CKPT = Path(os.getenv("VIDEO_MODEL_V1_CHECKPOINT") or BACKEND / "checkpoints" / "video" / "epoch_11_model_only.pt")
AUDIO_CKPT = Path(os.getenv("AUDIO_MODEL_V1_CHECKPOINT") or BACKEND / "checkpoints" / "audio" / "wav2vec2_v9.pt")
TOL = 1e-5


def _images(kind, n):
    return sorted((TEST_DATA / kind).glob("*.png"))[:n]


@pytest.fixture(scope="module")
def convnext():
    if not IMAGE_CKPT.is_file():
        pytest.skip(f"ConvNeXt checkpoint not found: {IMAGE_CKPT}")
    from app.pipelines.image.inference import ImagePipeline
    return ImagePipeline(checkpoint_path=str(IMAGE_CKPT))


@pytest.fixture(scope="module")
def clip():
    if not CLIP_CKPT.is_file():
        pytest.skip(f"CLIP checkpoint not found: {CLIP_CKPT}")
    from app.pipelines.image_clip.inference import ClipPipeline
    return ClipPipeline(checkpoint_path=str(CLIP_CKPT))


def test_image_explainer_reproduces_the_production_probability(convnext):
    from app.services.explain.image_explainer import explain_image
    for p in _images("portrait_app", 3) + _images("chatgpt_everyday", 2):
        e = explain_image(str(p), convnext.model, convnext.device)
        assert abs(e.fake_probability - convnext.predict(str(p))["fake_probability"]) < TOL
        assert 0.0 <= e.evidence_strength <= 1.0 and e.overlay_jpeg[:2] == b"\xff\xd8"


def test_evidence_fades_for_images_the_model_thinks_are_real(convnext):
    from app.services.explain.image_explainer import explain_image
    real = [p for p in _images("chatgpt_everyday", 8)
            if convnext.predict(str(p))["fake_probability"] < 0.05]     # the documented blind spot: scored REAL
    if not real:
        pytest.skip("no confidently-REAL sample available")
    assert explain_image(str(real[0]), convnext.model, convnext.device).evidence_strength < 0.2


def test_documented_blind_spots_still_hold(convnext, clip):
    """CLAUDE.md sec 3 records: ConvNeXt is blind on portrait-app images and both models miss the ChatGPT set.
    If this starts failing after a model change, the docs (and the report numbers) need updating - it is a tripwire."""
    gpt = _images("chatgpt_everyday", 8)
    if len(gpt) < 4:
        pytest.skip("not enough ChatGPT samples")
    missed = sum(max(convnext.predict(str(p))["fake_probability"], clip.predict(str(p))["fake_probability"]) < 0.5 for p in gpt)
    assert missed >= len(gpt) - 1


def test_ensemble_probability_is_the_max_of_the_two_models(client, auth, upload, convnext, clip, monkeypatch):
    """End to end through the real detector: stored ensemble score == max(ConvNeXt, CLIP)."""
    monkeypatch.setenv("IMAGE_MODEL_CHECKPOINT", str(IMAGE_CKPT))
    monkeypatch.setenv("CLIP_MODEL_CHECKPOINT", str(CLIP_CKPT))
    import app.services.image.detector as det
    monkeypatch.setattr(det, "_image_pipeline", convnext)
    monkeypatch.setattr(det, "_clip_pipeline", clip)
    sample = _images("portrait_app", 1)[0]
    uid = upload(auth, sample.read_bytes(), sample.name, "image/png")
    body = client.post(f"/api/v1/detect/{uid}", headers=auth).json()
    a = body["image_analysis"]
    assert a["fake_probability"] == pytest.approx(max(a["convnext_fake_probability"], a["clip_fake_probability"]))
    assert body["model_used"] == "ConvNeXt-Tiny + CLIP ViT-B/16 (ensemble)"


@pytest.mark.parametrize("path", [r"C:\fake photos\fake video 3.mp4", r"C:\real photos\real video 1.mp4"])
def test_video_explainer_reproduces_the_production_score(path):
    if not VIDEO_CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.explain import video_explainer
    from app.services.video.model_v1 import optimized
    ex = video_explainer.explain_video(path, str(VIDEO_CKPT))
    ref = optimized.predict(path, str(VIDEO_CKPT))
    assert len(ex.frames) == 16
    assert abs(ex.probability - ref.probability) < TOL
    assert max(abs(f.logit - r) for f, r in zip(ex.frames, ref.frame_logits)) < TOL


@pytest.mark.parametrize("path", [r"C:\fake photos\fake video 3.mp4", r"C:\real photos\real video 1.mp4"])
def test_video_progress_hook_is_observation_only(path):
    """The progress callback used by background jobs must not change what the model computes."""
    if not VIDEO_CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized
    events = []
    plain = optimized.predict(path, str(VIDEO_CKPT))
    watched = optimized.predict(path, str(VIDEO_CKPT), progress=lambda f, stage: events.append((f, stage)))
    assert watched.probability == plain.probability and watched.frame_logits == plain.frame_logits
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert {"Reading frames", "Finding faces", "Scoring frames"} <= {st.split(" (")[0] for _, st in events}


def test_progress_callback_exception_aborts_the_run():
    """Cancellation works by raising inside the callback; the model must not swallow it."""
    path = r"C:\fake photos\fake video 3.mp4"
    if not VIDEO_CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized

    class Stop(Exception):
        pass

    def cb(f, stage):
        if f > 0.2:
            raise Stop()

    with pytest.raises(Stop):
        optimized.predict(path, str(VIDEO_CKPT), progress=cb)


def _write_wav(path, seconds=6, sr=16000, freq=220.0, seed=0):
    import numpy as np
    import wave

    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * freq * t)
    noise = 0.02 * rng.standard_normal(t.shape[0])
    samples = ((tone + noise) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def test_audio_explainer_reproduces_the_production_score(tmp_path):
    if not AUDIO_CKPT.is_file():
        pytest.skip("audio checkpoint not present")
    from app.services.explain import audio_explainer
    from app.services.audio.model_v1 import optimized

    path = tmp_path / "sample.wav"
    _write_wav(path)
    ex = audio_explainer.explain_audio(str(path), str(AUDIO_CKPT))
    ref = optimized.predict(str(path), str(AUDIO_CKPT))
    assert len(ex.windows) == ref.windows_analyzed
    assert abs(ex.mean_probability - ref.deepfake_probability_mean) < TOL
    assert abs(ex.duration_s - ref.duration_s) < TOL
    assert 0 <= ex.saliency_window_order < len(ex.windows)
    assert len(ex.spectrogram_jpeg) > 0


def test_audio_explainability_renders_in_the_pdf_report(client, scan, auth, tmp_path):
    if not AUDIO_CKPT.is_file():
        pytest.skip("audio checkpoint not present")
    import io

    from pypdf import PdfReader
    from tests.conftest import wav_bytes

    path = tmp_path / "a.wav"
    path.write_bytes(wav_bytes(seconds=6.0))
    uid = scan(auth, path.read_bytes(), "a.wav", "audio/wav")
    r = client.get(f"/api/v1/report/{uid}", headers=auth)
    assert r.status_code == 200
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
    assert "Explainability" in text and "Windows as scored by the model" in text
    assert "Spectrogram" in text


def test_audio_window_timestamps_cover_the_clip_in_order(tmp_path):
    if not AUDIO_CKPT.is_file():
        pytest.skip("audio checkpoint not present")
    from app.services.explain import audio_explainer

    path = tmp_path / "sample.wav"
    _write_wav(path, seconds=10)
    ex = audio_explainer.explain_audio(str(path), str(AUDIO_CKPT))
    starts = [w.start_s for w in ex.windows]
    assert starts == sorted(starts)
    assert all(w.end_s - w.start_s == pytest.approx(4.0) for w in ex.windows)
    assert ex.windows[-1].end_s <= ex.duration_s + 1e-6


def test_preprocess_frame_with_info_matches_preprocess_frame_exactly():
    """preprocess_frame() was refactored 2026-10-03 to delegate to preprocess_frame_with_info() (CLAUDE.md
    section 21) -- this pins that the refactor changed nothing about the existing function's output."""
    import numpy as np
    from app.services.video.model_v1 import common

    cascade = common.get_face_cascade()
    frame = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    plain = common.preprocess_frame(frame, cascade)
    with_info, info = common.preprocess_frame_with_info(frame, cascade)
    assert np.array_equal(plain, with_info)
    assert "face_bbox" in info and "crop_bounds" in info


@pytest.mark.parametrize("path,expected_frames_with_face", [
    (r"C:\fake photos\fake video 1.mp4", 16),
    (r"C:\fake photos\fake video 4.mp4", 0),
])
def test_frames_with_face_matches_the_diagnosed_clips(path, expected_frames_with_face):
    """Regression pin for the two clips diagnosed in CLAUDE.md sections 19/21: video 1 finds a face on every
    sampled frame, video 4 (the sideways-encoded Akool/Magic Hour blind spot) finds none."""
    if not VIDEO_CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.model_v1 import optimized
    r = optimized.predict(path, str(VIDEO_CKPT))
    assert r.frames_with_face == expected_frames_with_face


def test_video_v1_backend_surfaces_frames_with_face_in_raw_scores():
    path = r"C:\fake photos\fake video 1.mp4"
    if not VIDEO_CKPT.is_file() or not os.path.isfile(path):
        pytest.skip("video checkpoint or sample video not present")
    from app.services.video.backend import VideoModelV1Backend
    result = VideoModelV1Backend().score(path)
    assert result.raw_scores["frames_with_face"] == 16
