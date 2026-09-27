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
