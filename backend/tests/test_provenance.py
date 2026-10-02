"""Provenance service on files whose provenance is known (genuine C2PA, tampered, stripped, SD parameters, EXIF...)."""

import types

import pytest
from PIL import Image, PngImagePlugin

from app.services.provenance.c2pa_reader import read_c2pa
from app.services.provenance.metadata import read_metadata
from app.services.provenance.service import build_provenance
from tests.conftest import TEST_DATA

SAMPLE = TEST_DATA / "chatgpt_everyday" / "image1.png"          # carries genuine OpenAI Content Credentials
needs_sample = pytest.mark.skipif(not SAMPLE.is_file(), reason="C2PA sample image not present")


def up(path, media="image"):
    return types.SimpleNamespace(media_type=media, storage_url=str(path), upload_id="t")


def verdict(v):
    return types.SimpleNamespace(verdict=v)


@needs_sample
def test_genuine_credentials_are_read_and_validated():
    c = read_c2pa(str(SAMPLE))
    assert c.present and c.ai_declared is True and c.issuer
    assert c.signature_valid is True and c.content_intact is True
    assert c.signer_trusted is False        # OpenAI is not on the C2PA trust list: identity is self-asserted, not "forged"


@needs_sample
def test_conflict_note_only_when_the_verdict_disagrees():
    assert build_provenance(up(SAMPLE), verdict("REAL")).assessment.conflict_note
    assert build_provenance(up(SAMPLE), verdict("UNCERTAIN")).assessment.conflict_note
    assert build_provenance(up(SAMPLE), verdict("FAKE")).assessment.conflict_note is None


@needs_sample
def test_tampering_after_signing_is_detected(tmp_path):
    raw = bytearray(SAMPLE.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                                  # flip one byte of pixel data
    (tmp_path / "t.png").write_bytes(raw)
    c = read_c2pa(str(tmp_path / "t.png"))
    assert c.present and c.content_intact is False
    assert "assertion.dataHash.mismatch" in c.failure_codes
    titles = [s.title for s in build_provenance(up(tmp_path / "t.png"), None).signals]
    assert any("modified after" in t for t in titles)


@needs_sample
def test_stripped_credentials_leave_nothing_and_claim_nothing(tmp_path):
    with Image.open(SAMPLE) as im:
        im.convert("RGB").save(tmp_path / "s.png")
    pv = build_provenance(up(tmp_path / "s.png"), verdict("REAL"))
    assert not read_c2pa(str(tmp_path / "s.png")).present
    assert pv.assessment.level == "none" and pv.assessment.conflict_note is None      # stripping cannot be seen, so cannot accuse


def test_stable_diffusion_style_parameters_are_recognised(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a castle at dawn\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345")
    Image.new("RGB", (64, 64), (120, 90, 60)).save(tmp_path / "sd.png", pnginfo=info)
    m = read_metadata(str(tmp_path / "sd.png"))
    assert "parameters" in m.ai_param_fields and "Stable Diffusion" in (m.ai_tool or "")
    assert build_provenance(up(tmp_path / "sd.png"), verdict("REAL")).assessment.level == "declared_ai"


def test_prompt_and_seed_chunks_are_recognised(tmp_path):
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", "beautiful portrait cat")
    info.add_text("seed", "335431420")
    Image.new("RGB", (32, 32)).save(tmp_path / "d.png", pnginfo=info)
    m = read_metadata(str(tmp_path / "d.png"))
    assert {"prompt", "seed"} <= set(m.ai_param_fields) and m.prompt_excerpt == "beautiful portrait cat"


def test_camera_exif_is_a_weak_signal_and_never_an_ai_declaration(tmp_path):
    exif = Image.Exif()
    exif[0x010F], exif[0x0110], exif[0x0132] = "Canon", "EOS R6", "2026:05:01 10:11:12"
    exif.get_ifd(0x8769)[0x9003] = "2026:05:01 10:11:12"
    Image.new("RGB", (64, 64), (10, 80, 200)).save(tmp_path / "c.jpg", exif=exif, quality=90)
    m = read_metadata(str(tmp_path / "c.jpg"))
    assert m.camera_capture_metadata and m.camera_model == "EOS R6"
    pv = build_provenance(up(tmp_path / "c.jpg"), verdict("FAKE"))
    assert pv.assessment.level == "camera_metadata"
    assert all(s.strength == "weak" for s in pv.signals if s.kind == "capture")
    assert not any(s.kind == "ai" for s in pv.signals)


def test_gps_presence_is_reported_but_coordinates_are_never_exposed(tmp_path):
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (19.0, 4.0, 30.0), "E", (72.0, 52.0, 10.0)
    Image.new("RGB", (32, 32)).save(tmp_path / "g.jpg", exif=exif)
    m = read_metadata(str(tmp_path / "g.jpg"))
    assert m.has_gps is True
    blob = repr(m) + repr(build_provenance(up(tmp_path / "g.jpg"), None))
    assert "19.0" not in blob and "72.0" not in blob


def test_no_metadata_means_no_information_not_evidence(tmp_path):
    Image.new("RGB", (64, 64), (200, 200, 200)).save(tmp_path / "p.jpg", quality=90)
    pv = build_provenance(up(tmp_path / "p.jpg"), verdict("REAL"))
    assert pv.assessment.level == "none" and "nothing" in pv.assessment.headline.lower()
    assert all(s.strength == "none" for s in pv.signals)


def test_ela_only_for_jpeg(tmp_path):
    Image.new("RGB", (64, 64)).save(tmp_path / "a.png")
    Image.new("RGB", (64, 64)).save(tmp_path / "a.jpg", quality=90)
    assert build_provenance(up(tmp_path / "a.png"), None).ela.applicable is False
    assert build_provenance(up(tmp_path / "a.jpg"), None).ela.applicable is True


def test_bad_inputs_never_raise(tmp_path):
    (tmp_path / "junk.jpg").write_bytes(b"this is not an image")
    assert build_provenance(up(tmp_path / "junk.jpg"), None) is not None
    assert build_provenance(up(tmp_path / "missing.jpg"), None).available is False
    assert build_provenance(up(tmp_path / "x.wav", "audio"), None).available is False


def test_video_reports_container_facts_and_no_credentials(tmp_path):
    import cv2
    import numpy as np

    path = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for _ in range(12):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    pv = build_provenance(up(path, "video"), verdict("REAL"))
    assert pv.available and pv.container["width"] == 64 and pv.container["height"] == 48
    assert pv.assessment.level == "none"


def _write_wav(path, seconds=2.0, sr=16000):
    import numpy as np
    import wave

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def test_audio_reports_container_facts_and_no_credentials(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, seconds=2.0)
    pv = build_provenance(up(path, "audio"), verdict("REAL"))
    assert pv.available
    assert pv.container["sample_rate"] == 16000 and pv.container["channels"] == 1
    assert pv.container["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert pv.assessment.level == "none"
    assert pv.metadata is None and pv.ela is None and pv.watermark is None
    assert any(s.kind == "absent" for s in pv.signals)


def test_audio_c2pa_signals_feed_the_same_assessment_path_as_image(tmp_path):
    """Audio has no pixel-level signals, but a C2PA AI declaration should assess identically to image/video -
    read_c2pa is media-agnostic, so this only has to confirm the audio branch actually wires it in."""
    import types
    import app.services.provenance.service as svc

    path = tmp_path / "a.wav"
    _write_wav(path)
    fake_c2pa = types.SimpleNamespace(
        present=True, ai_declared=True, issuer="Test Issuer", signer_name=None, generator="TestGen",
        signed_at=None, actions=[], content_intact=True, signature_valid=True, signer_trusted=False,
    )
    original = svc.read_c2pa
    svc.read_c2pa = lambda p: fake_c2pa
    try:
        pv = build_provenance(up(path, "audio"), verdict("REAL"))
    finally:
        svc.read_c2pa = original
    assert pv.available and pv.assessment.level == "declared_ai"
    assert pv.assessment.conflict_note and "REAL" in pv.assessment.conflict_note
    assert not any(s.kind == "absent" for s in pv.signals)
