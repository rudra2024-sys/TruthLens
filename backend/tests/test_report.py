"""PDF report generation: valid output, honest labels, graceful degradation of the best-effort sections."""

import io

import pytest
from pypdf import PdfReader

from tests.conftest import TEST_DATA, wav_bytes


def _pdf_text(content: bytes) -> str:
    assert content[:5] == b"%PDF-"
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)


def _report(client, headers, uid):
    r = client.get(f"/api/v1/report/{uid}", headers=headers)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    return _pdf_text(r.content)


def test_image_report_contains_the_verdict_and_both_sub_scores(client, scan, auth):
    text = _report(client, auth, scan(auth))
    assert "Verdict" in text and "FAKE" in text
    assert "ConvNeXt-Tiny Sub-score" in text and "CLIP Second-Opinion Sub-score" in text


def test_report_still_builds_when_the_explanation_cannot_be_computed(client, scan, auth, monkeypatch):
    """The explainability section is best-effort: a broken model must never take the whole report down."""
    def broken():
        raise RuntimeError("model exploded")
    monkeypatch.setattr("app.services.image.detector.get_image_pipeline", broken)
    text = _report(client, auth, scan(auth))
    assert "Verdict" in text and "Explainability" not in text


def test_report_still_builds_when_the_provenance_section_fails(client, scan, auth, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("c2pa exploded")
    monkeypatch.setattr("app.services.provenance.service.build_provenance", broken)
    text = _report(client, auth, scan(auth))
    assert "Verdict" in text and "Provenance" not in text


def test_video_v1_report_uses_the_correct_labels_not_the_old_heuristic_ones(client, auth, upload, stub_detectors):
    """Regression: with Video Model v1 the PDF still said 'Structure Entropy Score' / 'Byte Windows Sampled'."""
    stub_detectors("video", verdict="UNCERTAIN", confidence=0.48,
                   model_used="TruthLens Video Model v1 (EfficientNet-B0, epoch 11)")
    uid = upload(auth, b"\x00" * 64, "a.mp4", "video/mp4")
    assert client.post(f"/api/v1/detect/{uid}", headers=auth).status_code == 201
    text = _report(client, auth, uid)
    assert "Video Model v1 FAKE Probability" in text and "Frames Analyzed" in text
    assert "Structure Entropy" not in text and "Byte Windows" not in text


def test_heuristic_video_report_keeps_its_own_labels(client, auth, upload, stub_detectors):
    stub_detectors("video", verdict="UNCERTAIN", confidence=0.5, model_used="Video forensic heuristics v1")
    uid = upload(auth, b"\x00" * 64, "a.mp4", "video/mp4")
    client.post(f"/api/v1/detect/{uid}", headers=auth)
    text = _report(client, auth, uid)
    assert "Structure Entropy Score" in text and "Byte Windows Sampled" in text


def test_report_shows_the_provenance_conflict_for_a_real_verdict_on_an_ai_declared_file(client, auth, upload, stub_detectors):
    sample = TEST_DATA / "chatgpt_everyday" / "image1.png"
    if not sample.is_file():
        pytest.skip("C2PA sample image not present")
    stub_detectors("image", verdict="REAL", confidence=0.02)
    uid = upload(auth, sample.read_bytes(), "gpt.png", "image/png")
    client.post(f"/api/v1/detect/{uid}", headers=auth)
    text = _report(client, auth, uid)
    assert "Provenance" in text and "conflicts with the model verdict" in text
    assert "declare this file AI-generated" in text
    assert "does not change the detection verdict" in text        # the supplementary-evidence caveat


def test_audio_report_builds(client, scan, auth):
    text = _report(client, auth, scan(auth, b"\x00" * 64, "a.wav", "audio/wav"))
    assert "Audio Spoof Probability" in text


def test_audio_report_shows_provenance_container_facts_for_a_real_wav(client, scan, auth):
    """Audio provenance (section added 2026-10-02) is C2PA + container facts only; confirms it renders without
    the old 'not available for this media type' early-out, using a real decodable WAV (unlike the all-zero
    stub above, which has no readable container)."""
    text = _report(client, auth, scan(auth, wav_bytes(), "a.wav", "audio/wav"))
    assert "Provenance" in text
    assert "16000 Hz" in text and "1 channel(s)" in text
