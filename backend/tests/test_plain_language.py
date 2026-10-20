"""The plain-English parts of the single-file PDF report (services/report/plain_language.py)."""

from types import SimpleNamespace

from app.services.report import plain_language as pl
from tests.test_report import _report
from tests.conftest import wav_bytes


def _result(verdict, **kw):
    base = dict(verdict=verdict, confidence_score=0.9, calibrated_confidence=None, image_analysis=None,
                video_analysis=None, detected_at="2026-10-20T19:39:51.399052")
    base.update(kw)
    return SimpleNamespace(**base)


def _upload(media="image", path="/nonexistent"):
    return SimpleNamespace(media_type=media, storage_url=path)


def test_friendly_date_is_readable_and_never_raises():
    assert pl.friendly_date("2026-10-20T19:39:51.399052") == "20 October 2026, 7:39 PM (UTC)"
    assert pl.friendly_date("2026-01-05T08:05:00") == "5 January 2026, 8:05 AM (UTC)"
    assert pl.friendly_date("not a date") == "not a date" and pl.friendly_date(None) == "unknown"


def test_flagged_image_says_face_swap_only_when_the_face_check_is_the_highest_score():
    ia = SimpleNamespace(convnext_fake_probability=0.2, clip_fake_probability=0.4, face_swap_fake_probability=0.8)
    assert pl.image_fake_kind(_result("FAKE", image_analysis=ia)) == "face swap"
    ia2 = SimpleNamespace(convnext_fake_probability=0.2, clip_fake_probability=0.99, face_swap_fake_probability=0.8)
    assert pl.image_fake_kind(_result("FAKE", image_analysis=ia2)) == "AI-generated"
    ia3 = SimpleNamespace(convnext_fake_probability=0.2, clip_fake_probability=0.99, face_swap_fake_probability=None)
    assert pl.image_fake_kind(_result("FAKE", image_analysis=ia3)) == "AI-generated"
    assert pl.image_fake_kind(_result("REAL", image_analysis=ia)) is None        # only ever for a flagged image


def test_summary_sentences_are_hedged_not_absolute():
    s = {v: pl.summary_sentence(_upload(), _result(v)) for v in ("FAKE", "REAL", "UNCERTAIN")}
    assert "probably" in s["FAKE"] and "thinks" in s["FAKE"]
    assert "did not find" in s["REAL"] and "could not tell" in s["UNCERTAIN"]
    assert all("is fake" not in t.lower() and "proves" not in t.lower() for t in s.values())


def test_reliability_wording_depends_on_whether_calibration_differs():
    close = dict(pl.plain_points(_upload(), _result("REAL", calibrated_confidence=0.89)))["How sure is it?"]
    far = dict(pl.plain_points(_upload(), _result("REAL", calibrated_confidence=0.70)))["How sure is it?"]
    none = dict(pl.plain_points(_upload(), _result("REAL")))["How sure is it?"]
    assert "about as often" in close and "70%" in far and "more realistic" in far
    assert "not measured" in none


def test_cautions_only_appear_when_they_apply():
    real_img = dict(pl.plain_points(_upload(), _result("REAL")))["Things to keep in mind"]
    assert "newer AI image generators" in real_img
    fake_img = dict(pl.plain_points(_upload(), _result("FAKE")))
    assert "Things to keep in mind" not in fake_img                    # nothing applies to a flagged, normal image
    lowres = SimpleNamespace(low_resolution=True, frames_with_face=None, frames_analyzed=16,
                             calibrated_fake_probability=0.5)
    vid = dict(pl.plain_points(_upload("video"), _result("UNCERTAIN", video_analysis=lowres)))["Things to keep in mind"]
    assert "too low in resolution" in vid and "Sora" in vid
    aud = dict(pl.plain_points(_upload("audio"), _result("REAL")))["Things to keep in mind"]
    assert "not yet tested" in aud


def test_glossary_leaves_out_terms_for_absent_sections():
    names = [t for t, _ in pl.glossary_terms("image", has_provenance=False, has_explanation=False, has_face_swap=False)]
    assert names == ["Confidence", "Estimated reliability", "Uncertain"]
    full = [t for t, _ in pl.glossary_terms("audio")]
    assert "Spectrogram" in full and "Heatmap (Grad-CAM)" not in full


def test_real_report_has_plain_words_glossary_and_no_internal_references(client, scan, auth):
    text = _report(client, auth, scan(auth))
    assert "IN PLAIN WORDS" in text and "How sure is it?" in text and "What you can do next" in text
    assert "What the terms in this report mean" in text and "About this report" in text
    assert "CLAUDE.md" not in text and "temperature-scaled" not in text
    assert "T19:" not in text and "(UTC)" in text                            # no raw ISO timestamp
    assert "Manipulated (FAKE)" in text                                       # stub detector returns FAKE


def test_audio_report_glossary_is_scoped_to_audio(client, scan, auth):
    text = _report(client, auth, scan(auth, wav_bytes(), "a.wav", "audio/wav"))
    assert "Windows." in text and "Heatmap (Grad-CAM)." not in text and "Face-swap check." not in text
