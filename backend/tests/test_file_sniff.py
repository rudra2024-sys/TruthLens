"""Magic-byte upload validation (app/services/file_sniff.py, added 2026-10-03) - see its module docstring
for the known ISO-BMFF (MP4/MOV/M4A) disambiguation limitation this accepts rather than hides."""
from app.services.file_sniff import sniff_matches_declared_family
from tests.conftest import jpeg_bytes, png_bytes, video_bytes, wav_bytes


def test_real_png_matches_image():
    assert sniff_matches_declared_family(png_bytes(), "image") is True


def test_real_jpeg_matches_image():
    assert sniff_matches_declared_family(jpeg_bytes(), "image") is True


def test_real_wav_matches_audio():
    assert sniff_matches_declared_family(wav_bytes(), "audio") is True


def test_isobmff_header_matches_video():
    assert sniff_matches_declared_family(video_bytes(), "video") is True


def test_isobmff_header_also_matches_audio_known_limitation():
    """Documented limitation: MP4/M4A share the ISO-BMFF container, so the sniffer can't tell them apart by
    magic bytes alone - this is accepted, not a bug (see file_sniff.py's module docstring)."""
    assert sniff_matches_declared_family(video_bytes(), "audio") is True


def test_plain_text_disguised_as_image_is_rejected():
    assert sniff_matches_declared_family(b"this is not an image, just text" * 4, "image") is False


def test_plain_text_disguised_as_video_is_rejected():
    assert sniff_matches_declared_family(b"<html><body>not a video</body></html>", "video") is False


def test_all_zero_bytes_match_nothing():
    assert sniff_matches_declared_family(b"\x00" * 64, "image") is False
    assert sniff_matches_declared_family(b"\x00" * 64, "video") is False
    assert sniff_matches_declared_family(b"\x00" * 64, "audio") is False


def test_empty_bytes_match_nothing():
    assert sniff_matches_declared_family(b"", "image") is False


def test_unknown_media_type_always_false():
    assert sniff_matches_declared_family(png_bytes(), "unknown") is False


def test_png_does_not_match_video_or_audio():
    data = png_bytes()
    assert sniff_matches_declared_family(data, "video") is False
    assert sniff_matches_declared_family(data, "audio") is False


def test_wav_does_not_match_image_or_video():
    data = wav_bytes()
    assert sniff_matches_declared_family(data, "image") is False
    assert sniff_matches_declared_family(data, "video") is False
