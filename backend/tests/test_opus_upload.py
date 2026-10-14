"""Opus/Ogg audio uploads (added 2026-10-09): accepted by MIME type, by extension when the browser sends no
usable MIME type, magic-byte checked like every other format, and decodable by the real audio pipeline."""

import io

import av
import numpy as np
import pytest

from app.api.routes.upload import resolve_content_type
from app.services.file_sniff import sniff_matches_declared_family
from app.services.models.audio_model import decode_audio_mono_16k


def opus_bytes(seconds: float = 2.0) -> bytes:
    """A real Ogg/Opus file: a 220 Hz tone encoded with libopus."""
    buf = io.BytesIO()
    out = av.open(buf, "w", format="ogg")
    stream = out.add_stream("libopus", rate=48000)
    stream.layout = "mono"
    t = np.arange(int(48000 * seconds)) / 48000
    samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for i in range(0, len(samples), 960):
        frame = av.AudioFrame.from_ndarray(samples[i:i + 960][None, :], format="flt", layout="mono")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    return buf.getvalue()


def test_ogg_opus_matches_audio_family_only():
    data = opus_bytes()
    assert data[:4] == b"OggS"
    assert sniff_matches_declared_family(data, "audio") is True
    assert sniff_matches_declared_family(data, "image") is False
    assert sniff_matches_declared_family(data, "video") is False


@pytest.mark.parametrize("ctype, name, expected", [
    ("audio/ogg", "x.opus", "audio/ogg"),                       # a real MIME type is never overridden
    (None, "voice.opus", "audio/opus"),
    ("", "voice.OPUS", "audio/opus"),
    ("application/octet-stream", "note.ogg", "audio/ogg"),
    ("application/ogg", "note.oga", "audio/ogg"),
    ("application/octet-stream", "file.bin", "application/octet-stream"),   # unknown extension: unchanged
    ("text/plain", "fake.opus", "text/plain"),                  # only missing/generic types fall back
])
def test_resolve_content_type(ctype, name, expected):
    assert resolve_content_type(ctype, name) == expected


@pytest.mark.parametrize("ctype, name", [
    ("audio/ogg", "clip.opus"),
    ("audio/opus", "clip.opus"),
    ("application/octet-stream", "clip.opus"),
    ("application/octet-stream", "voice-note.ogg"),
])
def test_opus_upload_is_accepted_as_audio(client, auth, ctype, name):
    r = client.post("/api/v1/upload/", files={"file": (name, opus_bytes(), ctype)}, headers=auth)
    assert r.status_code == 201, r.text
    assert r.json()["media_type"] == "audio"


def test_non_audio_bytes_named_opus_are_rejected(client, auth):
    r = client.post("/api/v1/upload/", files={"file": ("fake.opus", b"not audio at all" * 8, "application/octet-stream")},
                    headers=auth)
    assert r.status_code == 415


def test_unknown_extension_with_generic_type_is_still_rejected(client, auth):
    r = client.post("/api/v1/upload/", files={"file": ("clip.bin", opus_bytes(), "application/octet-stream")},
                    headers=auth)
    assert r.status_code == 415


def test_opus_decodes_through_the_production_audio_decoder(tmp_path):
    path = tmp_path / "clip.opus"
    path.write_bytes(opus_bytes(seconds=3.0))
    samples = decode_audio_mono_16k(str(path))
    assert samples.dtype == np.float32
    assert abs(len(samples) - 3 * 16000) < 400          # ~3 s at 16 kHz (Opus pre-skip trims a few ms)
    assert 0.25 < float(np.abs(samples).max()) < 0.35   # the 0.3-amplitude tone survives the round trip
