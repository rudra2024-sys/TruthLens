"""Lightweight magic-byte sniffing for uploaded files (added 2026-10-03).

Before this, `upload.py` trusted the client-supplied `Content-Type` header alone -- trivially spoofable
(rename a `.exe` to `video.mp4`, send `Content-Type: video/mp4`, the old check never looked at the bytes).
This checks the file's actual leading bytes against known signatures for the allowed formats and rejects a
declared media family that the bytes don't support.

**Known limitation, stated rather than hidden**: MP4 video, MOV and M4A audio all share the same ISO Base
Media File Format container (an `ftyp` box at byte offset 4) -- the container magic bytes alone cannot tell
an MP4 video apart from an M4A audio file without parsing which track types it declares, which the real
decoders (PyAV, OpenCV) already do downstream and more reliably than a byte-sniffer should try to. So this
check validates the broader family ("is this actually some kind of audio/video/image container", not "is
this exactly video and not audio") -- it still catches the main attack (an unrelated file disguised as
media), it just doesn't disambiguate within the ISO-BMFF family. No new dependency: hand-written signatures,
not `python-magic` (which needs a native libmagic binding, awkward on Windows).
"""

from __future__ import annotations

HEADER_BYTES = 64  # more than enough for every signature below, including the ftyp box at offset 4-8


def _is_image(head: bytes) -> bool:
    if head.startswith(b"\xff\xd8\xff"):                                   # JPEG
        return True
    if head.startswith(b"\x89PNG\r\n\x1a\n"):                              # PNG
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":                      # WebP
        return True
    return False


def _is_isobmff(head: bytes) -> bool:
    """MP4/MOV/M4A/M4V - all ISO Base Media File Format, sharing an `ftyp` (or, for some QuickTime files,
    `moov`/`free`/`wide`/`mdat`) box near the start. Covers both allowed video and allowed audio MP4-family
    types - see module docstring for why finer disambiguation is left to the real decoders."""
    if head[4:8] == b"ftyp":
        return True
    return head[4:8] in (b"moov", b"free", b"wide", b"mdat", b"skip")


def _is_video(head: bytes) -> bool:
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":                      # AVI
        return True
    if head.startswith(b"\x1a\x45\xdf\xa3"):                               # WebM / Matroska (EBML header)
        return True
    return _is_isobmff(head)                                               # MP4, MOV


def _is_audio(head: bytes) -> bool:
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":                      # WAV
        return True
    if head.startswith(b"fLaC"):                                           # FLAC
        return True
    if head.startswith(b"ID3"):                                            # MP3 with an ID3v2 tag
        return True
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:    # MPEG audio frame sync (MP3/AAC)
        return True
    return _is_isobmff(head)                                               # M4A


_CHECKERS = {"image": _is_image, "video": _is_video, "audio": _is_audio}


def sniff_matches_declared_family(content: bytes, media_type: str) -> bool:
    """media_type: "image" | "video" | "audio" (the already-resolved family, not the raw MIME string).
    Returns True if the file's actual leading bytes are consistent with that family's known container/
    signature formats, False if they clearly aren't (e.g. a text file, an executable, an unrelated format).
    Unknown media_type always returns False (nothing to validate against)."""
    checker = _CHECKERS.get(media_type)
    if checker is None:
        return False
    return checker(content[:HEADER_BYTES])
