"""Provenance service: C2PA Content Credentials + embedded metadata (+ ELA visual aid).

Supplementary evidence shown next to the detection verdict. It is NOT part of the verdict: this module never
changes a stored result, a threshold or a model score. Its one interaction with the verdict is a plain-language
note when provenance and the models disagree (see `assessment.conflict_note`).

Rules of interpretation baked into the wording:
  * absent metadata = no information (screenshots, chat apps and social platforms strip it);
  * present metadata can be edited or forged, so camera metadata is only ever a weak signal;
  * an explicit AI declaration (Content Credentials, XMP digital-source-type, embedded prompt/seed) is a strong
    signal that the file came from a generator - what is not proven is the signer's identity when the certificate
    is off the C2PA trust list, and that is stated alongside it.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import cv2

from app.models.models import DetectionResult, Upload
from app.services.provenance.c2pa_reader import C2paInfo, read_c2pa
from app.services.provenance.ela import ElaResult, error_level_analysis
from app.services.provenance.metadata import MetadataInfo, read_metadata
from app.services.provenance.watermark import (
    VideoWatermarkResult,
    WatermarkResult,
    detect_sd_watermark,
    detect_sd_watermark_video,
)

logger = logging.getLogger(__name__)

CAVEAT = (
    "Provenance is supplementary evidence and does not change the detection verdict. Missing metadata says nothing "
    "about authenticity (it is stripped by screenshots, messaging apps and social media), and metadata that is "
    "present can be edited or forged."
)


@dataclass
class Signal:
    kind: str          # "ai" | "capture" | "edit" | "integrity" | "trust" | "info" | "absent"
    strength: str      # "strong" | "moderate" | "weak" | "none"
    title: str
    detail: str = ""


@dataclass
class Assessment:
    level: str                       # "declared_ai" | "camera_metadata" | "none"
    headline: str
    conflict_note: str | None = None


@dataclass
class Provenance:
    available: bool
    media_type: str
    reason: str | None = None
    signals: list[Signal] = field(default_factory=list)
    assessment: Assessment | None = None
    c2pa: C2paInfo | None = None
    metadata: MetadataInfo | None = None
    ela: ElaResult | None = None
    watermark: WatermarkResult | VideoWatermarkResult | None = None
    container: dict | None = None
    caveat: str = CAVEAT


def _video_container(path: str) -> dict:
    out: dict = {}
    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   "fps": fps, "frames": frames, "duration_s": (frames / fps) if fps else None}
    finally:
        cap.release()
    try:
        with open(path, "rb") as f:
            head = f.read(2 * 1024 * 1024)
        m = re.search(rb"(Lavf[\d.]+|Lavc[\d.]+|HandBrake[ \d.]*|Adobe [A-Za-z ]{3,20}|x264 - core \d+)", head)
        if m:
            out["encoder_tag"] = m.group(1).decode("utf-8", "ignore").strip()
    except OSError:
        pass
    return out


def _audio_container(path: str) -> dict:
    """Container-level facts only (no full decode - that's the detector's job, not provenance's)."""
    try:
        import av

        container = av.open(path)
        try:
            if not container.streams.audio:
                return {}
            stream = container.streams.audio[0]
            duration_s = None
            if stream.duration and stream.time_base:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration:
                duration_s = float(container.duration) / 1_000_000
            return {
                "duration_s": duration_s,
                "sample_rate": stream.sample_rate,
                "channels": stream.channels,
                "codec": stream.codec_context.name if stream.codec_context else None,
            }
        finally:
            container.close()
    except Exception:
        return {}


def _c2pa_signals(c: C2paInfo, s: list[Signal]) -> None:
    if not c.present:
        return
    who = c.issuer or c.signer_name or "an unnamed signer"
    made_with = f" using {c.generator}" if c.generator else ""
    when = f" on {c.signed_at[:10]}" if c.signed_at else ""
    acts = f" Recorded actions: {', '.join(c.actions)}." if c.actions else ""
    if c.ai_declared:
        s.append(Signal("ai", "strong", "Content Credentials declare this file AI-generated",
                        f"Signed by {who}{made_with}{when}.{acts}"))
    else:
        s.append(Signal("info", "none", "Content Credentials (C2PA) are attached",
                        f"Signed by {who}{made_with}{when}. They do not declare AI generation.{acts}"))
    if c.content_intact is True:
        s.append(Signal("integrity", "none", "File is unchanged since its credentials were issued",
                        "The content hash bound into the credentials still matches the file."))
    elif c.content_intact is False:
        s.append(Signal("integrity", "moderate", "File was modified after its credentials were issued",
                        "The content hash in the credentials no longer matches the file, so pixels or data were "
                        "changed after signing."))
    if c.signature_valid is False:
        s.append(Signal("trust", "moderate", "Credential signature did not verify",
                        "The manifest's signature could not be validated."))
    if c.signer_trusted is False:
        s.append(Signal("trust", "none", "Signer is not on the C2PA trust list",
                        "The signer's identity is self-asserted rather than confirmed by a listed authority. That "
                        "does not mean the credentials are forged."))
    elif c.signer_trusted is True and c.signature_valid:
        s.append(Signal("trust", "none", "Signer is on the C2PA trust list",
                        "The signing certificate chains to an authority on the official C2PA trust list."))


def _metadata_signals(m: MetadataInfo, s: list[Signal]) -> None:
    if m.xmp_declares_ai:
        s.append(Signal("ai", "strong", "XMP metadata declares an AI-generated source",
                        "The file's XMP block carries an IPTC digital-source-type for algorithmic / AI media."
                        + (f" Tool named: {m.ai_tool}." if m.ai_tool else "")))
    if m.ai_param_fields:
        strong = any(k in m.ai_param_fields for k in ("prompt", "parameters", "workflow", "seed", "generation-parameters",
                                                     "sd-metadata", "invokeai_metadata", "negative_prompt"))
        detail = f"Embedded fields: {', '.join(m.ai_param_fields)}."
        if m.ai_tool:
            detail += f" Looks like output from {m.ai_tool}."
        if m.prompt_excerpt:
            detail += f' Prompt excerpt: "{m.prompt_excerpt}".'
        s.append(Signal("ai", "strong" if strong else "moderate", "Generation parameters are embedded in the file", detail))
    elif m.ai_tool and not m.xmp_declares_ai:
        s.append(Signal("ai", "moderate", f"Software field names an AI tool ({m.ai_tool})", f'Software: "{m.software}".'))
    if m.camera_capture_metadata:
        bits = [f"{m.camera_make} {m.camera_model}".strip()]
        if m.lens:
            bits.append(m.lens)
        if m.taken_at:
            bits.append(f"taken {m.taken_at}")
        s.append(Signal("capture", "weak", "Camera capture metadata is present", "; ".join(bits) +
                        ". Consistent with a camera photo, but EXIF is easy to copy or fabricate."))
    if m.editing_software:
        s.append(Signal("edit", "weak", "Editing software is recorded", f'Software: "{m.editing_software}".'))
    if m.has_gps:
        s.append(Signal("info", "none", "Contains a GPS location", "Coordinates are not shown here."))


def _watermark_signals(wm: WatermarkResult, s: list[Signal]) -> None:
    if not wm.present:
        return
    s.append(Signal(
        "ai", "strong", "Fixed Stable Diffusion / SDXL watermark detected in pixel data",
        "The image's wavelet coefficients match the fixed 48-bit watermark diffusers embeds by default "
        "(https://github.com/ShieldMnt/invisible-watermark), exactly. Unlike embedded metadata this cannot be "
        "stripped by accident and survives ordinary saving, but a miss proves nothing (many front-ends disable "
        "it, and it does not survive resizing, cropping or heavy recompression) - and because the pattern is "
        "public, someone could deliberately paste it into a real photo, the same way EXIF can be forged.",
    ))


def _video_watermark_signals(wm: VideoWatermarkResult, s: list[Signal]) -> None:
    if not wm.present:
        return
    s.append(Signal(
        "ai", "strong", "Fixed Stable Diffusion / SDXL watermark detected in sampled video frames",
        f"{wm.frames_matched} of {wm.frames_checked} sampled frames carry an exact match to diffusers' fixed "
        "48-bit watermark (https://github.com/ShieldMnt/invisible-watermark) - consistent with this clip being "
        "composed of, or re-encoding, SD/SDXL-generated frames. Unlike embedded metadata this cannot be "
        "stripped by accident, but a miss proves nothing (many front-ends disable it, and re-encoding can "
        "destroy it frame by frame) - and because the pattern is public, it is in principle forgeable onto "
        "real footage.",
    ))


def _assess(signals: list[Signal], verdict: str | None) -> Assessment:
    ai = [x for x in signals if x.kind == "ai" and x.strength == "strong"]
    cam = [x for x in signals if x.kind == "capture"]
    if ai:
        note = None
        if verdict == "REAL":
            note = ("The detection models judged this file REAL, but it carries strong direct evidence of AI "
                    "generation (embedded metadata and/or a matched generator watermark). Detection models can "
                    "miss generators they were not trained on, so this direct evidence is usually the stronger "
                    "read here - though such evidence can in principle be added to a real file on purpose (see "
                    "the trust/signal notes above for how much that applies here).")
        elif verdict == "UNCERTAIN":
            note = "The detection models were undecided; the embedded evidence points to AI generation."
        return Assessment("declared_ai", "This file carries direct evidence of AI generation (metadata and/or a watermark).", note)
    if cam:
        return Assessment("camera_metadata", "Camera capture metadata is present (weak evidence).", None)
    return Assessment("none", "No provenance metadata found - this says nothing either way.", None)


def build_provenance(upload: Upload, result: DetectionResult | None = None) -> Provenance:
    media, path = upload.media_type, upload.storage_url
    if media not in ("image", "video", "audio"):
        return Provenance(False, media, reason="Provenance checks are not available for this media type.")
    if not path or not os.path.isfile(path):
        return Provenance(False, media, reason="The uploaded file is no longer available on the server.")
    try:
        c2pa_info = read_c2pa(path)
        signals: list[Signal] = []
        meta = ela = container = watermark = None
        if media == "image":
            meta = read_metadata(path)
            ela = error_level_analysis(path)
            watermark = detect_sd_watermark(path)
            _c2pa_signals(c2pa_info, signals)
            _metadata_signals(meta, signals)
            _watermark_signals(watermark, signals)
            if (not c2pa_info.present and not meta.has_exif and not meta.ai_param_fields and not meta.xmp_declares_ai
                    and not watermark.present):
                signals.append(Signal("absent", "none", "No provenance metadata found",
                                      "No EXIF, generation parameters, Content Credentials or generator watermark. "
                                      "Common after screenshots, messaging apps and social-media re-uploads; it "
                                      "carries no information about authenticity."))
        elif media == "video":
            container = _video_container(path)
            watermark = detect_sd_watermark_video(path)
            _c2pa_signals(c2pa_info, signals)
            _video_watermark_signals(watermark, signals)
            if not c2pa_info.present and not watermark.present:
                signals.append(Signal("absent", "none", "No Content Credentials or generator watermark found",
                                      "Most videos carry neither; this says nothing about authenticity."))
        else:  # audio - no pixel-level signal applies (no EXIF/ELA/SD watermark for audio), C2PA only
            container = _audio_container(path)
            _c2pa_signals(c2pa_info, signals)
            if not c2pa_info.present:
                signals.append(Signal("absent", "none", "No Content Credentials found",
                                      "Most audio files carry none; this says nothing about authenticity."))
        return Provenance(
            available=True, media_type=media, signals=signals,
            assessment=_assess(signals, getattr(result, "verdict", None)),
            c2pa=c2pa_info, metadata=meta, ela=ela, watermark=watermark, container=container,
        )
    except Exception:
        logger.exception("Provenance failed (upload_id=%s)", getattr(upload, "upload_id", "?"))
        return Provenance(False, media, reason="Provenance could not be read for this file.")
