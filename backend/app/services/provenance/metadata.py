"""Embedded-metadata reader: EXIF / PNG text chunks / XMP, looking for capture and generation evidence.

Purely descriptive. Metadata is trivially stripped or forged, so what it says is reported as a *signal* with the
appropriate caveat by the caller; the absence of metadata is never treated as evidence either way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from PIL import ExifTags, Image

# Field names that AI image tools write into PNG text chunks / EXIF. Presence of prompt/seed/sampler-style
# parameters is typical of Stable-Diffusion-family front ends.
_PARAM_KEYS = {"parameters", "prompt", "negative_prompt", "workflow", "seed", "sd-metadata", "invokeai_metadata",
               "dream", "comment", "description", "software", "generation_time", "steps", "sampler", "cfg_scale"}

_AI_TOOLS = [
    (r"midjourney", "Midjourney"), (r"dall[\-·. ]?e", "DALL·E"), (r"stable[ _-]?diffusion|sdxl|automatic1111|a1111", "Stable Diffusion"),
    (r"comfyui|class_type", "ComfyUI"), (r"invokeai", "InvokeAI"), (r"novelai", "NovelAI"),
    (r"firefly", "Adobe Firefly"), (r"leonardo", "Leonardo.ai"), (r"imagen", "Google Imagen"),
    (r"gemini", "Google Gemini"), (r"flux", "FLUX"), (r"gpt-image|openai", "OpenAI"), (r"ideogram", "Ideogram"),
    (r"playground", "Playground"), (r"nightcafe", "NightCafe"), (r"bing image creator|designer", "Microsoft Designer"),
]

_EDITORS = ["photoshop", "lightroom", "gimp", "snapseed", "canva", "paint.net", "affinity", "pixelmator", "facetune",
            "picsart", "capture one", "darktable", "luminar", "fotor", "remini", "meitu"]

_XMP_AI_RE = re.compile(rb"trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|algorithmicMedia", re.I)


@dataclass
class MetadataInfo:
    format: str | None = None
    width: int | None = None
    height: int | None = None
    has_exif: bool = False
    camera_make: str | None = None
    camera_model: str | None = None
    lens: str | None = None
    software: str | None = None
    taken_at: str | None = None
    exposure: dict = field(default_factory=dict)         # exposure_time / f_number / iso / focal_length (strings)
    has_gps: bool = False                                # presence only - coordinates are never returned
    ai_param_fields: list[str] = field(default_factory=list)
    ai_tool: str | None = None
    prompt_excerpt: str | None = None
    xmp_declares_ai: bool = False
    editing_software: str | None = None
    error: str | None = None

    @property
    def camera_capture_metadata(self) -> bool:
        """Make + model plus at least one exposure/time field: what a camera pipeline writes."""
        return bool(self.camera_make and self.camera_model and (self.taken_at or self.exposure))


def _text(v) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")
    return str(v).replace("\x00", "").strip()


def _match_tool(blob: str) -> str | None:
    low = blob.lower()
    for pat, name in _AI_TOOLS:
        if re.search(pat, low):
            return name
    return None


def read_metadata(path: str) -> MetadataInfo:
    info = MetadataInfo()
    try:
        with Image.open(path) as im:
            info.format, (info.width, info.height) = im.format, im.size
            exif = im.getexif()
            named = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            # Exif sub-IFD (exposure, lens, DateTimeOriginal live here)
            try:
                sub = exif.get_ifd(ExifTags.IFD.Exif)
                named.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
                info.has_gps = bool(exif.get_ifd(ExifTags.IFD.GPSInfo))
            except Exception:
                pass
            info.has_exif = bool(named)
            info.camera_make = _text(named["Make"]) if named.get("Make") else None
            info.camera_model = _text(named["Model"]) if named.get("Model") else None
            info.lens = _text(named["LensModel"]) if named.get("LensModel") else None
            info.software = _text(named["Software"]) if named.get("Software") else None
            info.taken_at = _text(named.get("DateTimeOriginal") or named.get("DateTime") or "") or None
            for label, key in (("exposure_time", "ExposureTime"), ("f_number", "FNumber"),
                               ("iso", "ISOSpeedRatings"), ("focal_length", "FocalLength")):
                if named.get(key) is not None:
                    info.exposure[label] = _text(named[key])[:24]

            # PNG text chunks / JPEG comment / WEBP metadata all surface in im.info
            fields: dict[str, str] = {}
            for k, v in im.info.items():
                if k in ("icc_profile", "exif", "xmp", "XML:com.adobe.xmp") or isinstance(v, (tuple, int, float)):
                    continue
                fields[str(k).lower()] = _text(v)
            uc = named.get("UserComment")
            if uc:
                fields["usercomment"] = _text(uc)
            if named.get("ImageDescription"):
                fields["imagedescription"] = _text(named["ImageDescription"])

            info.ai_param_fields = sorted(k for k in fields if k in _PARAM_KEYS and k not in ("software", "description", "comment"))
            blob = " ".join(f"{k} {v[:400]}" for k, v in fields.items() if k in _PARAM_KEYS | {"usercomment", "imagedescription"})
            blob += " " + (info.software or "")
            # Parameter-style text without an explicit key (e.g. A1111 puts "Steps: 20, Sampler: ..." in one field)
            sd_style = any(re.search(r"\bsteps:\s*\d+|\bsampler:|\bcfg scale:|\bseed:\s*\d+", v, re.I) for v in fields.values())
            if sd_style and "parameters" not in info.ai_param_fields:
                info.ai_param_fields.append("generation-parameters")
            for src in ("prompt", "parameters"):
                if fields.get(src):
                    first = fields[src].splitlines()[0][:140]
                    if not first.lstrip().startswith("{"):        # skip ComfyUI JSON graph dumps
                        info.prompt_excerpt = first
                    break
            # Only name an AI tool if there are generation fields or an explicit tool name in Software/comment
            if info.ai_param_fields or info.software:
                info.ai_tool = _match_tool(blob)
            if sd_style and not info.ai_tool:
                # "Steps / Sampler / CFG scale / Seed" is the parameter layout of Stable-Diffusion-family front ends
                # (AUTOMATIC1111 and compatible); it never spells out the model name, so name the family, not a product.
                info.ai_tool = "Stable Diffusion-family front end"

            xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
            if xmp:
                xb = xmp if isinstance(xmp, bytes) else str(xmp).encode("utf-8", "ignore")
                info.xmp_declares_ai = bool(_XMP_AI_RE.search(xb))
                if not info.ai_tool:
                    info.ai_tool = _match_tool(xb.decode("utf-8", "ignore")[:20000]) if info.xmp_declares_ai else None
    except Exception as e:
        info.error = f"{type(e).__name__}: {e}"[:200]
        return info

    soft = (info.software or "").lower()
    for ed in _EDITORS:
        if ed in soft:
            info.editing_software = info.software
            break
    return info
