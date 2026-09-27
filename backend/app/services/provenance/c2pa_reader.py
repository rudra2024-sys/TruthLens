"""C2PA Content Credentials reader (https://c2pa.org).

Reads and cryptographically validates an embedded C2PA manifest with the official `c2pa-python` SDK, using the
bundled C2PA trust list (trust/c2pa_trust_list.pem). Everything reported comes from the SDK's own validation
results; nothing is inferred.

What the fields mean (and do NOT mean):
  * signature_valid  - the manifest's claim signature verified.
  * content_intact   - the file's bytes still match the hash bound into the manifest, i.e. the pixels/audio were
                       not changed after signing. False means the file was modified after credentials were issued.
  * signer_trusted   - the signing certificate chains to an issuer on the C2PA trust list. Many real issuers
                       (e.g. OpenAI at the time of writing) are NOT on it, so False does not mean forged - it
                       means the signer's identity is self-asserted. A missing/stripped manifest proves nothing.
  * ai_declared      - an action in the manifest carries an IPTC digital-source-type of AI generation.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TRUST_LIST = Path(__file__).resolve().parent / "trust" / "c2pa_trust_list.pem"

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".tif": "image/tiff", ".tiff": "image/tiff", ".avif": "image/avif",
    ".heic": "image/heic", ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".wav": "audio/wav", ".mp3": "audio/mpeg",
}

# IPTC digital source types that mean "produced (at least in part) by a generative model".
_AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia")

_c2pa = None
_context = None
_import_failed = False


@dataclass
class C2paInfo:
    present: bool                               # a C2PA manifest was found in the file
    sdk_available: bool = True                  # False => only a raw byte marker scan was possible
    signature_valid: bool | None = None
    content_intact: bool | None = None
    signer_trusted: bool | None = None
    ai_declared: bool | None = None
    issuer: str | None = None
    signer_name: str | None = None
    generator: str | None = None                # claim generator / software agent
    signed_at: str | None = None
    actions: list[str] = field(default_factory=list)
    ingredient_count: int = 0
    validation_state: str | None = None
    failure_codes: list[str] = field(default_factory=list)
    error: str | None = None


def mime_for(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    return _MIME_BY_EXT.get(ext) or mimetypes.guess_type(path)[0]


def _load_sdk():
    """Import the SDK once and build a validation context that trusts the bundled C2PA trust list."""
    global _c2pa, _context, _import_failed
    if _c2pa is not None or _import_failed:
        return _c2pa
    try:
        import c2pa

        settings = {}
        if TRUST_LIST.is_file():
            settings = {"trust": {"trust_anchors": TRUST_LIST.read_text(encoding="utf-8")}}
        _context = c2pa.Context(c2pa.Settings.from_dict(settings)) if settings else None
        _c2pa = c2pa
    except Exception:                                   # SDK missing or unusable on this platform
        logger.exception("c2pa SDK unavailable; falling back to raw marker scan")
        _import_failed = True
    return _c2pa


def _raw_marker_scan(path: str) -> bool:
    """Fallback when the SDK is unavailable: does the file contain a C2PA/JUMBF box at all? (Unverified.)"""
    with open(path, "rb") as f:
        head = f.read(4 * 1024 * 1024)
    return b"c2pa" in head and (b"jumb" in head or b"caBX" in head)


def _is_ai_source(value: str | None) -> bool:
    return bool(value) and any(t in value.lower() for t in _AI_SOURCE_TYPES)


def read_c2pa(path: str) -> C2paInfo:
    sdk = _load_sdk()
    if sdk is None:
        try:
            return C2paInfo(present=_raw_marker_scan(path), sdk_available=False)
        except OSError as e:
            return C2paInfo(present=False, sdk_available=False, error=str(e))

    mime = mime_for(path)
    if not mime:
        return C2paInfo(present=False, error="unsupported file type for C2PA")
    try:
        with open(path, "rb") as f:
            reader = sdk.Reader(mime, f, context=_context) if _context is not None else sdk.Reader(mime, f)
            store = json.loads(reader.json())
    except Exception as e:
        msg = str(e)
        # "no manifest" is the normal case for most files - not an error worth surfacing.
        if re.search(r"ManifestNotFound|no claim|JumbfNotFound|not found", msg, re.I):
            return C2paInfo(present=False)
        logger.warning("C2PA read failed for %s: %s", path, msg[:200])
        return C2paInfo(present=False, error=msg[:200])

    active = (store.get("manifests") or {}).get(store.get("active_manifest") or "", {})
    if not active:
        return C2paInfo(present=False)

    results = (store.get("validation_results") or {}).get("activeManifest") or {}
    ok_codes = [s.get("code", "") for s in results.get("success", [])]
    fail_codes = [s.get("code", "") for s in results.get("failure", [])]
    # legacy field used by older SDK versions
    fail_codes += [s.get("code", "") for s in (store.get("validation_status") or []) if s.get("code") not in fail_codes]

    actions: list[str] = []
    ai_declared = False
    agent = None
    for a in active.get("assertions", []):
        if str(a.get("label", "")).startswith("c2pa.actions"):
            for act in (a.get("data") or {}).get("actions", []):
                name = act.get("action")
                if name:
                    actions.append(str(name).replace("c2pa.", ""))
                if _is_ai_source(act.get("digitalSourceType")):
                    ai_declared = True
                sa = act.get("softwareAgent")
                agent = agent or (sa.get("name") if isinstance(sa, dict) else sa)
        elif str(a.get("label", "")).startswith("cawg.training-mining") or "ai_generative" in str(a.get("label", "")):
            pass
    info = (active.get("claim_generator_info") or [{}])[0] if isinstance(active.get("claim_generator_info"), list) else {}
    sig = active.get("signature_info") or {}

    integrity_bad = any(("mismatch" in c.lower()) or ("malformed" in c.lower()) or c.startswith("claim.") for c in fail_codes)
    return C2paInfo(
        present=True,
        signature_valid=("claimSignature.validated" in ok_codes) and not any("claimSignature" in c for c in fail_codes),
        content_intact=("assertion.dataHash.match" in ok_codes or "assertion.bmffHash.match" in ok_codes
                        or "assertion.boxesHash.match" in ok_codes) and not integrity_bad,
        signer_trusted=not any(c.startswith("signingCredential.") for c in fail_codes),
        ai_declared=ai_declared,
        issuer=sig.get("issuer"),
        signer_name=sig.get("common_name"),
        generator=agent or info.get("name") or active.get("claim_generator"),
        signed_at=sig.get("time"),
        actions=list(dict.fromkeys(actions)),
        ingredient_count=len(active.get("ingredients") or []),
        validation_state=store.get("validation_state"),
        failure_codes=sorted(set(fail_codes)),
    )
