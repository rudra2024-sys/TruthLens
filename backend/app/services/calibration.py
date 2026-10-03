"""Applies a pre-fit temperature-scaling calibration to a detector's raw confidence score (added 2026-10-03).

The temperature per media type is fit OFFLINE by eval/fit_calibration.py against the eval scores already
measured in eval/data/ (image: 6,500 scores across the full b1-b4 eval batches; video: the 300-video unseen
RTFS set) and written to app/core/calibration.json -- nothing here fits anything at request time, this module
only applies the fixed, pre-measured number. No model weight, threshold or verdict changes: this produces a
SEPARATE `calibrated_confidence` value, shown alongside (never instead of) the existing raw confidence_score
-- see CLAUDE.md section 25 for the measured improvement (ECE 0.158->0.056 image, 0.254->0.059 video) and why
this stays additive rather than replacing the primary number.

Audio has no entry in calibration.json (no labeled audio dataset exists locally to fit one against, same gap
noted for audio robustness in section 24) -- calibrate() returns None for it, same as for any media type with
no fitted temperature, and callers must treat None as "not available," not "assume 1.0".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "core" / "calibration.json"
_EPS = 1e-6

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def clear_cache() -> None:
    """Test hook: force calibration.json to be re-read."""
    global _cache
    _cache = None


def calibrate(probability: float, media_type: str) -> float | None:
    """Returns the calibrated probability for `media_type`, or None if no temperature has been fit for it
    (e.g. audio, or a media type with no eval data yet)."""
    cfg = _load().get(media_type)
    if cfg is None:
        return None
    temperature = cfg["temperature"]
    p = min(max(probability, _EPS), 1 - _EPS)
    z = math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-z / temperature))
