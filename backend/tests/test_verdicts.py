"""Verdict banding: the +-0.2 UNCERTAIN margin around the threshold used by all three detectors."""

import pytest

from app.core.config import settings
from app.services.audio.detector import _verdict as audio_verdict
from app.services.image.detector import _verdict as image_verdict
from app.services.video.backend import _band_verdict
from app.services.video.model_v1 import common


def test_default_threshold_is_unchanged():
    """Detection thresholds are a protected ML setting (CLAUDE.md sec 8): changing one must be deliberate."""
    assert settings.FAKE_THRESHOLD == 0.5
    assert common.THRESHOLD == 0.525


@pytest.mark.parametrize("p,expected", [
    (0.0, "REAL"), (0.29, "REAL"), (0.3, "REAL"),
    (0.31, "UNCERTAIN"), (0.5, "UNCERTAIN"), (0.69, "UNCERTAIN"),
    (0.7, "FAKE"), (0.99, "FAKE"), (1.0, "FAKE"),
])
def test_image_verdict_bands(p, expected):
    assert image_verdict(p) == expected


@pytest.mark.parametrize("p,expected", [(0.05, "REAL"), (0.5, "UNCERTAIN"), (0.95, "FAKE")])
def test_audio_verdict_bands(p, expected):
    assert audio_verdict(p) == expected


@pytest.mark.parametrize("p,expected", [
    (0.30, "REAL"), (0.32, "REAL"), (0.33, "UNCERTAIN"), (0.525, "UNCERTAIN"), (0.72, "UNCERTAIN"),
    (0.73, "FAKE"), (0.90, "FAKE"),         # band edges are 0.325 / 0.725; not tested exactly (float rounding)
])
def test_video_band_is_centred_on_the_evaluated_threshold(p, expected):
    assert _band_verdict(p, common.THRESHOLD) == expected


def test_bands_are_monotonic():
    order = {"REAL": 0, "UNCERTAIN": 1, "FAKE": 2}
    seen = [order[image_verdict(i / 100)] for i in range(101)]
    assert seen == sorted(seen)
