"""The degradations behind the robustness numbers must actually do what their names say, deterministically."""

import numpy as np
import pytest
from PIL import Image

from eval import perturbations as P


def textured(size=(96, 80), seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    smooth = Image.fromarray(base).resize((size[0] // 4, size[1] // 4)).resize(size, Image.Resampling.BICUBIC)
    return smooth


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))


def hf_energy(img):
    g = np.asarray(img.convert("L"), np.float32)
    return float(np.var(g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]))   # Laplacian variance


def test_every_setting_runs_and_returns_rgb():
    img = textured()
    for fam, lvl in P.settings():
        out = P.apply(fam, lvl, img, "k")
        assert out.mode == "RGB" and min(out.size) >= 8, (fam, lvl)


def test_settings_cover_every_family_once_plus_clean():
    s = P.settings()
    assert s[0] == ("clean", 0)
    assert {f for f, _ in s[1:]} == set(P.FAMILIES)
    assert len(s) == 1 + sum(len(v["levels"]) for v in P.FAMILIES.values())


def test_clean_is_the_identity():
    img = textured()
    assert np.array_equal(np.asarray(P.apply("clean", 0, img)), np.asarray(img))


def test_lower_jpeg_quality_means_more_damage():
    img = textured()
    errors = [mse(img, P.apply("jpeg", q, img)) for q in (90, 70, 50, 30, 10)]
    assert errors == sorted(errors) and errors[-1] > errors[0]


def test_resize_keeps_size_and_loses_more_detail_when_smaller():
    img = textured()
    outs = [P.apply("resize", s, img) for s in (0.75, 0.5, 0.25)]
    assert all(o.size == img.size for o in outs)
    e = [mse(img, o) for o in outs]
    assert e == sorted(e)


def test_blur_removes_high_frequencies_progressively():
    img = textured()
    energy = [hf_energy(img)] + [hf_energy(P.apply("blur", s, img)) for s in (0.5, 1, 2, 3)]
    assert energy == sorted(energy, reverse=True)


@pytest.mark.parametrize("std", [5, 10, 20])
def test_noise_has_the_requested_strength_and_is_zero_mean(std):
    flat = Image.new("RGB", (200, 200), (128, 128, 128))          # mid-grey: no clipping, so the std is measurable
    diff = np.asarray(P.apply("noise", std, flat, "k"), np.float32) - 128.0
    assert abs(diff.std() - std) < 0.06 * std + 0.3 and abs(diff.mean()) < 0.5


def test_noise_and_screenshot_are_deterministic_but_differ_between_images():
    img = textured()
    for fam, lvl in (("noise", 10), ("social", "screenshot")):
        a, b = P.apply(fam, lvl, img, "same"), P.apply(fam, lvl, img, "same")
        assert np.array_equal(np.asarray(a), np.asarray(b))
    n1, n2 = P.apply("noise", 10, img, "one"), P.apply("noise", 10, img, "two")
    assert not np.array_equal(np.asarray(n1), np.asarray(n2))


def test_crop_keeps_the_centre_and_the_requested_share():
    img = textured((100, 60))
    out = P.apply("crop", 0.8, img)
    assert out.size == (80, 48)
    assert np.array_equal(np.asarray(out), np.asarray(img.crop((10, 6, 90, 54))))


def test_screenshot_rescales_within_range_and_whatsapp_caps_the_long_side():
    img = textured((100, 60))
    s = P.apply("social", "screenshot", img, "k")
    assert 0.79 <= s.size[0] / 100 <= 1.26
    big = Image.new("RGB", (3200, 1800), (10, 20, 30))
    assert max(P.apply("social", "whatsapp", big, "k").size) == 1600
    small = Image.new("RGB", (300, 200))
    assert P.apply("social", "whatsapp", small, "k").size == (300, 200)          # never upscaled


def test_social_webp_roundtrips():
    assert P.apply("social", "webp", textured(), "k").size == textured().size


def test_unknown_names_are_rejected_loudly():
    with pytest.raises(ValueError):
        P.apply("sharpen", 1, textured())
    with pytest.raises(ValueError):
        P.apply("social", "telegram", textured())
