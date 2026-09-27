"""Explainability maths on tiny synthetic models (no checkpoints needed)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from app.services.explain import core


class TinyNet(nn.Module):
    """features -> avgpool -> classifier, like the real models. Class 0 (FAKE) is driven by ONE channel only."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight.zero_()
            self.features.weight[0, 0] = 1.0     # channel 0 <- input channel 0 (the only thing the score reads)
            self.features.weight[1, 1] = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_grad_cam_peaks_where_the_score_is_earned():
    net = TinyNet().eval()
    x = torch.zeros(1, 3, 8, 8)
    x[0, 0, 2, 5] = 4.0                                     # the only evidence for the FAKE logit
    with torch.enable_grad():
        feats = net.features(x)
        logits = net.classifier(torch.flatten(net.avgpool(feats), 1))
        cam = core.cam_from_features(feats, logits[:, 0].sum())
    assert cam.shape == (1, 8, 8)
    assert torch.isclose(cam.max(), torch.tensor(1.0))     # normalised
    # Grad-CAM weights are spatially uniform for this net, so the map follows the activation: peak at (2, 5)
    assert np.unravel_index(int(cam[0].argmax()), (8, 8)) == (2, 5)


def test_grad_cam_is_all_zero_when_the_model_saw_no_positive_evidence():
    net = TinyNet().eval()
    x = torch.zeros(1, 3, 8, 8)
    with torch.enable_grad():
        feats = net.features(x)
        logits = net.classifier(torch.flatten(net.avgpool(feats), 1))
        cam = core.cam_from_features(feats, logits[:, 0].sum())
    assert float(cam.abs().sum()) == 0.0


def test_grad_cam_does_not_touch_model_parameter_gradients():
    """Safe next to the live model: no .grad may be left on its parameters."""
    net = TinyNet().eval()
    x = torch.rand(1, 3, 8, 8)
    with torch.enable_grad():
        feats = net.features(x)
        logits = net.classifier(torch.flatten(net.avgpool(feats), 1))
        core.cam_from_features(feats, logits[:, 0].sum())
    assert all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize("p,expected", [(0.0, 0.0), (0.1, 0.2), (0.25, 0.5), (0.5, 1.0), (0.99, 1.0), (-1, 0.0), (2, 1.0)])
def test_evidence_strength_fades_maps_for_confident_real_images(p, expected):
    assert core.evidence_strength(p) == pytest.approx(expected)


def test_colormap_range_and_shape():
    v = np.linspace(0, 1, 11).reshape(1, 11)
    rgb = core.colormap(v)
    assert rgb.shape == (1, 11, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (0, 0, 143) and tuple(rgb[0, -1]) == (255, 0, 0)      # cold end blue, hot end red
    assert core.colormap(np.array([-5.0, 9.0])).shape == (2, 3)                       # out-of-range values are clipped


def test_overlay_keeps_size_and_leaves_cold_regions_untouched():
    base = Image.new("RGB", (40, 30), (200, 200, 200))
    cam = np.zeros((7, 7), np.float32)
    cam[0, 0] = 1.0                                          # one hot corner, everything else cold
    out = core.overlay(base, cam)
    assert out.size == (40, 30)
    assert out.getpixel((39, 29)) == (200, 200, 200)         # far corner: no overlay drawn
    assert out.getpixel((0, 0)) != (200, 200, 200)           # hot corner: recoloured


def test_upsample_cam_is_clipped_to_unit_range():
    up = core.upsample_cam(np.array([[0.0, 1.0], [1.0, 0.0]], np.float32), (17, 13))
    assert up.shape == (13, 17) and up.min() >= 0.0 and up.max() <= 1.0


def test_fit_max_side_never_upscales():
    big, small = Image.new("RGB", (2000, 1000)), Image.new("RGB", (100, 50))
    assert core.fit_max_side(big, 512).size == (512, 256)
    assert core.fit_max_side(small, 512).size == (100, 50)


def test_data_uri_roundtrip():
    raw = core.encode(Image.new("RGB", (8, 8), (255, 0, 0)))
    uri = core.data_uri(raw)
    assert uri.startswith("data:image/jpeg;base64,")
    import base64
    assert base64.b64decode(uri.split(",", 1)[1]) == raw
