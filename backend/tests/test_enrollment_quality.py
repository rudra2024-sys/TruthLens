"""Unit tests for the enrollment quality gate's pure logic (added 2026-10-11) -- no pipeline,
no network, just real small images + crafted detection dicts (the same shape
IdentityPipeline.detect_faces() returns) run through the real blur/size/edge checks.

Route-level "does a rejection actually reach the caller as a 422" coverage lives in
test_identity.py::test_enroll_rejects_a_low_quality_photo.
"""

import numpy as np
from PIL import Image

from app.services.identity.enrollment_quality import assess_enrollment_quality


def _sharp_image(path, size=(640, 640)):
    """High-frequency noise -- large Laplacian variance, i.e. "sharp" by this heuristic."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr, "RGB").save(path)


def _flat_image(path, size=(640, 640), color=(120, 120, 120)):
    """Solid color -- zero Laplacian variance, i.e. "blurry" by this heuristic."""
    Image.new("RGB", size, color).save(path)


def test_well_framed_sharp_photo_passes(tmp_path):
    path = str(tmp_path / "good.png")
    _sharp_image(path)
    detection = {"count": 1, "box": [200.0, 200.0, 440.0, 440.0], "image_size": (640, 640)}
    assert assess_enrollment_quality(path, detection) == []


def test_multiple_faces_rejected_without_other_checks(tmp_path):
    path = str(tmp_path / "two_faces.png")
    _flat_image(path)  # would also fail blur, but multi-face should short-circuit to one issue
    detection = {"count": 2, "box": None, "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert len(issues) == 1
    assert "only you" in issues[0]


def test_tiny_face_rejected_for_size(tmp_path):
    path = str(tmp_path / "tiny_face.png")
    _sharp_image(path)
    # 10x10 box in a 640x640 image -> well under the 8% area-fraction threshold.
    detection = {"count": 1, "box": [300.0, 300.0, 310.0, 310.0], "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert any("too small" in issue for issue in issues)
    assert not any("cut off" in issue for issue in issues)
    assert not any("blurry" in issue for issue in issues)


def test_face_touching_edge_rejected_for_framing(tmp_path):
    path = str(tmp_path / "edge_face.png")
    _sharp_image(path)
    # Box starts at x=0 -> within the edge margin; still a reasonably large, sharp crop.
    detection = {"count": 1, "box": [0.0, 200.0, 240.0, 440.0], "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert any("cut off" in issue for issue in issues)
    assert not any("too small" in issue for issue in issues)


def test_blurry_photo_rejected(tmp_path):
    path = str(tmp_path / "blurry.png")
    _flat_image(path)
    detection = {"count": 1, "box": [200.0, 200.0, 440.0, 440.0], "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert any("blurry" in issue for issue in issues)
    assert not any("too small" in issue for issue in issues)
    assert not any("cut off" in issue for issue in issues)


def test_blurry_photo_rejected_even_without_a_confident_box(tmp_path):
    """Regression test for a real bug found via live browser testing: detect_faces()'s stricter
    0.95 confidence filter can fail on a blurry face (count=0, box=None) while the caller's
    separate, more lenient _embed() call still succeeds -- the blur check must not be skipped
    just because no box was found here."""
    path = str(tmp_path / "blurry_no_box.png")
    _flat_image(path)
    detection = {"count": 0, "box": None, "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert any("blurry" in issue for issue in issues)


def test_multiple_issues_all_reported_together(tmp_path):
    path = str(tmp_path / "bad.png")
    _flat_image(path)  # blurry
    # Also tiny and touching the edge.
    detection = {"count": 1, "box": [0.0, 0.0, 10.0, 10.0], "image_size": (640, 640)}
    issues = assess_enrollment_quality(path, detection)
    assert any("too small" in issue for issue in issues)
    assert any("cut off" in issue for issue in issues)
    assert any("blurry" in issue for issue in issues)
    assert len(issues) == 3
