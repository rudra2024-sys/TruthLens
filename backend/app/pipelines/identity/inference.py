"""TruthLens identity-match face embedding inference."""

from __future__ import annotations

from PIL import Image
import torch

from .model import load_models


class NoFaceDetectedError(Exception):
    """Raised when MTCNN finds no face in the given image. Deliberately not handled by
    falling back to a center-crop (unlike the video pipeline's face-detection fallback) --
    a non-face crop here would produce a meaningless embedding and a false-confidence match
    result, which matters a lot more for a verification feature than it does for a
    classifier that's at least somewhat tolerant of an imprecise crop."""


def _iou(a, b) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IdentityPipeline:

    def __init__(self):
        self.mtcnn, self.resnet, self.device = load_models()

    def detect_faces(self, image_path: str, prob_thresh: float = 0.95, iou_thresh: float = 0.3) -> dict:
        """How many distinct faces are in the image, plus the 5-point landmarks of the one face
        (only when there's exactly one -- ambiguous which face to report pose/mouth for
        otherwise). Second attempt at the count specifically -- the first (an OpenCV Haar
        cascade, see services/identity/monitoring.py's module docstring) was shipped and then
        found broken by real-webcam testing: 0 faces found on an obvious, well-lit single face.
        This version reuses the same MTCNN instance embed() already uses (no new model) via its
        lower-level .detect(), which returns every raw candidate box -- including many
        overlapping duplicates of the same face (confirmed directly: 7 boxes for 1 real face
        before any filtering). A confidence filter (prob_thresh) plus a greedy IoU merge
        (cluster boxes whose IoU with an existing cluster's first box exceeds iou_thresh)
        collapses those duplicates into a real count.

        Validated directly (2026-10-09, not just assumed) against real images: a composite of
        two different real people side by side -> 2; three separate real single-face webcam
        frames (including a near-black covered-lens shot and a blurry orange covered-lens shot,
        both confirmed genuinely faceless by eye) -> 1/0/0; a solid-color synthetic image -> 0.
        Not run through this project's larger eval harnesses (the ~300-image kind other
        detectors here get) -- a real but smaller-scale validation than that, stated plainly.

        Returns {"count": int, "landmarks": (5,2) array | None, "box": (4,) array | None,
        "image_size": (width, height)}. image_size is always returned (even with 0 faces)
        since callers may want it regardless. "box" (the kept face's [x1, y1, x2, y2], same
        pixel space as image_size) is added alongside landmarks purely for the monitoring
        session's live face-overlay visualization (section 3 item 3 of the FYP plan) -- it was
        already computed by this same .detect() call and simply discarded before, no new
        inference cost.
        """
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)

        if boxes is None:
            return {"count": 0, "landmarks": None, "box": None, "image_size": image_size}

        kept = [i for i, prob in enumerate(probs) if prob is not None and prob >= prob_thresh]

        clusters: list[int] = []
        for i in kept:
            if not any(_iou(boxes[i], boxes[j]) > iou_thresh for j in clusters):
                clusters.append(i)

        count = len(clusters)
        face_landmarks = landmarks[clusters[0]] if count == 1 else None
        face_box = boxes[clusters[0]] if count == 1 else None
        return {"count": count, "landmarks": face_landmarks, "box": face_box, "image_size": image_size}

    def embed(self, image_path: str) -> list[float]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            face_tensor = self.mtcnn(image)

        if face_tensor is None:
            raise NoFaceDetectedError(
                "No face detected in this photo. Please use a clear, "
                "front-facing photo with a single visible face."
            )

        with torch.inference_mode():
            embedding = self.resnet(
                face_tensor.unsqueeze(0).to(self.device)
            )[0]

        # Plain Python floats -- the raw tensor/numpy output isn't JSON-serializable.
        return [float(x) for x in embedding.cpu().tolist()]
