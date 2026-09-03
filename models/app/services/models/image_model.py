"""Real MesoNet-based face-forgery scoring for a single image (numpy array, RGB)."""
from __future__ import annotations

import numpy as np

from app.services.models.sessions import get_face_cascade, sigmoid_meso_predict

MESO_INPUT_SIZE = 256


def detect_largest_face(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest detected face, or None."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    # pick largest by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def crop_face_or_center(rgb: np.ndarray, margin: float = 0.35) -> tuple[np.ndarray, bool]:
    """Crop around the largest detected face with margin; fall back to full frame.
    Returns (crop_rgb, face_found)."""
    import cv2

    h_img, w_img = rgb.shape[:2]
    face = detect_largest_face(rgb)
    if face is None:
        return rgb, False

    x, y, w, h = face
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w_img, x + w + mx), min(h_img, y + h + my)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return rgb, False
    return crop, True


def preprocess_for_meso(rgb: np.ndarray) -> np.ndarray:
    import cv2

    resized = cv2.resize(rgb, (MESO_INPUT_SIZE, MESO_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def score_rgb_frame(rgb: np.ndarray) -> dict:
    """Run the MesoNet ensemble (Meso4 + MesoInception4, DF-trained) on one RGB frame.

    Returns dict with fake_probability (0..1, higher = more likely manipulated),
    face_found (bool), and the individual sub-model scores.
    """
    crop, face_found = crop_face_or_center(rgb)
    x = preprocess_for_meso(crop)[None, ...]  # (1, 256, 256, 3)

    real_meso4 = float(sigmoid_meso_predict("meso4", x)[0])
    real_meso_inception = float(sigmoid_meso_predict("meso_inception", x)[0])

    # Both models output P(real). Ensemble = simple average.
    real_prob = 0.5 * real_meso4 + 0.5 * real_meso_inception
    fake_prob = float(np.clip(1.0 - real_prob, 0.0, 1.0))

    return {
        "fake_probability": fake_prob,
        "face_found": face_found,
        "meso4_real": real_meso4,
        "meso_inception_real": real_meso_inception,
    }


def score_image_path(path: str) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
    return score_rgb_frame(rgb)
