"""Video forgery scoring: sample frames, score each with the MesoNet ensemble
(same real trained CNN used for images), then aggregate + measure temporal
flicker (frame-to-frame variance), which is itself a genuine forensic cue —
naive face-swaps often show unstable per-frame confidence."""
from __future__ import annotations

import numpy as np

from app.services.models.image_model import score_rgb_frame

MAX_FRAMES = 16


def sample_frames(path: str, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    try:
        if total <= 0:
            # Fall back to sequential reads if the container doesn't report a frame count.
            count = 0
            while count < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(frame_bgr[:, :, ::-1])  # BGR -> RGB
                count += 1
        else:
            n = min(max_frames, total)
            idxs = np.linspace(0, total - 1, n).astype(int)
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = cap.read()
                if not ok:
                    continue
                frames.append(frame_bgr[:, :, ::-1])
    finally:
        cap.release()

    return frames


def score_video_path(path: str) -> dict:
    frames = sample_frames(path)
    if not frames:
        raise ValueError("No decodable frames found in video.")

    per_frame_fake = []
    faces_found = 0
    for frame in frames:
        result = score_rgb_frame(frame)
        per_frame_fake.append(result["fake_probability"])
        if result["face_found"]:
            faces_found += 1

    arr = np.array(per_frame_fake, dtype=np.float32)
    mean_fake = float(arr.mean())
    temporal_std = float(arr.std())  # flicker / inconsistency across frames

    return {
        "mean_fake_probability": mean_fake,
        "temporal_std": temporal_std,
        "frames_analyzed": len(frames),
        "frames_with_face": faces_found,
    }
