"""Per-frame explanation for Video Model v1 (EfficientNet-B0, 16-frame mean-logit pooling).

Uses the production preprocessing (`model_v1.common.preprocess_frame`) and the cached production model, so
the 16 crops shown here are exactly the crops the verdict was computed from. Adds:
  * each frame's own FAKE logit / probability (the verdict is the sigmoid of their mean),
  * Grad-CAM heatmaps on the frames that pushed the score up the most.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from app.services.explain import core
from app.services.video.model_v1 import common, optimized

CROP_SIZE = 160          # size of the thumbnails sent to the client
TOP_HEATMAP_FRAMES = 3   # Grad-CAM overlays are produced for this many most-suspicious frames


@dataclass
class FrameExplanation:
    order: int                     # 0..15
    frame_index: int               # index in the source video
    timestamp_s: float | None      # None if the container reports no fps
    logit: float
    probability: float             # sigmoid(logit) of this single frame
    crop_jpeg: bytes               # the exact 224x224 model input (un-normalised), resized for display
    heatmap_jpeg: bytes | None     # Grad-CAM overlay, only for the most suspicious frames


@dataclass
class VideoExplanation:
    frames: list[FrameExplanation]
    mean_logit: float
    probability: float             # sigmoid(mean logit) - the quantity the verdict thresholds
    threshold: float
    frames_above_threshold: int
    fps: float | None
    duration_s: float | None


def _crop_from_tensor(chw: np.ndarray) -> Image.Image:
    """Invert the ImageNet normalisation to recover the 224x224 RGB crop the model actually saw."""
    rgb = np.transpose(chw, (1, 2, 0)) * common.IMAGENET_STD + common.IMAGENET_MEAN
    return Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8))


def explain_video(path: str, checkpoint_path: str | None = None) -> VideoExplanation:
    model, device, _ = optimized._get_model_and_device(checkpoint_path)
    cascade = common.get_face_cascade()

    cap = cv2.VideoCapture(path)
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
    finally:
        cap.release()

    frames_bgr = common.decode_selected_frames(path, use_grab_skip=True)   # raises ShortVideoError / ValueError
    positions = common.evenly_spaced_frame_indices(frame_count)
    batch = np.stack([common.preprocess_frame(f, cascade) for f in frames_bgr], axis=0)
    x = torch.from_numpy(batch).contiguous().float().to(device)

    with torch.enable_grad():
        feats = model.features(x)                                   # (16, 1280, 7, 7)
        pooled = torch.flatten(model.avgpool(feats), 1)             # same order as torchvision's forward
        logits = model.classifier(pooled).squeeze(-1)               # (16,)
        cams = core.cam_from_features(feats, logits.sum()).cpu().numpy()   # per-frame maps, independent samples

    frame_logits = logits.detach().cpu().numpy().astype(np.float64)
    mean_logit = float(frame_logits.mean())
    probability = float(1.0 / (1.0 + np.exp(-mean_logit)))

    top = set(np.argsort(-frame_logits)[:TOP_HEATMAP_FRAMES].tolist())
    out: list[FrameExplanation] = []
    for i in range(len(frames_bgr)):
        crop = _crop_from_tensor(batch[i])
        thumb = crop.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS)
        p_i = float(1.0 / (1.0 + np.exp(-frame_logits[i])))
        heat = core.overlay(thumb, cams[i] * core.evidence_strength(p_i)) if i in top else None
        out.append(
            FrameExplanation(
                order=i,
                frame_index=int(positions[i]),
                timestamp_s=(positions[i] / fps) if fps else None,
                logit=float(frame_logits[i]),
                probability=float(1.0 / (1.0 + np.exp(-frame_logits[i]))),
                crop_jpeg=core.encode(thumb),
                heatmap_jpeg=core.encode(heat) if heat is not None else None,
            )
        )

    return VideoExplanation(
        frames=out,
        mean_logit=mean_logit,
        probability=probability,
        threshold=float(common.THRESHOLD),
        frames_above_threshold=int(sum(1 for f in out if f.probability >= common.THRESHOLD)),
        fps=fps,
        duration_s=(frame_count / fps) if fps else None,
    )
