"""TruthLens CLIP second-opinion pipeline.

Frozen CLIP ViT-B/16 backbone (never trained on real-vs-fake) + a small
trained MLP head on top. Chosen as a second opinion alongside ConvNeXt-Tiny
because CLIP's features weren't shaped by any particular generator's
artifacts, so it tends to catch generators ConvNeXt (trained end-to-end on
a fixed set of fake sources) has never seen -- see chat/session notes
2026-09-11/12 for the real-world case (a modern AI-generated image) that
motivated this: ConvNeXt alone missed it, CLIP alone caught it.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "image_clip"
    / "clip_head_best.pth"
)


class ClipHead(nn.Module):
    """Matches the architecture trained in the CLIP second-opinion notebook."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def get_device() -> torch.device:
    requested = os.getenv("CLIP_MODEL_DEVICE", "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CLIP_MODEL_DEVICE=cuda was requested, "
                "but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(checkpoint_path: str | Path | None = None):
    """Returns (clip_backbone, preprocess_transform, head, device, checkpoint)."""

    import open_clip

    path = Path(
        checkpoint_path
        or os.getenv(
            "CLIP_MODEL_CHECKPOINT",
            str(DEFAULT_CHECKPOINT),
        )
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"CLIP head checkpoint not found: {path}"
        )

    device = get_device()

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    backbone_name = checkpoint.get("clip_backbone", "ViT-B-16")
    pretrained_tag = checkpoint.get("clip_pretrained", "openai")
    embed_dim = checkpoint["embed_dim"]

    clip_backbone, _, preprocess = open_clip.create_model_and_transforms(
        backbone_name,
        pretrained=pretrained_tag,
    )
    clip_backbone = clip_backbone.to(device).eval()
    for param in clip_backbone.parameters():
        param.requires_grad = False

    head = ClipHead(embed_dim)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.to(device)
    head.eval()

    return clip_backbone, preprocess, head, device, checkpoint
