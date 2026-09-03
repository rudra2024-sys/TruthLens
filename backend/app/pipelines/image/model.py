"""TruthLens Image Pipeline - ConvNeXt-Tiny."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "image"
    / "convnext_tiny_best.pth"
)


def get_device() -> torch.device:
    requested = os.getenv("IMAGE_MODEL_DEVICE", "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "IMAGE_MODEL_DEVICE=cuda was requested, "
                "but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def build_model() -> nn.Module:
    model = convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        2,
    )

    return model


def load_model(
    checkpoint_path: str | Path | None = None,
) -> tuple[nn.Module, torch.device, dict]:

    path = Path(
        checkpoint_path
        or os.getenv(
            "IMAGE_MODEL_CHECKPOINT",
            str(DEFAULT_CHECKPOINT),
        )
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"ConvNeXt checkpoint not found: {path}"
        )

    device = get_device()

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model, device, checkpoint