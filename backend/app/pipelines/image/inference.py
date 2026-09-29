"""TruthLens ConvNeXt-Tiny image inference."""

from __future__ import annotations

from PIL import Image
import torch

from .model import load_model
from .preprocessing import IMAGE_TRANSFORM


class ImagePipeline:

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ):
        self.model, self.device, self.checkpoint = load_model(
            checkpoint_path
        )

    def predict(self, image_path: str) -> dict:

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = IMAGE_TRANSFORM(image)

        tensor = tensor.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        fake_probability = float(
            probabilities[0].item()
        )

        real_probability = float(
            probabilities[1].item()
        )

        verdict = (
            "FAKE"
            if fake_probability >= real_probability
            else "REAL"
        )

        confidence = max(
            fake_probability,
            real_probability,
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "model_used": "ConvNeXt-Tiny",
            "benchmark": "CIFAKE + AI-vs-Human-Generated + 140k-Real-Fake-Faces",
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "validation_accuracy": self.checkpoint.get(
                "best_val_accuracy"
            ),
        }