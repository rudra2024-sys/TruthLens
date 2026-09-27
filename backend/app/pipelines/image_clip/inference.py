"""TruthLens CLIP second-opinion inference."""

from __future__ import annotations

from PIL import Image
import torch

from .model import load_model


class ClipPipeline:

    def __init__(
        self,
        checkpoint_path: str | None = None,
    ):
        (
            self.clip_backbone,
            self.preprocess,
            self.head,
            self.device,
            self.checkpoint,
        ) = load_model(checkpoint_path)

    def predict(self, image_path: str) -> dict:

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)

        tensor = tensor.unsqueeze(0).to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            embed = self.clip_backbone.encode_image(tensor)
            embed = embed / embed.norm(dim=-1, keepdim=True)

            logits = self.head(embed)
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
            "model_used": "CLIP ViT-B/16 (frozen) + MLP head",
            "checkpoint_val_accuracy": self.checkpoint.get(
                "best_val_accuracy"
            ),
        }
