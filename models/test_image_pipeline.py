from pathlib import Path
import torch

from app.pipelines.image.inference import ImagePipeline


ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = (
    ROOT
    / "models"
    / "checkpoints"
    / "image"
    / "convnext_tiny_best.pth"
)

TEST_IMAGE = ROOT / "test_sample.jpg"


print("=" * 60)
print("TRUTHLENS CONVNEXT-TINY IMAGE TEST")
print("=" * 60)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Checkpoint:", CHECKPOINT)
print("Checkpoint exists:", CHECKPOINT.exists())

print("Test image:", TEST_IMAGE)
print("Test image exists:", TEST_IMAGE.exists())

if not TEST_IMAGE.exists():
    raise FileNotFoundError(
        f"Test image not found: {TEST_IMAGE}"
    )

print()
print("Loading model...")

pipeline = ImagePipeline(
    str(CHECKPOINT)
)

print("Model loaded successfully.")
print("Device:", pipeline.device)
print("Checkpoint epoch:", pipeline.checkpoint["epoch"])
print(
    "Validation accuracy:",
    pipeline.checkpoint["best_val_accuracy"],
)

print()
print("Running inference...")

result = pipeline.predict(
    str(TEST_IMAGE)
)

print()
print("=" * 60)
print("INFERENCE RESULT")
print("=" * 60)

print("Verdict:", result["verdict"])
print(
    "FAKE probability:",
    f'{result["fake_probability"] * 100:.2f}%'
)
print(
    "REAL probability:",
    f'{result["real_probability"] * 100:.2f}%'
)
print(
    "Confidence:",
    f'{result["confidence"] * 100:.2f}%'
)
print("Model:", result["model_used"])
print("Benchmark:", result["benchmark"])
print("Checkpoint epoch:", result["checkpoint_epoch"])
print(
    "Validation accuracy:",
    result["validation_accuracy"]
)

print("=" * 60)