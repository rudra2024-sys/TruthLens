"""
Standalone comparison: deployed (round-2) vs round-3 CLIP head, plus deployed
ConvNeXt-Tiny v2, on two blind-spot genres:

- portrait_app: sample of the curated DiffusionDB vintage/costume-portrait set
  (round 3 was fine-tuned on this genre -- expect improvement)
- chatgpt_everyday: 8 real photos ChatGPT was asked to recreate as photorealistic
  "deepfakes" -- ordinary/mundane context, no stylization. NOT part of round 3's
  training data -- expect this to still fail regardless of CLIP head used.

Not wired into the backend or CI -- run manually:
    ../.venv/Scripts/python.exe test_blind_spot_checkpoints.py
"""

import glob
import os

from app.pipelines.image.inference import ImagePipeline
from app.pipelines.image_clip.inference import ClipPipeline

TEST_ROOT = os.path.join(os.path.dirname(__file__), "test_data", "blind_spot_images")

CONVNEXT_CHECKPOINT = r"C:\TrueLense\tl\models\checkpoints\image\convnext_tiny_diversified_v2.pth"
CLIP_ROUND2_CHECKPOINT = r"C:\TrueLense\tl\models\checkpoints\image_clip\clip_head_best.pth"
CLIP_ROUND3_CHECKPOINT = r"C:\Users\admin\Downloads\clip_head_best (2).pth"

os.environ.setdefault("IMAGE_MODEL_DEVICE", "cpu")
os.environ.setdefault("CLIP_MODEL_DEVICE", "cpu")


def load_images(category):
    paths = sorted(glob.glob(os.path.join(TEST_ROOT, category, "*.png")))
    return paths


def evaluate(name, predict_fn, categories):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    for category, paths in categories.items():
        fake_count = 0
        for path in paths:
            fake_probability = predict_fn(path)
            verdict = "FAKE" if fake_probability >= 0.5 else "REAL"
            if verdict == "FAKE":
                fake_count += 1
            print(f"  [{category}] {os.path.basename(path):40} "
                  f"{verdict:5} (fake_prob={fake_probability:.4f})")
        total = len(paths)
        pct = (fake_count / total * 100) if total else 0.0
        print(f"  -- {category}: {fake_count}/{total} correctly flagged FAKE ({pct:.1f}%)")


def main():
    categories = {
        "portrait_app": load_images("portrait_app"),
        "chatgpt_everyday": load_images("chatgpt_everyday"),
    }
    for name, paths in categories.items():
        print(f"{name}: {len(paths)} test images")

    convnext = ImagePipeline(checkpoint_path=CONVNEXT_CHECKPOINT)
    clip_round2 = ClipPipeline(checkpoint_path=CLIP_ROUND2_CHECKPOINT)
    clip_round3 = ClipPipeline(checkpoint_path=CLIP_ROUND3_CHECKPOINT)

    evaluate(
        "ConvNeXt-Tiny v2 (deployed, unchanged)",
        lambda p: convnext.predict(p)["fake_probability"],
        categories,
    )
    evaluate(
        "CLIP head -- ROUND 2 (currently deployed)",
        lambda p: clip_round2.predict(p)["fake_probability"],
        categories,
    )
    evaluate(
        "CLIP head -- ROUND 3 (fine-tuned tonight w/ portrait_app source)",
        lambda p: clip_round3.predict(p)["fake_probability"],
        categories,
    )
    evaluate(
        "Ensemble average -- ConvNeXt + CLIP ROUND 3",
        lambda p: (convnext.predict(p)["fake_probability"] + clip_round3.predict(p)["fake_probability"]) / 2,
        categories,
    )


if __name__ == "__main__":
    main()
