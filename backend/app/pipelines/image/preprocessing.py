"""Preprocessing for the TruthLens ConvNeXt-Tiny image model.

CIFAKE_NATIVE_SIZE / the two-step resize below: the checkpoint was trained
on the CIFAKE benchmark, whose images are natively 32x32 (upscaled to the
model's 224x224 input during training, not shot at that resolution). Fed a
real-world photo resized directly to 224x224, the model sees real
high-frequency camera detail it never encountered for the REAL class during
training and confidently mislabels it FAKE (verified 2026-09-10 through this
exact module: a real user photo scored 96% FAKE with a direct 224x224
resize; the same photo through this 32x32-then-224x224 path scored 58%
FAKE, correctly banded UNCERTAIN instead of a forced wrong FAKE, while a
genuine AI-generated image still correctly scored FAKE at 94% through the
same path -- see git history / session notes for the full before/after
comparison). This is a preprocessing correction to match the checkpoint's
actual training-time input distribution, not a change to weights,
architecture, or the decision threshold.
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

CIFAKE_NATIVE_SIZE = 32
MODEL_INPUT_SIZE = 224

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (CIFAKE_NATIVE_SIZE, CIFAKE_NATIVE_SIZE),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
