"""Preprocessing for the TruthLens ConvNeXt-Tiny image model.

The 32x32-then-224x224 double resize was originally added because the old
CIFAKE-only checkpoint mislabelled real high-resolution photos as FAKE. When
the `convnext_tiny_diversified.pth` checkpoint was introduced (fine-tuned on
CIFAKE + `alessandrasala79/ai-vs-human-generated-dataset` +
`xhlulu/140k-real-and-fake-faces`, each at native resolution), this module
was briefly changed to a direct 224x224 resize, then reverted back to the
crush step based on a same-day 2-image spot check that (misleadingly)
favored keeping it.

That reversion was wrong and has been undone. A proper per-source held-out
evaluation (3000 images x 3 sources, 2026-09-11) showed the crush step is
actively harmful at scale with this checkpoint: `ai_vs_human` accuracy drops
from 99.67% (direct resize) to 71.27% (crushed), and `faces` drops from
99.80% to 57.17% -- barely better than chance. Only CIFAKE is unaffected,
because CIFAKE images are natively 32x32 already, so crushing them is a
no-op. For `ai_vs_human` and `faces`, crushing real high-resolution images
down to 32x32 destroys almost all the detail the model actually relies on.
Direct resize is correct for this checkpoint; do not reintroduce the crush
step without a full per-source evaluation (a 1-2 image spot check is not
sufficient evidence either way).
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

MODEL_INPUT_SIZE = 224

IMAGE_TRANSFORM = transforms.Compose([
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
