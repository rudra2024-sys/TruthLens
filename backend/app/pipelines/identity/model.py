"""TruthLens Identity Pipeline - MTCNN (face detect+align) + InceptionResnetV1 (embedding).

Pretrained only (not trained by this team) -- same category as AASIST/MesoNet elsewhere in
this codebase. MTCNN's own detector weights ship bundled inside the facenet-pytorch package
(no network needed); InceptionResnetV1(pretrained='vggface2') downloads its weights on first
construction and caches them under TORCH_HOME/checkpoints/ -- see weights_present() below,
which exists so this is never constructed without that cache already warm (e.g. in CI).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


DEFAULT_TORCH_HOME = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "identity"
    / "torch_cache"
)


def get_device() -> torch.device:
    requested = os.getenv("IDENTITY_MODEL_DEVICE", "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "IDENTITY_MODEL_DEVICE=cuda was requested, "
                "but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def _torch_home() -> Path:
    return Path(os.getenv("TORCH_HOME", str(DEFAULT_TORCH_HOME)))


def weights_present() -> bool:
    """Whether InceptionResnetV1's pretrained weights are already cached locally.

    Checked before ever constructing the model, the same way the audio explainer checks
    its own checkpoint file before building wav2vec2 (CLAUDE.md sec 18/25) -- otherwise a
    network-less environment (CI, most importantly) would hang or fail trying to download
    them on first use instead of failing clearly and fast.
    """
    checkpoints_dir = _torch_home() / "checkpoints"
    return checkpoints_dir.is_dir() and any(checkpoints_dir.glob("*.pt"))


def load_models(
    device: torch.device | None = None,
) -> tuple[MTCNN, InceptionResnetV1, torch.device]:
    if not weights_present():
        raise FileNotFoundError(
            "InceptionResnetV1 pretrained weights not found under "
            f"{_torch_home() / 'checkpoints'}. Run the one-time setup step to "
            "download them (see backend/app/pipelines/identity/README or the "
            "project's CLAUDE.md) before using the identity pipeline."
        )

    # Vendor the download under this repo's own checkpoints tree instead of the
    # user's home directory, same convention as every other model's checkpoint.
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))

    device = device or get_device()

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        post_process=True,
        device=device,
    )

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    return mtcnn, resnet, device
