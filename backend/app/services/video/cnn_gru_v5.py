"""Experimental CNN+BiGRU temporal head (Colab, Sep-2026 v5 run) -- opt-in only, NOT the
production default. See CLAUDE.md sections 3/17/18.

Reuses the exact same frozen EfficientNet-B0 feature extractor and frame/crop/resize/
normalize pipeline as Video Model v1 (app.services.video.model_v1.common), with a trained
BiGRU head on top of the per-frame 1280-d features. Architecture matches
eval/video_head_experiments/build_notebook_v5.py's TemporalHead exactly.

Known weakness (measured against backend.py's CnnGruV5Backend docstring and CLAUDE.md):
the checkpoint's self-reported real-world AUC (0.76) was computed using the same 10 local
ground-truth videos its own checkpoint_selection process picked the epoch from -- not an
independent hold-out -- and the underlying per-video probabilities are degenerate (9 of 10
videos score within ~0.06 of 0.0 regardless of true label; the AUC is propped up almost
entirely by one fake video scoring 0.99). Likely cause: none of its three training sources
(pranabkc cropped-faces, WildDeepfake subset, RTFS-10k) resemble real consumer face-swap
app footage, the same gap already documented for the deployed CNN (Akool/Magic Hour blind
spot, CLAUDE.md section 3).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.services.video.model_v1 import common

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "video"
    / "cnn_gru_v5_head.pt"
)


class TemporalHead(nn.Module):
    """BiGRU over per-frame CNN features -> single FAKE logit. Matches the Colab v5
    training notebook's head exactly (bidirectional GRU, concat final fwd/bwd hidden
    states, dropout, linear)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(h_last)).squeeze(-1)


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    return Path(
        checkpoint_path
        or os.getenv("VIDEO_MODEL_CNN_GRU_V5_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    )


_cnn_cache: nn.Module | None = None
_head_cache: tuple[TemporalHead, dict] | None = None


def _get_cnn() -> nn.Module:
    """The same frozen EfficientNet-B0 backbone/checkpoint Video Model v1 uses -- loaded
    and cached separately here since this module reads pre-pool features, not logits."""
    global _cnn_cache
    if _cnn_cache is None:
        model = common.build_model()
        common.load_checkpoint(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        _cnn_cache = model
    return _cnn_cache


def _get_head(checkpoint_path: str | Path | None = None) -> tuple[TemporalHead, dict]:
    global _head_cache
    if _head_cache is None:
        path = resolve_checkpoint_path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"CNN+GRU v5 head checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        head = TemporalHead(input_dim=cfg["feature_dim"], hidden_dim=cfg["gru_hidden_dim"])
        head.load_state_dict(ckpt["model_state_dict"])
        head.eval()
        _head_cache = (head, cfg)
    return _head_cache


class Prediction:
    def __init__(self, probability: float, num_frames: int):
        self.probability = probability
        self.num_frames = num_frames


def predict(video_path: str, checkpoint_path: str | Path | None = None) -> Prediction:
    """Raises ValueError / model_v1.common.ShortVideoError exactly like
    model_v1.optimized.predict() -- same decode_selected_frames call underneath."""
    cnn = _get_cnn()
    head, _cfg = _get_head(checkpoint_path)
    cascade = common.get_face_cascade()

    frames = common.decode_selected_frames(video_path, use_grab_skip=True)
    batch = np.stack([common.preprocess_frame(f, cascade) for f in frames], axis=0)
    tensor = torch.from_numpy(batch).contiguous().float()

    with torch.inference_mode():
        feats = cnn.features(tensor)
        feats = cnn.avgpool(feats)
        feats = torch.flatten(feats, 1)
        logit = head(feats.unsqueeze(0))
        probability = torch.sigmoid(logit).item()

    return Prediction(probability=probability, num_frames=len(frames))
