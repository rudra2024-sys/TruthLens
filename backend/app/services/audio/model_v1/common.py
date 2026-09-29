"""Frozen, shared semantics for TruthLens Audio Model v1 (wav2vec2-base,
fine-tuned genuine/deepfake voice-cloning classifier).

Preprocessing here MUST match training exactly: RMS-normalize the whole
decoded clip once (not per-window -- training normalized the full clip
before windowing), 16kHz mono, 4.0s (64000-sample) windows. This is a
distinct window length from AASIST's 64600 (~4.04s) in
app/services/models/audio_model.py -- do not reuse that constant, it's
AASIST's own trained input length, not this model's.

Training/eval history (v2 through v9, full detail in the project's audio
training notebooks): fine-tuned from a wav2vec2-base checkpoint on ASVspoof
voice-conversion attacks, DEEP-VOICE, In-The-Wild (speaker-split),
LibriSpeech (genuine-only, fixes false positives on unfamiliar recording
conditions), and a system-diverse VCC2020 voice-conversion source
(system-split, fixes false negatives on unseen voice-cloning systems).
Held-out results: 5/6 independent test sets pass, including a 59-system
unseen-voice-conversion stress test (98.3%) and fresh unseen-speaker
genuine audio (10/10). See CLAUDE.md for the full breakdown.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

TARGET_SR = 16000
WINDOW_SAMPLES = TARGET_SR * 4  # 64000 -- this model's own trained window length (4.0s)
MAX_WINDOWS = 6  # lower than AASIST's 12: a wav2vec2-base forward pass is materially
                 # heavier per window on CPU-only inference, so this bounds worst-case
                 # request latency while still covering long clips.

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "checkpoints"
    / "audio"
    / "wav2vec2_v9.pt"
)


def get_device() -> torch.device:
    requested = os.getenv("AUDIO_MODEL_V1_DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "AUDIO_MODEL_V1_DEVICE=cuda was requested, but CUDA is unavailable."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioClassifier(nn.Module):
    """Binary genuine/deepfake classifier: wav2vec2-base backbone (frozen feature
    extractor), mean-pooled hidden states, small linear head. Architecture and label
    order (['genuine', 'deepfake']) must match training exactly -- this is not a
    guess, it's the same class definition used throughout the training notebooks and
    every standalone validation run."""

    def __init__(self, num_classes: int = 2, freeze_feature_extractor: bool = True):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self.backbone(waveform).last_hidden_state
        return self.head(out.mean(dim=1))


def build_model() -> nn.Module:
    return AudioClassifier(num_classes=2)


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    return Path(
        checkpoint_path
        or os.getenv("AUDIO_MODEL_V1_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path | None = None) -> None:
    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio model v1 checkpoint not found: {path}")
    # Exported as a raw state_dict (torch.save(model.state_dict(), ...)), not wrapped
    # in a {"model_state_dict": ...} dict like the video checkpoint -- load directly.
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)


def rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9
    return (audio * (10 ** (target_dbfs / 20) / rms)).astype(np.float32)
