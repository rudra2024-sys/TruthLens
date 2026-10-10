"""YOLOv8n ONNX checkpoint location + presence check (item 8, added 2026-10-09).

The ONNX file itself is NOT produced by any runtime dependency in requirements.txt -- it's
exported once, locally, via a throwaway `pip install ultralytics` (a much heavier dependency
than belongs in this project's normal runtime stack: pulls in pandas-like tooling, matplotlib,
etc.) and removed again afterward. See this package's README (or CLAUDE.md's identity-match
monitoring section) for the exact one-time export steps actually run:

    pip install onnxslim ultralytics-thop ultralytics-platform polars psutil nvidia-ml-py
    pip install ultralytics --no-deps   # avoid pulling opencv-python; already have
                                         # opencv-python-headless, installing both breaks cv2
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', nms=True)"
    # move the resulting yolov8n.onnx to backend/checkpoints/objects/yolov8n.onnx
    pip uninstall -y ultralytics ultralytics-thop ultralytics-platform onnxslim polars \
        polars-runtime-32 nvidia-ml-py psutil

IMPORTANT, confirmed by direct testing: installing ultralytics-thop silently upgraded this
project's pinned torch==2.8.0+cpu to a generic 2.13.0+cpu and torchvision similarly (pip's
resolver evidently doesn't reliably recognize the "+cpu" local-version-matching torch already
installed when resolving a transitive "torch>=1.8.0" constraint from the default PyPI index).
If redoing this export, immediately re-run
    pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
afterward and re-verify the identity/image pipelines still load before trusting the environment
again -- don't assume the export step left torch alone.

Runtime dependency is only onnxruntime (already in requirements.txt for the AASIST audio model).
First candidate tried (tiny-yolov3, ONNX Model Zoo, Apache-2.0) loaded and ran but its raw
confidence topped out at ~10% even on an easy, clear frontal photo -- too weak to trust.
YOLOv8n, tested the same way, gave 72-94% confidence on the same kind of photo.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "checkpoints"
    / "objects"
    / "yolov8n.onnx"
)


def checkpoint_path() -> Path:
    return Path(os.getenv("OBJECT_MODEL_CHECKPOINT", str(DEFAULT_CHECKPOINT)))


def weights_present() -> bool:
    return checkpoint_path().is_file()
