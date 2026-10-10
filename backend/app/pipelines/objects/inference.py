"""YOLOv8n object detection, scoped to phone/notes only (item 8, added 2026-10-09)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image

from .model import checkpoint_path, weights_present

# Only the two COCO classes actually asked for ("phone/notes"), not a general-purpose object
# detector -- deliberately scoped, same reasoning as identity-match's "cell phone + book only"
# framing in the design conversation.
_RELEVANT_CLASSES = {67: "cell phone", 73: "book"}

_INPUT_SIZE = 640


class ObjectPipeline:
    def __init__(self):
        if not weights_present():
            raise FileNotFoundError(
                f"YOLOv8n ONNX checkpoint not found at {checkpoint_path()}. "
                "See pipelines/objects/model.py's module docstring for the one-time export steps."
            )
        self.session = ort.InferenceSession(str(checkpoint_path()))

    def detect(self, image_path: str, conf_thresh: float = 0.4) -> list[str]:
        """Returns the deduplicated list of relevant class names found (e.g. ["cell phone"]),
        empty if none above conf_thresh."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            iw, ih = image.size
            scale = min(_INPUT_SIZE / iw, _INPUT_SIZE / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            resized = image.resize((nw, nh), Image.BILINEAR)
            canvas = Image.new("RGB", (_INPUT_SIZE, _INPUT_SIZE), (114, 114, 114))
            canvas.paste(resized, (0, 0))

        x = np.array(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x = np.expand_dims(x, 0)

        output = self.session.run(None, {"images": x})[0][0]  # (300, 6): x1,y1,x2,y2,conf,cls

        found: set[str] = set()
        for row in output:
            conf, cls = float(row[4]), int(row[5])
            if conf >= conf_thresh and cls in _RELEVANT_CLASSES:
                found.add(_RELEVANT_CLASSES[cls])

        return sorted(found)
