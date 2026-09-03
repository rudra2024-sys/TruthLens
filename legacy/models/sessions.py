"""
Shared, lazily-initialised ML inference sessions.

Models bundled here are real, published, pretrained networks (weights included
in this repo under ./weights, converted to ONNX for a lightweight runtime
dependency — only `onnxruntime` is needed, not the original training
frameworks):

- MesoNet (Meso4 + MesoInception4), Afchar et al., WIFS 2018
  "MesoNet: a Compact Facial Video Forgery Detection Network"
  https://arxiv.org/abs/1809.00888  (weights: DariusAf/MesoNet, Apache-2.0)
  Used for IMAGE and VIDEO (per-frame) face-forgery detection.

- AASIST, Jung et al., ICASSP 2022
  "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph
  Attention Networks", pretrained on ASVspoof2019-LA.
  https://arxiv.org/abs/2110.01200  (weights: clovaai/aasist, MIT)
  Used for AUDIO spoof / synthetic-speech detection.

Both models predict "bonafide vs manipulated" for the *specific* manipulation
families they were trained on (face-swap/reenactment for MesoNet, TTS/voice
conversion spoofing for AASIST). They are real trained DNNs, not heuristics —
but like any classifier they generalize imperfectly to forgery methods far
outside their training distribution. Scores should be read as evidence, not
absolute ground truth.
"""
from __future__ import annotations

import os
import threading
import numpy as np

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

_lock = threading.Lock()
_sessions: dict[str, "object"] = {}
_face_cascade = None


def _make_session(onnx_filename: str):
    import onnxruntime as ort

    path = os.path.join(_WEIGHTS_DIR, onnx_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model weight file missing: {path}. "
            "Expected it to ship inside app/services/models/weights/."
        )
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def get_session(name: str):
    """name in {'meso4', 'meso_inception', 'aasist'}"""
    if name in _sessions:
        return _sessions[name]
    with _lock:
        if name not in _sessions:
            filename = {
                "meso4": "meso4_df.onnx",
                "meso_inception": "meso_inception_df.onnx",
                "aasist": "aasist.onnx",
            }[name]
            _sessions[name] = _make_session(filename)
    return _sessions[name]


def get_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    with _lock:
        if _face_cascade is None:
            import cv2

            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise RuntimeError("Failed to load OpenCV face cascade.")
            _face_cascade = cascade
    return _face_cascade


def sigmoid_meso_predict(name: str, batch_nhwc: np.ndarray) -> np.ndarray:
    """batch_nhwc: float32 array (N, 256, 256, 3) in [0, 1]. Returns (N,) 'real' scores."""
    sess = get_session(name)
    out = sess.run(None, {"input": batch_nhwc.astype(np.float32)})[0]
    return out.reshape(-1)


def aasist_predict_logits(waveform_batch: np.ndarray) -> np.ndarray:
    """waveform_batch: float32 array (N, 64600). Returns (N, 2) logits [spoof, bonafide]."""
    sess = get_session("aasist")
    _, logits = sess.run(None, {"waveform": waveform_batch.astype(np.float32)})
    return logits
