import os
import json
import logging
import urllib.request
import urllib.error

from app.services.models.image_model import score_image_path
from app.services.models.video_model import score_video_path
from app.services.models.audio_model import score_audio_path

logger = logging.getLogger(__name__)
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://models:8001")

def _post_json(url: str, data: dict, timeout: float = 60.0) -> dict | None:
    try:
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Model service call to {url} failed: {e}")
    return None

def run_image_model(path: str) -> dict:
    if MODEL_SERVICE_URL:
        res = _post_json(f"{MODEL_SERVICE_URL}/predict/image", {"file_path": path})
        if res is not None:
            return res
    return score_image_path(path)

def run_video_model(path: str) -> dict:
    if MODEL_SERVICE_URL:
        res = _post_json(f"{MODEL_SERVICE_URL}/predict/video", {"file_path": path}, timeout=120.0)
        if res is not None:
            return res
    return score_video_path(path)

def run_audio_model(path: str) -> dict:
    if MODEL_SERVICE_URL:
        res = _post_json(f"{MODEL_SERVICE_URL}/predict/audio", {"file_path": path})
        if res is not None:
            return res
    return score_audio_path(path)
