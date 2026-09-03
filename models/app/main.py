import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.models.image_model import score_image_path
from app.services.models.video_model import score_video_path
from app.services.models.audio_model import score_audio_path

logger = logging.getLogger(__name__)

app = FastAPI(title="TruthLense Real Model Inference Service", version="1.0.0")

class PredictRequest(BaseModel):
    file_path: str

def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    # Relative path, e.g. "uploads/abc.jpg" -> "/app/uploads/abc.jpg"
    abs_path = os.path.abspath(os.path.join("/app", path))
    if os.path.isfile(abs_path):
        return abs_path
    return path

@app.get("/health")
def health():
    return {"status": "ok", "service": "truthlense-models"}

@app.post("/predict/image")
def predict_image(req: PredictRequest):
    full_path = _resolve_path(req.file_path)
    if not os.path.isfile(full_path):
        logger.error(f"Image not found at {full_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    try:
        return score_image_path(full_path)
    except Exception as e:
        logger.error(f"Error predicting image: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video")
def predict_video(req: PredictRequest):
    full_path = _resolve_path(req.file_path)
    if not os.path.isfile(full_path):
        logger.error(f"Video not found at {full_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    try:
        return score_video_path(full_path)
    except Exception as e:
        logger.error(f"Error predicting video: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/audio")
def predict_audio(req: PredictRequest):
    full_path = _resolve_path(req.file_path)
    if not os.path.isfile(full_path):
        logger.error(f"Audio not found at {full_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")
    try:
        return score_audio_path(full_path)
    except Exception as e:
        logger.error(f"Error predicting audio: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
