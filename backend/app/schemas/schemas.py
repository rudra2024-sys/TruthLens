from pydantic import BaseModel
from typing import Optional, List


class UploadOut(BaseModel):
    upload_id: str
    file_name: str
    media_type: str
    storage_url: str
    file_size_kb: float
    uploaded_at: str
    class Config:
        from_attributes = True


class ImageAnalysisOut(BaseModel):
    efficientnet_score: Optional[float] = None
    fft_score: Optional[float] = None
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None

    class Config:
        from_attributes = True

class VideoAnalysisOut(BaseModel):
    xception_score: float
    face_voice_sync: Optional[float] = None
    frames_analyzed: int
    class Config:
        from_attributes = True

class AudioAnalysisOut(BaseModel):
    wav2vec_score: float
    lcnn_score: float
    class Config:
        from_attributes = True


class DetectionResultOut(BaseModel):
    result_id: str
    upload_id: str
    confidence_score: float
    verdict: str
    model_used: str
    processing_time_ms: float
    detected_at: str
    image_analysis: Optional[ImageAnalysisOut] = None
    video_analysis: Optional[VideoAnalysisOut] = None
    audio_analysis: Optional[AudioAnalysisOut] = None
    class Config:
        from_attributes = True


class HistoryItemOut(BaseModel):
    upload_id: str
    file_name: str
    media_type: str
    uploaded_at: str
    verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_scans: int
    fake_count: int
    real_count: int
    uncertain_count: int
    by_media_type: dict
