from pydantic import BaseModel, Field, computed_field
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
    # DEPRECATED field names: kept only for backward compatibility with existing
    # API consumers and the frontend. These columns are model-agnostic in the DB
    # (mean/std spoof-probability across analyzed windows) but the field names
    # ("wav2vec"/"lcnn") don't describe whichever backend actually produced them
    # -- see AUDIO_MODEL_BACKEND in services/audio/backend.py for which one is
    # active. See spoof_probability / spoof_probability_std below for the same
    # values under honest, model-agnostic names.
    wav2vec_score: float = Field(
        description="DEPRECATED name. Mean spoof/deepfake probability across analyzed "
                     "windows, from whichever audio backend is active. See spoof_probability."
    )
    lcnn_score: float = Field(
        description="DEPRECATED name. Spoof/deepfake-probability standard deviation across "
                     "analyzed windows, from whichever audio backend is active. "
                     "See spoof_probability_std."
    )
    duration_s: Optional[float] = Field(
        default=None, description="Duration of the analyzed audio, in seconds."
    )
    windows_analyzed: Optional[int] = Field(
        default=None, description="Number of fixed-length windows the audio was split into for analysis."
    )

    class Config:
        from_attributes = True

    @computed_field(description="Mean spoof/deepfake probability across analyzed windows (0-1), from whichever audio backend is active. Same value as wav2vec_score, under a model-agnostic name.")
    @property
    def spoof_probability(self) -> float:
        return self.wav2vec_score

    @computed_field(description="Spoof/deepfake-probability standard deviation across analyzed windows (0-1), from whichever audio backend is active. Same value as lcnn_score, under a model-agnostic name.")
    @property
    def spoof_probability_std(self) -> float:
        return self.lcnn_score


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
