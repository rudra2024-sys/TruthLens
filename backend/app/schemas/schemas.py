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
    convnext_fake_probability: Optional[float] = None
    clip_fake_probability: Optional[float] = None

    class Config:
        from_attributes = True

class VideoAnalysisOut(BaseModel):
    xception_score: float
    face_voice_sync: Optional[float] = None
    frames_analyzed: int
    class Config:
        from_attributes = True

class AudioAnalysisOut(BaseModel):
    # DEPRECATED field names: kept only for backward compatibility with
    # existing API consumers and the frontend. The active audio detector is
    # AASIST, not Wav2Vec2 or LCNN -- neither of those models is run anywhere
    # in this pipeline. See aasist_spoof_probability / aasist_spoof_probability_std
    # below for the same values under honest names. Do not read these two as
    # evidence that Wav2Vec2/LCNN were used.
    wav2vec_score: float = Field(
        description="DEPRECATED name. Actually holds AASIST's mean spoof probability. "
                     "See aasist_spoof_probability."
    )
    lcnn_score: float = Field(
        description="DEPRECATED name. Actually holds AASIST's cross-window spoof-probability "
                     "std deviation. See aasist_spoof_probability_std."
    )

    class Config:
        from_attributes = True

    @computed_field(description="AASIST mean spoof probability across analyzed windows (0-1). Same value as wav2vec_score, under its real name.")
    @property
    def aasist_spoof_probability(self) -> float:
        return self.wav2vec_score

    @computed_field(description="AASIST spoof-probability standard deviation across analyzed windows (0-1). Same value as lcnn_score, under its real name.")
    @property
    def aasist_spoof_probability_std(self) -> float:
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
