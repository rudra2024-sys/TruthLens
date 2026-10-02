from pydantic import BaseModel, Field, computed_field
from typing import Literal, Optional, List


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


class FrameExplanationOut(BaseModel):
    order: int
    frame_index: int
    timestamp_s: Optional[float] = None
    logit: float
    probability: float
    crop: str                          # data: URI (JPEG) - the exact crop the model saw
    heatmap: Optional[str] = None      # data: URI, only for the most suspicious frames


class WindowExplanationOut(BaseModel):
    order: int
    start_s: float
    end_s: float
    probability: float                 # this window's own deepfake probability


class ExplanationOut(BaseModel):
    upload_id: str
    media_type: str
    available: bool
    reason: Optional[str] = None
    method: Optional[str] = None
    note: Optional[str] = None
    # image
    original: Optional[str] = None
    heatmap: Optional[str] = None
    fake_probability: Optional[float] = None            # ConvNeXt-Tiny alone
    evidence_strength: Optional[float] = None
    convnext_fake_probability: Optional[float] = None   # stored scan values
    clip_fake_probability: Optional[float] = None
    verdict_driver: Optional[str] = None
    # video
    frames: Optional[List[FrameExplanationOut]] = None
    mean_probability: Optional[float] = None
    threshold: Optional[float] = None
    frames_above_threshold: Optional[int] = None
    duration_s: Optional[float] = None
    # audio
    windows: Optional[List[WindowExplanationOut]] = None
    windows_above_threshold: Optional[int] = None
    saliency_window_order: Optional[int] = None
    spectrogram: Optional[str] = None   # data: URI (JPEG), the most suspicious window's spectrogram
    saliency: Optional[str] = None      # data: URI (JPEG), saliency overlay on the spectrogram above


class ProvenanceSignalOut(BaseModel):
    kind: str            # ai | capture | edit | integrity | trust | info | absent
    strength: str        # strong | moderate | weak | none
    title: str
    detail: str = ""


class ProvenanceOut(BaseModel):
    upload_id: str
    media_type: str
    available: bool
    reason: Optional[str] = None
    level: Optional[str] = None            # declared_ai | camera_metadata | none
    headline: Optional[str] = None
    conflict_note: Optional[str] = None    # set when provenance and the detection verdict disagree
    caveat: Optional[str] = None
    signals: List[ProvenanceSignalOut] = []
    c2pa: Optional[dict] = None
    metadata: Optional[dict] = None
    container: Optional[dict] = None
    ela: Optional[dict] = None             # includes a data: URI heatmap when applicable
    watermark: Optional[dict] = None       # fixed diffusers SD/SDXL pixel watermark check (images: single check;
                                            # video: sampled-frame check, adds frames_checked/frames_matched/best_bit_match)


class JobOut(BaseModel):
    """A background detection job (see app/services/jobs.py)."""
    job_id: str
    upload_id: str
    media_type: str
    state: str                              # queued | running | done | failed | cancelled
    progress: float                         # 0..1
    stage: str                              # human-readable, e.g. "Reading frames (7/16)"
    result_id: Optional[str] = None         # set when state == "done"; fetch via /detect/{upload_id}/result
    error: Optional[str] = None             # set when state == "failed"
    queue_position: Optional[int] = None    # 1 = next to run, only while queued
    elapsed_s: Optional[float] = None


class FeedbackIn(BaseModel):
    agrees: bool                                             # "was this result correct?"
    true_label: Optional[Literal["real", "ai", "unsure"]] = None   # what the file really is, if the user disagrees
    comment: Optional[str] = Field(default=None, max_length=500)
    allow_reuse: bool = False                                # explicit opt-in: keep the file for model evaluation/improvement


class FeedbackOut(BaseModel):
    upload_id: str
    verdict: str
    agrees: bool
    true_label: Optional[str] = None
    comment: Optional[str] = None
    allow_reuse: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
