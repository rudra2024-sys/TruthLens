from pydantic import BaseModel, Field, computed_field
from typing import Literal, Optional, List


class UploadOut(BaseModel):
    upload_id: str
    file_name: str
    media_type: str
    storage_url: str
    file_size_kb: float
    uploaded_at: str
    width: Optional[int] = None      # pixels, images only (read from the file header; None if unreadable)
    height: Optional[int] = None
    class Config:
        from_attributes = True


class ScannedMatchOut(BaseModel):
    """An earlier scan of identical file bytes by the same user (batch scans reuse it instead of rescanning)."""
    upload_id: str
    verdict: str
    confidence_score: float
    class Config:
        from_attributes = True


class ImageAnalysisOut(BaseModel):
    efficientnet_score: Optional[float] = None
    fft_score: Optional[float] = None
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    convnext_fake_probability: Optional[float] = None
    clip_fake_probability: Optional[float] = None
    face_swap_fake_probability: Optional[float] = None
    face_swap_face_px: Optional[int] = None

    class Config:
        from_attributes = True

class VideoAnalysisOut(BaseModel):
    xception_score: float
    face_voice_sync: Optional[float] = None
    frames_analyzed: int
    frames_with_face: Optional[int] = None  # of frames_analyzed, how many had an actual face detection
                                             # (not the center-square fallback) -- see CLAUDE.md section 21
    calibrated_fake_probability: Optional[float] = None  # P(FAKE) on the scale the verdict uses (CLAUDE.md s. 26)
    short_side_px: Optional[int] = None
    low_resolution: Optional[bool] = None    # True -> verdict withheld as UNCERTAIN: video too small to judge
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
    calibrated_confidence: Optional[float] = None  # post-hoc temperature-scaled P(FAKE)-derived confidence
                                                     # (image/video only, see app/services/calibration.py and
                                                     # CLAUDE.md section 25) -- supplementary, never replaces
                                                     # confidence_score above
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


class UploadIdIn(BaseModel):
    upload_id: str


class EnrollReferenceIn(BaseModel):
    upload_id: str
    # Required, explicit opt-in (added 2026-10-11) -- mirrors the per-item consent discipline
    # the feedback-export feature already uses (see CLAUDE.md section 15). The route rejects a
    # request where this isn't exactly true; there's no way to enroll without it.
    consent: bool


class EnrollOrganizationIn(BaseModel):
    join_code: str


class OrganizationOut(BaseModel):
    organization_id: str
    organization_name: str


class JoinCodeOut(BaseModel):
    join_code: str


class RosterStudentOut(BaseModel):
    user_id: str
    name: str
    email: str


class MySessionListItemOut(BaseModel):
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    total_checks: int
    flagged_checks: int


class IdentityReferenceOut(BaseModel):
    reference_id: str
    upload_id: str
    enrolled_at: str
    consent_given_at: Optional[str] = None
    government_id_verified: bool = False
    # Deliberately no embedding_json field here -- never serialize the raw embedding
    # back out over the API.

    class Config:
        from_attributes = True


class IdentityMatchOut(BaseModel):
    match_id: str
    reference_id: str
    upload_id: str
    similarity_score: float
    verdict: str
    checked_at: str

    class Config:
        from_attributes = True


class MonitoringCheckOut(BaseModel):
    check_id: str
    session_id: str
    upload_id: str
    face_count: int
    similarity_score: Optional[float] = None
    identity_verdict: Optional[str] = None
    fake_probability: float
    deepfake_verdict: str
    flagged: bool
    flag_reasons: List[str]
    checked_at: str
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    mouth_width_px: Optional[float] = None
    speech_ratio: Optional[float] = None
    object_detections: List[str] = []
    face_box: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    class Config:
        from_attributes = True


class CheckJobOut(BaseModel):
    """A background monitoring-check job (added 2026-10-11, item 6; see app/services/
    monitoring_jobs.py). The finished check is embedded directly in `result` once state == "done"
    -- small enough to hold inline, unlike a video job's heavier DetectionResult, so there's no
    separate fetch endpoint."""
    job_id: str
    session_id: str
    state: str                        # queued | running | done | failed
    result: Optional[MonitoringCheckOut] = None
    error: Optional[str] = None


class MonitoringCheckIn(BaseModel):
    upload_id: str
    # Optional ~3s audio clip, already uploaded via the generic /upload endpoint, for the
    # speech/talking-detection signal (item 7). Omitted entirely on browsers/sessions where
    # microphone access wasn't granted -- the check still runs, just without that signal.
    audio_upload_id: Optional[str] = None


class MonitoringEventIn(BaseModel):
    event_type: Literal[
        "tab_hidden", "window_blurred", "clipboard_paste", "devtools_suspected", "camera_interrupted",
        # Real-time, client-side signals (added 2026-10-12) -- fired by LiveFaceMesh.jsx the moment
        # a condition (no face / looking away / a second face) has held continuously for ~2s,
        # instead of waiting on the backend's own 5-20s periodic check cadence. Always paired with
        # a forced immediate real check (see MonitoringSession.jsx), so a "live_*" event is never
        # the only evidence on its own -- a real captured frame backs it up.
        "live_no_face", "live_looking_away", "live_multiple_faces",
    ]
    detail: Optional[str] = Field(default=None, max_length=200)


class MonitoringEventOut(BaseModel):
    event_id: str
    session_id: str
    event_type: str
    detail: Optional[str] = None
    occurred_at: str

    class Config:
        from_attributes = True


class MonitoringSessionOut(BaseModel):
    session_id: str
    reference_id: str
    started_at: str
    ended_at: Optional[str] = None

    class Config:
        from_attributes = True


class MonitoringSessionDetailOut(MonitoringSessionOut):
    checks: List[MonitoringCheckOut] = []
    events: List[MonitoringEventOut] = []

    class Config:
        from_attributes = True


class AdminSessionListItemOut(BaseModel):
    """One row of GET /admin/sessions (added 2026-10-11) -- a session plus just enough about its
    owner to identify them in a list, not the full MonitoringSessionDetailOut (too heavy for a
    list of many users' sessions)."""
    session_id: str
    user_id: str
    user_name: str
    user_email: str
    started_at: str
    ended_at: Optional[str] = None
    total_checks: int
    flagged_checks: int


class AdminSessionDetailOut(MonitoringSessionDetailOut):
    user_id: str
    user_name: str
    user_email: str


class EnrollGovernmentIdIn(BaseModel):
    upload_id: str
    consent: bool


class ExamIn(BaseModel):
    title: str
    pass_threshold: int
    max_flags: int = 4


class ExamOut(BaseModel):
    exam_id: str
    organization_id: str
    title: str
    pass_threshold: int
    max_flags: int
    is_active: bool
    created_at: str
    question_count: int = 0
    # Populated only by the student-facing GET /exams/mine listing (see exams.py) -- left None
    # everywhere else (org create/list/update, the single-exam get_exam) since those callers
    # don't have a specific student's attempts in scope.
    my_attempt_id: Optional[str] = None
    my_attempt_status: Optional[str] = None

    class Config:
        from_attributes = True


class ExamQuestionIn(BaseModel):
    text: str
    options: List[str] = Field(min_length=2)
    correct_option_index: int
    order_index: Optional[int] = None


class ExamQuestionAdminOut(BaseModel):
    """Organisation-facing view -- includes the answer key."""
    question_id: str
    exam_id: str
    order_index: int
    text: str
    options: List[str]
    correct_option_index: int


class ExamQuestionPublicOut(BaseModel):
    """Student-facing view while taking an exam -- correct_option_index is deliberately absent,
    never serialize the answer key here."""
    question_id: str
    order_index: int
    text: str
    options: List[str]


class ExamAttemptStartIn(BaseModel):
    details: dict[str, str] = {}


class ExamIdentityVerifyIn(BaseModel):
    """A fresh government-ID photo plus a live selfie, submitted together right before an exam
    attempt starts -- the per-attempt confidence check (see services/identity/detector.py::
    verify_identity_with_selfie), not a one-time enrollment."""
    id_upload_id: str
    live_upload_id: str
    consent: bool = False


class ExamIdentityVerifyOut(BaseModel):
    similarity: float
    verdict: str  # MATCH | UNCERTAIN | NO_MATCH, same banding as IdentityMatchOut
    passed: bool
    threshold: float


class ExamAttemptOut(BaseModel):
    """The exam-taking page's polling schema -- includes the live flagged_checks count so the
    frontend can show a running total, and status, which is what it actually watches to decide
    when to lock the UI."""
    attempt_id: str
    exam_id: str
    session_id: str
    status: str
    score: Optional[int] = None
    started_at: str
    ended_at: Optional[str] = None
    flagged_checks: int
    max_flags: int


class ExamAnswerIn(BaseModel):
    question_id: str
    selected_option_index: int


class ExamAnswerOut(BaseModel):
    answer_id: str
    question_id: str
    selected_option_index: int
    answered_at: str

    class Config:
        from_attributes = True


class ExamAttemptListItemOut(BaseModel):
    """One row of GET /admin/exams/{exam_id}/attempts -- an attempt plus enough about its exam
    and owner to review it in a list, mirroring AdminSessionListItemOut's shape."""
    attempt_id: str
    exam_id: str
    exam_title: str
    user_id: str
    user_name: str
    user_email: str
    session_id: str
    status: str
    score: Optional[int] = None
    started_at: str
    ended_at: Optional[str] = None


class BatchItemIn(BaseModel):
    """One file of a batch, as the page knows it: an upload that was scanned (upload_id set) or one that could not be
    (upload_id empty, `error` says why)."""
    upload_id: Optional[str] = None
    file_name: str = Field(max_length=300)
    error: Optional[str] = Field(default=None, max_length=300)


class BatchExportIn(BaseModel):
    items: List[BatchItemIn] = Field(min_length=1, max_length=100)
    format: Literal["csv", "pdf"]
