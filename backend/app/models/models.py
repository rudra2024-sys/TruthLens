import json
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, ForeignKey, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    email: Mapped[str] = mapped_column(
        String,
        nullable=False,
        unique=True,
        index=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    created_at: Mapped[str] = mapped_column(
        String,
        default=lambda: datetime.utcnow().isoformat(),
    )
    
class Upload(Base):
    __tablename__ = "uploads"
    upload_id:   Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_name:   Mapped[str] = mapped_column(String, nullable=False)
    media_type:  Mapped[str] = mapped_column(String, nullable=False)
    storage_url: Mapped[str] = mapped_column(String, nullable=False)
    file_size_kb: Mapped[float] = mapped_column(Float, default=0)
    uploaded_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    # Nullable so pre-existing rows created before ownership was enforced don't
    # break; a NULL owner is never matched by an authenticated user's history
    # or ownership check, so old unowned rows just become inaccessible rather
    # than visible to everyone.
    user_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.user_id"), nullable=True)

    result: Mapped["DetectionResult"] = relationship(
        "DetectionResult", back_populates="upload", uselist=False,
        lazy="selectin",
    )


class DetectionResult(Base):
    __tablename__ = "detection_results"
    result_id:          Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_id:          Mapped[str]   = mapped_column(String, ForeignKey("uploads.upload_id"))
    confidence_score:   Mapped[float] = mapped_column(Float)
    calibrated_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict:             Mapped[str]  = mapped_column(String)
    model_used:          Mapped[str]  = mapped_column(String, default="ensemble-v1")
    processing_time_ms:  Mapped[float] = mapped_column(Float, default=0)
    detected_at:         Mapped[str]  = mapped_column(String, default=lambda: datetime.utcnow().isoformat())

    upload:         Mapped["Upload"]        = relationship("Upload", back_populates="result")
    image_analysis: Mapped["ImageAnalysis"] = relationship("ImageAnalysis", back_populates="result", uselist=False, lazy="selectin")
    video_analysis: Mapped["VideoAnalysis"] = relationship("VideoAnalysis", back_populates="result", uselist=False, lazy="selectin")
    audio_analysis: Mapped["AudioAnalysis"] = relationship("AudioAnalysis", back_populates="result", uselist=False, lazy="selectin")


class ImageAnalysis(Base):
    __tablename__ = "image_analysis"

    analysis_id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    result_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("detection_results.result_id")
    )

    # Legacy fields — preserved for old database records.
    # New ConvNeXt detections leave these as NULL.
    efficientnet_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    fft_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Ensemble output (ConvNeXt-Tiny + CLIP second opinion, combined via max
    # -- see services/image/detector.py for why) -- this is the value
    # _verdict() and confidence_score are actually derived from.
    fake_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    real_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Individual sub-model scores, kept for transparency in the report --
    # do not repeat the wav2vec_score/lcnn_score mislabeling mistake (see
    # schemas.py AudioAnalysisOut) where a field's name stopped matching what
    # it actually holds. NULL for detections run before the CLIP ensemble
    # was added.
    convnext_fake_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    clip_fake_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    result: Mapped["DetectionResult"] = relationship(
        "DetectionResult",
        back_populates="image_analysis"
    )

class VideoAnalysis(Base):
    __tablename__ = "video_analysis"
    analysis_id:      Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    result_id:         Mapped[str]  = mapped_column(String, ForeignKey("detection_results.result_id"))
    xception_score:    Mapped[float] = mapped_column(Float)
    face_voice_sync:   Mapped[float] = mapped_column(Float, nullable=True)
    frames_analyzed:   Mapped[int]   = mapped_column(Integer)
    frames_with_face:  Mapped[int]   = mapped_column(Integer, nullable=True)
    result: Mapped["DetectionResult"] = relationship("DetectionResult", back_populates="video_analysis")


class AudioAnalysis(Base):
    __tablename__ = "audio_analysis"
    analysis_id:      Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    result_id:         Mapped[str]  = mapped_column(String, ForeignKey("detection_results.result_id"))
    wav2vec_score:     Mapped[float] = mapped_column(Float)
    lcnn_score:        Mapped[float] = mapped_column(Float)
    duration_s:        Mapped[float | None] = mapped_column(Float, nullable=True)
    windows_analyzed:  Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped["DetectionResult"] = relationship("DetectionResult", back_populates="audio_analysis")


class Feedback(Base):
    """A user's judgement of one detection result ("was this correct?").

    One row per (upload, user): resubmitting updates it. `allow_reuse` is an explicit opt-in to keep the uploaded file for
    evaluating / improving the models; the export tooling ignores the file of any feedback where it is False.
    """
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("upload_id", "user_id", name="uq_feedback_upload_user"),)

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    upload_id:   Mapped[str] = mapped_column(String, ForeignKey("uploads.upload_id"), index=True)
    user_id:     Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), index=True)
    result_id:   Mapped[str | None] = mapped_column(String, ForeignKey("detection_results.result_id"), nullable=True)
    verdict:     Mapped[str] = mapped_column(String)            # the verdict the user was shown
    agrees:      Mapped[bool] = mapped_column(Boolean)          # True = "the result was correct"
    true_label:  Mapped[str | None] = mapped_column(String, nullable=True)   # real | ai | unsure (only when disagreeing)
    comment:     Mapped[str | None] = mapped_column(String, nullable=True)   # <= 500 chars
    allow_reuse: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at:  Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    updated_at:  Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class FaceReference(Base):
    """A user's enrolled reference face embedding for identity-match verification (added
    2026-10-08). One active reference per user -- re-enrolling replaces it in place rather
    than keeping history. embedding_json is a JSON-encoded list of 512 floats (facenet-pytorch's
    InceptionResnetV1 output); plain Text is fine at this project's scale, no vector DB needed.
    Never serialized back out over the API -- see schemas.py's IdentityReferenceOut.
    """
    __tablename__ = "face_references"

    reference_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:      Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), unique=True, index=True)
    upload_id:    Mapped[str] = mapped_column(String, ForeignKey("uploads.upload_id"))
    embedding_json: Mapped[str] = mapped_column(String, nullable=False)
    enrolled_at:  Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class IdentityMatchResult(Base):
    """One identity-match check (a live-captured frame compared against a FaceReference)."""
    __tablename__ = "identity_match_results"

    match_id:         Mapped[str]  = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:          Mapped[str]  = mapped_column(String, ForeignKey("users.user_id"), index=True)
    reference_id:     Mapped[str]  = mapped_column(String, ForeignKey("face_references.reference_id"))
    upload_id:        Mapped[str]  = mapped_column(String, ForeignKey("uploads.upload_id"))
    similarity_score: Mapped[float] = mapped_column(Float)
    verdict:          Mapped[str]  = mapped_column(String)   # MATCH | NO_MATCH | UNCERTAIN
    checked_at:       Mapped[str]  = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class MonitoringSession(Base):
    """A bounded window of periodic identity/deepfake/presence checks against one enrolled
    FaceReference (added 2026-10-09) -- the continuous-monitoring extension of one-shot
    identity-match. ended_at is nullable: null means still active. No separate status column --
    the single nullable timestamp is both simpler and can't drift out of sync with itself."""
    __tablename__ = "monitoring_sessions"

    session_id:   Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:      Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), index=True)
    reference_id: Mapped[str] = mapped_column(String, ForeignKey("face_references.reference_id"))
    started_at:   Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
    ended_at:     Mapped[str | None] = mapped_column(String, nullable=True)
    # Set once, from the session's first face-detected check -- the "looking at the screen"
    # reference point later checks' yaw/pitch are compared against (added 2026-10-09, item 10).
    # Self-calibrating per person/camera-angle rather than a fixed universal angle threshold,
    # since there's no labeled head-pose dataset here to calibrate a universal one against.
    baseline_yaw:   Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_pitch: Mapped[float | None] = mapped_column(Float, nullable=True)


class MonitoringCheck(Base):
    """One periodic check within a MonitoringSession. Independent signals, any of which can flag
    the check on their own -- flagged can be True even when identity_verdict is MATCH, which is
    the point ("abnormal activity even if they have matched"). similarity_score/identity_verdict
    are null when face_count != 1 (no single clear face to compare).

    face_count's history: a first attempt at an exact count (OpenCV Haar cascade) was shipped
    and then found broken by real-webcam testing (0 faces found on an obvious, well-lit single
    face) and pulled back to a 0/1-only presence signal. Redone 2026-10-09 using MTCNN's own
    .detect() (the detector actually used for the embedding, already loaded) with a confidence
    filter + a hand-written IoU box-merge to deduplicate overlapping candidate boxes -- directly
    validated against real photos (a composite of two different real people -> 2, single real
    faces -> 1, faceless/covered-lens frames -> 0) before being trusted here. See
    app/pipelines/identity/inference.py::IdentityPipeline.count_faces for the implementation and
    full validation notes, and app/services/identity/monitoring.py's module docstring for the
    complete history.
    flag_reasons_json is a JSON list of strings, e.g. ["identity_mismatch","deepfake_signal"].
    """
    __tablename__ = "monitoring_checks"

    check_id:          Mapped[str]   = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id:        Mapped[str]   = mapped_column(String, ForeignKey("monitoring_sessions.session_id"), index=True)
    upload_id:         Mapped[str]   = mapped_column(String, ForeignKey("uploads.upload_id"))
    face_count:        Mapped[int]   = mapped_column(Integer)
    similarity_score:  Mapped[float | None] = mapped_column(Float, nullable=True)
    identity_verdict:  Mapped[str | None]   = mapped_column(String, nullable=True)
    fake_probability:  Mapped[float] = mapped_column(Float)
    deepfake_verdict:  Mapped[str]   = mapped_column(String)
    flagged:           Mapped[bool]  = mapped_column(Boolean)
    flag_reasons_json: Mapped[str]   = mapped_column(String, default="[]")
    checked_at:        Mapped[str]   = mapped_column(String, default=lambda: datetime.utcnow().isoformat())

    # Items 7-10 (added 2026-10-09). All nullable -- only populated when the relevant signal
    # could actually be computed for this check (e.g. yaw/pitch need exactly one face found;
    # speech_ratio/object detections need the frontend to have sent that frame/clip at all).
    yaw_deg:        Mapped[float | None] = mapped_column(Float, nullable=True)
    pitch_deg:      Mapped[float | None] = mapped_column(Float, nullable=True)
    # Mouth-CORNER distance (item 9), not true vertical mouth-aperture -- MTCNN's 5-point
    # landmarks have no top/bottom-lip point, confirmed during planning. A weaker proxy for
    # talking than real lip-sync work would use; informational only, never auto-flagged (see
    # monitoring.py's module docstring and session_report.py's lip-sync section).
    mouth_width_px: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Fraction of an accompanying ~3s audio clip classified as speech by short-term-energy VAD
    # (item 7) -- tells you speech-level audio activity happened, NOT whose voice it was. See
    # audio_vad.py's module docstring.
    speech_ratio:   Mapped[float | None] = mapped_column(Float, nullable=True)
    # JSON list of COCO class names found in this frame, filtered to ["cell phone", "book"]
    # only (item 8) -- see pipelines/objects/inference.py. Decoded via the object_detections
    # property below, same convention as flag_reasons.
    object_detections_json: Mapped[str] = mapped_column(String, default="[]")

    # Live face-overlay visualization (added 2026-10-10, follow-up item 3): the single kept
    # face's MTCNN box ([x1,y1,x2,y2]) and 5-point landmarks ([[x,y],...]) in the captured
    # frame's own pixel space, plus that frame's dimensions so the frontend can scale them onto
    # the displayed video element. Null whenever face_count != 1 -- same condition that already
    # gates yaw_deg/mouth_width_px above. Genuine per-check MTCNN output, not a client-side
    # approximation -- see inference.py::detect_faces's docstring.
    face_box_json:  Mapped[str | None] = mapped_column(String, nullable=True)
    landmarks_json: Mapped[str | None] = mapped_column(String, nullable=True)
    image_width:    Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height:   Mapped[int | None] = mapped_column(Integer, nullable=True)

    @property
    def flag_reasons(self) -> list[str]:
        """Decoded view of flag_reasons_json -- schemas.py's MonitoringCheckOut reads this
        property (not the raw JSON string column) so the API returns a real list."""
        return json.loads(self.flag_reasons_json)

    @property
    def object_detections(self) -> list[str]:
        """Decoded view of object_detections_json, same convention as flag_reasons."""
        return json.loads(self.object_detections_json)

    @property
    def face_box(self) -> list[float] | None:
        """Decoded view of face_box_json, same convention as flag_reasons."""
        return json.loads(self.face_box_json) if self.face_box_json is not None else None

    @property
    def landmarks(self) -> list[list[float]] | None:
        """Decoded view of landmarks_json, same convention as flag_reasons."""
        return json.loads(self.landmarks_json) if self.landmarks_json is not None else None


class MonitoringEvent(Base):
    """A behavioral event within a MonitoringSession, separate from MonitoringCheck -- these
    don't involve a camera frame or any model inference at all (added 2026-10-09): tab/window-
    focus loss, a clipboard paste, a devtools-open heuristic firing, or the camera track ending/
    muting unexpectedly. No `flagged` column -- an event existing at all IS the flag; the
    frontend decides locally what's worth posting (e.g. only a tab-hidden spell longer than a
    couple of seconds, not every incidental blur). event_type is one of: tab_hidden,
    window_blurred, clipboard_paste, devtools_suspected, camera_interrupted. detail is a free-
    form string (e.g. an away-duration in seconds for tab_hidden/window_blurred).

    Honesty note (carried into the UI too): these only ever see activity inside the browser tab
    this session is running in -- a second physical device, a different browser, or clipboard
    activity in a separate exam-platform tab are all invisible to this signal. devtools_suspected
    is a window-size heuristic, not a hard guarantee, and is bypassable by someone who knows it.
    """
    __tablename__ = "monitoring_events"

    event_id:    Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id:  Mapped[str] = mapped_column(String, ForeignKey("monitoring_sessions.session_id"), index=True)
    event_type:  Mapped[str] = mapped_column(String)
    detail:      Mapped[str | None] = mapped_column(String, nullable=True)
    occurred_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class AuditLogEntry(Base):
    """A record of a sensitive action (added 2026-10-03) - login/signup/feedback-withdrawal/account-deletion,
    see app/services/audit_log.py. Deliberately NOT a foreign key to users.user_id: account deletion must
    still be able to log "this user deleted their account" as the very last thing that happens for them, and
    a hard FK would either block that delete or force the log row to be deleted right along with the account
    it's supposed to be a record of. user_id here is a plain, unenforced string instead - see
    docs/ETHICS_AND_LIMITATIONS.md section 5, which has flagged "no audit log" as a gap since it was written.
    """
    __tablename__ = "audit_log"
    entry_id:   Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id:    Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    action:     Mapped[str] = mapped_column(String, index=True)     # e.g. "login", "signup", "account_deleted"
    detail:     Mapped[str | None] = mapped_column(String, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: datetime.utcnow().isoformat())
