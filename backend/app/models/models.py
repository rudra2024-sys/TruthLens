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
