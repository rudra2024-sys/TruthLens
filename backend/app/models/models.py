import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, ForeignKey, Boolean
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

    # New trained ConvNeXt-Tiny outputs.
    fake_probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    real_probability: Mapped[float | None] = mapped_column(
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
