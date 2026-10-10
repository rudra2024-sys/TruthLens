from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "TruthLens"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = "sqlite+aiosqlite:///./truthlens.db"

    UPLOAD_DIR: str = "uploads"
    REPORT_DIR: str = "reports"
    MAX_FILE_SIZE_MB: int = 100

    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/quicktime", "video/webm"]
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mpeg", "audio/flac", "audio/x-wav", "audio/mp3",
                                  "audio/mp4", "audio/x-m4a", "audio/m4a",
                                  # Chrome's MediaRecorder records audio/webm (Opus) by default --
                                  # it doesn't natively produce WAV. Added for the monitoring
                                  # session's audio-clip uploads (item 7, 2026-10-09).
                                  "audio/webm"]

    FAKE_THRESHOLD: float = 0.5

    # Cosine-similarity threshold for the identity-match feature (added 2026-10-08), banded
    # +-0.1 the same way FAKE_THRESHOLD is banded +-0.2 elsewhere. This is a literature-typical
    # default for facenet-pytorch's InceptionResnetV1(pretrained='vggface2') embeddings, NOT
    # empirically validated against real face pairs in this environment -- there is no labeled
    # same/different-person photo dataset here yet to calibrate it against (same honesty norm
    # as AudioModelMetadata.validated elsewhere in this codebase).
    IDENTITY_MATCH_THRESHOLD: float = 0.6

    # Monitoring-session signals (items 7/8/10, added 2026-10-09) -- all placeholder thresholds,
    # same honesty tier as IDENTITY_MATCH_THRESHOLD above: no labeled dataset exists locally to
    # calibrate any of these against, stated plainly rather than implied otherwise.
    LOOKING_AWAY_YAW_THRESHOLD_DEG: float = 25.0
    LOOKING_AWAY_PITCH_THRESHOLD_DEG: float = 20.0
    TALKING_SPEECH_RATIO_THRESHOLD: float = 0.4
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD: float = 0.4

    # Comma-separated browser origins allowed to call the API. "*" (the default) keeps the historical open behaviour
    # for development; set it to your site's origin in production (see docs/DEPLOYMENT.md).
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"

settings = Settings()
