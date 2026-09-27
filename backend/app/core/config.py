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
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mpeg", "audio/flac", "audio/x-wav", "audio/mp3"]

    FAKE_THRESHOLD: float = 0.5

    # Comma-separated browser origins allowed to call the API. "*" (the default) keeps the historical open behaviour
    # for development; set it to your site's origin in production (see docs/DEPLOYMENT.md).
    CORS_ORIGINS: str = "*"

    class Config:
        env_file = ".env"

settings = Settings()
