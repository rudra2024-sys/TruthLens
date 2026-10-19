from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "TruthLens"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = "sqlite+aiosqlite:///./truthlens.db"

    UPLOAD_DIR: str = "uploads"
    REPORT_DIR: str = "reports"
    MAX_FILE_SIZE_MB: int = 100

    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/quicktime", "video/webm"]
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mpeg", "audio/flac", "audio/x-wav", "audio/mp3",
                                  "audio/mp4", "audio/x-m4a", "audio/m4a",
                                  "audio/ogg", "audio/opus",
                                  # Chrome's MediaRecorder records audio/webm (Opus) by default --
                                  # it doesn't natively produce WAV. Added for the monitoring
                                  # session's audio-clip uploads (item 7, 2026-10-09).
                                  "audio/webm"]

    FAKE_THRESHOLD: float = 0.5

    # Cosine-similarity threshold for the identity-match feature, banded +-0.1 the same way
    # FAKE_THRESHOLD is banded +-0.2 elsewhere. Empirically calibrated 2026-10-11 (item 5) against
    # 1000 real labeled pairs from the LFW "DevTest" split (jessicali9530/lfw-dataset on Kaggle,
    # 500 same-person + 500 different-person), scored with this exact pipeline
    # (IdentityPipeline.embed + _cosine_similarity, see eval/run_identity_predictions.py) -- not a
    # re-implementation and not a guess. Measured: ROC AUC 0.974, same-person similarity mean 0.72
    # (std 0.20), different-person mean 0.03 (std 0.16), clearly separated (see
    # eval/results/identity/similarity_distributions.png). 0.44 is the accuracy-maximizing point
    # (96.5% accuracy, 0.4% false-accept rate, 6.6% false-reject rate at that exact point -- see
    # eval/results/identity/report.md for the full table). Caveat carried over from that report:
    # LFW is frontal, studio/press-quality photos, not webcam footage -- a real enrollment/
    # monitoring photo (worse lighting, off-angle, compression) will plausibly show more overlap
    # between the two distributions than this measurement does, so treat this as a real,
    # data-grounded starting point, not a guaranteed production false-accept/false-reject rate.
    IDENTITY_MATCH_THRESHOLD: float = 0.44

    # Monitoring-session signals (items 7/8/10, added 2026-10-09) -- all placeholder thresholds,
    # same honesty tier as IDENTITY_MATCH_THRESHOLD above: no labeled dataset exists locally to
    # calibrate any of these against, stated plainly rather than implied otherwise.
    LOOKING_AWAY_YAW_THRESHOLD_DEG: float = 25.0
    LOOKING_AWAY_PITCH_THRESHOLD_DEG: float = 20.0
    TALKING_SPEECH_RATIO_THRESHOLD: float = 0.4
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD: float = 0.4

    # Motion-based liveness heuristic (added 2026-10-11, item 9) -- NOT blink detection: MTCNN's
    # 5-point landmarks (2 eye centers, nose, 2 mouth corners) have no eyelid points, so a true
    # blink signal isn't computable from them. This instead looks at frame-to-frame landmark
    # motion across consecutive checks already stored in MonitoringCheck -- a printed photo held
    # up shows near-zero motion; a live face shows natural micro-movement even trying to stay
    # still. Real, imperfect signal: a genuinely still real face can occasionally trigger it, and
    # a photo subtly joggled by whoever is holding it can evade it. Both values are stated
    # placeholders -- no labeled dataset exists to validate them, same as the thresholds above.
    LIVENESS_MOTION_EPSILON_FRACTION: float = 0.004
    LIVENESS_MIN_CONSECUTIVE_CHECKS: int = 3

    # Enrollment quality gate (added 2026-10-11) -- same honesty tier as the thresholds above:
    # placeholder values based on general photo-quality heuristics, not empirically tuned against
    # a labeled dataset of real enrollment photos. See services/identity/enrollment_quality.py.
    ENROLLMENT_MIN_FACE_AREA_FRACTION: float = 0.08
    ENROLLMENT_EDGE_MARGIN_PX: float = 4.0
    ENROLLMENT_MIN_BLUR_VARIANCE: float = 5.0

    # Comma-separated browser origins allowed to call the API. "*" (the default) keeps the historical open behaviour
    # for development; set it to your site's origin in production (see docs/DEPLOYMENT.md).
    CORS_ORIGINS: str = "*"

    # Monitoring-check background jobs (added 2026-10-11, item 6) -- a separate pool from
    # DETECTION_JOB_CONCURRENCY (jobs.py, read straight from the env var of that name): checks are
    # lighter per-call than a video scan but still CPU-bound (MTCNN + ConvNeXt + CLIP + YOLOv8n per
    # check), and under several concurrent proctored sessions each posting a check every 5-20s,
    # letting them all run unbounded would contend for CPU. 4 is a reasonable-sounding default, not
    # load-tested.
    MONITORING_CHECK_JOB_CONCURRENCY: int = 4

    class Config:
        env_file = ".env"

settings = Settings()
