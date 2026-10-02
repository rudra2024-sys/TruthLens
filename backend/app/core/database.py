from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# No migration framework in this project (Base.metadata.create_all only creates tables
# that don't exist yet -- it never alters an existing table). For small, purely-additive
# schema changes on an already-live SQLite DB, run a plain idempotent ALTER TABLE here
# instead of introducing Alembic for a couple of columns. Each statement is wrapped so a
# "duplicate column" error (already applied on a previous startup) is silently skipped.
_LIGHT_MIGRATIONS = [
    "ALTER TABLE audio_analysis ADD COLUMN duration_s FLOAT",
    "ALTER TABLE audio_analysis ADD COLUMN windows_analyzed INTEGER",
    "ALTER TABLE video_analysis ADD COLUMN frames_with_face INTEGER",
]


async def _run_light_migrations():
    # Each statement gets its own transaction -- if one is a no-op (column already
    # exists) and raises, it must not poison a shared transaction for the next one.
    for stmt in _LIGHT_MIGRATIONS:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _run_light_migrations()
    print("Database ready (SQLite)")
