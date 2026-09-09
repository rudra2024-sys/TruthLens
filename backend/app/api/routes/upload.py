import uuid, os
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.config import settings
from app.models.models import Upload, User
from app.schemas.schemas import UploadOut
from app.api.routes.auth import get_current_user

router = APIRouter(prefix="/upload", tags=["Upload"])

ALL_ALLOWED = settings.ALLOWED_IMAGE_TYPES + settings.ALLOWED_VIDEO_TYPES + settings.ALLOWED_AUDIO_TYPES

def get_media_type(content_type: str) -> str:
    if content_type in settings.ALLOWED_IMAGE_TYPES: return "image"
    if content_type in settings.ALLOWED_VIDEO_TYPES: return "video"
    if content_type in settings.ALLOWED_AUDIO_TYPES: return "audio"
    return "unknown"

@router.post("/", response_model=UploadOut, status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or file.content_type not in ALL_ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Accepted: JPG/PNG/WebP, MP4/MOV/WebM, WAV/MP3/FLAC.")

    content = await file.read()
    size_kb = len(content) / 1024
    if size_kb / 1024 > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit.")
    if size_kb == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    upload_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "file")[1]
    path = os.path.join(settings.UPLOAD_DIR, f"{upload_id}{ext}")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async with aiofiles.open(path, "wb") as f:
        await f.write(content)

    record = Upload(
        upload_id=upload_id,
        file_name=file.filename or f"{upload_id}{ext}",
        media_type=get_media_type(file.content_type),
        storage_url=path,
        file_size_kb=round(size_kb, 1),
        user_id=current_user.user_id,
    )
    db.add(record)
    await db.flush()
    return record

@router.get("/{upload_id}", response_model=UploadOut)
async def get_upload(
    upload_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    u = r.scalar_one_or_none()
    # 404 (not 403) for both "doesn't exist" and "not yours" -- otherwise the
    # distinct error code itself would leak that an upload_id belongs to
    # someone else.
    if not u or u.user_id != current_user.user_id:
        raise HTTPException(404, "Upload not found")
    return u
