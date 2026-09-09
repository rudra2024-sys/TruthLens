from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.core.database import get_db
from app.models.models import Upload, DetectionResult, User
from app.schemas.schemas import HistoryItemOut, StatsOut
from app.api.routes.auth import get_current_user

router = APIRouter(tags=["Dashboard"])


@router.get("/history", response_model=list[HistoryItemOut])
async def get_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.execute(
        select(Upload)
        .where(Upload.user_id == current_user.user_id)
        .order_by(desc(Upload.uploaded_at))
        .limit(limit)
    )
    uploads = r.scalars().all()
    items = []
    for u in uploads:
        items.append(HistoryItemOut(
            upload_id=u.upload_id,
            file_name=u.file_name,
            media_type=u.media_type,
            uploaded_at=u.uploaded_at,
            verdict=u.result.verdict if u.result else None,
            confidence_score=u.result.confidence_score if u.result else None,
        ))
    return items


@router.get("/stats", response_model=StatsOut)
async def get_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    r = await db.execute(
        select(DetectionResult)
        .join(Upload, Upload.upload_id == DetectionResult.upload_id)
        .where(Upload.user_id == current_user.user_id)
    )
    results = r.scalars().all()

    r2 = await db.execute(
        select(Upload).where(Upload.user_id == current_user.user_id)
    )
    uploads = r2.scalars().all()

    by_type = {"image": 0, "video": 0, "audio": 0}
    for u in uploads:
        if u.media_type in by_type:
            by_type[u.media_type] += 1

    return StatsOut(
        total_scans=len(results),
        fake_count=sum(1 for x in results if x.verdict == "FAKE"),
        real_count=sum(1 for x in results if x.verdict == "REAL"),
        uncertain_count=sum(1 for x in results if x.verdict == "UNCERTAIN"),
        by_media_type=by_type,
    )
