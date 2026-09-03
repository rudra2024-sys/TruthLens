from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.models import Upload, DetectionResult
from app.services.report.generator import generate_pdf_report

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("/{upload_id}")
async def download_report(upload_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Upload).where(Upload.upload_id == upload_id))
    upload = r.scalar_one_or_none()
    if not upload:
        raise HTTPException(404, "Upload not found")

    r2 = await db.execute(select(DetectionResult).where(DetectionResult.upload_id == upload_id))
    result = r2.scalar_one_or_none()
    if not result:
        raise HTTPException(404, "No detection result available for this upload yet")

    path = generate_pdf_report(upload, result)
    return FileResponse(path, media_type="application/pdf", filename=f"TruthLens_Report_{upload.file_name}.pdf")
