"""Batch scan export (added 2026-10-21): CSV or PDF summary for a list of scanned uploads. The scanning itself is
the normal upload + detect flow, driven file by file from the Verify page's Batch mode (see services/batch_export.py)."""

from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.models import User
from app.schemas.schemas import BatchExportIn
from app.services.batch_export import build_rows, to_csv
from app.services.report.batch_report import build_batch_pdf

router = APIRouter(prefix="/batch", tags=["Batch"])


@router.post("/export")
async def export_batch(
    body: BatchExportIn,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """CSV (opens in Excel) or PDF summary of the given uploads, each read back from its stored result. Only the
    caller's own uploads (404 otherwise)."""
    rows = await build_rows(db, current_user, body.items)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    if body.format == "csv":
        return Response(to_csv(rows), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="TruthLens_Batch_{stamp}.csv"'})
    return Response(build_batch_pdf(rows), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="TruthLens_Batch_{stamp}.pdf"'})
