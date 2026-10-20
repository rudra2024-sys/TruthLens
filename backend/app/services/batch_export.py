"""Batch scan results -> rows -> CSV / PDF (added 2026-10-21).

A batch is not stored as such: the page scans each file through the normal upload + detect flow and then sends the list
of upload ids here. Every row is read back from the stored results (nothing is recomputed or invented), restricted to
the caller's own uploads - an id that is not theirs is a 404, like everywhere else in the API.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import DetectionResult, Upload, User
from app.schemas.schemas import BatchItemIn
from app.services.report.plain_language import SMALL_IMAGE_PX, image_fake_kind, image_size

MAX_BATCH = 100

_VERDICT_WORD = {"FAKE": "Manipulated", "REAL": "Authentic", "UNCERTAIN": "Uncertain"}

CSV_COLUMNS = ["File", "Type", "Status", "Verdict", "Verdict code", "Note", "Tool confidence (%)",
               "Estimated reliability (%)", "Scanned at (UTC)", "Case ID", "Problem"]


@dataclass
class BatchRow:
    file_name: str
    media_type: str = ""
    scanned: bool = False
    verdict_code: str = ""
    note: str = ""
    tool_confidence: float | None = None          # 0..1
    reliability: float | None = None              # 0..1
    scanned_at: str = ""
    case_id: str = ""
    problem: str = ""

    @property
    def verdict_word(self) -> str:
        return _VERDICT_WORD.get(self.verdict_code, "") if self.scanned else "Not scanned"


def _note(upload: Upload, result: DetectionResult) -> str:
    parts: list[str] = []
    kind = image_fake_kind(result)
    if kind == "face swap":
        parts.append("Likely a face swap")
    elif kind == "AI-generated":
        parts.append("Likely AI-generated")
    size = image_size(upload)
    if size and min(size) < SMALL_IMAGE_PX:
        parts.append(f"Small image ({size[0]}x{size[1]} px), less reliable")
    va = result.video_analysis
    if va is not None and getattr(va, "low_resolution", None):
        parts.append("Video too low-resolution to judge")
    elif va is not None and va.frames_with_face is not None and va.frames_analyzed \
            and va.frames_with_face < va.frames_analyzed / 2:
        parts.append("Few clear faces found, less reliable")
    return "; ".join(parts)


async def build_rows(db: AsyncSession, user: User, items: list[BatchItemIn]) -> list[BatchRow]:
    ids = [i.upload_id for i in items if i.upload_id]
    uploads: dict[str, Upload] = {}
    results: dict[str, DetectionResult] = {}
    if ids:
        uploads = {u.upload_id: u for u in (await db.execute(select(Upload).where(Upload.upload_id.in_(ids)))).scalars()}
        if any(i not in uploads or uploads[i].user_id != user.user_id for i in ids):
            raise HTTPException(404, "Upload not found.")           # one rule for "unknown" and "not yours"
        results = {r.upload_id: r for r in
                   (await db.execute(select(DetectionResult).where(DetectionResult.upload_id.in_(ids)))).scalars()}

    rows: list[BatchRow] = []
    for it in items:
        res = results.get(it.upload_id) if it.upload_id else None
        if res is None:
            rows.append(BatchRow(file_name=it.file_name, problem=(it.error or "No result for this file.").strip(),
                                 case_id=(it.upload_id or "")[:8].upper(),
                                 media_type=uploads[it.upload_id].media_type if it.upload_id in uploads else ""))
            continue
        up = uploads[it.upload_id]
        rows.append(BatchRow(
            file_name=it.file_name or up.file_name, media_type=up.media_type, scanned=True,
            verdict_code=res.verdict, note=_note(up, res), tool_confidence=res.confidence_score,
            reliability=res.calibrated_confidence, scanned_at=str(res.detected_at), case_id=up.upload_id[:8].upper(),
        ))
    return rows


def _safe_cell(value: str) -> str:
    """Spreadsheet formula injection guard: a text cell starting with = + - @ (or a tab/CR) would be run as a formula
    by Excel/Sheets, and file names are chosen by whoever made the file."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def _pct(x: float | None) -> str:
    return "" if x is None else f"{x * 100:.1f}"


def to_csv(rows: list[BatchRow]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow([_safe_cell(r.file_name), r.media_type, "Scanned" if r.scanned else "Not scanned", r.verdict_word,
                    r.verdict_code, _safe_cell(r.note), _pct(r.tool_confidence), _pct(r.reliability), r.scanned_at,
                    r.case_id, _safe_cell(r.problem)])
    return ("﻿" + buf.getvalue()).encode("utf-8")          # BOM so Excel reads non-ASCII file names correctly
