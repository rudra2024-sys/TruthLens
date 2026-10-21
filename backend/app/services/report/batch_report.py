"""One-page-per-~25-files PDF summary of a batch scan (added 2026-10-21). Same look and the same plain-English voice as
the single-file report; every figure comes from the stored results via services/batch_export.py."""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.services.batch_export import BatchRow
from app.services.report.pdf_theme import (
    CONTENT_WIDTH, INK, LINE, MUTED, PAGE_MARGIN, PAGE_SIZE, VERDICT_COLOR, footer_block, glossary_block,
    report_header, section_heading, tl_styles,
)
from app.services.report.plain_language import glossary_terms


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def counts(rows: list[BatchRow]) -> dict[str, int]:
    return {
        "total": len(rows),
        "authentic": sum(r.scanned and r.verdict_code == "REAL" for r in rows),
        "manipulated": sum(r.scanned and r.verdict_code == "FAKE" for r in rows),
        "uncertain": sum(r.scanned and r.verdict_code == "UNCERTAIN" for r in rows),
        "not_scanned": sum(not r.scanned for r in rows),
    }


def overview_sentence(c: dict[str, int]) -> str:
    noun = "file" if c["total"] == 1 else "files"
    parts = [f"{c['authentic']} look authentic", f"{c['manipulated']} were flagged as probably made or edited by AI",
             f"{c['uncertain']} could not be judged"]
    if c["not_scanned"]:
        parts.append(f"{c['not_scanned']} could not be scanned")
    return f"TruthLens looked at {c['total']} {noun}. Of these, " + ", ".join(parts[:-1]) + f" and {parts[-1]}."


def _file_cell(path: str, cell, small):
    """File name on top, its folder (when the batch came from a folder) in small grey text beneath."""
    folder, _, name = path.rpartition("/")
    parts = [Paragraph(_esc(name or path), cell)]
    if folder:
        parts.append(Paragraph(_esc(folder), small))
    return parts


def build_batch_pdf(rows: list[BatchRow], now: datetime | None = None) -> bytes:
    tl = tl_styles()
    now = now or datetime.utcnow()
    cell = tl["value"]
    small = tl["small"]
    c = counts(rows)

    els = report_header(tl, "Batch Summary Report",
                        f"{c['total']} files &nbsp;&middot;&nbsp; Created {now.day} {now:%B %Y}, "
                        f"{now.hour % 12 or 12}:{now:%M %p} (UTC)")

    # the counts at a glance
    tiles = [("Authentic", c["authentic"], VERDICT_COLOR["REAL"]), ("Manipulated", c["manipulated"], VERDICT_COLOR["FAKE"]),
             ("Uncertain", c["uncertain"], VERDICT_COLOR["UNCERTAIN"]), ("Not scanned", c["not_scanned"], MUTED)]
    tile_tbl = Table([[Paragraph(f'<font size=20 color="{col.hexval()}"><b>{n}</b></font>', cell) for _, n, col in tiles],
                      [Paragraph(label, small) for label, _, _ in tiles]], colWidths=[CONTENT_WIDTH / 4] * 4)
    tile_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BOX", (0, 0), (-1, -1), 0.75, LINE),
                                  ("LINEAFTER", (0, 0), (-2, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    els += [Paragraph(_esc(overview_sentence(c)), tl["summary"]), Spacer(1, 0.4 * cm), tile_tbl, Spacer(1, 0.5 * cm)]

    els += section_heading(tl, "In plain words")
    for line in (
        "Each file was checked on its own, exactly as it would be on the Verify page. The table below lists every file "
        "with the tool's answer. Open any file's full report from History (use the Case ID) to see the explanation.",
        "An answer of <b>Manipulated</b> means the tool thinks the file was probably made or edited by AI. "
        "<b>Authentic</b> means it found no such signs. <b>Uncertain</b> means it could not tell. None of these is "
        "proof. The tool can miss files made by newer AI generators or apps it has not seen, so Authentic and "
        "Uncertain never prove a file is genuine.",
        "The <b>reliability</b> column is the tool's confidence adjusted by how often it was actually right in our "
        "own tests, so it is usually the more realistic figure.",
    ):
        els += [Paragraph(line, tl["body"]), Spacer(1, 0.15 * cm)]

    els += [Spacer(1, 0.3 * cm)] + section_heading(tl, "Results")
    head = [Paragraph(f"<b>{h}</b>", small) for h in ("File", "Type", "Verdict", "Confidence", "Reliability", "Note")]
    data = [head]
    for r in rows:
        col = VERDICT_COLOR.get(r.verdict_code, MUTED) if r.scanned else MUTED
        data.append([
            _file_cell(r.file_name, cell, small), Paragraph(_esc(r.media_type or "-"), cell),
            Paragraph(f'<font color="{col.hexval()}"><b>{r.verdict_word}</b></font>', cell),
            Paragraph("-" if r.tool_confidence is None else f"{r.tool_confidence * 100:.0f}%", cell),
            Paragraph("-" if r.reliability is None else f"{r.reliability * 100:.0f}%", cell),
            Paragraph(_esc(r.note or r.problem or ""), small),
        ])
    widths = [5.4 * cm, 1.4 * cm, 2.3 * cm, 2.3 * cm, 2.1 * cm, CONTENT_WIDTH - 13.5 * cm]
    tbl = Table(data, colWidths=widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FAF6EF")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, LINE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    els.append(tbl)
    els += [Spacer(1, 0.15 * cm), Paragraph("A dash (-) means that figure is not available for that file, for example the "
                                           "reliability adjustment has not been measured for audio.", small)]

    media = {r.media_type for r in rows if r.media_type}
    terms = [t for t in glossary_terms("image", has_provenance=False, has_explanation=False, has_face_swap=False)]
    if "video" in media:
        terms += [(t, d) for t, d in glossary_terms("video", has_provenance=False, has_explanation=False) if t == "Frames"]
    els += glossary_block(tl, terms)
    els += footer_block(tl, "About this report",
                        "This summary was produced automatically by TruthLens's detection tools. It is an automated "
                        "opinion, not proof and not legal evidence. For anything important, confirm with the original "
                        "source of each file.")

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=PAGE_SIZE, topMargin=PAGE_MARGIN, bottomMargin=PAGE_MARGIN,
                      leftMargin=PAGE_MARGIN, rightMargin=PAGE_MARGIN, title="TruthLens batch summary").build(els)
    return buf.getvalue()
