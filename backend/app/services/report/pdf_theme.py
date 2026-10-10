"""Shared visual language for every TruthLens PDF report (added 2026-10-10, at the user's
request to make reports "more simple... may be read by people who are not into tech" and to
stop each report looking like a separate, generic auto-generated document).

Both generator.py (single-file detection report) and session_report.py (monitoring-session
report) import from here so a reader sees ONE consistent brand/format, not two different-looking
PDFs. This module only changes presentation (colors, type, layout, a plain-English summary
sentence restating numbers that were already computed) -- it never computes or alters a verdict,
score, or any other figure. The plain-English sentences below are a restatement of real,
already-computed values, not a new claim (section 8 rule 5: no fabricated output).

Print palette, not the app's dark "ground" theme -- reports are read/printed on white, so this
reuses the site's brass accent and verdict colors but on a light background, deepened/saturated
enough to stay readable on paper.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

INK = colors.HexColor("#241F1A")
MUTED = colors.HexColor("#746B5E")
BRASS = colors.HexColor("#C89361")
BRASS_DEEP = colors.HexColor("#8F6238")
CARD_BG = colors.HexColor("#FAF6EF")
LINE = colors.HexColor("#E8DFD0")

VERDICT_COLOR = {
    "REAL": colors.HexColor("#16a34a"),
    "FAKE": colors.HexColor("#dc2626"),
    "UNCERTAIN": colors.HexColor("#d97706"),
    "MATCH": colors.HexColor("#16a34a"),
    "NO_MATCH": colors.HexColor("#dc2626"),
    "CLEAN": colors.HexColor("#16a34a"),
    "FLAGGED": colors.HexColor("#d97706"),
}
DEFAULT_VERDICT_COLOR = colors.HexColor("#6b7280")

PAGE_SIZE = A4
PAGE_MARGIN = 2 * cm
CONTENT_WIDTH = A4[0] - 2 * PAGE_MARGIN


def tl_styles() -> dict:
    """One shared stylesheet, built fresh per call (ReportLab styles aren't safely reusable
    across documents) -- Times for headings (the site's serif headline voice), Helvetica for
    body copy (the site's clean sans body voice)."""
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "tl_kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.5,
            textColor=BRASS_DEEP, spaceAfter=2, leading=11,
        ),
        "title": ParagraphStyle(
            "tl_title", parent=base["Title"], fontName="Times-Bold", fontSize=22,
            textColor=INK, spaceAfter=0, leading=26, alignment=0,
        ),
        "h2": ParagraphStyle(
            "tl_h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=13.5,
            textColor=INK, spaceBefore=0, spaceAfter=6, leading=16,
        ),
        "h3": ParagraphStyle(
            "tl_h3", parent=base["Heading3"], fontName="Times-Bold", fontSize=11.5,
            textColor=INK, spaceBefore=0, spaceAfter=4, leading=14,
        ),
        "body": ParagraphStyle(
            "tl_body", parent=base["Normal"], fontName="Helvetica", fontSize=10,
            textColor=INK, leading=14.5,
        ),
        "summary": ParagraphStyle(
            "tl_summary", parent=base["Normal"], fontName="Helvetica", fontSize=12,
            textColor=INK, leading=17,
        ),
        "small": ParagraphStyle(
            "tl_small", parent=base["Normal"], fontName="Helvetica", fontSize=8.5,
            textColor=MUTED, leading=12,
        ),
        "muted_label": ParagraphStyle(
            "tl_muted_label", parent=base["Normal"], fontName="Helvetica", fontSize=9,
            textColor=MUTED, leading=12,
        ),
        "value": ParagraphStyle(
            "tl_value", parent=base["Normal"], fontName="Helvetica", fontSize=9,
            textColor=INK, leading=12,
        ),
        "technical_note": ParagraphStyle(
            "tl_technical_note", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=9,
            textColor=MUTED, leading=13,
        ),
    }


def report_header(styles: dict, title_text: str, case_label: str) -> list:
    """Wordmark + report title + a case/session identifier + a brass hairline -- the same
    header shape on every report this project produces, so two reports read as one product."""
    rule = Table([[""]], colWidths=[CONTENT_WIDTH], rowHeights=[1.3])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRASS), ("LINEBELOW", (0, 0), (-1, -1), 0, BRASS)]))
    return [
        Paragraph("TRUTHLENS &nbsp;&middot;&nbsp; AI MEDIA FORENSICS", styles["kicker"]),
        Paragraph(title_text, styles["title"]),
        Spacer(1, 0.15 * cm),
        Paragraph(case_label, styles["small"]),
        Spacer(1, 0.3 * cm),
        rule,
        Spacer(1, 0.6 * cm),
    ]


def verdict_card(styles: dict, verdict_label: str, headline: str, summary_sentence: str) -> list:
    """The big colored "what you need to know" card at the top of every report -- a plain-
    English sentence restating the verdict/confidence that's detailed everywhere else in the
    document, aimed at a reader who isn't going to parse a table of sub-model scores."""
    color = VERDICT_COLOR.get(verdict_label, DEFAULT_VERDICT_COLOR)
    inner = Table(
        [[Paragraph(f'<font color="{color.hexval()}"><b>{headline}</b></font>', ParagraphStyle(
            "tl_verdict_headline", parent=styles["h2"], fontSize=18, textColor=color, spaceAfter=4))],
         [Paragraph(summary_sentence, styles["summary"])]],
        colWidths=[CONTENT_WIDTH - 1.2 * cm],
    )
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (0, 0), 0), ("BOTTOMPADDING", (0, 0), (0, 0), 2),
        ("TOPPADDING", (0, 1), (0, 1), 2), ("BOTTOMPADDING", (0, 1), (0, 1), 0),
    ]))
    card = Table([[inner]], colWidths=[CONTENT_WIDTH])
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, LINE),
        ("LINEBEFORE", (0, 0), (0, 0), 3, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 16), ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("TOPPADDING", (0, 0), (-1, -1), 14), ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    return [card, Spacer(1, 0.7 * cm)]


def info_card(rows: list[tuple[str, str]], label_width: float = 4.5 * cm) -> list:
    """A light-card key/value table for "at a glance" facts (file name, scan date, duration,
    etc.) -- the same visual container used for metadata on every report."""
    s = tl_styles()
    table_rows = [[Paragraph(label, s["muted_label"]), Paragraph(str(value), s["value"])] for label, value in rows]
    table = Table(table_rows, colWidths=[label_width, CONTENT_WIDTH - label_width - 0.8 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, LINE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (0, -1), 14), ("RIGHTPADDING", (-1, 0), (-1, -1), 14),
        ("LEFTPADDING", (1, 0), (1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return [table]


def section_heading(styles: dict, text: str) -> list:
    """A section heading with a short brass underline -- replaces a bare Heading3 everywhere,
    so every section of every report reads with the same rhythm."""
    rule = Table([[""]], colWidths=[2.2 * cm], rowHeights=[1.1])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BRASS)]))
    return [Paragraph(text, styles["h2"]), Spacer(1, 0.08 * cm), rule, Spacer(1, 0.3 * cm)]


def technical_divider(styles: dict, note: str) -> list:
    """Marks the boundary between the plain-English summary and the raw model/technical data --
    lets a non-technical reader stop here with confidence, and a technical reader know exactly
    where the detail starts."""
    return [
        Spacer(1, 0.3 * cm),
        Paragraph("TECHNICAL DETAILS", ParagraphStyle(
            "tl_tech_kicker", parent=styles["kicker"], textColor=MUTED)),
        Paragraph(note, styles["technical_note"]),
        Spacer(1, 0.3 * cm),
    ]


def footer_block(styles: dict, heading: str, body: str) -> list:
    """The standardized closing disclaimer block, same shape on every report."""
    rule = Table([[""]], colWidths=[CONTENT_WIDTH], rowHeights=[0.5])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.5, LINE)]))
    return [
        Spacer(1, 0.8 * cm),
        rule,
        Spacer(1, 0.25 * cm),
        Paragraph(heading, ParagraphStyle(
            "tl_footer_heading", parent=styles["small"], fontName="Helvetica-Bold", textColor=BRASS_DEEP)),
        Paragraph(body, styles["small"]),
    ]
