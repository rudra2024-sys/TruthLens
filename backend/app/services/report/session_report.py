"""Session summary PDF for a MonitoringSession (added 2026-10-09) -- a sibling to
generator.py::generate_pdf_report, not an overload of it: that function's signature is bound to
a single Upload+DetectionResult, not reusable for a session's many checks/events. Same
conventions throughout: getSampleStyleSheet(), the teal #0F6B66 title / red-green-amber verdict
color scheme, a file written to settings.REPORT_DIR (path returned, not bytes -- FileResponse
serves it directly), and best-effort try/except per section.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import settings
from app.models.models import MonitoringCheck, MonitoringEvent, MonitoringSession

logger = logging.getLogger(__name__)

EVENT_LABELS = {
    "tab_hidden": "Tab hidden",
    "window_blurred": "Window lost focus",
    "clipboard_paste": "Clipboard paste",
    "devtools_suspected": "Devtools suspected",
    "camera_interrupted": "Camera interrupted",
}

REASON_LABELS = {
    "no_face": "No face detected",
    "multiple_faces": "Multiple faces detected",
    "identity_mismatch": "Identity mismatch",
    "identity_uncertain": "Identity uncertain",
    "deepfake_signal": "Possible AI manipulation",
    "deepfake_uncertain": "Manipulation uncertain",
}


def _parse_iso(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _duration_label(session: MonitoringSession) -> str:
    start, end = _parse_iso(session.started_at), _parse_iso(session.ended_at)
    if start is None or end is None:
        return "In progress"
    seconds = max(0, int((end - start).total_seconds()))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs}s"


def _flag_breakdown_table(checks: list[MonitoringCheck], small) -> list:
    counts: Counter = Counter()
    for check in checks:
        for reason in check.flag_reasons:
            counts[reason] += 1
    if not counts:
        return [Paragraph("No checks were flagged during this session.", small)]
    rows = [[REASON_LABELS.get(reason, reason), str(count)] for reason, count in counts.most_common()]
    table = Table(rows, colWidths=[10 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [table]


def _event_summary_table(events: list[MonitoringEvent], small) -> list:
    if not events:
        return [Paragraph("No behavioral events were recorded during this session.", small)]
    counts: Counter = Counter(e.event_type for e in events)
    away_total = sum(
        float(e.detail) for e in events
        if e.event_type in ("tab_hidden", "window_blurred") and e.detail
        and e.detail.replace(".", "", 1).isdigit()
    )
    rows = [[EVENT_LABELS.get(etype, etype), str(count)] for etype, count in counts.most_common()]
    table = Table(rows, colWidths=[10 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    out = [table]
    if away_total > 0:
        out.append(Spacer(1, 0.15 * cm))
        out.append(Paragraph(f"Total time away from the tab/window: {away_total:.0f}s", small))
    return out


def _timeline_rows(checks: list[MonitoringCheck], events: list[MonitoringEvent]) -> list[tuple]:
    """Merges checks and events into one chronological list of (timestamp, is_flagged, label)."""
    rows = []
    for c in checks:
        label = f"Camera check -- face count {c.face_count}"
        if c.identity_verdict:
            label += f", identity {c.identity_verdict}"
        label += f", fake {c.fake_probability * 100:.0f}%"
        if c.flagged:
            label += " -- " + ", ".join(REASON_LABELS.get(r, r) for r in c.flag_reasons)
        rows.append((c.checked_at, c.flagged, label))
    for e in events:
        label = EVENT_LABELS.get(e.event_type, e.event_type)
        if e.detail:
            label += f" ({e.detail})"
        rows.append((e.occurred_at, True, label))
    rows.sort(key=lambda r: r[0] or "")
    return rows


def _timeline_table(checks: list[MonitoringCheck], events: list[MonitoringEvent], small) -> list:
    rows = _timeline_rows(checks, events)
    if not rows:
        return [Paragraph("No activity was recorded during this session.", small)]

    table_rows = [["Time", "Event"]]
    style_commands = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for i, (ts, flagged, label) in enumerate(rows, start=1):
        time_label = ts.split("T")[-1].split(".")[0] if ts and "T" in ts else (ts or "")
        table_rows.append([time_label, Paragraph(label, small)])
        if flagged:
            style_commands.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fef2f2")))
            style_commands.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#dc2626")))

    table = Table(table_rows, colWidths=[2.5 * cm, 13.5 * cm])
    table.setStyle(TableStyle(style_commands))
    return [table]


def generate_session_report(
    session: MonitoringSession,
    checks: list[MonitoringCheck],
    events: list[MonitoringEvent],
) -> str:
    os.makedirs(settings.REPORT_DIR, exist_ok=True)
    path = os.path.join(settings.REPORT_DIR, f"{session.session_id}_session_report.pdf")

    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TLTitle", parent=styles["Title"], textColor=colors.HexColor("#0F6B66"))
    small = ParagraphStyle("sess_small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    flagged_checks = sum(1 for c in checks if c.flagged)

    elements = [
        Paragraph("TruthLens Monitoring Session Report", title_style),
        Spacer(1, 0.5 * cm),
    ]

    meta_table = Table(
        [
            ["Session ID", session.session_id],
            ["Started At", session.started_at],
            ["Ended At", session.ended_at or "Still active"],
            ["Duration", _duration_label(session)],
            ["Total Checks", str(len(checks))],
            ["Flagged Checks", f"{flagged_checks} / {len(checks)}"],
            ["Behavioral Events", str(len(events))],
        ],
        colWidths=[5 * cm, 10 * cm],
    )
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [meta_table, Spacer(1, 0.8 * cm)]

    try:
        elements.append(Paragraph("Flag Breakdown", styles["Heading3"]))
        elements += _flag_breakdown_table(checks, small)
        elements.append(Spacer(1, 0.6 * cm))
    except Exception:
        logger.exception("Flag breakdown section skipped (session_id=%s)", session.session_id)

    try:
        elements.append(Paragraph("Behavioral Events", styles["Heading3"]))
        elements += _event_summary_table(events, small)
        elements.append(Spacer(1, 0.6 * cm))
    except Exception:
        logger.exception("Event summary section skipped (session_id=%s)", session.session_id)

    try:
        elements.append(Paragraph("Timeline", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * cm))
        elements += _timeline_table(checks, events, small)
    except Exception:
        logger.exception("Timeline section skipped (session_id=%s)", session.session_id)

    elements += [
        Spacer(1, 1 * cm),
        Paragraph(
            "TruthLens Model Information",
            ParagraphStyle("stubTitle", parent=styles["Normal"], fontSize=9,
                           textColor=colors.HexColor("#B45309"), spaceAfter=4),
        ),
        Paragraph(
            "Camera checks compare a live-captured frame against the enrolled reference photo "
            "(identity match), run the same ConvNeXt-Tiny + CLIP deepfake ensemble used "
            "elsewhere in TruthLens, and count faces in frame. Behavioral events (tab/window "
            "focus, clipboard, devtools, camera interruption) only ever see activity inside the "
            "browser tab this session ran in -- a second device, a different browser, or "
            "activity in a separate tab are invisible to them. Nothing here is proof of "
            "misconduct on its own; it is a record for human review.",
            ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
        ),
    ]

    doc.build(elements)
    return path
