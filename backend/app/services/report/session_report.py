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
from app.services.report.pdf_theme import (
    footer_block,
    info_card,
    report_header,
    section_heading,
    tl_styles,
    verdict_card,
)

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
    "looking_away": "Looking away",
    "talking_detected": "Talking detected",
    "object_detected": "Unauthorized object detected",
}

# How much a check's mouth-corner width has to change (as a fraction of the session's own
# average) to be called "changed" rather than "flat" in the lip-sync section below -- a
# placeholder, same honesty tier as every other threshold introduced in this feature.
_MOUTH_WIDTH_CHANGE_FRACTION = 0.08


def _top_reason(checks: list[MonitoringCheck]) -> str | None:
    counts: Counter = Counter()
    for check in checks:
        for reason in check.flag_reasons:
            counts[reason] += 1
    if not counts:
        return None
    reason, _ = counts.most_common(1)[0]
    return REASON_LABELS.get(reason, reason)


def _plain_summary_sentence(session: MonitoringSession, checks: list[MonitoringCheck], events: list[MonitoringEvent]) -> str:
    """One jargon-free sentence restating what this session's numbers mean, for a reader (e.g.
    an exam proctor) who isn't going to parse a flag-breakdown table -- a plain restatement of
    figures computed elsewhere in this report, never a new claim (see pdf_theme.py)."""
    flagged = sum(1 for c in checks if c.flagged)
    total = len(checks)
    in_progress = session.ended_at is None
    if total == 0:
        return "No automated checks have been recorded for this session yet."
    if flagged == 0 and not events:
        state = "so far" if in_progress else "during this session"
        return f"Every automated check came back clean {state} -- nothing was flagged."
    top = _top_reason(checks)
    bits = [f"{flagged} of {total} automated checks were flagged"]
    if events:
        was_were = "was" if len(events) == 1 else "were"
        plural_s = "" if len(events) == 1 else "s"
        bits.append(f"{len(events)} behavioral event{plural_s} {was_were} recorded "
                     "(such as switching tabs or opening developer tools)")
    sentence = " and ".join(bits) + "."
    if in_progress:
        sentence += " This session is still in progress."
    if top:
        sentence += f" The most common reason was “{top}”."
    return sentence


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


def _lip_sync_section(checks: list[MonitoringCheck], small) -> list:
    """Informational only -- never a flag (see monitoring.py's module docstring for why:
    MTCNN's 5-point landmarks give mouth-corner WIDTH, not true vertical mouth-aperture, a much
    weaker proxy for talking than real lip-sync work would use). Shown here, for a human to
    weigh, as a table of consecutive checks that both have speech_ratio and mouth_width_px:
    whether the mouth width changed meaningfully between them while speech was detected, or
    stayed flat -- a flat mouth width during detected speech is the pattern worth a human's
    attention (e.g. a pre-recorded audio clip playing with no one visibly talking), not
    something this code decides on its own.
    """
    usable = [c for c in checks if c.speech_ratio is not None and c.mouth_width_px is not None]
    caveat = Paragraph(
        "Informational only, not a flag. Mouth-corner WIDTH is a weak proxy for talking -- "
        "MTCNN's landmarks have no top/bottom-lip point for true mouth-opening measurement. "
        "Not validated against real talking footage. A human should judge this, not the system.",
        small,
    )
    if len(usable) < 2:
        return [caveat, Spacer(1, 0.15 * cm),
                Paragraph("Not enough checks with both audio and a detected face to compare.", small)]

    avg_width = sum(c.mouth_width_px for c in usable) / len(usable)
    rows = [["Time", "Speech Ratio", "Mouth Width (px)", "Change"]]
    style_commands = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(usable)):
        prev, cur = usable[i - 1], usable[i]
        delta = cur.mouth_width_px - prev.mouth_width_px
        changed = abs(delta) >= _MOUTH_WIDTH_CHANGE_FRACTION * avg_width
        note = "changed" if changed else "flat"
        if cur.speech_ratio >= settings.TALKING_SPEECH_RATIO_THRESHOLD and not changed:
            note = "flat while speech detected"
        time_label = cur.checked_at.split("T")[-1].split(".")[0] if "T" in cur.checked_at else cur.checked_at
        rows.append([time_label, f"{cur.speech_ratio:.2f}", f"{cur.mouth_width_px:.1f}", note])

    table = Table(rows, colWidths=[2.5 * cm, 3 * cm, 3.5 * cm, 7 * cm])
    table.setStyle(TableStyle(style_commands))
    return [caveat, Spacer(1, 0.2 * cm), table]


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
    tl = tl_styles()
    small = ParagraphStyle("sess_small", parent=styles["Normal"], fontSize=9, textColor=colors.grey)

    flagged_checks = sum(1 for c in checks if c.flagged)
    status_label = "CLEAN" if flagged_checks == 0 and not events else "FLAGGED"
    status_headline = "No flagged activity." if status_label == "CLEAN" else "Flagged activity found -- for human review."

    elements = report_header(
        tl, "Monitoring Session Report",
        f"Session: {session.session_id} &nbsp;&middot;&nbsp; Started {session.started_at}",
    )

    elements += verdict_card(
        tl, status_label, status_headline,
        _plain_summary_sentence(session, checks, events) +
        "<br/><br/>Nothing on this page is proof of misconduct on its own -- it is a record for a human to review.",
    )

    elements += info_card([
        ("Started At", session.started_at),
        ("Ended At", session.ended_at or "Still active"),
        ("Duration", _duration_label(session)),
        ("Total Checks", str(len(checks))),
        ("Flagged Checks", f"{flagged_checks} / {len(checks)}"),
        ("Behavioral Events", str(len(events))),
    ])
    elements.append(Spacer(1, 0.6 * cm))

    try:
        elements += section_heading(tl, "Flag Breakdown")
        elements += _flag_breakdown_table(checks, small)
        elements.append(Spacer(1, 0.6 * cm))
    except Exception:
        logger.exception("Flag breakdown section skipped (session_id=%s)", session.session_id)

    try:
        elements += section_heading(tl, "Behavioral Events")
        elements += _event_summary_table(events, small)
        elements.append(Spacer(1, 0.6 * cm))
    except Exception:
        logger.exception("Event summary section skipped (session_id=%s)", session.session_id)

    try:
        elements += section_heading(tl, "Lip-Sync Signal (informational, not a flag)")
        elements += _lip_sync_section(checks, small)
        elements.append(Spacer(1, 0.6 * cm))
    except Exception:
        logger.exception("Lip-sync section skipped (session_id=%s)", session.session_id)

    try:
        elements += section_heading(tl, "Timeline")
        elements += _timeline_table(checks, events, small)
    except Exception:
        logger.exception("Timeline section skipped (session_id=%s)", session.session_id)

    elements += footer_block(
        tl, "TruthLens Model Information",
        "Camera checks compare a live-captured frame against the enrolled reference photo "
        "(identity match), run the same ConvNeXt-Tiny + CLIP deepfake ensemble used "
        "elsewhere in TruthLens, and count faces in frame. Behavioral events (tab/window "
        "focus, clipboard, devtools, camera interruption) only ever see activity inside the "
        "browser tab this session ran in -- a second device, a different browser, or "
        "activity in a separate tab are invisible to them. Nothing here is proof of "
        "misconduct on its own; it is a record for human review.",
    )

    doc.build(elements)
    return path
