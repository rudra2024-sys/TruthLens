import io
import logging
import os

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.core.config import settings
from app.models.models import Upload, DetectionResult

logger = logging.getLogger(__name__)


def _rl_image(raw: bytes, width_cm: float, max_height_cm: float | None = None):
    """ReportLab Image from JPEG bytes, scaled to a target width keeping the aspect ratio (and shrunk further
    if needed so its height never exceeds max_height_cm - tall photos must not force a page break)."""
    from PIL import Image as PILImage

    with PILImage.open(io.BytesIO(raw)) as im:
        w, h = im.size
    width, height = width_cm * cm, width_cm * cm * h / w
    if max_height_cm is not None and height > max_height_cm * cm:
        width, height = width * (max_height_cm * cm) / height, max_height_cm * cm
    return RLImage(io.BytesIO(raw), width=width, height=height)


def _frame_timeline(video, width_cm: float = 16.0, height_cm: float = 5.0) -> Drawing:
    """Bar chart of the 16 per-frame FAKE probabilities with the decision threshold marked. Vector drawing."""
    W, H = width_cm * cm, height_cm * cm
    left, bottom, top_pad = 1.2 * cm, 1.0 * cm, 0.8 * cm
    plot_w, plot_h = W - left - 0.3 * cm, H - bottom - top_pad
    d = Drawing(W, H)
    d.add(Rect(left, bottom, plot_w, plot_h, strokeColor=colors.HexColor("#cbd5e1"), fillColor=None))
    n = len(video.frames)
    slot = plot_w / n
    for i, f in enumerate(video.frames):
        bar_h = max(0.5, f.probability * plot_h)
        above = f.probability >= video.threshold
        d.add(Rect(left + i * slot + slot * 0.15, bottom, slot * 0.7, bar_h, strokeColor=None,
                   fillColor=colors.HexColor("#dc2626" if above else "#0F6B66")))
        label = f"{f.timestamp_s:.1f}s" if f.timestamp_s is not None else str(f.frame_index)
        d.add(String(left + i * slot + slot / 2, bottom - 0.45 * cm, label, fontSize=6, textAnchor="middle",
                     fillColor=colors.HexColor("#475569")))
    y_thr = bottom + video.threshold * plot_h
    d.add(Line(left, y_thr, left + plot_w, y_thr, strokeColor=colors.HexColor("#d97706"), strokeWidth=1,
               strokeDashArray=[3, 2]))
    d.add(String(left + plot_w, y_thr + 2, f"threshold {video.threshold:.3f}", fontSize=6, textAnchor="end",
                 fillColor=colors.HexColor("#d97706")))
    for tick in (0.0, 0.5, 1.0):
        d.add(String(left - 4, bottom + tick * plot_h - 2, f"{tick:.1f}", fontSize=6, textAnchor="end",
                     fillColor=colors.HexColor("#475569")))
    d.add(String(left, H - 9, "P(fake) per frame", fontSize=7, fillColor=colors.HexColor("#334155")))
    return d


def _explainability_flowables(upload: Upload, result: DetectionResult, styles) -> list:
    """Explainability section (heatmap / per-frame timeline). Returns [] when nothing can be shown."""
    from app.services.explain.service import build_explanation

    small = ParagraphStyle("xpl_small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ex = build_explanation(upload, result)
    if not ex.available:
        return []
    out = [Spacer(1, 0.6 * cm), Paragraph("Explainability", styles["Heading3"]),
           Paragraph(f"Method: {ex.method}", small), Spacer(1, 0.2 * cm)]

    if ex.image is not None:
        img = ex.image
        pair = Table(
            [[_rl_image(img.original_jpeg, 7.6, 9.5), _rl_image(img.overlay_jpeg, 7.6, 9.5)],
             [Paragraph("Submitted image", small),
              Paragraph(f"Grad-CAM (ConvNeXt-Tiny FAKE probability {img.fake_probability * 100:.1f}%)", small)]],
            colWidths=[8 * cm, 8 * cm],
        )
        pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        out.append(KeepTogether([pair]))
        if ex.verdict_driver:
            out.append(Paragraph(f"Sub-model driving the deployed score: {ex.verdict_driver}", small))
    if ex.video is not None:
        v = ex.video
        out.append(_frame_timeline(v))
        out.append(Paragraph(
            f"Mean-logit probability {v.probability * 100:.1f}% (decision threshold {v.threshold:.3f}); "
            f"{v.frames_above_threshold} of {len(v.frames)} frames individually above the threshold.", small))
        out.append(Spacer(1, 0.2 * cm))
        cells = [_rl_image(f.crop_jpeg, 1.85) for f in v.frames]
        caps = [Paragraph(f"{f.probability * 100:.0f}%", small) for f in v.frames]
        per_row = 8
        rows = []
        for i in range(0, len(cells), per_row):
            rows.append(cells[i:i + per_row])
            rows.append(caps[i:i + per_row])
        strip = Table(rows, colWidths=[2 * cm] * per_row)
        strip.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("FONTSIZE", (0, 0), (-1, -1), 6)]))
        out.append(KeepTogether([Paragraph("Frames as seen by the model", small), strip]))
        hot = [f for f in v.frames if f.heatmap_jpeg]
        if hot:
            hot.sort(key=lambda f: -f.probability)
            row = [[_rl_image(f.heatmap_jpeg, 5.0) for f in hot],
                   [Paragraph(f"frame {f.order + 1}: {f.probability * 100:.0f}% fake", small) for f in hot]]
            t = Table(row, colWidths=[5.4 * cm] * len(hot))
            out.append(KeepTogether([Spacer(1, 0.2 * cm),
                                     Paragraph("Most suspicious frames (Grad-CAM)", small), t]))
    out.append(Spacer(1, 0.2 * cm))
    out.append(Paragraph(ex.note or "", small))
    return out


def _provenance_flowables(upload: Upload, result: DetectionResult, styles) -> list:
    """Provenance & metadata section (C2PA credentials, embedded metadata, ELA visual aid)."""
    from app.services.provenance.service import build_provenance

    small = ParagraphStyle("prov_small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    body = ParagraphStyle("prov_body", parent=styles["Normal"], fontSize=9)
    pv = build_provenance(upload, result)
    if not pv.available or pv.assessment is None:
        return []

    tone = {"strong": "#dc2626", "moderate": "#d97706", "weak": "#64748b", "none": "#64748b"}
    out = [Spacer(1, 0.6 * cm), Paragraph("Provenance &amp; Metadata", styles["Heading3"]),
           Paragraph(pv.assessment.headline, body), Spacer(1, 0.15 * cm)]

    if pv.assessment.conflict_note:
        box = Table([[Paragraph("<b>Provenance conflicts with the model verdict.</b> "
                                + pv.assessment.conflict_note, body)]], colWidths=[16 * cm])
        box.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#d97706")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        out += [box, Spacer(1, 0.2 * cm)]

    for sig in pv.signals:
        colour = tone.get(sig.strength, "#64748b")
        out.append(Paragraph(f"<font color='{colour}'><b>{sig.title}</b></font>", body))
        if sig.detail:
            out.append(Paragraph(sig.detail.replace("&", "&amp;").replace("<", "&lt;"), small))
        out.append(Spacer(1, 0.1 * cm))

    if pv.container:
        c = pv.container
        bits = [f"{c.get('width')}x{c.get('height')} px"]
        if c.get("fps"):
            bits.append(f"{c['fps']:.1f} fps")
        if c.get("duration_s"):
            bits.append(f"{c['duration_s']:.1f} s")
        if c.get("encoder_tag"):
            bits.append(f"encoder tag: {c['encoder_tag']}")
        out.append(Paragraph("Container: " + ", ".join(bits), small))

    if pv.ela is not None and pv.ela.applicable and pv.ela.heatmap_jpeg:
        out += [Spacer(1, 0.2 * cm),
                Paragraph("Error Level Analysis (visual aid only)", body),
                _rl_image(pv.ela.heatmap_jpeg, 7.0, 7.5),
                Paragraph(f"Mean recompression error {pv.ela.mean_error:.2f}, 95th percentile {pv.ela.p95_error:.2f}. "
                          "ELA is not a detector: resizing, screenshots and normal image texture all change it.",
                          small)]
    out += [Spacer(1, 0.15 * cm), Paragraph(pv.caveat, small)]
    return out


def generate_pdf_report(upload: Upload, result: DetectionResult) -> str:
    os.makedirs(settings.REPORT_DIR, exist_ok=True)

    path = os.path.join(
        settings.REPORT_DIR,
        f"{upload.upload_id}_report.pdf",
    )

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TLTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0F6B66"),
    )

    verdict_hex = {
        "FAKE": "#dc2626",
        "REAL": "#16a34a",
        "UNCERTAIN": "#d97706",
    }.get(result.verdict, "#000000")

    elements = []

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            "TruthLens Forensic Detection Report",
            title_style,
        )
    )

    elements.append(Spacer(1, 0.5 * cm))

    # ---------------------------------------------------------
    # FILE / SCAN INFORMATION
    # ---------------------------------------------------------

    meta_table = Table(
        [
            ["File Name", upload.file_name],
            ["Media Type", upload.media_type.upper()],
            ["File Size", f"{upload.file_size_kb:.1f} KB"],
            ["Scanned At", result.detected_at],
            ["Model Used", result.model_used],
            ["Processing Time", f"{result.processing_time_ms:.0f} ms"],
        ],
        colWidths=[5 * cm, 10 * cm],
    )

    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(meta_table)

    elements.append(Spacer(1, 0.8 * cm))

    # ---------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------

    elements.append(
        Paragraph(
            f"Verdict: "
            f"<font color='{verdict_hex}'>{result.verdict}</font>",
            styles["Heading1"],
        )
    )

    elements.append(
        Paragraph(
            f"Confidence Score: "
            f"{result.confidence_score * 100:.1f}%",
            styles["Heading3"],
        )
    )

    elements.append(Spacer(1, 0.5 * cm))

    # ---------------------------------------------------------
    # IMAGE ANALYSIS
    # ---------------------------------------------------------

    if result.image_analysis:

        elements.append(
            Paragraph(
                "Image Pipeline Breakdown",
                styles["Heading3"],
            )
        )

        image_analysis = result.image_analysis

        # New ConvNeXt pipeline
        if (
            image_analysis.fake_probability is not None
            or image_analysis.real_probability is not None
        ):

            if image_analysis.fake_probability is not None:
                elements.append(
                    Paragraph(
                        f"FAKE Probability: "
                        f"{image_analysis.fake_probability * 100:.1f}%",
                        styles["Normal"],
                    )
                )

            if image_analysis.real_probability is not None:
                elements.append(
                    Paragraph(
                        f"REAL Probability: "
                        f"{image_analysis.real_probability * 100:.1f}%",
                        styles["Normal"],
                    )
                )

        # Ensemble sub-model scores (ConvNeXt-Tiny + CLIP second opinion),
        # shown separately from the combined FAKE/REAL Probability above so
        # the two models' individual disagreement stays visible.
        if image_analysis.convnext_fake_probability is not None:
            elements.append(
                Paragraph(
                    f"ConvNeXt-Tiny Sub-score (FAKE probability): "
                    f"{image_analysis.convnext_fake_probability * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if image_analysis.clip_fake_probability is not None:
            elements.append(
                Paragraph(
                    f"CLIP Second-Opinion Sub-score (FAKE probability): "
                    f"{image_analysis.clip_fake_probability * 100:.1f}%",
                    styles["Normal"],
                )
            )

        # Legacy heuristic pipeline
        if image_analysis.efficientnet_score is not None:
            elements.append(
                Paragraph(
                    f"Noise Residual Score: "
                    f"{image_analysis.efficientnet_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if image_analysis.fft_score is not None:
            elements.append(
                Paragraph(
                    f"FFT Frequency Score: "
                    f"{image_analysis.fft_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

    # ---------------------------------------------------------
    # VIDEO ANALYSIS
    # ---------------------------------------------------------

    if result.video_analysis:

        elements.append(
            Paragraph(
                "Video Pipeline Breakdown",
                styles["Heading3"],
            )
        )

        video_analysis = result.video_analysis

        # DB columns keep their legacy names (xception_score / frames_analyzed). Under Video Model v1 they hold
        # the model's FAKE probability and the number of frames scored, not byte-entropy heuristics.
        if "Video Model v1" in (result.model_used or ""):
            if video_analysis.xception_score is not None:
                elements.append(
                    Paragraph(
                        f"Video Model v1 FAKE Probability (sigmoid of mean frame logit): "
                        f"{video_analysis.xception_score * 100:.1f}%",
                        styles["Normal"],
                    )
                )
            if video_analysis.frames_analyzed is not None:
                elements.append(
                    Paragraph(
                        f"Frames Analyzed: {video_analysis.frames_analyzed}",
                        styles["Normal"],
                    )
                )
        elif video_analysis.xception_score is not None:
            elements.append(
                Paragraph(
                    f"Structure Entropy Score: "
                    f"{video_analysis.xception_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if "Video Model v1" not in (result.model_used or "") and video_analysis.face_voice_sync is not None:
            elements.append(
                Paragraph(
                    f"Byte Consistency Score: "
                    f"{video_analysis.face_voice_sync * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if "Video Model v1" not in (result.model_used or "") and video_analysis.frames_analyzed is not None:
            elements.append(
                Paragraph(
                    f"Byte Windows Sampled: "
                    f"{video_analysis.frames_analyzed}",
                    styles["Normal"],
                )
            )

    # ---------------------------------------------------------
    # AUDIO ANALYSIS
    # ---------------------------------------------------------

    if result.audio_analysis:

        elements.append(
            Paragraph(
                "Audio Pipeline Breakdown",
                styles["Heading3"],
            )
        )

        audio_analysis = result.audio_analysis

        # Note: these DB fields are named wav2vec_score/lcnn_score for legacy
        # reasons; they hold mean/std spoof probability from whichever audio
        # backend is active (see backend/app/services/audio/backend.py). Labels
        # below describe what was actually computed, not the field names or any
        # specific model name, so this stays accurate across backend swaps.
        if audio_analysis.wav2vec_score is not None:
            elements.append(
                Paragraph(
                    f"Audio Spoof Probability: "
                    f"{audio_analysis.wav2vec_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if audio_analysis.lcnn_score is not None:
            elements.append(
                Paragraph(
                    f"Audio Score Variability (std. across windows): "
                    f"{audio_analysis.lcnn_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if audio_analysis.duration_s is not None:
            elements.append(
                Paragraph(
                    f"Audio Duration Analyzed: {audio_analysis.duration_s:.1f}s",
                    styles["Normal"],
                )
            )

        if audio_analysis.windows_analyzed is not None:
            elements.append(
                Paragraph(
                    f"Windows Analyzed: {audio_analysis.windows_analyzed} "
                    f"(the file is split into fixed-length windows; the scores above "
                    f"are the mean/std of the spoof probability across all of them)",
                    styles["Normal"],
                )
            )

    # ---------------------------------------------------------
    # EXPLAINABILITY (image heatmap / video per-frame timeline)
    # Best-effort: if anything fails the rest of the report is unaffected.
    # ---------------------------------------------------------

    try:
        elements.extend(_explainability_flowables(upload, result, styles))
    except Exception:
        logger.exception("Explainability section skipped (upload_id=%s)", upload.upload_id)

    # ---------------------------------------------------------
    # PROVENANCE & METADATA (best-effort, supplementary; never alters the verdict)
    # ---------------------------------------------------------

    try:
        elements.extend(_provenance_flowables(upload, result, styles))
    except Exception:
        logger.exception("Provenance section skipped (upload_id=%s)", upload.upload_id)

    # ---------------------------------------------------------
    # FOOTER / DISCLAIMER
    # ---------------------------------------------------------

    elements.append(
        Spacer(1, 1 * cm)
    )

    elements.append(
        Paragraph(
            "TruthLens Model Information",
            ParagraphStyle(
                "stubTitle",
                parent=styles["Normal"],
                fontSize=9,
                textColor=colors.HexColor("#B45309"),
                spaceAfter=4,
            ),
        )
    )

    elements.append(
        Paragraph(
            "This report contains the output produced by the "
            "TruthLens detection pipeline. For image analysis, "
            "the FAKE/REAL Probability is the higher of the FAKE "
            "probabilities reported by ConvNeXt-Tiny and a CLIP-based "
            "second opinion, reported individually as sub-scores when "
            "available. Legacy heuristic scores are included only when "
            "present.",
            ParagraphStyle(
                "footer",
                parent=styles["Normal"],
                fontSize=8,
                textColor=colors.grey,
            ),
        )
    )

    # ---------------------------------------------------------
    # BUILD PDF
    # ---------------------------------------------------------

    doc.build(elements)

    return path