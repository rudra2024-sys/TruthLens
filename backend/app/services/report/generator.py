import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.core.config import settings
from app.models.models import Upload, DetectionResult


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

        if video_analysis.xception_score is not None:
            elements.append(
                Paragraph(
                    f"Structure Entropy Score: "
                    f"{video_analysis.xception_score * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if video_analysis.face_voice_sync is not None:
            elements.append(
                Paragraph(
                    f"Byte Consistency Score: "
                    f"{video_analysis.face_voice_sync * 100:.1f}%",
                    styles["Normal"],
                )
            )

        if video_analysis.frames_analyzed is not None:
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
            "ConvNeXt-Tiny probabilities are reported when "
            "available. Legacy heuristic scores are included "
            "only when present.",
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