"""Plain-English wording for the single-file PDF report (added 2026-10-21, at the user's request: the report was
"very techy" for people who are not).

Everything here restates numbers that were already computed - it never produces a verdict, score or claim of its own
(CLAUDE.md section 8 rule 5). The wording is deliberately hedged: "probably", "the tool thinks". Kept separate from
generator.py so the text can be tested without building a PDF.
"""

from __future__ import annotations

from datetime import datetime

from app.models.models import DetectionResult, Upload

SMALL_IMAGE_PX = 300        # same judgement-call cut-off as the web page (frontend SmallImageNote.jsx)

_KIND = {"image": "image", "video": "video", "audio": "audio clip"}


def friendly_date(iso: str | None) -> str:
    """'2026-10-20T19:39:51.399052' -> '20 October 2026, 7:39 PM (UTC)'. Falls back to the raw text."""
    if not iso:
        return "unknown"
    try:
        d = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    return f"{d.day} {d:%B %Y}, {d.hour % 12 or 12}:{d:%M %p} (UTC)"


def verdict_label(verdict: str) -> str:
    return {"FAKE": "Manipulated (FAKE)", "REAL": "Authentic (REAL)", "UNCERTAIN": "Uncertain"}.get(verdict, verdict)


def image_fake_kind(result: DetectionResult) -> str | None:
    """'face swap' or 'AI-generated' for a flagged image, from the stored component scores (same rule as the web page's
    getImageFakeHint): the face-swap check wins only when its score is the highest of the image's components."""
    a = result.image_analysis
    if a is None or result.verdict != "FAKE":
        return None
    face = getattr(a, "face_swap_fake_probability", None)
    whole = max(a.convnext_fake_probability or 0.0, a.clip_fake_probability or 0.0)
    return "face swap" if face is not None and face >= whole else "AI-generated"


def summary_sentence(upload: Upload, result: DetectionResult) -> str:
    kind = _KIND.get(upload.media_type, "file")
    if result.verdict == "FAKE":
        extra = ""
        what = image_fake_kind(result)
        if what == "face swap":
            extra = " Most likely a face swap."
        elif what == "AI-generated":
            extra = " Most likely AI-generated."
        return f"The tool thinks this {kind} was probably made or edited by AI.{extra}"
    if result.verdict == "REAL":
        return f"The tool did not find signs that this {kind} was made or edited by AI."
    return f"The tool could not tell whether this {kind} is genuine or made by AI."


def image_size(upload: Upload) -> tuple[int, int] | None:
    if upload.media_type != "image":
        return None
    try:
        from PIL import Image
        with Image.open(upload.storage_url) as im:
            return im.size
    except Exception:
        return None


def plain_points(upload: Upload, result: DetectionResult) -> list[tuple[str, str]]:
    """The 'In plain words' section: (heading, text) pairs, most important first."""
    kind = _KIND.get(upload.media_type, "file")
    pts: list[tuple[str, str]] = []

    own = f"{result.confidence_score * 100:.0f}%"
    va = result.video_analysis
    video_scaled = va is not None and getattr(va, "calibrated_fake_probability", None) is not None
    if video_scaled:
        pts.append(("How sure is it?",
                    f"The tool's confidence is {own}. For videos this figure has already been adjusted using how the "
                    "tool performed on test videos where the answer was known. It is an estimate, not a guarantee."))
    elif result.calibrated_confidence is not None:
        cal = f"{result.calibrated_confidence * 100:.0f}%"
        if abs(result.calibrated_confidence - result.confidence_score) < 0.03:
            pts.append(("How sure is it?",
                        f"The tool is about {own} sure. When we tested it on files where we knew the answer, results "
                        "this sure were right about as often, so the figure is a fair guide. It is not a guarantee."))
        else:
            pts.append(("How sure is it?",
                        f"The tool says it is {own} sure. But when we tested it on files where we knew the answer, "
                        f"results that sure were right roughly {cal} of the time, so {cal} is the more realistic "
                        "figure. Neither number is a guarantee."))
    else:
        pts.append(("How sure is it?",
                    f"The tool says it is {own} sure. This is the tool's own estimate, not a guarantee, and we have "
                    f"not measured how often this figure matches reality for {kind}s of this type."))

    care: list[str] = []
    size = image_size(upload)
    if size and min(size) < SMALL_IMAGE_PX:
        care.append(f"This image is small ({size[0]} x {size[1]} pixels). Small images carry little detail, and in our "
                    "own tests genuine photos this small were sometimes flagged. Be extra careful with this result.")
    if va is not None and getattr(va, "low_resolution", None):
        care.append("This video is too low in resolution for the tool to judge reliably, so the result is shown as "
                    "uncertain whatever the score.")
    if va is not None and va.frames_with_face is not None and va.frames_analyzed \
            and va.frames_with_face < va.frames_analyzed / 2:
        care.append("The tool could find a clear face in fewer than half of the video frames it looked at, so its "
                    "result rests on less evidence than usual.")
    if upload.media_type == "image" and result.verdict != "FAKE":
        care.append("The tool can miss images from newer AI image generators it was not trained on, so a result of "
                    "authentic or uncertain does not prove an image is genuine.")
    if upload.media_type == "video":
        care.append("The video check is built to spot face swaps and lip-sync fakes. Videos created entirely by AI "
                    "(for example by Sora or Kling) are outside what it was trained for and can be reported as "
                    "authentic.")
    if upload.media_type == "audio":
        care.append("We have not yet tested how well the audio check holds up on compressed, noisy or re-shared "
                    "recordings, so treat results on such files with extra care.")
    if care:
        pts.append(("Things to keep in mind", "\n".join(care)))

    if result.verdict == "FAKE":
        nxt = ("Do not treat this file as genuine, and do not share it as fact until you have checked where it came "
               "from. Look for the original source, and compare it with trusted versions if they exist.")
    elif result.verdict == "REAL":
        nxt = ("If the file matters for something important (news, legal or school decisions), still confirm where it "
               "came from. A tool result is one piece of evidence, not proof.")
    else:
        nxt = ("Try a clearer or larger copy of the file if you can, and check where it came from. Until then, treat "
               "it as unconfirmed.")
    pts.append(("What you can do next", nxt))
    return pts


_GLOSSARY = {
    "Confidence": "How sure the tool says it is about its answer. It is the tool's own estimate and can be wrong.",
    "Estimated reliability": "The same score adjusted using how often the tool was actually right in our own tests. "
                             "Usually the more realistic of the two numbers.",
    "Uncertain": "The tool's signals were too mixed to call the file genuine or manipulated. It is not a 'maybe fake'; "
                 "it means the tool could not tell.",
    "Heatmap (Grad-CAM)": "A coloured overlay showing which parts of the picture the tool's score depended on most. It "
                          "explains the tool's behaviour; it is not proof that those parts were edited.",
    "Face-swap check": "A separate check that looks only at the face in a picture, because face swaps can fool a "
                       "whole-image check.",
    "Frames": "Still pictures taken from a video. The tool looks at 16 of them and averages what it sees.",
    "Windows": "Short slices of an audio clip. The tool scores each slice and averages the results.",
    "Spectrogram": "A picture of a sound, showing how its pitch and loudness change over time.",
    "Provenance and metadata": "Information stored inside a file, such as the camera, editing software or notes from an "
                               "AI tool. It can be removed or faked, so missing information says nothing either way.",
    "Content Credentials (C2PA)": "A digital signature some cameras and AI tools attach to say where a file came from "
                                  "and whether it has been changed since.",
    "Error Level Analysis (ELA)": "A picture that shows how differently parts of an image respond to being re-saved. "
                                  "It is a visual hint only, not a detector.",
    "Invisible watermark check": "Looks for a hidden pattern that some AI image tools add. Finding one is strong "
                                 "evidence; not finding one proves nothing.",
}

_GLOSSARY_BY_MEDIA = {
    "image": ["Confidence", "Estimated reliability", "Uncertain", "Heatmap (Grad-CAM)", "Face-swap check",
              "Provenance and metadata", "Content Credentials (C2PA)", "Error Level Analysis (ELA)",
              "Invisible watermark check"],
    "video": ["Confidence", "Estimated reliability", "Uncertain", "Frames", "Heatmap (Grad-CAM)",
              "Provenance and metadata", "Content Credentials (C2PA)", "Invisible watermark check"],
    "audio": ["Confidence", "Uncertain", "Windows", "Spectrogram", "Provenance and metadata",
              "Content Credentials (C2PA)"],
}


_PROVENANCE_TERMS = {"Provenance and metadata", "Content Credentials (C2PA)", "Error Level Analysis (ELA)",
                     "Invisible watermark check"}
_EXPLANATION_TERMS = {"Heatmap (Grad-CAM)", "Spectrogram", "Frames", "Windows"}


def glossary_terms(media_type: str, has_provenance: bool = True, has_explanation: bool = True,
                   has_face_swap: bool = True) -> list[tuple[str, str]]:
    """Definitions for the terms the report can contain; sections that are absent from this report (they are
    best-effort and can be skipped) take their terms with them, so the glossary never explains something unseen."""
    names = _GLOSSARY_BY_MEDIA.get(media_type, ["Confidence", "Uncertain"])
    if not has_provenance:
        names = [n for n in names if n not in _PROVENANCE_TERMS]
    if not has_explanation:
        names = [n for n in names if n not in _EXPLANATION_TERMS]
    if not has_face_swap:
        names = [n for n in names if n != "Face-swap check"]
    return [(t, _GLOSSARY[t]) for t in names]
