"""Renders paper_content.py into IEEE conference format:
  - TruthLens_IEEE.tex  (IEEEtran class; compile on Overleaf or with pdflatex)
  - TruthLens_IEEE.docx (IEEE two-column Word layout, US Letter, Times New Roman)

Usage (python-docx must be importable; pass an extra sys.path entry as argv[1] if needed):
  python build_paper.py [extra_site_dir]
"""
import re
import sys
from pathlib import Path

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_content as C  # noqa: E402

# ---------------------------------------------------------------- numbering
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
TOKEN = re.compile(r"\{([a-z0-9:,]+)\}")


def _all_text():
    for b in C.BODY:
        if b[0] in ("p", "h1", "h2"):
            yield b[1]
        elif b[0] == "list":
            yield from b[1]
        elif b[0] == "table":
            for row in b[4]:
                yield from row
        elif b[0] == "fig":
            yield b[3]


cite_order, fig_no, tab_no = [], {}, {}
for b in C.BODY:
    if b[0] == "fig":
        fig_no[b[1]] = len(fig_no) + 1
    elif b[0] == "table":
        tab_no[b[1]] = ROMAN[len(tab_no)]
for text in _all_text():
    for m in TOKEN.finditer(text):
        for k in m.group(1).split(","):
            if not k.startswith(("fig:", "tab:")) and k not in cite_order:
                cite_order.append(k)
missing = [k for k in cite_order if k not in C.REFERENCES]
unused = [k for k in C.REFERENCES if k not in cite_order]
assert not missing, f"cited but not defined: {missing}"
assert not unused, f"defined but never cited: {unused}"
cite_no = {k: i + 1 for i, k in enumerate(cite_order)}


def _compress(nums):
    nums = sorted(nums)
    return ", ".join(f"[{n}]" for n in nums)


# ---------------------------------------------------------------- LaTeX
TEX_CHARS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "~": r"$\sim$",
             "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "σ": r"$\sigma$", "°": r"$^\circ$",
             "ü": r"\"u", "ö": r"\"o"}


def tex(text):
    out, i = [], 0
    for m in TOKEN.finditer(text):
        out.append(_tex_plain(text[i:m.start()]))
        keys = m.group(1).split(",")
        if keys[0].startswith(("fig:", "tab:")):
            out.append(r"\ref{" + keys[0].replace(":", "-") + "}")
        else:
            out.append(r"~\cite{" + ",".join(keys) + "}")
        i = m.end()
    out.append(_tex_plain(text[i:]))
    s = "".join(out).replace(" ~\\cite", "~\\cite")
    return s


def _tex_plain(s):
    s = "".join(TEX_CHARS.get(ch, ch) for ch in s)
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    s = re.sub(r"\"([^\"]+)\"", r"``\1''", s)
    s = re.sub(r"(\d)x(\d)", r"\1$\\times$\2", s)
    s = re.sub(r"(\d)-(\d)", r"\1--\2", s)
    s = s.replace("p_C", "$p_C$").replace("p_L", "$p_L$")
    s = re.sub(r"\bT = ", r"$T=$ ", s)
    s = s.replace("(T > 1)", "($T>1$)").replace("(T < 1)", "($T<1$)").replace("where T is", "where $T$ is")
    s = s.replace("threshold t ", "threshold $t$ ").replace("score p to", "score $p$ to")
    s = s.replace("2e-16", r"$2\times10^{-16}$").replace("4x down", r"4$\times$ down")
    return s


def build_tex():
    L = [r"\documentclass[conference]{IEEEtran}",
         r"\IEEEoverridecommandlockouts",
         r"\usepackage{cite}",
         r"\usepackage{amsmath,amssymb,amsfonts}",
         r"\usepackage{graphicx}",
         r"\usepackage{array}",
         r"\usepackage{textcomp}",
         r"\usepackage{xcolor}",
         r"\usepackage[hidelinks]{hyperref}",
         r"\begin{document}",
         "",
         r"\title{" + C.TITLE + "}",
         "",
         r"\author{"]
    L.append(r"\IEEEauthorblockN{" + ", ".join(C.AUTHORS) + "}")
    L.append(r"\IEEEauthorblockA{\textit{" + C.AFFILIATION[0] + r"} \\ \textit{" + C.AFFILIATION[1]
             + r"} \\ " + C.AFFILIATION[2] + "}")
    L += ["}", "", r"\maketitle", "",
          r"\begin{abstract}", tex(C.ABSTRACT), r"\end{abstract}", "",
          r"\begin{IEEEkeywords}", C.INDEX_TERMS, r"\end{IEEEkeywords}", ""]
    first_para = True
    for b in C.BODY:
        kind = b[0]
        if kind == "h1":
            L += ["", r"\section{" + b[1] + "}"]
        elif kind == "h2":
            L += [r"\subsection{" + b[1] + "}"]
        elif kind == "p":
            t = tex(b[1])
            if first_para:
                # IEEE drop-cap style opening for the first paragraph.
                word, rest = t.split(" ", 1)
                t = r"\IEEEPARstart{" + word[0] + "}{" + word[1:].upper() + "} " + rest
                first_para = False
            L += [t, ""]
        elif kind == "list":
            L += [r"\begin{itemize}"] + [r"\item " + tex(it) for it in b[1]] + [r"\end{itemize}", ""]
        elif kind == "eq":
            L += [r"\begin{equation}", b[1], r"\end{equation}", ""]
        elif kind == "fig":
            _, label, path, cap, width, wide = b
            env = "figure*" if wide else "figure"
            L += [r"\begin{" + env + "}[!t]", r"\centering",
                  r"\includegraphics[width=\columnwidth]{" + path + "}",
                  r"\caption{" + tex(cap) + "}", r"\label{" + label.replace(":", "-") + "}",
                  r"\end{" + env + "}", ""]
        elif kind == "table":
            _, label, cap, header, rows, spec = b
            cols = [float(x) for x in re.findall(r"p\{([\d.]+)\}", spec)]
            colspec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w * 0.92:.2f}" + r"\columnwidth}" for w in cols)
            L += [r"\begin{table}[!t]", r"\renewcommand{\arraystretch}{1.15}",
                  r"\caption{" + tex(cap) + "}", r"\label{" + label.replace(":", "-") + "}",
                  r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                  r"\begin{tabular}{" + colspec + "}", r"\hline",
                  " & ".join(r"\textbf{" + tex(h) + "}" for h in header) + r" \\", r"\hline"]
            L += [" & ".join(tex(c) for c in row) + r" \\" for row in rows]
            L += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    L += [r"\section*{Acknowledgment}", tex(C.ACKNOWLEDGMENT), "",
          r"\begin{thebibliography}{00}"]
    L += [r"\bibitem{" + k + "} " + C.REFERENCES[k] for k in cite_order]
    L += [r"\end{thebibliography}", "", r"\end{document}", ""]
    out = HERE / "TruthLens_IEEE.tex"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


# ---------------------------------------------------------------- Word
def build_docx():
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Inches, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(10)
    pf = st.paragraph_format
    pf.space_before = Pt(0); pf.space_after = Pt(0); pf.line_spacing = 1.0

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
    sec.left_margin = sec.right_margin = Inches(0.625)

    def set_cols(section, n):
        sectPr = section._sectPr
        for c in sectPr.findall(qn("w:cols")):
            sectPr.remove(c)
        cols = OxmlElement("w:cols")
        cols.set(qn("w:num"), str(n))
        cols.set(qn("w:space"), "360")  # 0.25 in
        sectPr.append(cols)

    def para(align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True, before=0, after=0):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.first_line_indent = Inches(0.14) if indent else None
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        return p

    def run(p, text, size=None, bold=False, italic=False, smallcaps=False):
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if smallcaps:
            r.font.small_caps = True
        return r

    def rich(p, text, size=None, bold=False):
        """Adds text with *italic* spans and resolved {cite}/{fig}/{tab} tokens."""
        def resolve(m):
            keys = m.group(1).split(",")
            if keys[0].startswith("fig:"):
                return str(fig_no[keys[0]])
            if keys[0].startswith("tab:"):
                return tab_no[keys[0]]
            return _compress([cite_no[k] for k in keys])
        text = TOKEN.sub(resolve, text)
        for i, part in enumerate(re.split(r"\*([^*]+)\*", text)):
            if not part:
                continue
            for j, piece in enumerate(re.split(r"\bp_([A-Z])\b", part)):
                if j % 2 == 1:
                    run(p, "p", size=size, italic=True)
                    run(p, piece, size=size, italic=True).font.subscript = True
                elif piece:
                    run(p, piece, size=size, bold=bold, italic=(i % 2 == 1))

    # Title + authors (single column)
    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=12)
    run(p, C.TITLE, size=24)
    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=2)
    run(p, ", ".join(C.AUTHORS), size=11)
    for i, line in enumerate(C.AFFILIATION):
        q = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False)
        run(q, line, size=10, italic=(i < 2))
    para(indent=False, after=10)

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(body, 2)

    p = para(indent=True, after=6)
    run(p, "Abstract", size=9, bold=True, italic=True)
    run(p, "\u2014" + C.ABSTRACT, size=9, bold=True)
    p = para(indent=True, after=6)
    run(p, "Index Terms", size=9, bold=True, italic=True)
    run(p, "\u2014" + C.INDEX_TERMS, size=9, bold=True)

    h1 = h2 = 0
    eq_no = 0
    first_para = True
    for b in C.BODY:
        kind = b[0]
        if kind == "h1":
            h1 += 1; h2 = 0
            p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=8, after=4)
            run(p, f"{ROMAN[h1 - 1]}. ", smallcaps=True)
            run(p, b[1], smallcaps=True)
            p.paragraph_format.keep_with_next = True
        elif kind == "h2":
            h2 += 1
            p = para(WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=4, after=2)
            run(p, f"{chr(64 + h2)}. {b[1]}", italic=True)
            p.paragraph_format.keep_with_next = True
        elif kind == "p":
            p = para()
            text = b[1]
            if first_para:
                # Drop-cap substitute: enlarged first letter + small-caps rest of first word.
                word, text = text.split(" ", 1)
                run(p, word[0], size=20)
                run(p, word[1:].upper() + " ")
                p.paragraph_format.first_line_indent = None
                first_para = False
            rich(p, text)
        elif kind == "list":
            for it in b[1]:
                p = para(indent=False)
                p.paragraph_format.left_indent = Inches(0.2)
                p.paragraph_format.first_line_indent = Inches(-0.12)
                run(p, "\u2022 ")
                rich(p, it)
        elif kind == "eq":
            eq_no += 1
            img, w_in = _equation_png(b[2], eq_no)
            p = para(WD_ALIGN_PARAGRAPH.LEFT, indent=False, before=4, after=4)
            ts = p.paragraph_format.tab_stops
            ts.add_tab_stop(Inches(1.72), WD_TAB_ALIGNMENT.CENTER)
            ts.add_tab_stop(Inches(3.42), WD_TAB_ALIGNMENT.RIGHT)
            p.add_run("\t").add_picture(str(img), width=Inches(w_in))
            run(p, f"\t({eq_no})")
        elif kind == "fig":
            _, label, path, cap, width, wide = b
            p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=6)
            p.add_run().add_picture(str(HERE / path), width=Inches(width))
            p.paragraph_format.keep_with_next = True
            p = para(WD_ALIGN_PARAGRAPH.JUSTIFY, indent=False, before=3, after=8)
            run(p, f"Fig. {fig_no[label]}. ", size=8)
            rich(p, cap, size=8)
        elif kind == "table":
            _, label, cap, header, rows, spec = b
            widths = [float(x) for x in re.findall(r"p\{([\d.]+)\}", spec)]
            p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=6)
            run(p, f"TABLE {tab_no[label]}", size=8)
            p.paragraph_format.keep_with_next = True
            p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=3)
            rich(p, cap, size=8)
            for r_ in p.runs:
                r_.font.small_caps = True
            p.paragraph_format.keep_with_next = True
            t = doc.add_table(rows=len(rows) + 1, cols=len(header))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            _rule_table(t, qn, OxmlElement)
            for ci, w in enumerate(widths):
                for row in t.rows:
                    row.cells[ci].width = Inches(3.45 * w)
            for ci, h in enumerate(header):
                cp = t.rows[0].cells[ci].paragraphs[0]
                cp.paragraph_format.first_line_indent = None
                rich(cp, h, size=8, bold=True)
            for ri, row in enumerate(rows, start=1):
                for ci, val in enumerate(row):
                    cp = t.rows[ri].cells[ci].paragraphs[0]
                    cp.paragraph_format.first_line_indent = None
                    rich(cp, val, size=8)
            for ri, row in enumerate(t.rows):
                trPr = row._tr.get_or_add_trPr()
                cant = OxmlElement("w:cantSplit"); trPr.append(cant)
                if ri < len(t.rows) - 1:
                    for c in row.cells:
                        for cp in c.paragraphs:
                            cp.paragraph_format.keep_with_next = True
            para(indent=False, after=6)

    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=8, after=4)
    run(p, "Acknowledgment", smallcaps=True)
    p = para(); rich(p, C.ACKNOWLEDGMENT)

    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=8, after=4)
    run(p, "References", smallcaps=True)
    for k in cite_order:
        p = para(WD_ALIGN_PARAGRAPH.LEFT, indent=False)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
        run(p, f"[{cite_no[k]}]\t", size=8)
        _ref_runs(p, C.REFERENCES[k], run)

    # A continuous section break after the last paragraph makes Word balance the final two columns.
    tail = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(tail, 2)
    out = HERE / "TruthLens_IEEE.docx"
    doc.save(out)
    return out


def _equation_png(lines, n):
    """Renders mathtext lines to a tight PNG; returns (path, width_in at 10-pt body scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(4, 0.3 * len(lines)))
    for i, line in enumerate(lines):
        fig.text(0, 1 - (i + 0.5) / len(lines), line, fontsize=10, va="center", ha="left")
    out = HERE / "figures" / f"eq{n}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w_px, _ = Image.open(out).size
    return out, w_px / 600


def _rule_table(t, qn, OxmlElement):
    """IEEE-style table: top/bottom rules and a rule under the header, no vertical lines."""
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single"); el.set(qn("w:sz"), "6" if edge != "insideH" else "2")
        el.set(qn("w:color"), "000000" if edge != "insideH" else "BFBFBF")
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement(f"w:{edge}"); el.set(qn("w:val"), "nil"); borders.append(el)
    tblPr.append(borders)


def _ref_runs(p, latex, run):
    s = latex.replace("``", "\u201c").replace("''", "\u201d").replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace(r"\"o", "ö").replace(r"\"u", "ü").replace(r"{\ss}", "ß")
    for i, part in enumerate(re.split(r"\\emph\{([^}]*)\}", s)):
        if part:
            run(p, part, size=8, italic=(i % 2 == 1))


if __name__ == "__main__":
    print("tex :", build_tex())
    print("docx:", build_docx())
    print(f"{len(cite_order)} references, {len(fig_no)} figures, {len(tab_no)} tables")
