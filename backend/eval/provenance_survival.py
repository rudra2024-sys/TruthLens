"""Does an embedded AI declaration survive the degradations in eval/perturbations.py?

Takes the repository's blind-spot images that carry an explicit declaration (ChatGPT: C2PA Content Credentials;
portrait-app: prompt + seed PNG text chunks), pushes each through every degradation as an image editor / sharing pipeline
would (decode -> change pixels -> re-encode), and re-reads the provenance. A byte-for-byte file copy is the control.

  python -m eval.provenance_survival
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.provenance.c2pa_reader import read_c2pa          # noqa: E402
from app.services.provenance.metadata import read_metadata         # noqa: E402
from eval import perturbations as P                                # noqa: E402

DATA = BACKEND / "test_data" / "blind_spot_images"


def declared(path: str) -> bool:
    c = read_c2pa(path)
    m = read_metadata(path)
    return bool((c.present and c.ai_declared) or m.xmp_declares_ai or m.ai_param_fields)


def main():
    sets = {
        "ChatGPT (C2PA Content Credentials)": sorted((DATA / "chatgpt_everyday").glob("*.png")),
        "Portrait-app / DiffusionDB (prompt + seed text chunks)": sorted((DATA / "portrait_app").glob("*.png")),
    }
    tmp = Path(tempfile.mkdtemp(prefix="prov_survival_"))
    lines = ["# Does provenance survive re-encoding?\n",
             "Each image starts with an explicit AI declaration in its own metadata. Every degradation below is applied the "
             "way an editor or sharing pipeline would (decode, change pixels, re-encode); the file is then re-read.\n"]
    try:
        for name, files in sets.items():
            n = len(files)
            control = 0
            resave = 0
            rows = []
            for f in files:
                c = tmp / f"copy_{f.name}"
                shutil.copyfile(f, c)                                       # byte-for-byte: metadata untouched
                control += declared(str(c))
            for fam, lvl in P.settings():
                ok = 0
                for f in files:
                    with Image.open(f) as im:
                        out = P.apply(fam, lvl, im.convert("RGB"), f.name)
                    p = tmp / "x.png"
                    out.save(p, format="PNG")                              # what any re-encode without metadata copying does
                    ok += declared(str(p))
                if fam == "clean":
                    resave = ok
                else:
                    rows.append((f"{fam} {lvl}", ok))
            lines += [f"## {name} — {n} images\n",
                      f"- Byte-for-byte copy (control): **{control}/{n}** still declare AI.",
                      f"- Plain decode and re-save with no pixel change: **{resave}/{n}**.\n",
                      "| degradation | still declares AI |", "|---|---|"]
            lines += [f"| {r} | {k}/{n} |" for r, k in rows]
            lines.append("")
        lines += ["**Reading:** the declaration lives in file metadata, not in the pixels, so anything that decodes and re-saves "
                  "the image (a screenshot, a messaging app, a resize, a re-compression, an editor's \"Save as\") discards it. "
                  "Provenance therefore only helps on files that reached the detector untouched, and its absence proves nothing. "
                  "Caveat: some platforms deliberately preserve Content Credentials; this measures ordinary re-encoding.\n"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = BACKEND / "eval" / "results" / "provenance_survival.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))
    print("wrote", out)


if __name__ == "__main__":
    main()
