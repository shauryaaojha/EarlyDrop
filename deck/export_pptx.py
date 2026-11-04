"""Turn a rendered deck PDF into a PowerPoint file: one full-bleed 16:9 image per slide, with the
speaker notes (data-notes) copied into PowerPoint's notes pane.

    powershell -File deck/export_pdf.ps1 -Page pbl3.html -Out EarlyDrop_PBL3_Deck.pdf
    python deck/export_pptx.py deck/pbl3.html deck/EarlyDrop_PBL3_Deck.pdf deck/EarlyDrop_PBL3_Deck.pptx

Needs pymupdf and python-pptx (pip install pymupdf python-pptx). Slides are images, so edit the HTML, not the .pptx.
"""
import html
import io
import re
import sys

import fitz  # pymupdf
from pptx import Presentation
from pptx.util import Emu


def notes_from_html(path):
    src = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r'<section class="slide[^"]*"(.*?)>', src, re.S):
        attrs = m.group(1)
        title = re.search(r'data-title="([^"]*)"', attrs)
        speaker = re.search(r'data-speaker="([^"]*)"', attrs)
        notes = re.search(r'data-notes="([^"]*)"', attrs)
        out.append((html.unescape(title.group(1)) if title else "",
                    html.unescape(speaker.group(1)) if speaker else "",
                    html.unescape(notes.group(1)) if notes else ""))
    return out


def main(page, pdf, pptx_out, dpi=144):
    notes = notes_from_html(page)
    doc = fitz.open(pdf)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)       # 13.333 × 7.5 in, 16:9
    blank = prs.slide_layouts[6]
    for i, pg in enumerate(doc):
        img = io.BytesIO(pg.get_pixmap(dpi=dpi).pil_tobytes(format="JPEG", quality=90, optimize=True))
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(img, 0, 0, width=prs.slide_width, height=prs.slide_height)
        if i < len(notes):
            title, speaker, text = notes[i]
            s.notes_slide.notes_text_frame.text = f"{title} · Speaker (suggested): {speaker}\n\n{text}"
    prs.save(pptx_out)
    print(f"Wrote {pptx_out}: {len(doc)} slides")


if __name__ == "__main__":
    main(*sys.argv[1:4])
