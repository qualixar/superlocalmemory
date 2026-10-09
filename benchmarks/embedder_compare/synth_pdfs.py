"""Deterministic test PDFs: public repo docs (text layer) and one image-only PDF."""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw

from synth_images import _font, font_dir

# pdf name -> (source doc relative to repo root, max source characters)
SOURCES: dict[str, tuple[str, int]] = {
    "answer-check": ("docs/answer-check.md", 18000),
    "configuration": ("docs/configuration.md", 16000),
    "auto-memory": ("docs/auto-memory.md", 9900),
    "getting-started": ("docs/getting-started.md", 8900),
}

# Image-only PDF: no text layer, so recall needs OCR. Invented content.
IMAGE_ONLY_NAME = "ferry-notice"
IMAGE_ONLY_PAGES: list[list[str]] = [
    ["Harbour Ferry Notice", "Winter timetable starts on 4 November.",
     "The 07:15 sailing to Alder Island is cancelled on Sundays.",
     "Bicycles travel free on weekday sailings."],
    ["Lost Property Policy", "Items are held for 45 days.",
     "Umbrellas are donated after 14 days.", "Claims need a photo identity card."],
    ["Safety Drill Schedule", "Fire drill: first Monday of each month at 10:30.",
     "Life jacket checks every March and September.",
     "Assembly point: the blue shelter by gate 3."],
]


def _clean(line: str) -> str:
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    return re.sub(r"[*`]{1,3}", "", line).strip()


def _flowables(md: str):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Preformatted, Spacer

    body = ParagraphStyle("b", fontName="DejaVu", fontSize=10, leading=14, spaceAfter=4)
    head = ParagraphStyle("h", parent=body, fontName="DejaVu-Bold", fontSize=13, leading=17, spaceBefore=8)
    mono = ParagraphStyle("m", fontName="DejaVuMono", fontSize=8, leading=10)
    out, in_code = [], False
    for raw in md.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            out.append(Preformatted("\n".join(textwrap.wrap(raw, 90)) or " ", mono))
            continue
        s = raw.strip()
        if not s or re.fullmatch(r"[|\-: ]+", s):
            out.append(Spacer(1, 4))
            continue
        if s.startswith("#"):
            out.append(Paragraph(escape(_clean(s.lstrip("# "))), head))
            continue
        s = "• " + s[2:] if s[:2] in ("- ", "* ") else s
        s = " | ".join(c.strip() for c in s.strip("|").split("|")) if s.startswith("|") else s
        out.append(Paragraph(escape(_clean(s)), body))
    return out


def _register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for name, file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                       ("DejaVuMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(font_dir() / file)))


def _text_pdf(path: Path, md: str, title: str) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate

    doc = SimpleDocTemplate(str(path), pagesize=letter, title=title, author="slm-bench",
                            invariant=1, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    doc.build(_flowables(md))


def _image_only_pdf(path: Path) -> None:
    pages = []
    for lines in IMAGE_ONLY_PAGES:
        im = Image.new("RGB", (1240, 1754), "white")
        d = ImageDraw.Draw(im)
        d.text((100, 120), lines[0], fill="black", font=_font(56, bold=True))
        for i, line in enumerate(lines[1:]):
            d.text((100, 300 + i * 110), line, fill=(30, 30, 30), font=_font(38))
        pages.append(im)
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150.0)


def generate(out_dir: Path, repo_root: Path) -> list[str]:
    """Write every test PDF into out_dir; return their names (without .pdf)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    for name, (rel, limit) in SOURCES.items():
        md = (Path(repo_root) / rel).read_text(encoding="utf-8")[:limit]
        _text_pdf(out_dir / f"{name}.pdf", md, name)
    _image_only_pdf(out_dir / f"{IMAGE_ONLY_NAME}.pdf")
    return [*SOURCES, IMAGE_ONLY_NAME]


if __name__ == "__main__":
    print("\n".join(generate(Path(sys.argv[1]), Path(sys.argv[2]))))
