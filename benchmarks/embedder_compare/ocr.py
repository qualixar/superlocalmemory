"""OCR text for images and PDF pages, cached by sha256 of the source file."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

LONG_EDGE = 1280
MIN_TEXT_LAYER = 20  # chars; fewer means the page has no usable text layer


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_page(pdf_path: Path, index: int, out_png: Path, long_edge: int = LONG_EDGE) -> None:
    """Render one PDF page (0-based index) to PNG, long edge = long_edge px."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    page = pdf[index]
    w, h = page.get_size()
    image = page.render(scale=long_edge / max(w, h)).to_pil().convert("RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)
    pdf.close()


def pdf_text_layer(pdf_path: Path, index: int) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return pdf[index].get_textpage().get_text_range()
    finally:
        pdf.close()


class Ocr:
    """RapidOCR wrapper with an on-disk cache keyed by file hash."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._engine = None
        self.errors: list[str] = []
        self.last_ms = 0.0  # OCR time of the last image_text call (cached value if cached)

    def _read(self, path: Path) -> str:
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
        out = self._engine(str(path))
        return "\n".join(out.txts or ())

    def image_text(self, path: Path) -> str:
        """OCR text of an image file; failures are logged and recorded, never silent."""
        cache = self.cache_dir / f"{file_sha256(path)}.json"
        if cache.exists():
            cached = json.loads(cache.read_text())
            self.last_ms = cached.get("ocr_ms", 0.0)
            return cached["text"]
        t0 = time.perf_counter()
        try:
            text = self._read(path)
            self.last_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # unreadable image: log loudly, keep going
            msg = f"OCR failed for {path}: {exc!r}"
            self.errors.append(msg)
            print(msg, file=sys.stderr)
            self.last_ms = 0.0
            return ""
        cache.write_text(json.dumps({"text": text, "ocr_ms": self.last_ms}))
        return text


def attach_text(corpus: list[dict], dataset_dir: Path, ocr: Ocr) -> list[dict]:
    """Fill item['text'] and item['ocr_ms'] for image and pdf_page items (text layer first, then OCR)."""
    for item in corpus:
        if item["kind"] == "text":
            continue
        png = dataset_dir / item["path"]
        text, ocr.last_ms = "", 0.0
        if item["kind"] == "pdf_page":
            meta = item["meta"]
            text = pdf_text_layer(dataset_dir / meta["pdf"], meta["page"] - 1)
        if len(text.strip()) < MIN_TEXT_LAYER:
            text = ocr.image_text(png)
        item["text"] = " ".join(text.split())
        item["ocr_ms"] = ocr.last_ms
    return corpus
