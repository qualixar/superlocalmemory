# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Read one PDF page by page: text layer plus a rendered picture of each page.

Run by file path inside the managed environment (``python -I pdf_parse.py``), which
does not have superlocalmemory installed: this file imports nothing from it. pypdfium2
and Pillow are imported inside the functions.

stdin, first line: ``{"path", "out_dir", "max_pages", "render_long_edge", "min_text_chars",
"skip": [page numbers already done]}``. After each page line the script waits for one
line on stdin before it renders the next page (a closed stdin ends the run), so a slow
reader never lets pictures pile up on disk.

stdout, JSON lines:
  {"opened": true, "page_count": N}
  {"page_no", "png_path", "text", "has_text_layer", "width", "height"}   (one per page)
  {"done": true, "page_count", "title", "created"}
  {"error": "<short code>"}                                              (ends the run)

Errors are short codes; they never carry file contents or paths. Embedded files and
links in the PDF are never followed.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys

MAX_TEXT_CHARS = 2_000_000
MAX_SCALE = 8.0
_DATE = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?")


class _Fail(Exception):
    """A run-ending condition; the text is a safe short code."""


def _emit(obj: dict) -> None:
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


def _load_request() -> dict:
    try:
        req = json.loads(sys.stdin.readline())
        ok = (isinstance(req, dict) and isinstance(req.get("path"), str)
              and isinstance(req.get("out_dir"), str) and os.path.isdir(req["out_dir"]))
    except ValueError:
        ok = False
    if not ok:
        raise _Fail("bad_request")
    for key, default in (("max_pages", 500), ("render_long_edge", 1280), ("min_text_chars", 20)):
        value = req.get(key, default)
        req[key] = value if isinstance(value, int) and value > 0 else default
    skip = req.get("skip")
    req["skip"] = {n for n in skip if isinstance(n, int)} if isinstance(skip, list) else set()
    return req


def _open(path: str):
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise _Fail("pdf_reader_missing") from None
    try:
        return pdfium.PdfDocument(path)
    except Exception as exc:  # noqa: BLE001 - classified, never echoed
        if "password" in str(exc).lower():
            raise _Fail("encrypted") from None
        raise _Fail("unreadable") from None


def _meta(pdf, key: str) -> str:
    try:
        return str(pdf.get_metadata_value(key) or "").strip()
    except Exception:  # noqa: BLE001 - metadata is optional
        return ""


def _iso_date(raw: str) -> str:
    found = _DATE.match(raw)
    if not found:
        return ""
    year, month, day = found[1], found[2] or "01", found[3] or "01"
    return f"{year}-{month}-{day}" if 1 <= int(month) <= 12 and 1 <= int(day) <= 31 else ""


def _page_text(page) -> str:
    textpage = page.get_textpage()
    try:
        return (textpage.get_text_range() or "")[:MAX_TEXT_CHARS]
    finally:
        textpage.close()


def _render(page, out_dir: str, long_edge: int) -> tuple[str, int, int]:
    width, height = page.get_size()
    scale = min(MAX_SCALE, long_edge / max(width, height, 1.0))
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    try:
        longest = max(image.size)
        if longest != long_edge:
            ratio = long_edge / longest
            image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))))
        path = os.path.join(out_dir, "page-" + secrets.token_hex(8) + ".png")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            image.save(fh, "PNG")
        return path, image.width, image.height
    finally:
        image.close()


def _one_page(pdf, index: int, req: dict) -> dict:
    page = pdf[index]
    try:
        text = _page_text(page)
        path, width, height = _render(page, req["out_dir"], req["render_long_edge"])
    except Exception:  # noqa: BLE001 - one bad page ends the run with a short code
        raise _Fail("page_failed") from None
    finally:
        page.close()
    layer = len("".join(text.split())) >= req["min_text_chars"]
    return {"page_no": index + 1, "png_path": path, "text": text if layer else "",
            "has_text_layer": layer, "width": width, "height": height}


def _run() -> None:
    req = _load_request()
    pdf = _open(req["path"])
    try:
        count = len(pdf)
        if count > req["max_pages"]:
            raise _Fail("too_many_pages")
        _emit({"opened": True, "page_count": count})
        for index in range(count):
            if index + 1 in req["skip"]:
                continue
            _emit(_one_page(pdf, index, req))
            if not sys.stdin.readline():
                return
        _emit({"done": True, "page_count": count, "title": _meta(pdf, "Title")[:300],
               "created": _iso_date(_meta(pdf, "CreationDate"))})
    finally:
        pdf.close()


def main() -> int:
    sys.stdout = sys.stderr  # stray library prints must not corrupt the protocol
    try:
        _run()
    except _Fail as exc:
        _emit({"error": str(exc)})
    except Exception:  # noqa: BLE001 - never echo what went wrong inside the reader
        _emit({"error": "failed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
