"""The standalone PDF parse script, run by path under an interpreter that has pypdfium2."""

from __future__ import annotations

import json
import struct
import subprocess
import zlib
from pathlib import Path

from superlocalmemory.runtimes import pdf_parse
from tests.test_documents.pdfs import make_pdf

SCRIPT = Path(pdf_parse.__file__)


def run(py, tmp_path, pdf: bytes, **extra):
    src = tmp_path / "in.pdf"
    src.write_bytes(pdf)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    req = {"path": str(src), "out_dir": str(out), "max_pages": 500, "render_long_edge": 1280,
           "min_text_chars": 20, **extra}
    proc = subprocess.run([str(py), "-I", str(SCRIPT)], input=json.dumps(req) + "\n" + "\n" * 50,
                          capture_output=True, text=True, timeout=120)
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()], proc, out


def png_size(path: str) -> tuple[int, int]:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_pages_text_layer_and_png_long_edge(pdf_python, tmp_path):
    pdf = make_pdf(["Hello from page one, this is long enough", ""], title="My Title", created="D:20240131120000Z")
    events, proc, out = run(pdf_python, tmp_path, pdf)
    pages = [e for e in events if "page_no" in e]
    assert [p["page_no"] for p in pages] == [1, 2]
    assert pages[0]["has_text_layer"] is True and "Hello from page one" in pages[0]["text"]
    assert pages[1]["has_text_layer"] is False and pages[1]["text"].strip() == ""
    for p in pages:
        assert max(png_size(p["png_path"])) == 1280 and Path(p["png_path"]).parent == out
        assert p["width"] > 0 and p["height"] > 0
    done = events[-1]
    assert done["done"] is True and done["page_count"] == 2
    assert done["title"] == "My Title" and done["created"] == "2024-01-31"


def test_short_text_is_not_a_text_layer(pdf_python, tmp_path):
    events, _, _ = run(pdf_python, tmp_path, make_pdf(["tiny"]))
    page = [e for e in events if "page_no" in e][0]
    assert page["has_text_layer"] is False


def test_skip_list_leaves_pages_out(pdf_python, tmp_path):
    events, _, _ = run(pdf_python, tmp_path, make_pdf(["a" * 30, "b" * 30, "c" * 30]), skip=[1, 3])
    assert [e["page_no"] for e in events if "page_no" in e] == [2]
    assert events[-1]["page_count"] == 3


def test_encrypted_pdf_is_an_error_without_content(pdf_python, tmp_path):
    events, proc, _ = run(pdf_python, tmp_path, make_pdf(["secret words here please"], encrypted=True))
    assert events[-1] == {"error": "encrypted"}
    assert "secret words" not in proc.stdout + proc.stderr


def test_page_cap(pdf_python, tmp_path):
    events, _, _ = run(pdf_python, tmp_path, make_pdf(["x"] * 3), max_pages=2)
    assert events[-1] == {"error": "too_many_pages"} and not [e for e in events if "page_no" in e]


def test_garbage_and_missing_file_errors_carry_no_content(pdf_python, tmp_path):
    events, proc, _ = run(pdf_python, tmp_path, b"%PDF-1.4 not really a pdf TOPSECRETWORDS")
    assert list(events[-1]) == ["error"] and "TOPSECRETWORDS" not in proc.stdout + proc.stderr
    events, _, _ = run(pdf_python, tmp_path, make_pdf(["x"]), path=str(tmp_path / "nope.pdf"))
    assert list(events[-1]) == ["error"]


def test_bad_request_is_an_error(pdf_python, tmp_path):
    proc = subprocess.run([str(pdf_python), "-I", str(SCRIPT)], input="not json\n", capture_output=True,
                          text=True, timeout=60)
    assert json.loads(proc.stdout.splitlines()[-1]) == {"error": "bad_request"}


def test_script_imports_nothing_from_the_package():
    import ast

    names = []
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not [n for n in names if n.startswith("superlocalmemory")]
