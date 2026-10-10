"""Optional arXiv PDF pages in the dataset build (offline: media stages stubbed)."""
from __future__ import annotations

import json
from pathlib import Path

import build_dataset as bd


def _make(dirpath: Path, empty: str | None = None, skip: str | None = None) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in bd.ARXIV_IDS:
        if i != skip:
            (dirpath / f"{i}.pdf").write_bytes(b"" if i == empty else b"%PDF-1.4 x")
    return dirpath


def test_arxiv_pdfs_all_present(tmp_path):
    got = bd.arxiv_pdfs(_make(tmp_path / "a"))
    assert got == [tmp_path / "a" / f"{i}.pdf" for i in bd.ARXIV_IDS]


def test_arxiv_pdfs_none_or_missing_dir_is_empty(tmp_path):
    assert bd.arxiv_pdfs(None) == []
    assert bd.arxiv_pdfs(tmp_path / "nope") == []


def test_arxiv_pdfs_one_missing_is_empty(tmp_path):
    assert bd.arxiv_pdfs(_make(tmp_path / "a", skip=bd.ARXIV_IDS[1])) == []


def test_arxiv_pdfs_one_empty_is_empty(tmp_path):
    assert bd.arxiv_pdfs(_make(tmp_path / "a", empty=bd.ARXIV_IDS[2])) == []


def _stub_media(monkeypatch, seen: dict):
    def corpus(ds, arxiv=()):
        seen["arxiv"] = list(arxiv)
        extra = [{"doc_id": f"pdf:arxiv-{p.stem}#p1", "kind": "pdf_page", "path": "x.png"}
                 for p in arxiv]
        return extra, {}

    def queries(path, doc_ids, page_text):
        seen.setdefault("paths", []).append(Path(path).name)
        return [], {}

    monkeypatch.setattr(bd, "_media_corpus", corpus)
    monkeypatch.setattr(bd, "_media_queries", queries)


def _build(tiny_home, out, arxiv):
    bd._build_into(bd.FIXTURE if hasattr(bd, "FIXTURE") else Path(__file__).parent / "fixtures" / "fake_locomo.json",
                   tiny_home / "selection.json", out, True, None, arxiv)


def test_build_into_without_arxiv_skips_arxiv_queries(tiny_home, tmp_path, monkeypatch):
    seen: dict = {}
    _stub_media(monkeypatch, seen)
    _build(tiny_home, tmp_path / "ds", [])
    assert seen["paths"] == ["media_queries.jsonl"]
    ids = [json.loads(x)["doc_id"] for x in (tmp_path / "ds" / "corpus.jsonl").read_text().splitlines()]
    assert not any(i.startswith("pdf:arxiv-") for i in ids)


def test_build_into_with_arxiv_adds_pages_and_queries(tiny_home, tmp_path, monkeypatch):
    seen: dict = {}
    _stub_media(monkeypatch, seen)
    pdfs = bd.arxiv_pdfs(_make(tmp_path / "a"))
    _build(tiny_home, tmp_path / "ds", pdfs)
    assert seen["arxiv"] == pdfs
    assert seen["paths"] == ["media_queries.jsonl", "arxiv_queries.jsonl"]
    ids = [json.loads(x)["doc_id"] for x in (tmp_path / "ds" / "corpus.jsonl").read_text().splitlines()]
    assert f"pdf:arxiv-{bd.ARXIV_IDS[0]}#p1" in ids
