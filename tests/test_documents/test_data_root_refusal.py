"""A document path inside SuperLocalMemory's own data folder is never read, for anyone."""

from __future__ import annotations

import os

import pytest

from superlocalmemory.documents.submit import _Stop, _stage_path
from superlocalmemory.media.ingest import MediaInput
from tests.test_documents.pdfs import make_pdf


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    (r / "media" / "ab").mkdir(parents=True)
    (r / "media" / "ab" / "x.pdf").write_bytes(make_pdf(["secret"]))
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


def _refused(path) -> str:
    with pytest.raises(_Stop) as stop, open(os.devnull, "wb") as sink:
        _stage_path(MediaInput(path=path), sink)
    assert stop.value.receipt.status == "refused"
    return stop.value.receipt.reason


def test_a_stored_original_is_refused(root):
    assert "own data" in _refused(root / "media" / "ab" / "x.pdf")


def test_the_data_folder_and_its_parent_are_refused(root):
    assert "own data" in _refused(root)
    assert "own data" in _refused(root.parent)


def test_a_link_into_the_data_folder_is_refused(root, tmp_path):
    link = tmp_path / "report.pdf"
    os.symlink(root / "media" / "ab" / "x.pdf", link)
    assert "own data" in _refused(link)


def test_a_document_elsewhere_still_reads(root, tmp_path):
    f = tmp_path / "ok.pdf"
    f.write_bytes(make_pdf(["fine"]))
    with open(os.devnull, "wb") as sink:
        assert _stage_path(MediaInput(path=f), sink)[1] > 0
