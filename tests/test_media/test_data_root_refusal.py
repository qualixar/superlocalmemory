"""A path that points at SuperLocalMemory's own data folder is never read, for anyone."""

from __future__ import annotations

import os

import pytest

from superlocalmemory.media import ingest
from superlocalmemory.media.ingest import MediaInput, _Stop

PNG = b"\x89PNG\r\n\x1a\nxx"


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    (r / "media" / "ab").mkdir(parents=True)
    (r / "media" / "ab" / "x.png").write_bytes(PNG)
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


def _refused(path) -> str:
    with pytest.raises(_Stop) as stop:
        ingest._read_input(MediaInput(path=path))
    assert stop.value.receipt.status == "refused"
    return stop.value.receipt.reason


def test_an_original_inside_the_data_folder_is_refused(root):
    assert "own data" in _refused(root / "media" / "ab" / "x.png")


def test_the_data_folder_itself_and_its_parent_are_refused(root):
    assert "own data" in _refused(root)
    assert "own data" in _refused(root.parent)


def test_a_link_into_the_data_folder_is_refused(root, tmp_path):
    link = tmp_path / "innocent.png"
    os.symlink(root / "media" / "ab" / "x.png", link)
    assert "own data" in _refused(link)


def test_a_dotdot_path_into_the_data_folder_is_refused(root, tmp_path):
    (tmp_path / "other").mkdir()
    assert "own data" in _refused(tmp_path / "other" / ".." / "slm" / "media" / "ab" / "x.png")


def test_a_picture_elsewhere_still_reads(root, tmp_path):
    pic = tmp_path / "pic.png"
    pic.write_bytes(PNG)
    assert ingest._read_input(MediaInput(path=pic)) == PNG
