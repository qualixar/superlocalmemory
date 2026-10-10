"""A Python that cannot load SQLite extensions gets a clear refusal, never a stray media.db."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.media import media_db_path, open_media_store
from superlocalmemory.media import store as media_store
from superlocalmemory.media.store import MediaVectorsUnavailable


@pytest.fixture()
def no_extensions(monkeypatch):
    monkeypatch.setattr(media_store, "extensions_supported", lambda: False)


def test_create_is_refused_before_any_file_is_made(tmp_path, no_extensions):
    root = tmp_path / "slm"
    with pytest.raises(MediaVectorsUnavailable):
        open_media_store(create=True, data_root=root)
    assert not media_db_path(root).exists()


def test_opening_an_existing_file_is_refused_with_the_same_error(tmp_path, no_extensions):
    root = tmp_path / "slm"
    root.mkdir()
    media_db_path(root).write_bytes(b"")
    with pytest.raises(MediaVectorsUnavailable):
        open_media_store(data_root=root)


def test_the_refusal_is_a_sqlite_error():
    assert issubclass(MediaVectorsUnavailable, sqlite3.NotSupportedError)
