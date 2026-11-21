"""Metadata that parses but is not an object still hides a memory when the context hides sources."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.retrieval import visibility


class _Db:
    def __init__(self, conn):
        self._c = conn

    def execute(self, sql, params=()):
        return self._c.execute(sql, params).fetchall()


def _db(metadata: object) -> _Db:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE memories (memory_id TEXT, metadata_json TEXT)")
    conn.execute("INSERT INTO memories VALUES ('m1', ?)", (metadata,))
    return _Db(conn)


@pytest.mark.parametrize("metadata", ['[1]', '"x"', '5', b"\xff\xfe\x00", "{not json"])
@pytest.mark.parametrize("ctx", [
    visibility.VisibilityContext(hide_sources=True),
    visibility.VisibilityContext(hide_media=True),
])
def test_unreadable_or_non_object_metadata_is_hidden(metadata, ctx) -> None:
    assert visibility._hidden_source_memories(_db(metadata), ["m1"], ctx) == {"m1"}


@pytest.mark.parametrize("metadata", [None, "", "{}", "[]", '{"a": 1}'])
def test_empty_or_plain_metadata_stays_visible(metadata) -> None:
    ctx = visibility.VisibilityContext(hide_sources=True, hide_media=True)
    assert visibility._hidden_source_memories(_db(metadata), ["m1"], ctx) == set()
