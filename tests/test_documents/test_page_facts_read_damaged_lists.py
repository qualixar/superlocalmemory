"""A doc_pages.fact_ids_json that holds non-UTF-8 bytes reads as no facts instead of raising."""

from __future__ import annotations

import sqlite3

from superlocalmemory.documents.pagefacts import fact_ids_by_document
from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS


class _Store:
    def __init__(self, conn):
        self._conn = conn

    def _read(self):
        return self._conn


def test_a_damaged_page_list_reads_as_empty_and_the_others_still_count() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE doc_pages (document_id TEXT, fact_ids_json TEXT)")
    conn.executemany("INSERT INTO doc_pages VALUES (?, ?)", [
        ("d1", b"\xff\xfe\x00"), ("d1", '["f1"]'), ("d1", b'["f2"]'), ("d2", None)])

    assert fact_ids_by_document(_Store(conn), ["d1", "d2"]) == {"d1": ["f1", "f2"], "d2": []}


def test_the_column_is_one_the_doctor_and_the_ratchet_cover() -> None:
    assert ("doc_pages", "fact_ids_json") in TOLERATED_COLUMNS
