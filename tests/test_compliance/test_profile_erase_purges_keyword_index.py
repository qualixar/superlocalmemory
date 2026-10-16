# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A profile erasure leaves the erased words nowhere in the keyword index file.

With FTS5 ``secure-delete`` off (SQLite older than 3.42, or 3.44-3.46.0 where SLM
switches it off because it corrupts the index), a DELETE only marks the row;
the words stay in ``atomic_facts_fts_data`` until the index is rewritten. The
4.2.8 release gate caught this through a real daemon on SQLite 3.45.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.compliance.gdpr import GDPRCompliance


class _DB:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def execute(self, sql: str, params: tuple = ()) -> list:
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        self.conn.commit()
        return rows


def _index_holds(conn: sqlite3.Connection, word: bytes) -> bool:
    return any(word in bytes(row[0]) for row in
               conn.execute("SELECT block FROM atomic_facts_fts_data") if row[0])


def test_the_purge_rewrites_the_index_without_deleted_words(tmp_path) -> None:
    conn = sqlite3.connect(tmp_path / "memory.db")
    conn.execute("CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(content)")
    # secure-delete stays off: the path every older or known-broken SQLite takes.
    conn.execute("INSERT INTO atomic_facts_fts(rowid, content) VALUES (1, 'zorvexa keeps')")
    conn.execute("INSERT INTO atomic_facts_fts(rowid, content) VALUES (2, 'other words')")
    conn.commit()
    conn.execute("DELETE FROM atomic_facts_fts WHERE rowid = 1")
    conn.commit()
    assert _index_holds(conn, b"zorvexa"), "precondition: a plain delete leaves the word"

    gdpr = GDPRCompliance.__new__(GDPRCompliance)
    gdpr._db = _DB(conn)
    counts: dict = {}
    gdpr._purge_keyword_indexes(counts)

    assert not _index_holds(conn, b"zorvexa")
    assert "fts_purge_failed" not in counts
    assert conn.execute("SELECT rowid FROM atomic_facts_fts WHERE atomic_facts_fts MATCH 'other'"
                        ).fetchall() == [(2,)]


def test_a_purge_that_fails_is_reported_not_claimed(tmp_path) -> None:
    class _Broken:
        def execute(self, sql: str, params: tuple = ()) -> list:
            if "sqlite_master" in sql:
                return [("atomic_facts_fts",)]
            raise sqlite3.OperationalError("database is locked")

    gdpr = GDPRCompliance.__new__(GDPRCompliance)
    gdpr._db = _Broken()
    counts: dict = {}
    gdpr._purge_keyword_indexes(counts)
    assert counts["fts_purge_failed"] == 1
