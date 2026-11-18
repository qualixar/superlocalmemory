# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M083: compiled pages and the Markdown projection's records: registered, additive, repairable."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage import _schema_version, profile_fold
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M082_fact_signatures as M082
from superlocalmemory.storage.migrations import M083_compiled_pages as M083


def _page(conn, page_id="p1", origin="derived") -> None:
    conn.execute("INSERT INTO compiled_pages (page_id, profile_id, kind, subject_key, body_json, "
                 "content_hash, built_at, origin) VALUES (?, 'default', 'topic', ?, '[]', 'h', 'n', ?)",
                 (page_id, "entity:" + page_id, origin))


def test_m083_is_deferred_after_m082_additive_and_repairable() -> None:
    assert _MODULES[M083.NAME] is M083
    catalogued = {m.name: m for m in DEFERRED_MIGRATIONS}
    assert M083.NAME in catalogued and catalogued[M083.NAME].dependencies == (M082.NAME,)
    assert M083.DOWNGRADE_FLOOR == 51 and M083.DB_TARGET == "memory"
    assert _schema_version.SUPPORTED_SCHEMA_VERSION == 83
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M083.verify(conn)
    M083.apply(conn)
    M083.apply(conn)
    assert M083.verify(conn)
    conn.execute("DROP TRIGGER trg_compiled_pages_fact_erased")
    assert not M083.verify(conn)
    M083.repair(conn)
    assert M083.verify(conn)


def test_a_page_is_derived_always_and_never_admitted_as_memory() -> None:
    conn = sqlite3.connect(":memory:")
    M083.apply(conn)
    _page(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _page(conn, "p2", origin="user_direct")
    with pytest.raises(sqlite3.IntegrityError):  # one page per profile, kind and subject
        _page(conn, "p1")


def test_deleting_a_page_takes_its_citations_and_erasing_a_fact_takes_its_pages() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M083.apply(conn)
    for n in ("a", "b"):
        conn.execute("INSERT INTO atomic_facts VALUES (?)", (f"f-{n}",))
        _page(conn, n)
        conn.execute("INSERT INTO page_citations VALUES (?, 0, ?)", (n, f"f-{n}"))
    conn.execute("INSERT INTO wiki_edits VALUES ('k', 'default', 'f-a', 's', 'n')")
    conn.execute("INSERT INTO wiki_files VALUES ('default', 'topic/a.md', 'a', 'h', 'n')")
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f-a'")
    assert [r[0] for r in conn.execute("SELECT page_id FROM compiled_pages")] == ["b"]
    assert [r[0] for r in conn.execute("SELECT page_id FROM page_citations")] == ["b"]
    assert conn.execute("SELECT COUNT(*) FROM wiki_edits").fetchone()[0] == 0
    conn.execute("DELETE FROM compiled_pages")
    assert conn.execute("SELECT COUNT(*) FROM page_citations").fetchone()[0] == 0


def test_the_profile_fold_deletes_every_table_it_adds() -> None:
    for table in ("compiled_pages", "page_citations", "wiki_files", "wiki_edits"):
        assert profile_fold.DECISIONS[table][0] == profile_fold.DELETE, table


def test_a_store_without_facts_builds_the_tables_without_the_erase_trigger() -> None:
    conn = sqlite3.connect(":memory:")
    M083.apply(conn)
    assert M083.verify(conn)
