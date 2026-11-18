# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M083: compiled pages, their citations and the Markdown projection (4.2.30, WS-6, D-17, D-18).

**Pages (D-18).** A compiled page is derived text: every sentence is a verbatim
span of a cited live fact or a deterministic template over slot values, and
carries the ids of the facts it cites. ``compiled_pages`` keeps the page as one
row (the sentences in ``body_json``, so a page is one artifact that versions and
rolls back whole) with how many candidate sentences were withheld, and
``page_citations`` indexes sentence -> fact so a reader can follow a sentence to
the memory and so erasing a fact finds the pages that quote it. ``origin`` is
always ``derived`` (CHECK): a page is never admitted as memory.

**Projection (D-17).** ``wiki_files`` records the hash of each Markdown file the
projection wrote, so the watcher ignores its own writes by hash. ``wiki_edits``
remembers each edited block already turned into a proposal (a key of fact, the
block's hash and the edited text's hash), so the same edit is not proposed
twice. Both hold ids, paths and hashes, never text.

Erasing a fact removes the pages that cite it (a page is rebuilt without it by
the next dream cycle) and the edit records that name it (I-11); the projected
files are rewritten by ``compiler/erasure``. CHECKs hold invariants only, never
a list of kinds, which SQLite could not extend after release (W0-29).
"""

from __future__ import annotations

import sqlite3

NAME = "M083_compiled_pages"
DB_TARGET = "memory"
#: Additive: new tables an older build never reads.
DOWNGRADE_FLOOR = 51

DDL = """
CREATE TABLE IF NOT EXISTS compiled_pages (
    page_id        TEXT PRIMARY KEY,
    profile_id     TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (length(kind) > 0),
    subject_key    TEXT NOT NULL CHECK (length(subject_key) > 0),
    title          TEXT NOT NULL DEFAULT '',
    origin         TEXT NOT NULL DEFAULT 'derived' CHECK (origin = 'derived'),
    version        INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    body_json      TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    shown_count    INTEGER NOT NULL DEFAULT 0 CHECK (shown_count >= 0),
    withheld_count INTEGER NOT NULL DEFAULT 0 CHECK (withheld_count >= 0),
    withheld_json  TEXT NOT NULL DEFAULT '{}',
    built_at       TEXT NOT NULL,
    set_id         TEXT,
    UNIQUE (profile_id, kind, subject_key)
);
CREATE INDEX IF NOT EXISTS idx_compiled_pages_profile ON compiled_pages (profile_id, kind);
CREATE TABLE IF NOT EXISTS page_citations (
    page_id TEXT NOT NULL,
    seq     INTEGER NOT NULL CHECK (seq >= 0),
    fact_id TEXT NOT NULL,
    PRIMARY KEY (page_id, seq, fact_id)
);
CREATE INDEX IF NOT EXISTS idx_page_citations_fact ON page_citations (fact_id);
CREATE TABLE IF NOT EXISTS wiki_files (
    profile_id TEXT NOT NULL,
    rel_path   TEXT NOT NULL CHECK (length(rel_path) > 0),
    page_id    TEXT,
    file_hash  TEXT NOT NULL,
    written_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, rel_path)
);
CREATE TABLE IF NOT EXISTS wiki_edits (
    edit_key   TEXT PRIMARY KEY,
    profile_id TEXT NOT NULL,
    fact_id    TEXT NOT NULL,
    set_id     TEXT,
    seen_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wiki_edits_fact ON wiki_edits (fact_id)
"""

#: A page goes with its citations.
_PAGE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS trg_compiled_pages_links
AFTER DELETE ON compiled_pages
BEGIN
    DELETE FROM page_citations WHERE page_id = OLD.page_id;
END
"""

#: I-11: erasing a fact removes the pages that cite it and the edit records that name it.
_FACT_TRIGGERS = (
    """
CREATE TRIGGER IF NOT EXISTS trg_compiled_pages_fact_erased
AFTER DELETE ON atomic_facts
BEGIN
    DELETE FROM compiled_pages WHERE page_id IN
        (SELECT page_id FROM page_citations WHERE fact_id = OLD.fact_id);
    DELETE FROM wiki_edits WHERE fact_id = OLD.fact_id;
END
""",
)

_TABLES = {"compiled_pages", "page_citations", "wiki_files", "wiki_edits"}
_TRIGGER_NAMES = ("trg_compiled_pages_links", "trg_compiled_pages_fact_erased")


def _has_facts(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='atomic_facts'").fetchone())


def apply(conn: sqlite3.Connection) -> None:
    for statement in DDL.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.execute(_PAGE_TRIGGER)
    if _has_facts(conn):
        for trigger in _FACT_TRIGGERS:
            conn.execute(trigger)


def verify(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if not _TABLES <= names:
        return False
    wanted = _TRIGGER_NAMES if _has_facts(conn) else _TRIGGER_NAMES[:1]
    return set(wanted) <= names


def repair(conn: sqlite3.Connection) -> None:
    apply(conn)
