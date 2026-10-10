# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""What the interpreter running these tests can actually do.

Some of this suite needs capabilities that are a property of how Python was
BUILT, not of what is installed. When one is missing the tests that need it fail
somewhere deep and unhelpfully — two of them failed on a vector count being 0,
four rows of stack away from the reason.

This names the capability once so a missing one reads as "this interpreter
cannot do X, here is how to get one that can" instead of as a product defect.
"""

from __future__ import annotations

import sqlite3


def sqlite_can_load_extensions() -> bool:
    """Whether this interpreter's sqlite3 can load a loadable extension.

    ``enable_load_extension`` is compiled in or it is not: a Python built
    against a SQLite without ``SQLITE_ENABLE_LOAD_EXTENSION`` does not have the
    method at all. Nothing installable fixes it — the interpreter has to change.

    Without it ``sqlite_vec`` cannot load, so ``VectorStore.available`` is False,
    the engine's vector store is None, and every "is this findable by meaning"
    question answers no. The product degrades correctly; the tests that assert
    the non-degraded path cannot run.
    """
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        return False
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError, sqlite3.OperationalError):
        return False
    else:
        return True
    finally:
        conn.close()


def vector_search_available() -> bool:
    """Whether semantic-vector search can actually run here."""
    if not sqlite_can_load_extensions():
        return False
    try:
        import sqlite_vec
    except Exception:
        return False
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("CREATE VIRTUAL TABLE t USING vec0(embedding float[4])")
        return True
    except Exception:
        return False
    finally:
        conn.close()


#: Reason string for ``skipif``, naming the remedy rather than the symptom.
NO_VECTOR_SEARCH_REASON = (
    "this interpreter cannot load SQLite extensions, so sqlite-vec is "
    "unavailable and the vector store is disabled. Rebuild the test "
    "environment with a Python whose sqlite3 has enable_load_extension "
    "(the homebrew python@3.14 on this machine does): "
    "python3.14 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'"
)


def keyword_index_forgets_at_once() -> bool:
    """Whether FTS5 ``secure-delete`` exists here (SQLite 3.42+).

    Without it a deleted row's words stay inside the keyword index file until
    the next ``slm db repair`` purges them. That is the product's documented
    behaviour on an older SQLite (storage/fts_residue.py, GitHub #153), not a
    defect: Ubuntu 22.04's system SQLite is 3.37.2.
    """
    from superlocalmemory.storage.fts_residue import SECURE_DELETE_MIN_SQLITE

    return sqlite3.sqlite_version_info >= SECURE_DELETE_MIN_SQLITE


NO_SECURE_DELETE_REASON = (
    f"SQLite {sqlite3.sqlite_version} predates FTS5 secure-delete (3.42); there "
    "deleted words leave the keyword index at the next 'slm db repair', by design"
)


def purge_keyword_index_on_old_sqlite(db_path) -> None:
    """Apply the repair's keyword-index purge where secure-delete cannot exist.

    A test asserting that an erasure leaves the words nowhere then checks the
    whole documented contract on every SQLite: at once on 3.42+, after the
    repair's purge before that. A no-op on 3.42+.
    """
    if keyword_index_forgets_at_once():
        return
    from superlocalmemory.storage import fts_residue

    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        for table in fts_residue.FTS_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?", (table,),
            ).fetchone()
            if exists:
                fts_residue.purge_deleted_terms(conn, table)
        conn.commit()
    finally:
        conn.close()
