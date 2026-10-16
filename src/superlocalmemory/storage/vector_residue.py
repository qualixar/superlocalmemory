# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Vectors that no memory can be reached through any more.

Meaning search joins each nearest-neighbour hit to ``embedding_metadata``. A
vec0 row with no metadata and no ``vector_row_map`` row can never become a
result, yet it still takes one of the k slots every search asks for, so the
search has to widen to find real ones. ``VectorStore.gc_orphaned_vectors`` was
written for exactly this and never called; a real 22k-fact store had 809.

This module owns the sweep so both callers share one definition: the routine
maintenance pass (``sweep_unreferenced_vectors``) and the store repair
(``unreferenced_rowids`` + ``fetch_vectors`` + ``delete_rowids``, which keeps an
undo copy). Every write runs under the process write lock in a short
``BEGIN IMMEDIATE`` transaction, the same discipline every vec0 writer uses, so
a vector written together with its references can never be swept in between.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

logger = logging.getLogger(__name__)

#: Rows removed per transaction; keeps each write-lock hold short.
BATCH = 200


@contextmanager
def vec_connection(db_path: str | Path) -> Iterator[sqlite3.Connection | None]:
    """A connection with the vector extension loaded, or None if it cannot be.

    Opens an existing store only (``mode=rw``): a path with no store behind it
    yields None instead of creating an empty database there."""
    try:
        conn = sqlite3.connect(f"file:{quote(str(db_path))}?mode=rw", uri=True, timeout=30)
    except sqlite3.OperationalError as exc:
        logger.debug("no store to open at this path: %s", exc)
        yield None
        return
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:  # no extension on this Python: nothing to sweep
            logger.debug("vector extension unavailable: %s", exc)
            yield None
            return
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='fact_embeddings'").fetchone()
        yield conn if has_table else None
    finally:
        conn.close()


def _referenced(conn: sqlite3.Connection) -> set[int]:
    """Every vec0 rowid a fact can be reached through. Read once per call:
    ``vector_row_map`` has no index on ``vec_rowid``, so a per-row NOT EXISTS
    would be quadratic on a large store."""
    refs: set[int] = set()
    for table in ("embedding_metadata", "vector_row_map"):
        try:
            refs.update(int(r[0]) for r in conn.execute(f"SELECT vec_rowid FROM {table}"))  # noqa: S608
        except sqlite3.OperationalError as exc:  # an old store without the table
            if "no such table" not in str(exc).lower():
                raise
    return refs


def unreferenced_rowids(conn: sqlite3.Connection, limit: int | None = None) -> list[int]:
    """vec0 rowids named by neither ``embedding_metadata`` nor ``vector_row_map``."""
    refs = _referenced(conn)
    out = sorted(int(r[0]) for r in conn.execute("SELECT rowid FROM fact_embeddings")
                 if int(r[0]) not in refs)
    return out if limit is None else out[: int(limit)]


def fetch_vectors(conn: sqlite3.Connection, rowids: list[int]) -> list[tuple[int, str, bytes]]:
    """``(rowid, profile_id, embedding bytes)`` for an undo copy."""
    out = []
    for rowid in rowids:
        row = conn.execute(
            "SELECT rowid, profile_id, embedding FROM fact_embeddings WHERE rowid = ?",
            (rowid,)).fetchone()
        if row is not None:
            out.append((int(row[0]), str(row[1]), bytes(row[2])))
    return out


def delete_rowids(conn: sqlite3.Connection, db_path: str | Path, rowids: list[int]) -> int:
    """Delete the given vec0 rows, re-checking each is still unreferenced."""
    from superlocalmemory.storage.write_lock import get_write_lock

    removed = 0
    with get_write_lock(Path(db_path)):
        conn.execute("BEGIN IMMEDIATE")
        try:
            refs = _referenced(conn)
            for rowid in rowids:
                if rowid not in refs:
                    cur = conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (rowid,))
                    removed += max(0, cur.rowcount)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return removed



# -- index rows of memories that no longer exist -------------------------------

_MAPPINGS = ("embedding_metadata", "vector_row_map")


@dataclass(frozen=True)
class OrphanRemoval:
    """What :func:`remove_orphan_index_rows` removed, by kind."""

    vectors: int = 0
    metadata: int = 0
    row_map: int = 0

    @property
    def total(self) -> int:
        return self.vectors + self.metadata + self.row_map


def load_vec(conn: sqlite3.Connection) -> bool:
    """Load the vector extension onto ``conn``; False when this Python cannot."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as exc:
        logger.debug("vector extension unavailable: %s", exc)
        return False


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def _orphan_mapping_rows(conn: sqlite3.Connection) -> dict[str, list[tuple[str, int]]]:
    """``(fact_id, vec_rowid)`` of mapping rows whose fact is not stored, per table."""
    out: dict[str, list[tuple[str, int]]] = {}
    for table in _MAPPINGS:
        out[table] = [] if not _has_table(conn, table) else [
            (str(r[0]), int(r[1])) for r in conn.execute(
                f"SELECT fact_id, vec_rowid FROM {table} "  # noqa: S608
                "WHERE fact_id NOT IN (SELECT fact_id FROM atomic_facts)")]
    return out


def _live_rowids(conn: sqlite3.Connection) -> set[int]:
    """vec0 rowids that a stored fact's mapping row still points at."""
    live: set[int] = set()
    for table in _MAPPINGS:
        if _has_table(conn, table):
            live.update(int(r[0]) for r in conn.execute(
                f"SELECT vec_rowid FROM {table} "  # noqa: S608
                "WHERE fact_id IN (SELECT fact_id FROM atomic_facts)"))
    return live


def orphan_index_summary(conn: sqlite3.Connection) -> OrphanRemoval:
    """Read-only count of what :func:`remove_orphan_index_rows` would remove.

    ``vectors`` counts only vec0 rows that exist; it needs the extension loaded
    (see :func:`load_vec`) and is 0 without it.
    """
    if not _has_table(conn, "atomic_facts"):
        return OrphanRemoval()
    orphans = _orphan_mapping_rows(conn)
    return OrphanRemoval(
        vectors=len(_orphan_vectors(conn, orphans)),
        metadata=len(orphans["embedding_metadata"]),
        row_map=len(orphans["vector_row_map"]),
    )


def _orphan_vectors(conn: sqlite3.Connection, orphans: dict[str, list[tuple[str, int]]]) -> list[int]:
    wanted = {rowid for rows in orphans.values() for _fact, rowid in rows} - _live_rowids(conn)
    if not wanted or not _has_table(conn, "fact_embeddings"):
        return []
    try:
        return sorted(r for r in wanted if conn.execute(
            "SELECT 1 FROM fact_embeddings WHERE rowid = ?", (r,)).fetchone())
    except sqlite3.OperationalError as exc:  # vec0 not loaded on this connection
        logger.debug("orphan vectors not listed: %s", exc)
        return []


def remove_orphan_index_rows(conn: sqlite3.Connection) -> OrphanRemoval:
    """Remove every index row that belongs to a memory that is no longer stored.

    A delete that failed partway, or a cleanup that took only the metadata, left
    a mapping row and its vector behind for a memory that was gone. They answer
    nothing, but the vector stays referenced by its mapping row, so the sweep for
    unreferenced vectors never took it. Here a row is removed only when its fact
    is absent from ``atomic_facts``, and a vector only when no stored fact's
    mapping points at it: a live memory's index entry is never touched. The
    mapping rows go in the same transaction as the vectors; without the vector
    extension the mapping rows still go, and the vector becomes unreferenced for
    the routine sweep. Idempotent. The caller holds the write lock.

    ``conn`` must be able to open a transaction (not already in one).
    """
    if not _has_table(conn, "atomic_facts"):
        return OrphanRemoval()
    own = not conn.in_transaction
    if own:
        conn.execute("BEGIN IMMEDIATE")
    try:
        orphans = _orphan_mapping_rows(conn)  # read inside the write transaction
        vectors = _orphan_vectors(conn, orphans)
        for rowid in vectors:
            conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (rowid,))
        counts = {}
        for table, rows in orphans.items():
            removed = 0
            for fact_id, _rowid in rows:
                removed += max(0, conn.execute(
                    f"DELETE FROM {table} WHERE fact_id = ? "  # noqa: S608
                    "AND fact_id NOT IN (SELECT fact_id FROM atomic_facts)",
                    (fact_id,)).rowcount)
            counts[table] = removed
        if own:
            conn.execute("COMMIT")
    except BaseException:
        if own and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    result = OrphanRemoval(vectors=len(vectors), metadata=counts["embedding_metadata"],
                           row_map=counts["vector_row_map"])
    if result.total:
        logger.info("removed index rows of memories that no longer exist: %s", result)
    return result


def repair_orphan_index_rows(db_path: str | Path) -> OrphanRemoval:
    """Open the store, take the process write lock and run the removal. Raises on error."""
    from superlocalmemory.storage.write_lock import get_write_lock

    path = Path(db_path)
    if not path.exists():
        return OrphanRemoval()
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        load_vec(conn)
        with get_write_lock(path):
            return remove_orphan_index_rows(conn)
    finally:
        conn.close()


def _remove_orphans_locked(conn: sqlite3.Connection, db_path: str | Path) -> int:
    from superlocalmemory.storage.write_lock import get_write_lock

    with get_write_lock(Path(db_path)):
        return remove_orphan_index_rows(conn).total


def sweep_unreferenced_vectors(db_path: str | Path, *, limit: int = BATCH) -> int:
    """Maintenance: remove at most ``limit`` unreachable vectors, and the index rows
    of memories that no longer exist. Never raises."""
    try:
        with vec_connection(db_path) as conn:
            if conn is None:
                return 0
            gone = _remove_orphans_locked(conn, db_path)
            rowids = unreferenced_rowids(conn, limit)
            return gone + (delete_rowids(conn, db_path, rowids) if rowids else 0)
    except Exception as exc:
        logger.debug("unreferenced vector sweep skipped: %s", exc)
        return 0


__all__ = ["BATCH", "OrphanRemoval", "delete_rowids", "fetch_vectors", "load_vec",
           "orphan_index_summary", "remove_orphan_index_rows", "repair_orphan_index_rows",
           "sweep_unreferenced_vectors", "unreferenced_rowids", "vec_connection"]
