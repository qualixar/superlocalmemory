# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Staged writes, the catch-up diff, and the one-transaction swap.

The live embedding space is four things that must always agree: the vec0
table ``fact_embeddings``, ``embedding_metadata``, ``vector_row_map`` and the
canonical columns ``atomic_facts.embedding / fisher_mean / fisher_variance``
(kind-scoped recall, consolidation and the in-memory vector index read those).
:func:`activate` replaces all four in ONE transaction; everything before it
writes only to ``reembed_next_*`` and the ``*_next`` column twins. Every step
of the swap is O(1) or a pass over small rows: 22,284 facts swap in about half a
second where rewriting the vector columns alone took 6.2 s.

Every function here runs inside a transaction the caller opened with
``BEGIN IMMEDIATE`` while holding ``get_write_lock(memory.db)``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from superlocalmemory.storage import embedding_canonical_slots as slots
from superlocalmemory.storage import embedding_change_log as change_log
from superlocalmemory.storage import embedding_spaces as sp
from superlocalmemory.storage.embedding_reindex_jobs import update_job

logger = logging.getLogger(__name__)

_SCAN_CHUNK = 2000

#: The same projection DDL VectorStore uses, for a store that never had one.
_METADATA_DDL = (
    "CREATE TABLE IF NOT EXISTS embedding_metadata (vec_rowid INTEGER PRIMARY KEY, "
    "fact_id TEXT NOT NULL UNIQUE, profile_id TEXT NOT NULL DEFAULT 'default', "
    "model_name TEXT NOT NULL DEFAULT '', dimension INTEGER NOT NULL DEFAULT 768, "
    "created_at TEXT NOT NULL DEFAULT (datetime('now')))",
    "CREATE INDEX IF NOT EXISTS idx_embmeta_fact ON embedding_metadata (fact_id)",
    "CREATE INDEX IF NOT EXISTS idx_embmeta_profile ON embedding_metadata (profile_id)",
    "CREATE TABLE IF NOT EXISTS vector_row_map (fact_id TEXT NOT NULL PRIMARY KEY, "
    "profile_id TEXT NOT NULL, vec_rowid INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS idx_vector_row_map_profile ON vector_row_map (profile_id)",
)


class NotCaughtUp(RuntimeError):
    """The staged space no longer matches the live facts; catch up and retry."""

    def __init__(self, missing: int, gone: int, *, only_logged: bool = False) -> None:
        self.missing, self.gone = missing, gone
        #: True when every difference is in the change log (an incremental
        #: catch-up closes it); False means a full diff is needed.
        self.only_logged = only_logged
        super().__init__(f"{missing} facts new or changed, {gone} staged facts gone")


@dataclass(frozen=True, slots=True)
class StagedRow:
    fact_id: str
    profile_id: str
    content_hash: str
    vector: bytes  # float32, little-endian
    fisher_mean: bytes | None
    fisher_variance: bytes | None


def content_hash(content: Any) -> str:
    return hashlib.sha256(str(content if content is not None else "").encode("utf-8")).hexdigest()


def fisher_blobs(vector: Any) -> tuple[bytes, bytes]:
    """The Fisher-Rao pair for a vector: the same rule EmbeddingService uses."""
    from superlocalmemory.core.embeddings import EmbeddingService

    mean, variance = EmbeddingService.compute_fisher_params(None, list(vector))
    return (np.asarray(mean, dtype=np.float32).tobytes(),
            np.asarray(variance, dtype=np.float32).tobytes())


# -- reading facts ------------------------------------------------------------

def facts_after(conn: Any, cursor: int, limit: int) -> list[tuple[int, str, str, str]]:
    return [(int(r[0]), str(r[1]), str(r[2]), r[3] if r[3] is not None else "")
            for r in conn.execute(
                "SELECT rowid, fact_id, profile_id, content FROM atomic_facts "
                "WHERE rowid > ? ORDER BY rowid LIMIT ?", (cursor, limit))]


def count_facts(conn: Any) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0])


def _iter_live_vs_staged(conn: Any) -> Iterator[tuple]:
    twin = slots.next_name("embedding")
    after = 0
    while True:
        rows = conn.execute(
            f"SELECT f.rowid, f.fact_id, f.profile_id, f.content, m.content_hash, "
            f"m.profile_id, f.{twin} IS NULL FROM atomic_facts f LEFT JOIN {sp.NEXT_MAP} m "
            "ON m.fact_id = f.fact_id WHERE f.rowid > ? ORDER BY f.rowid LIMIT ?",
            (after, _SCAN_CHUNK)).fetchall()
        if not rows:
            return
        for r in rows:
            yield int(r[0]), str(r[1]), str(r[2]), r[3], r[4], r[5], bool(r[6])
        after = int(rows[-1][0])


def staged_diff(conn: Any) -> tuple[list[tuple[int, str, str, str]], list[str]]:
    """Facts the staged space lacks or has stale, and staged rows whose fact is gone."""
    need: list[tuple[int, str, str, str]] = []
    for rowid, fact_id, profile_id, content, staged_hash, staged_profile, twin_empty in (
            _iter_live_vs_staged(conn)):
        if (staged_hash is None or twin_empty or str(staged_profile) != profile_id
                or content_hash(content) != staged_hash):
            need.append((rowid, fact_id, profile_id, content if content is not None else ""))
    gone = [str(r[0]) for r in conn.execute(
        f"SELECT m.fact_id FROM {sp.NEXT_MAP} m LEFT JOIN atomic_facts f "
        "ON f.fact_id = m.fact_id WHERE f.fact_id IS NULL")]
    return need, gone


def quick_verify(conn: Any, clean_mark: int) -> None:
    """Under the swap's lock: O(changes) since the last clean full diff, plus two
    index-only counts. Raises NotCaughtUp when anything moved."""
    changed = change_log.changed_since(conn, clean_mark) if change_log.active(conn) else -1
    missing = int(conn.execute(
        f"SELECT COUNT(*) FROM atomic_facts f WHERE NOT EXISTS "
        f"(SELECT 1 FROM {sp.NEXT_MAP} m WHERE m.fact_id = f.fact_id)").fetchone()[0])
    gone = int(conn.execute(
        f"SELECT COUNT(*) FROM {sp.NEXT_MAP} m WHERE NOT EXISTS "
        "(SELECT 1 FROM atomic_facts f WHERE f.fact_id = m.fact_id)").fetchone()[0])
    if changed != 0 or missing or gone:
        raise NotCaughtUp(missing + max(changed, 1 if changed < 0 else 0), gone,
                          only_logged=changed > 0 and not missing and not gone)


# -- staged writes ------------------------------------------------------------

def _drop_staged_fact(conn: Any, fact_id: str) -> None:
    row = conn.execute(f"SELECT vec_rowid FROM {sp.NEXT_MAP} WHERE fact_id = ?",
                       (fact_id,)).fetchone()
    if row is not None:
        conn.execute(f"DELETE FROM {sp.NEXT_VEC} WHERE rowid = ?", (int(row[0]),))
        conn.execute(f"DELETE FROM {sp.NEXT_MAP} WHERE fact_id = ?", (fact_id,))


def write_batch(conn: Any, job: dict, rows: list[StagedRow], *, cursor: int | None,
                done_inc: int, copied_inc: int = 0, caught_up_inc: int = 0) -> int:
    """Stage ``rows``; skip any fact deleted since it was read. Returns rows written."""
    sp.purge_pending(conn)
    next_rowid = int(conn.execute(
        f"SELECT next_rowid FROM {sp.JOBS} WHERE job_id = ?", (job["job_id"],)).fetchone()[0])
    written = 0
    for row in rows:
        live = conn.execute("SELECT profile_id FROM atomic_facts WHERE fact_id = ?",
                            (row.fact_id,)).fetchone()
        _drop_staged_fact(conn, row.fact_id)
        if live is None:
            continue  # erased while it was being embedded: never staged
        next_rowid += 1
        conn.execute(f"INSERT INTO {sp.NEXT_VEC}(rowid, profile_id, embedding) VALUES (?, ?, ?)",
                     (next_rowid, row.profile_id, row.vector))
        conn.execute(
            f"INSERT INTO {sp.NEXT_MAP} (fact_id, profile_id, vec_rowid, content_hash) "
            "VALUES (?, ?, ?, ?)", (row.fact_id, row.profile_id, next_rowid, row.content_hash))
        slots.write_next(conn, row.fact_id, {"embedding": row.vector,
                                             "fisher_mean": row.fisher_mean,
                                             "fisher_variance": row.fisher_variance})
        written += 1
    values: dict[str, Any] = {"next_rowid": next_rowid}
    if cursor is not None:
        values["cursor"] = cursor
    current = conn.execute(f"SELECT done, copied, caught_up FROM {sp.JOBS} WHERE job_id = ?",
                           (job["job_id"],)).fetchone()
    values["done"] = int(current[0]) + done_inc
    values["copied"] = int(current[1]) + copied_inc
    values["caught_up"] = int(current[2]) + caught_up_inc
    update_job(conn, job["job_id"], **values)
    return written


def drop_gone(conn: Any, fact_ids: list[str]) -> None:
    for fact_id in fact_ids:
        _drop_staged_fact(conn, fact_id)


# -- the swap -----------------------------------------------------------------

def _rebuild_projection(conn: Any, source: str, model_name: str, dimension: int) -> None:
    for statement in _METADATA_DDL:
        conn.execute(statement)
    conn.execute("DELETE FROM embedding_metadata")
    conn.execute("DELETE FROM vector_row_map")
    conn.execute(
        "INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id, model_name, dimension) "
        f"SELECT vec_rowid, fact_id, profile_id, ?, ? FROM {source}", (model_name, dimension))
    conn.execute("INSERT INTO vector_row_map (fact_id, profile_id, vec_rowid) "
                 f"SELECT fact_id, profile_id, vec_rowid FROM {source}")


def _timed(timings: dict, name: str, started: float) -> float:
    now = time.perf_counter()
    timings[name] = round((now - started) * 1000, 1)
    return now


def _drop_derived_for(prev_cfg: dict) -> None:
    """Forget what was derived with the replaced model. Never fails a switch."""
    try:
        from superlocalmemory.cache import factory

        model = str((prev_cfg or {}).get("model_name") or "")
        if model:
            factory.invalidate_for_model(model)
    except Exception as exc:
        logger.debug("derivation cache not invalidated after the switch: %s", exc)


def activate(conn: Any, job: dict, *, model_name: str, dimension: int,
             live_cfg: dict, prev_cfg: dict, clean_mark: int) -> dict:
    """Make the staged space live and keep the old one as the previous space.

    ``clean_mark``: the change-log position at which a full content-hash diff
    (run before this, with no lock held) found nothing left to stage.
    """
    timings: dict[str, float] = {}
    started = mark = time.perf_counter()
    quick_verify(conn, clean_mark)
    mark = _timed(timings, "verify_ms", mark)
    if sp.vec_dimension(conn, sp.NEXT_VEC) != dimension:
        raise RuntimeError("staged vector table has the wrong dimension")
    sp.purge_pending(conn)
    if sp.table_exists(conn, sp.PREV_VEC):  # two switches ago: dropped after the swap
        sp.drop_vec(conn, sp.TRASH_VEC)
        sp.rename_vec(conn, sp.PREV_VEC, sp.TRASH_VEC)
    conn.execute(f"DELETE FROM {sp.PREV_MAP}")
    if sp.table_exists(conn, sp.LIVE_VEC):
        sp.rename_vec(conn, sp.LIVE_VEC, sp.PREV_VEC)
        if sp.table_exists(conn, "embedding_metadata"):
            conn.execute(
                f"INSERT INTO {sp.PREV_MAP} (fact_id, profile_id, vec_rowid, content_hash) "
                f"SELECT e.fact_id, e.profile_id, e.vec_rowid, n.content_hash "
                f"FROM embedding_metadata e JOIN {sp.NEXT_MAP} n ON n.fact_id = e.fact_id")
    probe = conn.execute(f"SELECT embedding, profile_id FROM {sp.NEXT_VEC} LIMIT 1").fetchone()
    sp.rename_vec(conn, sp.NEXT_VEC, sp.LIVE_VEC)
    mark = _timed(timings, "vector_tables_ms", mark)
    _rebuild_projection(conn, sp.NEXT_MAP, model_name, dimension)
    mark = _timed(timings, "projection_ms", mark)
    timings["columns_swapped"] = len(slots.swap(conn))
    mark = _timed(timings, "columns_ms", mark)
    if probe is not None and conn.execute(
            f"SELECT rowid FROM {sp.LIVE_VEC} WHERE embedding MATCH ? AND profile_id = ? "
            "AND k = 1", (probe[0], probe[1])).fetchone() is None:
        raise RuntimeError("the new vector index did not answer a search after the swap")
    staged = int(conn.execute(f"SELECT COUNT(*) FROM {sp.NEXT_MAP}").fetchone()[0])
    change_log.stop(conn)
    conn.execute(f"DROP TABLE {sp.NEXT_MAP}")
    sp.ensure_side_tables(conn)  # the trigger needs the (empty) table
    sp.write_space(conn, job["to_signature"], live_cfg, job["from_signature"], prev_cfg,
                   job["job_id"])
    now = time.time()
    update_job(conn, job["job_id"], state="activated", activated_at=now, finished_at=now)
    _timed(timings, "rest_ms", mark)
    _drop_derived_for(prev_cfg)
    return {"facts": staged, **timings,
            "swap_ms": round((time.perf_counter() - started) * 1000, 1)}


def reverse(conn: Any, job: dict, *, model_name: str, dimension: int,
            live_cfg: dict) -> None:
    """Undo an activation made a moment ago: every step is the swap backwards."""
    sp.rename_vec(conn, sp.LIVE_VEC, sp.NEXT_VEC)
    if sp.table_exists(conn, sp.PREV_VEC):
        sp.rename_vec(conn, sp.PREV_VEC, sp.LIVE_VEC)
    # The space from two switches ago lost its map in the swap: it stays as
    # trash for the runner to drop, and there is no previous space after this.
    _rebuild_projection(conn, sp.PREV_MAP, model_name, dimension)
    slots.swap(conn)  # the old values are still in the twins: nothing cleared them yet
    conn.execute(f"DELETE FROM {sp.PREV_MAP}")
    sp.drop_vec(conn, sp.NEXT_VEC)
    sp.write_space(conn, job["from_signature"], live_cfg)
    sp.drop_side_tables_if_unused(conn)


__all__ = ["NotCaughtUp", "StagedRow", "activate", "quick_verify", "content_hash", "count_facts",
           "drop_gone", "facts_after", "fisher_blobs", "reverse", "staged_diff",
           "write_batch"]
