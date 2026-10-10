# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair``: fix what the census proves, in small, receipted steps.

Steps, in order, each idempotent and resumable (a rerun recomputes what is
left, so an interrupted run is finished by running it again):

1. ``orphans`` — derived rows whose parent is gone (integrity_census REMOVE
   classes). Removed in batches; each batch re-checks the parent inside its own
   transaction and keeps an undo copy (none for an erased parent).
2. ``vectors`` — vectors no memory can be reached through. No undo copy: they
   belong to no memory, so there is nothing they could be restored for.
3. ``erased_text`` — the words of erased memories still held outside the
   projections (journal text, event previews, entity summaries, derived
   summaries): scrubbed by core/erasure_scrub.py. No undo copy, by design.
4. ``keyword_index`` — words of deleted rows still inside the keyword index
   blocks: the index is rewritten once, and set to drop deleted words at once.
5. ``unfinished_deletes`` — memories a delete started on and never finished:
   an ordinary delete's memory is made findable again (storage/unfinished_deletes);
   an unfinished erasure of a person or profile is never undone, only counted.
6. ``obligations`` — failed ledger entries settled by proof
   (integrity_obligations); admin cancellations with a live subject untouched.
7. ``own_facts`` — memories whose only searchable fact enrichment removed as a
   near-duplicate of another memory get it back (storage/own_fact_repair);
   erased or changed memories are held back and counted. Undo hides it again.
8. ``vector_parity`` — the two vector indexes made to say what the memories say
   (storage/vector_parity): a sqlite-vec row that is not its live memory's own
   embedding is rewritten from it; a Lance row whose memory is gone, withheld or
   soft-deleted is removed. No undo copy, by design: a stale vector and a
   withheld memory's vector are exactly what must not come back. Inside SLM the
   running Lance projection is used (the daemon stays its only writer); with SLM
   stopped the projection is opened here, only if the store is promoted to it.

Never: drop a constraint, invent a parent, delete history, delete content-
bearing rows, or delete anything the census did not prove. Every write holds
the process write lock for one short transaction (``batch_size`` rows), then
releases it and pauses, so remembers and recalls keep running.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from superlocalmemory.storage import integrity_census as census
from superlocalmemory.storage import integrity_obligations as obligations
from superlocalmemory.storage import integrity_receipts as receipts
from superlocalmemory.storage.integrity_scan import erased_text_targets, plan
from superlocalmemory.storage.write_lock import get_write_lock

logger = logging.getLogger(__name__)


@dataclass
class Limits:
    batch_size: int = 100
    pause_s: float = 0.05
    max_seconds: float | None = None
    confirm_s: float = 2.0


@dataclass
class RunStats:
    run_id: str
    started: float = field(default_factory=time.monotonic)
    holds_ms: list[float] = field(default_factory=list)
    step_max_ms: dict[str, float] = field(default_factory=dict)
    step: str = ""
    done: dict[str, int] = field(default_factory=dict)
    stopped: bool = False

    def hold(self, ms: float) -> None:
        self.holds_ms.append(ms)
        self.step_max_ms[self.step] = round(max(ms, self.step_max_ms.get(self.step, 0.0)), 1)

    def add(self, key: str, n: int) -> None:
        if n:
            self.done[key] = self.done.get(key, 0) + int(n)


class _OutOfTime(Exception):
    pass


class RepairBusy(RuntimeError):
    """Another repair is running in this process."""


#: One repair at a time per process (the daemon serves the HTTP route).
_ONE_RUN = threading.Lock()


class Repair:
    def __init__(self, db_path: str | Path, *, limits: Limits | None = None,
                 on_batch: Callable[[str, int], None] | None = None,
                 engine: Any = None, lance: Any = None) -> None:
        self.db_path = Path(db_path)
        self.limits = limits or Limits()
        self.on_batch = on_batch
        #: The running engine (daemon), needed to rebuild a restored memory's
        #: search entries. None: such memories are counted, not restored.
        self.engine = engine
        #: The vector projection (anything with ``fact_ids()`` and
        #: ``remove_vectors(ids)``). None: found as described in ``_lance``.
        self.lance = lance

    # -- plumbing -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            conn.close()  # a failed setup must not leave the file open
            raise
        return conn

    @contextmanager
    def _held(self, stats: RunStats, conn: sqlite3.Connection) -> Iterator[None]:
        """One short write transaction under the process write lock; timed."""
        if self.limits.max_seconds is not None and (
                time.monotonic() - stats.started) > self.limits.max_seconds:
            raise _OutOfTime
        with get_write_lock(self.db_path):
            t0 = time.monotonic()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                stats.hold((time.monotonic() - t0) * 1000.0)
        time.sleep(self.limits.pause_s)

    # -- steps ----------------------------------------------------------------

    def _orphans(self, stats: RunStats) -> None:
        conn = self._connect()
        try:
            for k in census.ORPHAN_CLASSES:
                if k.action != census.REMOVE:
                    continue
                # Listed once, then confirmed after a pause: a writer that
                # commits a child row a moment before its parent (another
                # transaction still open) is never mistaken for an orphan. The
                # orphan check is repeated again inside each write.
                candidates = census.orphan_rowids(conn, k, 10**9)
                if not candidates:
                    continue
                time.sleep(self.limits.confirm_s)
                step = self.limits.batch_size
                for start in range(0, len(candidates), step):
                    with self._held(stats, conn):
                        removed = self._remove_batch(conn, stats.run_id, k,
                                                     candidates[start:start + step])
                    stats.add(f"orphans.{k.table}", removed)
                    if self.on_batch:
                        self.on_batch(f"orphans.{k.table}", removed)
        finally:
            conn.close()

    def _remove_batch(self, conn, run_id: str, k: census.OrphanClass, rowids: list[int]) -> int:
        ph = ",".join("?" * len(rowids))
        cur = conn.execute(f"SELECT rowid AS rowid, * FROM {k.table} WHERE rowid IN ({ph}) "  # noqa: S608
                           f"AND {census.still_orphan_clause(k)}", tuple(rowids))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if not rows:
            return 0
        erased = {r[0] for r in conn.execute(
            "SELECT fact_id FROM projection_tombstones")} if k.parent == "atomic_facts" else set()
        key_at = cols.index(k.column)
        kept = 0
        for row in rows:
            if row[key_at] in erased:
                continue  # an erased parent: no copy of its derived data is kept
            receipts.keep_row(conn, run_id, k.table, receipts.row_json(cols, row))
            kept += 1
        ids = [r[0] for r in rows]
        conn.execute(f"DELETE FROM {k.table} WHERE rowid IN ({','.join('?' * len(ids))})",  # noqa: S608
                     tuple(ids))
        receipts.receipt(conn, run_id, "remove_orphans", k.table, k.why,
                         {"rows": len(ids), "rowids": ids}, {"rows": 0, "undo_copies": kept},
                         undoable=kept > 0)
        return len(ids)

    def _vectors(self, stats: RunStats) -> None:
        from superlocalmemory.storage.vector_residue import unreferenced_rowids, vec_connection

        with vec_connection(self.db_path) as conn:
            if conn is None:
                stats.add("vectors.skipped_no_extension", 1)
                return
            conn.isolation_level = None
            self._other_spaces(stats, conn)
            candidates = unreferenced_rowids(conn)
            if not candidates:
                return
            time.sleep(self.limits.confirm_s)  # see _orphans; re-checked in the write too
            step = self.limits.batch_size
            for start in range(0, len(candidates), step):
                rowids = candidates[start:start + step]
                with self._held(stats, conn):
                    refs = {int(r[0]) for t in ("embedding_metadata", "vector_row_map")
                            for r in conn.execute(f"SELECT vec_rowid FROM {t}")}  # noqa: S608
                    gone = [r for r in rowids if r not in refs]
                    for rowid in gone:
                        conn.execute("DELETE FROM fact_embeddings WHERE rowid = ?", (rowid,))
                    receipts.receipt(conn, stats.run_id, "remove_unreachable_vectors",
                                     "fact_embeddings", "no memory refers to these vectors",
                                     {"rows": len(gone), "rowids": gone}, {"rows": 0},
                                     undoable=False)
                stats.add("vectors.fact_embeddings", len(gone))

    def _other_spaces(self, stats: RunStats, conn: sqlite3.Connection) -> None:
        """Vectors of erased memories in a model switch's staged or previous
        space (storage/embedding_spaces): queued when the memory was deleted,
        removed here as well as by the switch's own idle pass."""
        from superlocalmemory.storage import embedding_spaces as sp

        if not sp.table_exists(conn, sp.PURGE):
            return
        with self._held(stats, conn):
            removed = sp.purge_pending(conn)
            if removed:
                receipts.receipt(conn, stats.run_id, "remove_unreachable_vectors",
                                 "embedding spaces", "their memory was erased",
                                 {"rows": removed}, {"rows": 0}, undoable=False)
        stats.add("vectors.other_spaces", removed)

    def _unfinished_deletes(self, stats: RunStats) -> None:
        from superlocalmemory.storage import unfinished_deletes as ud

        conn = self._connect()
        try:
            found = ud.find(conn)
        finally:
            conn.close()
        for item in found:
            if item.origin == ud.ERASURE:
                stats.add("unfinished_deletes.erasure_to_run_again", 1)
            elif self.engine is None:
                stats.add("unfinished_deletes.needs_slm_running", 1)
            elif ud.restore(self.engine, item):
                conn = self._connect()
                try:
                    with self._held(stats, conn):
                        receipts.receipt(conn, stats.run_id, "restore_unfinished_delete",
                                         f"atomic_facts:{item.fact_id}",
                                         "a delete started and never finished; the person was "
                                         "told it did not happen", {"findable": False},
                                         {"findable": True}, undoable=False)
                finally:
                    conn.close()
                stats.add("unfinished_deletes.restored", 1)
            else:
                stats.add("unfinished_deletes.not_restored", 1)

    def _erased_text(self, stats: RunStats) -> None:
        from superlocalmemory.core import erasure_scrub
        from superlocalmemory.storage.database import DatabaseManager

        db = DatabaseManager(self.db_path)
        conn = self._connect()
        try:
            targets = erased_text_targets(conn)
        finally:
            conn.close()
        for profile_id, fact_id in targets:
            if self.limits.max_seconds is not None and (
                    time.monotonic() - stats.started) > self.limits.max_seconds:
                raise _OutOfTime
            with db.transaction():
                t0 = time.monotonic()
                counts = erasure_scrub.scrub(db, profile_id, fact_id, None)
                if any(counts.values()):
                    db.execute(
                        "INSERT INTO integrity_repair_receipts (run_id, action, target, reason, "
                        "before_json, after_json, undoable, created_at) VALUES (?, 'scrub_erased_text', "
                        "?, 'words of an erased memory were still stored', ?, ?, 0, ?)",
                        (stats.run_id, fact_id, '{"copies": %d}' % sum(counts.values()),
                         json.dumps(counts, sort_keys=True), time.time()))
                stats.hold((time.monotonic() - t0) * 1000.0)
            stats.add("erased_text.copies_scrubbed", sum(counts.values()))
            time.sleep(self.limits.pause_s)

    def _keyword_index(self, stats: RunStats) -> None:
        from superlocalmemory.storage import fts_residue

        conn = self._connect()
        try:
            unrepaired = self._rebuild_damaged_indexes(stats, conn)
            done = conn.execute("SELECT COUNT(*) FROM integrity_repair_receipts WHERE action = "
                                "'purge_keyword_index'").fetchone()[0]
            on = all(fts_residue.secure_delete_on(conn, t) for t in fts_residue.FTS_TABLES
                     if census._has(conn, t))
            if done and on:
                return  # purged once, and every delete since removed its words at once
            for table in fts_residue.FTS_TABLES:
                if not census._has(conn, table) or table in unrepaired:
                    continue  # purging an index that cannot be read would only fail
                with self._held(stats, conn):
                    before = conn.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(block)), 0) "  # noqa: S608
                                          f"FROM {table}_data").fetchone()
                    state = fts_residue.ensure_secure_delete(conn).get(table)
                    fts_residue.purge_deleted_terms(conn, table)
                    after = conn.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(block)), 0) "  # noqa: S608
                                         f"FROM {table}_data").fetchone()
                    # Before SQLite 3.42 secure-delete never turns on, so every
                    # run comes back here; only a purge that removed something
                    # is a repair worth a receipt.
                    changed = tuple(before) != tuple(after)
                    if changed:
                        receipts.receipt(conn, stats.run_id, "purge_keyword_index", table,
                                         "words of deleted rows were still inside the index",
                                         {"blocks": before[0], "bytes": before[1]},
                                         {"blocks": after[0], "bytes": after[1],
                                          "secure_delete": state}, undoable=False)
                if changed:
                    stats.add("keyword_index.rewritten", 1)
        finally:
            conn.close()

    def _rebuild_damaged_indexes(self, stats: RunStats, conn: sqlite3.Connection) -> set[str]:
        """A malformed keyword index is derived data: rebuild it from the memories.

        The setting that damages the index is turned off first (a rebuild with it
        still on is damaged again by the next edit). Each repair is checked
        afterwards; an index that is still damaged is counted and reported, never
        recorded as fixed (GitHub #204).
        """
        from superlocalmemory.storage import fts_residue

        present = [t for t in fts_residue.FTS_TABLES if census._has(conn, t)]
        if not fts_residue.secure_delete_supported() and any(
                fts_residue.secure_delete_on(conn, t) for t in present):
            with self._held(stats, conn):
                fts_residue.disable_where_damaging(conn)
        unrepaired: set[str] = set()
        for table in fts_residue.FTS_TABLES:
            if census._has(conn, table) and fts_residue.keyword_index_damaged(conn, table):
                if not self._repair_index(stats, conn, table):
                    unrepaired.add(table)
        return unrepaired

    def _repair_index(self, stats: RunStats, conn: sqlite3.Connection, table: str) -> bool:
        """Rebuild, then check; if still damaged recreate the table, then check."""
        from superlocalmemory.storage import fts_residue

        before = self._index_size(conn, table)
        attempts = (("rebuilt", "rebuild_keyword_index", fts_residue.rebuild_keyword_index),
                    ("recreated", "recreate_keyword_index", fts_residue.recreate_keyword_index))
        for word, action, fix in attempts:
            try:
                with self._held(stats, conn):
                    fix(conn, table)
            except sqlite3.DatabaseError as exc:
                logger.warning("keyword index %s could not be %s: %s", table, word, exc)
                continue
            if self._index_is_damaged(conn, table):
                continue
            with self._held(stats, conn):
                receipts.receipt(conn, stats.run_id, action, table,
                                 f"keyword index was damaged; {word} from the stored memories",
                                 {"blocks": before[0], "bytes": before[1]},
                                 self._index_size_dict(conn, table), undoable=False)
            stats.add(f"keyword_index.{word}", 1)
            return True
        logger.error("keyword index %s is still damaged after a rebuild and a recreate", table)
        stats.add("keyword_index.still_damaged", 1)
        return False

    @staticmethod
    def _index_size(conn: sqlite3.Connection, table: str) -> tuple[int, int]:
        row = conn.execute(f"SELECT COUNT(*), COALESCE(SUM(LENGTH(block)), 0) "  # noqa: S608
                           f"FROM {table}_data").fetchone()
        return int(row[0]), int(row[1])

    def _index_size_dict(self, conn: sqlite3.Connection, table: str) -> dict[str, int]:
        blocks, size = self._index_size(conn, table)
        return {"blocks": blocks, "bytes": size}

    @staticmethod
    def _index_is_damaged(conn: sqlite3.Connection, table: str) -> bool:
        """FTS5's own check, then SQLite's quick_check (the one a person runs)."""
        from superlocalmemory.storage import fts_residue
        from superlocalmemory.storage.integrity_diagnosis import check_database

        return (fts_residue.keyword_index_damaged(conn, table)
                or table in check_database(conn).damaged_indexes)

    def _obligations(self, stats: RunStats) -> None:
        conn = self._connect()
        try:
            rows = obligations.failed_obligations(conn)
            for start in range(0, len(rows), self.limits.batch_size):
                with self._held(stats, conn):
                    for row in rows[start:start + self.limits.batch_size]:
                        settled = self._settle(conn, stats, row)
                        if settled:
                            stats.add(f"obligations.{settled}", 1)
        finally:
            conn.close()

    def _settle(self, conn, stats: RunStats, row: dict) -> str | None:
        current = conn.execute("SELECT state, detail, attempts FROM projection_obligations "
                               "WHERE obligation_id = ?", (row["obligation_id"],)).fetchone()
        if current is None or current[0] != "failed":
            return None
        cls, proof = obligations.classify(conn, row)
        new = obligations.settlement(cls, row, proof, stats.run_id)
        if new is None:
            return None
        state, detail, attempts = new
        conn.execute("UPDATE projection_obligations SET state = ?, detail = ?, attempts = ?, "
                     "updated_at = ? WHERE obligation_id = ? AND state = 'failed'",
                     (state, detail, attempts, time.time(), row["obligation_id"]))
        receipts.keep_row(conn, stats.run_id, "projection_obligations", json.dumps({
            "key": ["obligation_id", row["obligation_id"]],
            "written": {"state": state, "detail": detail},
            "old": {"state": current[0], "detail": current[1], "attempts": current[2]}}),
            kind="update")
        receipts.receipt(conn, stats.run_id, f"settle_obligation_{cls}",
                         f"projection_obligations:{row['obligation_id']}",
                         f"{row['kind']} {row['owner']} settled by proof",
                         {"state": "failed"}, {"state": state, "proof": proof}, undoable=True)
        return cls

    def _own_facts(self, stats: RunStats) -> None:
        """Memories whose own fact enrichment removed (storage/own_fact_repair)."""
        from superlocalmemory.storage import own_fact_repair as own
        from superlocalmemory.storage.database import DatabaseManager

        conn = self._connect()
        try:
            # Held-back memories are reported by plan() (before/after), not as work done.
            # promote() re-checks every exclusion inside its own write transaction.
            found, _held = own.classify(conn)
            db = DatabaseManager(self.db_path)
            for item in found:
                if self.limits.max_seconds is not None and (
                        time.monotonic() - stats.started) > self.limits.max_seconds:
                    raise _OutOfTime
                try:
                    fact_id = own.promote(db, None, item)
                except Exception as exc:  # noqa: BLE001 — one memory never stops the rest
                    logger.warning("own-fact repair failed for one memory: %s", type(exc).__name__)
                    with self._held(stats, conn):
                        receipts.receipt(conn, stats.run_id, "own_fact_failed",
                                         f"memories:{item.memory_id}", "the repair raised",
                                         {"facts": 0},
                                         {"error": f"{type(exc).__name__}: {exc}"[:200]},
                                         undoable=False)
                    stats.add("own_facts.failed", 1)
                    continue
                if fact_id is None:
                    stats.add("own_facts.not_repaired", 1)
                    continue
                with self._held(stats, conn):
                    self._own_fact_undo(conn, stats.run_id, item.memory_id)
                    receipts.receipt(conn, stats.run_id, "restore_own_fact",
                                     f"memories:{item.memory_id}",
                                     "enrichment removed this memory's only searchable fact",
                                     {"facts": 0}, {"facts": 1, "fact_id": fact_id},
                                     undoable=True)
                stats.add("own_facts.restored", 1)
        finally:
            conn.close()

    @staticmethod
    def _own_fact_undo(conn: sqlite3.Connection, run_id: str, memory_id: str) -> None:
        """Undo puts the memory back as it was (no visible fact): it archives
        EVERY live fact of the memory — the restored one and any enrichment
        derived from it since — and takes the queued enrichment off the queue."""
        from superlocalmemory.core.ingestion_command import _NEVER_RETRY_AT

        receipts.keep_row(conn, run_id, "atomic_facts", json.dumps({
            "key": ["memory_id", memory_id], "written": {"archive_status": "live"},
            "old": {"archive_status": "archived"}}), kind="update")
        op = conn.execute("SELECT operation_id FROM ingestion_operations WHERE idempotency_key = ?",
                          (f"own-fact-repair:{memory_id}",)).fetchone()
        if op is not None:
            receipts.keep_row(conn, run_id, "ingestion_operations", json.dumps({
                "key": ["operation_id", op[0]],
                "written": {"state": "queryable", "next_retry_at": 0},
                "old": {"state": "failed", "next_retry_at": _NEVER_RETRY_AT,
                        "last_error": "own-fact repair undone"}}), kind="update")

    def _lance(self) -> tuple[Any, Callable[[], None] | None]:
        """``(projection, close)``; the projection is None when there is nothing to fix.

        Inside SLM (an engine is given) only the projection SLM itself runs is
        used: opening a second writer beside it is never done. Without SLM it is
        opened here, and only for a store that is promoted to it.
        """
        if self.lance is not None:
            return self.lance, None
        if self.engine is not None:
            from superlocalmemory.core.backend_orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            return (orchestrator.get_vector_backend() if orchestrator else None), None
        from superlocalmemory.storage import vector_parity as vp

        if vp.lance_state(self.db_path) != vp.ACTIVE:
            return None, None
        from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend

        backend = LanceDBVectorBackend(str(self.db_path.parent / "lance"))
        return backend, backend.close

    def _vector_parity(self, stats: RunStats) -> None:
        """Both vector indexes made to agree with the memories (storage/vector_parity)."""
        self._parity_stale(stats)
        self._parity_lance(stats)

    def _parity_stale(self, stats: RunStats) -> None:
        from superlocalmemory.storage import vector_parity as vp
        from superlocalmemory.storage.vector_residue import vec_connection

        with vec_connection(self.db_path) as conn:
            if conn is None:
                stats.add("vector_parity.skipped_no_extension", 1)
                return
            conn.isolation_level = None
            if vp.reindex_in_progress(conn):
                stats.add("vector_parity.skipped_model_switch_running", 1)
                return
            found = list(vp.scan_index(conn).stale)
            if not found:
                return
            time.sleep(self.limits.confirm_s)  # see _orphans; re-checked in the write too
            step = self.limits.batch_size
            for start in range(0, len(found), step):
                with self._held(stats, conn):
                    done = vp.rewrite_stale(conn, found[start:start + step])
                    if done:
                        receipts.receipt(conn, stats.run_id, "resync_stale_vectors",
                                         "fact_embeddings",
                                         "the vector was not the memory's own embedding",
                                         {"rows": len(done), "fact_ids": done}, {"rows": 0},
                                         undoable=False)
                stats.add("vector_parity.stale_rewritten", len(done))
                if self.on_batch:
                    self.on_batch("vector_parity.stale", len(done))

    def _parity_lance(self, stats: RunStats) -> None:
        from superlocalmemory.storage import vector_parity as vp

        close = None
        conn = self._connect()
        try:
            lance, close = self._lance()
            if lance is None:
                return
            orphans = vp.orphan_ids(conn, lance.fact_ids())
            if not orphans:
                return
            time.sleep(self.limits.confirm_s)  # see _orphans; re-checked in the write too
            step = self.limits.batch_size
            for start in range(0, len(orphans), step):
                with self._held(stats, conn):
                    # The Lance write sits inside the SQLite write lock, so no
                    # memory can be restored between the check and the delete;
                    # if the delete fails nothing is receipted.
                    confirmed = vp.orphan_ids(conn, orphans[start:start + step])
                    if confirmed:
                        lance.remove_vectors(confirmed)
                        receipts.receipt(conn, stats.run_id, "remove_orphan_lance_vectors",
                                         "lance embeddings",
                                         "no live, visible memory behind the vector",
                                         {"rows": len(confirmed), "fact_ids": confirmed},
                                         {"rows": 0}, undoable=False)
                stats.add("vector_parity.lance_orphans_removed", len(confirmed))
                if self.on_batch:
                    self.on_batch("vector_parity.lance", len(confirmed))
        except _OutOfTime:
            raise
        except Exception as exc:  # noqa: BLE001 - the projection is derived data; the run goes on
            logger.warning("vector parity: Lance step failed: %s", type(exc).__name__)
            stats.add("vector_parity.lance_failed", 1)
        finally:
            conn.close()
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("closing the Lance projection failed", exc_info=True)

    # -- public -----------------------------------------------------------

    def _running_lance(self) -> Any:
        """The projection a scan may read directly: the injected one, or SLM's own."""
        if self.lance is not None:
            return self.lance
        if self.engine is None:
            return None  # found on disk, read-only
        from superlocalmemory.core.backend_orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        backend = orchestrator.get_vector_backend() if orchestrator else None
        return backend if backend is not None else False  # False: SLM runs none; do not look

    STEPS = ("orphans", "vectors", "erased_text", "keyword_index", "unfinished_deletes",
             "obligations", "own_facts", "vector_parity")

    def apply(self, run_id: str | None = None) -> dict[str, Any]:
        if not _ONE_RUN.acquire(blocking=False):
            raise RepairBusy("a repair is already running on this SLM; wait for it to finish")
        try:
            return self._apply(run_id)
        finally:
            _ONE_RUN.release()

    def _apply(self, run_id: str | None) -> dict[str, Any]:
        stats = RunStats(run_id or uuid.uuid4().hex[:16])
        conn = self._connect()
        try:
            receipts.ensure_tables(conn)
            # A run still marked running was interrupted (this process holds
            # the only-run lock): the steps are idempotent, this run finishes it.
            conn.execute("UPDATE integrity_repair_runs SET status = 'stopped', finished_at = ? "
                         "WHERE status = 'running'", (time.time(),))
            before = plan(conn, lance=self._running_lance())
            receipts.start_run(conn, stats.run_id)
        finally:
            conn.close()
        status = "finished"
        try:
            for step in self.STEPS:
                stats.step = step
                getattr(self, f"_{step}")(stats)
        except _OutOfTime:
            status = "stopped"
        except Exception:
            status = "failed"
            logger.exception("integrity repair failed")
            raise
        finally:
            conn = self._connect()
            try:
                after = plan(conn, lance=self._running_lance())
                summary = {
                    "run_id": stats.run_id, "status": status, "done": stats.done,
                    "batches": len(stats.holds_ms),
                    "max_lock_hold_ms": round(max(stats.holds_ms, default=0.0), 1),
                    "max_lock_hold_ms_by_step": stats.step_max_ms,
                    "seconds": round(time.monotonic() - stats.started, 2),
                    "before": before, "after": after,
                }
                receipts.finish_run(conn, stats.run_id, status, {
                    k: v for k, v in summary.items() if k not in ("before", "after")})
            finally:
                conn.close()
        return summary

    def undo(self, run_id: str) -> dict[str, int]:
        """Put back what a run removed or changed (undo copies only)."""
        conn = self._connect()
        try:
            receipts.ensure_tables(conn)
            # Removed orphans had no parent by definition: putting them back
            # restores that state exactly, so foreign keys are off for this.
            conn.execute("PRAGMA foreign_keys=OFF")
            with get_write_lock(self.db_path):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    restored = receipts.restore_rows(conn, run_id)
                    conn.execute("UPDATE integrity_repair_runs SET status = 'undone' "
                                 "WHERE run_id = ?", (run_id,))
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            return restored
        finally:
            conn.close()


__all__ = ["Limits", "Repair", "RepairBusy"]
