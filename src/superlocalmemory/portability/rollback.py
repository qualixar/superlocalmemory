# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Undo an import run exactly: remove what the run saved, and nothing else (D-19).

A run's ``import_run_items`` name every memory it saved. Rolling back removes those memories
with everything derived from them, the same way forgetting a memory does: its facts (through
the daemon's writer, which refuses to delete a fact a reviewed correction protects), the
keyword, vector, graph, scene, summary and cache copies of them (``core/mutations``), the
memory's own row, and the ingestion record that holds its original text. It is not an erasure
request, so it writes no tombstone and no erasure receipt: the run never existed. The run row
stays, marked ``rolled_back`` with who and when, because the decision log points at it.

The save path enriches a memory in the background after it is saved, so a memory may gain facts
while the rollback runs. Each pass removes what is there, waits for saves still being enriched
to settle (bounded), and looks again; the run is marked rolled back only when a final look finds
no memory, fact or ingestion record of the run. If something is still being processed after the
last pass the run stays ``rolling_back`` and undoing it again finishes the job.

Nothing the run did not save is touched. A memory saved by someone else, an existing memory,
and the profile's other imports are never in the list.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from superlocalmemory.portability import constants as C
from superlocalmemory.portability import rollback_derived as derived
from superlocalmemory.portability import runs

logger = logging.getLogger(__name__)
IN_FLIGHT = ("raw", "queryable", "enriching")


def _marks(n: int) -> str:
    return ",".join("?" * n)


def _facts_of(db: Any, profile_id: str, memory_ids: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(0, len(memory_ids), 400):
        part = memory_ids[i:i + 400]
        out += [(r["fact_id"], r["memory_id"]) for r in db.execute(
            f"SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? "
            f"AND memory_id IN ({_marks(len(part))})", (profile_id, *part))]
    return out


def _operations(db: Any, profile_id: str, run_id: str) -> list[tuple[str, str]]:
    """``(operation_id, state)`` of the saves this run made (by their keys)."""
    try:
        return [(r["operation_id"], r["state"]) for r in db.execute(
            "SELECT operation_id, state FROM ingestion_operations WHERE profile_id = ? "
            "AND source_type = 'http-import' AND idempotency_key LIKE ?",
            (profile_id, f"imp:{run_id}:%"))]
    except Exception:  # noqa: BLE001 - a store without the table has no records to remove
        return []


def _settle(db: Any, profile_id: str, run_id: str, stop: threading.Event) -> None:
    """Wait (bounded) for this run's saves that are still being enriched."""
    deadline = time.monotonic() + C.ROLLBACK_SETTLE_S
    while time.monotonic() < deadline and not stop.is_set():
        if not any(state in ("enriching",) for _id, state in _operations(db, profile_id, run_id)):
            return
        stop.wait(0.25)


def _delete_fact(engine: Any, runtime: Any, profile_id: str, run_id: str, fact_id: str,
                 memory_id: str) -> None:
    """One fact, with every copy of it; the writer first, then the derived projections."""
    from superlocalmemory.core import mutations

    try:
        if runtime is not None:
            runtime.delete_fact(profile_id, fact_id,
                                idempotency_key=f"import-undo:{run_id}:{fact_id}")
        else:  # no daemon writer (a library caller): the same delete on the database itself
            engine._db.delete_fact(fact_id, profile_id=profile_id)
    except Exception as exc:  # noqa: BLE001 - protected by a reviewed correction, or busy: stays
        logger.info("import undo: a fact stays for now (%s)", type(exc).__name__)
    if engine._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,)):
        return  # still there: the next pass looks again
    mutations._purge_delete_projections(engine, fact_id, profile_id, memory_id=memory_id)


def _sweep(engine: Any, runtime: Any, profile_id: str, run_id: str,
           memory_ids: list[str]) -> set[str]:
    """One pass over the run's memories. Returns every fact id it met."""
    db = engine._db
    met: set[str] = set()
    for i in range(0, len(memory_ids), C.ROLLBACK_BATCH):
        chunk = memory_ids[i:i + C.ROLLBACK_BATCH]
        pairs = _facts_of(db, profile_id, chunk)
        fact_ids = [f for f, _m in pairs]
        met.update(fact_ids)
        entities = derived.entities_naming(db, profile_id, fact_ids)
        for fact_id, memory_id in pairs:
            _delete_fact(engine, runtime, profile_id, run_id, fact_id, memory_id)
        gone = [f for f in fact_ids if not db.execute(
            "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (f,))]
        marks = _marks(len(chunk))
        with db.raw_connection() as conn:  # one short transaction per chunk
            conn.execute(f"DELETE FROM memories WHERE profile_id = ? AND memory_id IN ({marks}) "
                         "AND NOT EXISTS (SELECT 1 FROM atomic_facts f WHERE f.memory_id = "
                         "memories.memory_id)", (profile_id, *chunk))
        derived.after_facts_deleted(db, profile_id, gone, entities)
    return met


def _remove_records(db: Any, profile_id: str, run_id: str) -> list[str]:
    """The ingestion records of the run (they hold the original text) and the write
    obligations that point at them; returns their operation ids. The write receipts
    (``write_commits``) are an append-only ledger of what happened and stay."""
    ops = [op for op, _state in _operations(db, profile_id, run_id)]
    if not ops:
        return []
    with db.raw_connection() as conn:
        for i in range(0, len(ops), 400):
            part = ops[i:i + 400]
            marks = _marks(len(part))
            for table in ("projection_obligations", "ingestion_operations"):
                try:
                    conn.execute(f"DELETE FROM {table} WHERE operation_id IN ({marks})",  # noqa: S608
                                 part)
                except Exception:  # noqa: BLE001 - no such table: nothing to remove
                    pass
    return ops


def _left(db: Any, profile_id: str, run_id: str, memory_ids: list[str]) -> int:
    count = len(_facts_of(db, profile_id, memory_ids))
    for i in range(0, len(memory_ids), 400):
        part = memory_ids[i:i + 400]
        count += int(dict(db.execute(
            f"SELECT COUNT(*) AS c FROM memories WHERE profile_id = ? "
            f"AND memory_id IN ({_marks(len(part))})", (profile_id, *part))[0])["c"])
    return count + len(_operations(db, profile_id, run_id))


def rollback(engine: Any, runtime: Any, profile_id: str, run_id: str, stop: threading.Event,
             on_done: Any = None) -> str:
    """Remove a run's memories. Returns the final state (``rolled_back`` or ``rolling_back``)."""
    db = engine._db
    with db.raw_connection() as conn:
        memory_ids = runs.memory_ids(conn, run_id)
        started = conn.execute("SELECT COALESCE(started_at, created_at) FROM import_runs "
                               "WHERE run_id = ?", (run_id,)).fetchone()[0]
    ops: set[str] = set()
    facts: set[str] = set()
    for attempt in range(C.ROLLBACK_PASSES):
        _settle(db, profile_id, run_id, stop)
        ops.update(_remove_records(db, profile_id, run_id))  # a save not yet enriched never is now
        facts |= _sweep(engine, runtime, profile_id, run_id, memory_ids)
        if _left(db, profile_id, run_id, memory_ids) == 0:
            break
        if stop.wait(1.0 + attempt):
            break
    ops.update(_remove_records(db, profile_id, run_id))
    clean = _left(db, profile_id, run_id, memory_ids) == 0
    if clean:
        derived.at_the_end(db, profile_id, run_id, facts, started, sorted(ops))
    state = runs.ROLLED_BACK if clean else runs.ROLLING_BACK
    note = "" if clean else ("Some of it was still being processed. Undo it again in a minute "
                             "to finish.")
    with db.raw_connection() as conn:
        runs.set_state(conn, run_id, state, note=note)
        if clean and on_done is not None:
            on_done(conn)
    if not clean:
        logger.info("import run %s: rollback not finished", run_id)
    return state


__all__ = ["rollback"]
