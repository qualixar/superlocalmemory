# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair`` on a real store with the damage old builds left.

Damage is planted the way it happened on a real 22k-fact store: facts deleted
on a connection with foreign keys off (a deduplication), an erasure from before
4.1.22 that left the memory's text in the journal, and admin-cancelled failed
obligations whose work was in fact done.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import pytest


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str):
    from superlocalmemory.core.engine_ingestion import canonical_store

    return canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                           require_complete=True, return_receipt=True)


def _n(conn, sql, args=()):
    return int(conn.execute(sql, args).fetchone()[0])


@pytest.fixture
def damaged(engine_with_mock_deps):
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    pid = engine._profile_id
    kept = [list(_store(engine, f"Synthetic keeper {i} tends the orchard row {i}.")
                 .final_fact_ids)[0] for i in range(3)]
    merged = [list(_store(engine, f"Synthetic duplicate {i} of the lantern note.")
                   .final_fact_ids)[0] for i in range(4)]
    marker = f"vexmarrow{uuid.uuid4().hex[:6]}"
    erased_receipt = _store(engine, f"{marker} is a synthetic secret about the north gate.")
    for fact_id in erased_receipt.final_fact_ids:
        assert delete_fact_authorized(engine, fact_id, trusted_actor_id=_actor(),
                                      source_agent_id="test").get("ok")
    db_path = engine._db.db_path
    engine.close()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")  # how the 2026-06-10 deduplication ran
    for fact_id in merged:
        conn.execute("INSERT OR IGNORE INTO bm25_tokens (fact_id, profile_id, tokens) "
                     "VALUES (?, ?, '[]')", (fact_id, pid))
        conn.execute("INSERT INTO fact_outcome_score (fact_id, profile_id, score, play_count, "
                     "updated_at) VALUES (?, ?, 0.1, 1, 't')", (fact_id, pid))
        conn.execute("INSERT INTO fact_access_log (fact_id, profile_id, accessed_at, "
                     "access_type) VALUES (?, ?, 't', 'recall')", (fact_id, pid))
        conn.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    # An erasure from before 4.1.22: its keyword tokens survived, and the
    # journal kept the memory's whole text.
    conn.execute("INSERT INTO bm25_tokens (fact_id, profile_id, tokens) VALUES (?, ?, ?)",
                 (list(erased_receipt.final_fact_ids)[0], pid, json.dumps([marker, "gate"])))
    conn.execute("UPDATE ingestion_operations SET raw_content = ? WHERE operation_id = ?",
                 (f"{marker} is a synthetic secret about the north gate.",
                  erased_receipt.operation_id))
    now = time.time()
    for owner in ("bm25", "vector", "temporal"):
        conn.execute(
            "INSERT INTO projection_obligations (operation_id, profile_id, owner, kind, "
            "subject_id, state, detail, attempts, created_at, updated_at) VALUES "
            "(?, ?, ?, 'erase', ?, 'failed', ?, 3, ?, ?)",
            (f"op-{owner}", pid, owner, list(erased_receipt.final_fact_ids)[0],
             json.dumps({"admin_cancel": True}), now, now))
    conn.execute(
        "INSERT INTO projection_obligations (operation_id, profile_id, owner, kind, subject_id, "
        "state, detail, attempts, created_at, updated_at) VALUES ('op-live', ?, 'temporal', "
        "'apply', ?, 'failed', ?, 3, ?, ?)", (pid, kept[0], json.dumps({"admin_cancel": True}),
                                              now, now))
    conn.commit()
    conn.close()
    return {"db": db_path, "pid": pid, "kept": kept, "merged": merged, "marker": marker}


def test_plan_reports_and_changes_nothing(damaged):
    from superlocalmemory.storage.integrity_scan import plan

    with open(damaged["db"], "rb") as fh:
        before = fh.read()
    conn = sqlite3.connect(f"file:{damaged['db']}?mode=ro", uri=True)
    p = plan(conn)
    conn.close()
    orphans = {o["table"]: o["rows"] for o in p["orphans"]}
    assert orphans["bm25_tokens"] == 5 and orphans["fact_outcome_score"] == 4
    assert orphans["fact_access_log"] == 4  # history: reported, kept
    assert p["erased_text"]["journal_text"] == 1
    # The keyword erase is not proven while the erased fact's tokens remain;
    # the repair removes them first (step 1), then settles it (step 5).
    assert p["failed_obligations"] == {"proven_erased": 2, "obsolete": 0, "retryable": 0,
                                       "needs_review": 2}
    with open(damaged["db"], "rb") as fh:
        assert fh.read() == before


def test_repair_fixes_what_is_proven_keeps_history_and_is_idempotent(damaged):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    summary = Repair(damaged["db"], limits=Limits(batch_size=2, pause_s=0)).apply()

    assert summary["status"] == "finished"
    assert summary["done"]["orphans.bm25_tokens"] == 5
    assert summary["done"]["orphans.fact_outcome_score"] == 4
    assert summary["done"]["erased_text.copies_scrubbed"] >= 1
    assert summary["done"]["obligations.proven_erased"] == 3
    assert summary["batches"] >= 4  # batch_size=2: the work went in small writes
    after = {o["table"]: o["rows"] for o in summary["after"]["orphans"]}
    assert after["bm25_tokens"] == 0 and after["fact_outcome_score"] == 0
    assert after["fact_access_log"] == 4
    assert summary["after"]["erased_text"]["journal_text"] == 0
    assert summary["after"]["failed_obligations"]["needs_review"] == 1

    conn = sqlite3.connect(damaged["db"])
    marker = damaged["marker"]
    assert _n(conn, "SELECT COUNT(*) FROM ingestion_operations WHERE instr(raw_content, ?)",
              (marker,)) == 0
    cancelled = conn.execute("SELECT state, detail FROM projection_obligations WHERE "
                             "operation_id = 'op-live'").fetchone()
    assert cancelled[0] == "failed"  # an admin cancel with a live subject is left alone
    settled = json.loads(conn.execute("SELECT detail FROM projection_obligations WHERE "
                                      "operation_id = 'op-bm25'").fetchone()[0])
    assert settled["admin_cancel"] is True and settled["proof"]["tombstoned"] is True
    assert _n(conn, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id IN (?, ?, ?)",
              tuple(damaged["kept"])) == 3
    assert _n(conn, "SELECT COUNT(*) FROM integrity_repair_receipts WHERE run_id = ?",
              (summary["run_id"],)) >= 6
    conn.close()

    again = Repair(damaged["db"], limits=Limits(pause_s=0)).apply()
    # Notes of work NOT done ("vector_parity.skipped_no_extension" where the
    # interpreter cannot load sqlite-vec) are not a second repair.
    assert {k: v for k, v in again["done"].items()
            if not k.startswith("vectors.") and ".skipped_" not in k} == {}


def test_undo_puts_back_exactly_what_the_run_removed(damaged):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    repair = Repair(damaged["db"], limits=Limits(pause_s=0))
    summary = repair.apply()
    restored = repair.undo(summary["run_id"])

    # 4 of the 5 token rows: the erased memory's row is not kept for undo.
    assert restored["bm25_tokens"] == 4 and restored["fact_outcome_score"] == 4
    assert restored["projection_obligations"] == 3
    conn = sqlite3.connect(damaged["db"])
    assert _n(conn, "SELECT COUNT(*) FROM projection_obligations WHERE state = 'failed'") == 4
    # The erased memory's words are never part of an undo.
    assert _n(conn, "SELECT COUNT(*) FROM integrity_repair_undo WHERE instr(row_json, ?)",
              (damaged["marker"],)) == 0
    assert conn.execute("SELECT status FROM integrity_repair_runs WHERE run_id = ?",
                        (summary["run_id"],)).fetchone()[0] == "undone"
    conn.close()
    assert repair.undo(summary["run_id"]) == {}  # idempotent


def test_a_parent_that_returns_keeps_its_rows(damaged):
    """The orphan check is repeated inside the write: a restored fact keeps its rows."""
    from superlocalmemory.storage import integrity_census as census
    from superlocalmemory.storage import integrity_receipts
    from superlocalmemory.storage.integrity_repair import Repair

    k = next(c for c in census.ORPHAN_CLASSES if c.table == "bm25_tokens")
    conn = sqlite3.connect(damaged["db"], isolation_level=None)
    integrity_receipts.ensure_tables(conn)
    rowids = census.orphan_rowids(conn, k, 10)
    back = conn.execute("SELECT fact_id FROM bm25_tokens WHERE rowid = ?", (rowids[0],)).fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                 "SELECT ?, memory_id, profile_id, 'restored synthetic' FROM atomic_facts "
                 "LIMIT 1", (back,))
    conn.execute("BEGIN IMMEDIATE")
    removed = Repair(damaged["db"])._remove_batch(conn, "run-x", k, rowids)
    conn.execute("COMMIT")
    assert removed == len(rowids) - 1
    assert _n(conn, "SELECT COUNT(*) FROM bm25_tokens WHERE fact_id = ?", (back,)) == 1
    conn.close()


def test_health_has_five_separate_sections(damaged):
    from superlocalmemory.storage.integrity_health import health

    report = health(damaged["db"], pages=True)
    assert list(report) == ["page_integrity", "relational_integrity", "source_fidelity",
                            "projection_readiness", "active_repair"]
    assert report["page_integrity"]["quick_check"] == "ok"
    assert report["relational_integrity"]["removable_rows"] >= 8
    assert report["projection_readiness"]["failed_obligations"]["proven_erased"] == 2
    assert report["active_repair"]["last"] is None


def test_one_repair_at_a_time_and_an_interrupted_run_is_closed(damaged):
    from superlocalmemory.storage import integrity_receipts
    from superlocalmemory.storage import integrity_repair as ir

    conn = sqlite3.connect(damaged["db"])
    integrity_receipts.ensure_tables(conn)
    conn.execute("INSERT INTO integrity_repair_runs (run_id, started_at, status) "
                 "VALUES ('crashed', 1.0, 'running')")
    conn.commit()
    conn.close()
    assert ir._ONE_RUN.acquire(blocking=False)
    try:
        with pytest.raises(ir.RepairBusy):
            ir.Repair(damaged["db"]).apply()
    finally:
        ir._ONE_RUN.release()

    summary = ir.Repair(damaged["db"], limits=ir.Limits(pause_s=0, confirm_s=0)).apply()

    conn = sqlite3.connect(damaged["db"])
    assert conn.execute("SELECT status FROM integrity_repair_runs WHERE run_id = 'crashed'"
                        ).fetchone()[0] == "stopped"
    assert conn.execute("SELECT status FROM integrity_repair_runs WHERE run_id = ?",
                        (summary["run_id"],)).fetchone()[0] == "finished"
    conn.close()
