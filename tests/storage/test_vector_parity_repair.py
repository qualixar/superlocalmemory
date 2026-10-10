# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm db repair``: vector parity.

Two kinds of drift were measured on a copy of a real 23,900-memory store:

* 11 rows of the sqlite-vec index held a vector that no longer matched the
  embedding stored on the memory itself (older versions wrote them); the memory
  and the Lance row agreed, the sqlite-vec row was the odd one out;
* 7 rows of the Lance projection belonged to memories that no longer exist.

After syncing the 11 and deleting the 7 the two searches agreed on every query
tried. The repair does the same, from the memory's own embedding, and never
brings back, rewrites or keeps a vector of an erased, deleted or withheld
memory. The Lance side is a fake here (real Lance is in tests/vector, native).
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import numpy as np
import pytest

from superlocalmemory.storage.vector_residue import vec_connection

pytest.importorskip("sqlite_vec")

from tests.helpers.env_capabilities import (
    NO_VECTOR_SEARCH_REASON,
    vector_search_available,
)

# The fixtures build stores with sqlite-vec loaded. An interpreter whose sqlite3
# cannot load extensions (the python.org builds on macOS) has the package but
# cannot use it, which an import check does not see.
pytestmark = pytest.mark.skipif(
    not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON,
)


class FakeLance:
    """What the repair needs from the vector projection: list ids, remove ids."""

    def __init__(self, ids, *, fail_on_remove: bool = False) -> None:
        self.ids = list(ids)
        self.removed: list[str] = []
        self.fail_on_remove = fail_on_remove

    def fact_ids(self) -> list[str]:
        return list(self.ids)

    def remove_vectors(self, fact_ids) -> int:
        if self.fail_on_remove:
            raise RuntimeError("lance is unavailable")
        gone = set(fact_ids)
        self.removed.extend(fact_ids)
        self.ids = [i for i in self.ids if i not in gone]
        return len(gone)


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    receipt = canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)[0]


def _other_vector(seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    vec = rng.randn(768).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tobytes()


def _set_vec(conn, fact_id: str, blob: bytes) -> None:
    conn.execute("UPDATE fact_embeddings SET embedding = ? WHERE rowid = "
                 "(SELECT vec_rowid FROM embedding_metadata WHERE fact_id = ?)", (blob, fact_id))


def _vec(conn, fact_id: str) -> bytes:
    return bytes(conn.execute(
        "SELECT fe.embedding FROM fact_embeddings fe JOIN embedding_metadata em "
        "ON em.vec_rowid = fe.rowid WHERE em.fact_id = ?", (fact_id,)).fetchone()[0])


def _own(conn, fact_id: str) -> bytes:
    return bytes(conn.execute("SELECT embedding FROM atomic_facts WHERE fact_id = ?",
                              (fact_id,)).fetchone()[0])


@pytest.fixture
def drifted(engine_with_mock_deps):
    """A store with every case the repair must tell apart."""
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    marker = f"vexmarrow{uuid.uuid4().hex[:6]}"
    ids = {
        "ok": _store(engine, "Synthetic ok memory about the orchard row one."),
        "stale": _store(engine, "Synthetic stale memory about the orchard row two."),
        "json": _store(engine, "Synthetic legacy memory about the orchard row three."),
        "withheld": _store(engine, "Synthetic withheld memory about the orchard row four."),
        "soft": _store(engine, "Synthetic soft deleted memory about the orchard row five."),
        "short": _store(engine, "Synthetic short memory about the orchard row six."),
    }
    erased = _store(engine, f"{marker} is a synthetic secret about the north gate.")
    assert delete_fact_authorized(engine, erased, trusted_actor_id=_actor(),
                                  source_agent_id="test").get("ok")
    db_path = engine._db.db_path
    engine.close()

    with vec_connection(db_path) as conn:
        conn.isolation_level = None
        for seed, name in enumerate(("stale", "withheld", "soft"), start=101):
            _set_vec(conn, ids[name], _other_vector(seed))
        # A legacy memory keeps its embedding as JSON text; the float noise of
        # the round trip is not drift.
        own = np.frombuffer(_own(conn, ids["json"]), dtype=np.float32)
        conn.execute("UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                     (json.dumps([float(f"{x:.8g}") for x in own]), ids["json"]))
        # A memory whose own embedding has another width than the index: not
        # something this repair can fix (it needs a re-embed), never touched.
        conn.execute("UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                     (np.ones(4, dtype=np.float32).tobytes(), ids["short"]))
        conn.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?",
                     (ids["withheld"],))
        conn.execute("UPDATE atomic_facts SET archive_status = 'archived' WHERE fact_id = ?",
                     (ids["soft"],))
    ghost = "ghost" + uuid.uuid4().hex[:8]
    lance = FakeLance([ids["ok"], ids["stale"], ids["json"], ids["short"],
                       ids["withheld"], ids["soft"], erased, ghost])
    return {"db": db_path, "ids": ids, "erased": erased, "ghost": ghost, "lance": lance,
            "marker": marker}


def _plan(damaged, lance="fake"):
    from superlocalmemory.storage.integrity_scan import plan

    conn = sqlite3.connect(f"file:{damaged['db']}?mode=ro", uri=True)
    try:
        return plan(conn, lance=damaged["lance"] if lance == "fake" else lance)
    finally:
        conn.close()


def test_the_scan_counts_what_the_two_searches_disagree_on(drifted):
    with open(drifted["db"], "rb") as fh:
        before = fh.read()
    parity = _plan(drifted)["vector_parity"]

    # Only the live, visible memory counts. The withheld and the soft-deleted
    # memory's index rows are stale as well, and are left alone.
    assert parity["stale_vectors"] == 1
    assert parity["dimension_mismatch"] == 1
    assert parity["lance"]["state"] == "active"
    # erased, never-existed, withheld and soft-deleted memories
    assert parity["lance"]["orphans"] == 4
    with open(drifted["db"], "rb") as fh:
        assert fh.read() == before  # a preview changes nothing


def test_json_noise_and_identical_bytes_are_not_drift(drifted):
    from superlocalmemory.storage.vector_parity import is_stale

    exact = np.random.RandomState(1).randn(768).astype(np.float32)
    assert is_stale(exact, exact.copy()) is False
    noisy = np.array([float(f"{x:.7g}") for x in exact], dtype=np.float32)
    assert is_stale(noisy, exact) is False
    assert is_stale(np.random.RandomState(2).randn(768).astype(np.float32), exact) is True
    # a vector that is nearly, but not quite, the same one (cosine 0.99) is drift
    near = exact + np.random.RandomState(3).randn(768).astype(np.float32) * 0.1
    assert is_stale(near, exact) is True


def test_apply_syncs_the_stale_row_and_removes_the_orphans(drifted):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    ids = drifted["ids"]
    with vec_connection(drifted["db"]) as conn:
        rows_before = conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0]
        untouched = {n: _vec(conn, ids[n]) for n in ("withheld", "soft", "short", "ok", "json")}

    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0),
                     lance=drifted["lance"]).apply()

    assert summary["status"] == "finished"
    assert summary["done"]["vector_parity.stale_rewritten"] == 1
    assert summary["done"]["vector_parity.lance_orphans_removed"] == 4
    assert sorted(drifted["lance"].removed) == sorted(
        [ids["withheld"], ids["soft"], drifted["erased"], drifted["ghost"]])
    assert sorted(drifted["lance"].ids) == sorted(
        [ids["ok"], ids["stale"], ids["json"], ids["short"]])
    with vec_connection(drifted["db"]) as conn:
        assert _vec(conn, ids["stale"]) == _own(conn, ids["stale"])
        # nothing else moved: the withheld and soft-deleted memories keep the
        # (stale) row they had, the short one is not forced into the index
        for name, blob in untouched.items():
            assert _vec(conn, ids[name]) == blob
        # nothing was inserted: an erased memory's vector is not brought back
        assert conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0] == rows_before
        assert conn.execute("SELECT COUNT(*) FROM embedding_metadata WHERE fact_id = ?",
                            (drifted["erased"],)).fetchone()[0] == 0
    # the scan after the repair finds nothing left to do
    assert summary["after"]["vector_parity"]["stale_vectors"] == 0
    assert summary["after"]["vector_parity"]["lance"]["orphans"] == 0
    assert summary["before"]["vector_parity"]["stale_vectors"] == 1
    assert summary["before"]["vector_parity"]["lance"]["orphans"] == 4

    again = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0),
                   lance=drifted["lance"]).apply()
    assert not [k for k in again["done"] if k.startswith("vector_parity.")]


def test_receipts_hold_ids_and_counts_only_and_are_not_undoable(drifted):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    repair = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0), lance=drifted["lance"])
    summary = repair.apply()
    run_id = summary["run_id"]

    conn = sqlite3.connect(drifted["db"])
    receipts = conn.execute(
        "SELECT action, target, reason, before_json, after_json, undoable FROM "
        "integrity_repair_receipts WHERE run_id = ? AND action IN "
        "('resync_stale_vectors', 'remove_orphan_lance_vectors')", (run_id,)).fetchall()
    assert {r[0] for r in receipts} == {"resync_stale_vectors", "remove_orphan_lance_vectors"}
    assert all(r[5] == 0 for r in receipts)
    by_action = {r[0]: json.loads(r[3]) for r in receipts}
    assert by_action["resync_stale_vectors"]["rows"] == 1
    assert by_action["resync_stale_vectors"]["fact_ids"] == [drifted["ids"]["stale"]]
    assert by_action["remove_orphan_lance_vectors"]["rows"] == 4
    # no vector, no text: the only thing in a receipt besides ids is counts
    dump = json.dumps(receipts)
    assert drifted["marker"] not in dump and "orchard" not in dump
    assert "b64" not in dump
    # nothing is kept to put back: a stale or orphaned vector must not return
    assert conn.execute("SELECT COUNT(*) FROM integrity_repair_undo WHERE run_id = ?",
                        (run_id,)).fetchone()[0] == 0
    conn.close()

    restored = repair.undo(run_id)
    assert "fact_embeddings" not in restored and "lance" not in json.dumps(restored)
    with vec_connection(drifted["db"]) as c2:
        assert _vec(c2, drifted["ids"]["stale"]) == _own(c2, drifted["ids"]["stale"])


def test_a_memory_withheld_after_the_scan_is_not_rewritten(drifted, monkeypatch):
    """The check is repeated inside the write: the scan is only a list of names."""
    from superlocalmemory.storage import integrity_repair as ir
    from superlocalmemory.storage import vector_parity as vp

    stale_id = drifted["ids"]["stale"]
    real_scan = vp.scan_index
    calls = {"n": 0}

    def scan_then_withhold(conn):
        found = real_scan(conn)
        calls["n"] += 1
        if calls["n"] == 2:  # 1: the preview taken before the run; 2: the repair step
            assert [r.fact_id for r in found.stale] == [stale_id]
            other = sqlite3.connect(drifted["db"])
            other.execute("UPDATE atomic_facts SET quarantined = 1 WHERE fact_id = ?",
                          (stale_id,))
            other.commit()
            other.close()
        return found

    with vec_connection(drifted["db"]) as conn:
        stale_blob = _vec(conn, stale_id)
    monkeypatch.setattr(vp, "scan_index", scan_then_withhold)
    summary = ir.Repair(drifted["db"], limits=ir.Limits(pause_s=0, confirm_s=0),
                        lance=drifted["lance"]).apply()

    assert calls["n"] >= 2
    assert "vector_parity.stale_rewritten" not in summary["done"]
    with vec_connection(drifted["db"]) as conn:
        assert _vec(conn, stale_id) == stale_blob


def test_a_model_switch_in_progress_is_left_alone(drifted):
    from superlocalmemory.storage import embedding_spaces as sp
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    conn = sqlite3.connect(drifted["db"])
    sp.ensure_control_tables(conn)
    conn.execute(f"INSERT INTO {sp.JOBS} (kind, state, from_signature, to_signature, "
                 "from_config, to_config, created_at, updated_at) "
                 "VALUES ('switch', 'running', 'a', 'b', '{}', '{}', 1.0, 1.0)")
    conn.commit()
    conn.close()

    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0),
                     lance=drifted["lance"]).apply()
    assert summary["done"]["vector_parity.skipped_model_switch_running"] == 1
    assert "vector_parity.stale_rewritten" not in summary["done"]


def test_a_memory_restored_after_the_scan_keeps_its_lance_row(drifted, monkeypatch):
    """The orphan check is repeated inside the write, under the same lock."""
    from superlocalmemory.storage import vector_parity as vp
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    soft_id = drifted["ids"]["soft"]
    real = vp.orphan_ids
    calls = {"n": 0}

    def list_then_restore(conn, ids):
        found = real(conn, ids)
        calls["n"] += 1
        if calls["n"] == 2:  # 1: the preview before the run; 2: the repair's list
            assert soft_id in found
            other = sqlite3.connect(drifted["db"])
            other.execute("UPDATE atomic_facts SET archive_status = 'live' WHERE fact_id = ?",
                          (soft_id,))
            other.commit()
            other.close()
        return found

    monkeypatch.setattr(vp, "orphan_ids", list_then_restore)
    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0),
                     lance=drifted["lance"]).apply()

    assert calls["n"] >= 3
    assert soft_id in drifted["lance"].ids and soft_id not in drifted["lance"].removed
    assert summary["done"]["vector_parity.lance_orphans_removed"] == 3


def test_a_failing_lance_does_not_stop_the_rest_or_the_run(drifted):
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    broken = FakeLance(drifted["lance"].ids, fail_on_remove=True)
    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0), lance=broken).apply()

    assert summary["status"] == "finished"
    assert summary["done"]["vector_parity.stale_rewritten"] == 1
    assert summary["done"]["vector_parity.lance_failed"] == 1
    assert "vector_parity.lance_orphans_removed" not in summary["done"]
    conn = sqlite3.connect(drifted["db"])
    assert conn.execute("SELECT COUNT(*) FROM integrity_repair_receipts WHERE run_id = ? "
                        "AND action = 'remove_orphan_lance_vectors'",
                        (summary["run_id"],)).fetchone()[0] == 0
    conn.close()


def test_without_lance_only_the_sqlite_side_is_checked(drifted):
    """No active Lance (the usual store) is neither an error nor a finding."""
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    parity = _plan(drifted, lance=None)["vector_parity"]
    assert parity["stale_vectors"] == 1
    assert parity["lance"] == {"state": "not_active", "rows": None, "orphans": None}

    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0)).apply()
    assert summary["done"]["vector_parity.stale_rewritten"] == 1
    assert "vector_parity.lance_orphans_removed" not in summary["done"]


def test_inside_slm_only_the_running_projection_is_used(drifted, monkeypatch):
    """With an engine (the daemon) the repair never opens a second Lance writer."""
    from superlocalmemory.storage.integrity_repair import Limits, Repair

    class Orchestrator:
        def __init__(self, backend):
            self._backend = backend

        def get_vector_backend(self):
            return self._backend

    monkeypatch.setattr("superlocalmemory.core.backend_orchestrator.get_orchestrator",
                        lambda: Orchestrator(drifted["lance"]))
    summary = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0), engine=object()).apply()
    assert summary["done"]["vector_parity.lance_orphans_removed"] == 4
    assert summary["before"]["vector_parity"]["lance"]["orphans"] == 4

    # SLM runs no Lance: nothing is opened, found on disk or counted
    monkeypatch.setattr("superlocalmemory.core.backend_orchestrator.get_orchestrator",
                        lambda: Orchestrator(None))
    again = Repair(drifted["db"], limits=Limits(pause_s=0, confirm_s=0), engine=object()).apply()
    assert again["before"]["vector_parity"]["lance"]["state"] == "not_active"
    assert not [k for k in again["done"] if k.startswith("vector_parity.")]


def test_health_reports_it_under_projection_readiness(drifted):
    from superlocalmemory.storage.integrity_health import health

    report = health(drifted["db"])
    assert report["projection_readiness"]["vector_parity"]["stale_vectors"] == 1
