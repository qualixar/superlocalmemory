# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Issue #296: a store whose Cozo copy already drifted (4.3 M edges against 2.9 M) heals.

Fixing the drift source stops new drift; it does not remove the edges an older version left
behind, and re-queueing facts cannot reach an edge whose fact is gone. The daemon compares the
copy with the canonical store and, once the projection queue is drained and the difference
holds, rebuilds that profile's copy from SQLite while recall reads SQLite.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.backend_orchestrator import BackendOrchestrator
from superlocalmemory.core.graph_projection_check import check_graph_projection
from superlocalmemory.graph.cozo_backend import _COZO_AVAILABLE, CozoDBGraphBackend
from superlocalmemory.storage import projection_outbox
from superlocalmemory.storage.models import AtomicFact, EdgeType, GraphEdge, MemoryRecord

pytestmark = pytest.mark.skipif(not _COZO_AVAILABLE, reason="CozoDB not installed")


@pytest.fixture
def world(engine_with_mock_deps, tmp_path):
    db = engine_with_mock_deps._db
    for pid in ("default", "work"):
        if pid != "default":
            db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
        db.store_memory(MemoryRecord(memory_id=f"m-{pid}", profile_id=pid, content="parent"))
        facts = []
        for i in range(3):
            fact = AtomicFact(memory_id=f"m-{pid}", profile_id=pid, content=f"{pid} {i}",
                              embedding=[0.1] * 8)
            db.store_fact(fact)
            facts.append(fact.fact_id)
        for a, b in zip(facts, facts[1:]):
            db.store_edge(GraphEdge(profile_id=pid, source_id=a, target_id=b,
                                    edge_type=EdgeType.SEMANTIC, weight=0.9))
    db.execute("DELETE FROM projection_outbox")
    cozo = CozoDBGraphBackend(str(tmp_path / "cozo"))
    orchestrator = BackendOrchestrator(engine_with_mock_deps._config, db)
    orchestrator._cozo = cozo
    orchestrator._update_status("cozo", "active")
    for pid in ("default", "work"):
        with db.raw_connection() as conn:
            cozo.bulk_import_from_sqlite(conn, pid)
    yield db, cozo, orchestrator
    cozo.close()


def _count(cozo, pid: str) -> int:
    rows = cozo._db.run("?[count(a)] := *edge{from_id: a, profile_id: $pid}", {"pid": pid})
    return int(rows.values.tolist()[0][0]) if len(rows) else 0


def _drift(cozo, pid: str, n: int = 3) -> None:
    """What an older version left: edges the store no longer has, to facts that are gone."""
    for i in range(n):
        cozo.add_edge(f"gone-{pid}-{i}", f"gone-{pid}-{i + 1}", "semantic", 1.0, profile_id=pid)


def test_an_in_step_copy_is_left_alone(world):
    db, cozo, orch = world
    report = check_graph_projection(db, orch, "default", heal=True, settle_s=0)
    assert report.in_step and not report.rebuilt and report.stored == 2


def test_a_drifted_copy_is_rebuilt_and_the_other_profile_is_untouched(world):
    db, cozo, orch = world
    _drift(cozo, "default")
    _drift(cozo, "work", 1)
    report = check_graph_projection(db, orch, "default", heal=True, settle_s=0)
    assert report.rebuilt and report.copied == 5 and report.after == report.stored == 2
    assert _count(cozo, "default") == 2
    assert _count(cozo, "work") == 3  # its own drift is its own check's business
    assert orch.get_graph_backend() is cozo


def test_check_only_reports_and_changes_nothing(world):
    db, cozo, orch = world
    _drift(cozo, "default")
    report = check_graph_projection(db, orch, "default", heal=False, settle_s=0)
    assert not report.in_step and not report.rebuilt and _count(cozo, "default") == 5


def test_a_copy_still_catching_up_is_not_rebuilt(world):
    db, cozo, orch = world
    _drift(cozo, "default")
    facts = [r["fact_id"] for r in db.execute(
        "SELECT fact_id FROM atomic_facts WHERE profile_id = 'default'")]
    projection_outbox.enqueue(db, facts[0], "default")
    report = check_graph_projection(db, orch, "default", heal=True, settle_s=0)
    assert not report.rebuilt and report.pending == 1 and "queue" in report.skipped


def test_recall_reads_sqlite_while_the_copy_is_rebuilt(world, monkeypatch):
    db, cozo, orch = world
    _drift(cozo, "default")
    seen: list[object] = []
    real = cozo.rebuild_profile

    def watched(conn, pid):
        seen.append(orch.get_graph_backend())
        return real(conn, pid)

    monkeypatch.setattr(cozo, "rebuild_profile", watched)
    check_graph_projection(db, orch, "default", heal=True, settle_s=0)
    assert seen == [None] and orch.get_graph_backend() is cozo


def test_a_failed_rebuild_keeps_recall_on_sqlite(world, monkeypatch):
    db, cozo, orch = world
    _drift(cozo, "default")
    monkeypatch.setattr(cozo, "rebuild_profile",
                        lambda conn, pid: (_ for _ in ()).throw(RuntimeError("disk full")))
    report = check_graph_projection(db, orch, "default", heal=True, settle_s=0)
    assert not report.rebuilt and "disk full" in report.error
    assert orch.get_graph_backend() is None  # a half-built copy is never served


def test_no_projection_means_nothing_to_check(engine_with_mock_deps):
    orch = BackendOrchestrator(engine_with_mock_deps._config, engine_with_mock_deps._db)
    report = check_graph_projection(engine_with_mock_deps._db, orch, "default", heal=True)
    assert report.skipped and not report.rebuilt


def test_the_maintenance_cycle_checks_every_profile_and_heals(world, monkeypatch):
    from superlocalmemory.core import backend_orchestrator
    from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler

    db, cozo, orch = world
    _drift(cozo, "default")
    _drift(cozo, "work", 2)
    monkeypatch.setattr(backend_orchestrator, "get_orchestrator", lambda: orch)
    monkeypatch.setattr("superlocalmemory.core.graph_projection_check.SETTLE_S", 0)
    scheduler = MaintenanceScheduler(db, orch._config)
    monkeypatch.setattr(scheduler, "_profile_ids", lambda: ("default", "work"))
    scheduler._running = True
    scheduler._graph_projection_pass()
    assert _count(cozo, "default") == 2 and _count(cozo, "work") == 2


def test_regraph_covers_every_profile_with_memories(world):
    from superlocalmemory.core.graph_projection_check import regraph

    db, cozo, orch = world
    _drift(cozo, "work")
    reports = regraph(db, orch, profile="", heal=True, settle_s=0)
    assert {r["profile_id"] for r in reports} == {"default", "work"}
    assert [r["rebuilt"] for r in reports if r["profile_id"] == "work"] == [True]
    assert regraph(db, orch, profile="nobody", heal=True) == []


def test_the_check_never_takes_the_writer(world, monkeypatch):
    """Found live: counting 378k edges inside raw_connection() held the store's single writer
    for minutes on a busy machine, so every save waited behind a read-only check."""
    db, cozo, orch = world

    def writer(*_a, **_kw):
        raise AssertionError("a read-only check must not take the writer connection")

    monkeypatch.setattr(db, "raw_connection", writer)
    report = check_graph_projection(db, orch, "default", heal=False)
    assert report.in_step and report.stored == 2 and not report.error
