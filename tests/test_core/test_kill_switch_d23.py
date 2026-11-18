# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-23: the kill switch returns the store to pre-4.2.30 read behaviour, on every surface.

"A kill switch returns the store to pre-4.2.30 read behaviour without a downgrade: it stops
every 4.2.30 pass and makes reads ignore slots, leases, disputes and taint; it does not undo
retirements, which need a separate ledger command."

Every background pass is tried twice on the same store with every flag ON: once without the
switch (the control: the pass writes, so the test can fail) and once with it (nothing changes in
any table, no state file appears, no pass body is entered).
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest

from superlocalmemory.core import cue_binding
from superlocalmemory.core.feature_flags import (
    FLAGS,
    KILL_ENV,
    FeatureFlags,
    FlagState,
    env_name,
    set_kill_switch,
    write_flag,
)
from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import signing_key as sk
from tests.test_core.test_coverage_backfill import _db
from tests.test_core.test_session_brief import truth  # noqa: F401 (fixture: a store with a dispute and a stale source)
from tests.test_truth.test_slot_pass import NEW, OLD, _fact


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A migrated store holding work for every pass, and a clean data folder."""
    from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
    from superlocalmemory.storage.schema_v347 import apply_v347_schema
    from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
    from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

    home = tmp_path / "slm-home"
    home.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(home))
    for name in list(os.environ):
        if name.startswith("SLM_FEATURE_") or name == KILL_ENV:
            monkeypatch.delenv(name)
    db = _db(tmp_path / "memory.db")
    for apply_schema in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                         apply_v3410_schema, apply_v3411_schema):
        apply_schema(str(db.db_path))
    _fact(db, OLD, "2026-03-01T00:00:00+00:00")  # a slot head, then a dispute
    _fact(db, NEW, "2026-11-14T00:00:00+00:00")
    _fact(db, "The nightly backup runs at two in the morning.", "2026-02-01T00:00:00+00:00")
    _fact(db, "the nightly backup runs at two in the morning", "2026-02-09T00:00:00+00:00")
    record = MemoryRecord(content="Run the schema check before editing the schema.",
                          session_date="2026-11-16",
                          metadata={cue_binding.METADATA_KEY: ["repo:proj"]})
    db.store_memory(record)
    db.store_fact(AtomicFact(memory_id=record.memory_id, content=record.content))
    sk.install(sk.Signer(bytes(range(32))))
    yield SimpleNamespace(home=home, db=db)
    sk.install(None)


def _all_on(home) -> None:
    for flag_id in FLAGS:
        write_flag(home, flag_id, FlagState.ON)


def _snapshot(world) -> dict:
    """Every row of every table, and every file under the data folder."""
    names = [r[0] for r in world.db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
        "AND name NOT LIKE '%_fts%' AND name NOT LIKE '%_vec%' ORDER BY name")]
    rows = {n: [tuple(dict(r).values()) for r in world.db.execute(f"SELECT * FROM {n}")]  # noqa: S608
            for n in names}
    files = sorted(str(p.relative_to(world.home)) for p in world.home.rglob("*") if p.is_file())
    return {"rows": rows, "files": files}


def _count(world, table: str) -> int:
    return world.db.execute(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]  # noqa: S608


def _scheduler(world) -> SimpleNamespace:
    sched = SimpleNamespace(_running=True, _db=world.db, _judge_supplier=None, _slot_job=None,
                            steps=[])
    sched._record_step = lambda *a: sched.steps.append(a)
    sched._profile_ids = lambda: ("default",)
    sched._relation_scorer = lambda: None
    return sched


#: Every 4.2.30 pass in the scheduler: (method, the table its work lands in).
PASSES = (
    ("_trust_pass", "fact_trust"),
    ("_signing_pass", "fact_signatures"),
    ("_slot_pass", "slot_revisions"),
    ("_cue_pass", "memory_cues"),
    ("_dream_pass", "proposal_sets"),
)
#: Passes that were in the scheduler before 4.2.30 (the switch has nothing to stop in them).
BEFORE_4230 = {"_hub_baseline_pass", "_coverage_pass", "_graph_projection_pass"}


def test_every_scheduler_pass_is_either_in_the_table_or_older_than_4230() -> None:
    """A new background pass cannot be added without deciding what the kill switch does to it."""
    found = {name for name, _ in inspect.getmembers(MaintenanceScheduler, inspect.isfunction)
             if name.endswith("_pass")}
    assert found == {name for name, _ in PASSES} | BEFORE_4230


@pytest.mark.parametrize(("method", "table"), PASSES)
def test_a_pass_writes_with_the_switch_off(world, method, table) -> None:
    """The control: the same pass, same store, same flags, writes. Without this the next test proves nothing."""
    _all_on(world.home)
    getattr(MaintenanceScheduler, method)(_scheduler(world))
    assert _count(world, table) > 0


@pytest.mark.parametrize(("method", "table"), PASSES)
def test_no_pass_runs_or_writes_with_the_switch_on(world, method, table) -> None:
    _all_on(world.home)
    set_kill_switch(world.home, True)
    before = _snapshot(world)
    sched = _scheduler(world)
    getattr(MaintenanceScheduler, method)(sched)
    assert _snapshot(world) == before
    assert _count(world, table) == 0
    assert all(ok for _step, ok, *_ in sched.steps)  # stopped, not failed


@pytest.mark.parametrize("how", ["file", "environment"])
def test_the_environment_and_the_file_both_stop_the_passes(world, monkeypatch, how) -> None:
    _all_on(world.home)
    if how == "file":
        set_kill_switch(world.home, True)
    else:
        monkeypatch.setenv(KILL_ENV, "1")
    before = _snapshot(world)
    for method, _table in PASSES:
        getattr(MaintenanceScheduler, method)(_scheduler(world))
    assert _snapshot(world) == before


def test_the_dispute_and_backlog_passes_are_reached_with_the_switch_off_and_never_with_it_on(
        world, monkeypatch) -> None:
    from superlocalmemory.core import receipt_key
    from superlocalmemory.truth import backlog_pass, dispute_pass

    calls: list[str] = []
    monkeypatch.setattr(receipt_key, "load_or_create", lambda home: b"k" * 32)
    monkeypatch.setattr(dispute_pass.DisputePass, "run", lambda self: calls.append("dispute") or {})
    monkeypatch.setattr(backlog_pass.BacklogPass, "run", lambda self: calls.append("backlog") or {})
    _all_on(world.home)
    MaintenanceScheduler._slot_pass(_scheduler(world))
    assert calls == ["dispute", "backlog"]
    calls.clear()
    set_kill_switch(world.home, True)
    MaintenanceScheduler._slot_pass(_scheduler(world))
    assert calls == []


def test_a_whole_cycle_with_the_switch_on_writes_nothing_new(world) -> None:
    """The scheduler's own order of 4.2.30 steps, back to back."""
    _all_on(world.home)
    set_kill_switch(world.home, True)
    before = _snapshot(world)
    sched = _scheduler(world)
    for step in (MaintenanceScheduler._trust_pass, MaintenanceScheduler._signing_pass,
                 MaintenanceScheduler._slot_pass, MaintenanceScheduler._cue_pass,
                 MaintenanceScheduler._dream_pass):
        step(sched)
    assert _snapshot(world) == before


# -- reads ---------------------------------------------------------------------------------------

def test_recall_results_carry_no_standing_origin_dispute_or_source_notes(world) -> None:
    from superlocalmemory.server.answer_receipt import with_why
    from superlocalmemory.truth import recall_notes

    _all_on(world.home)
    items = [{"fact_id": "f1", "content": "x", "disputed": True,
              "origin": {"class": "agent", "taint": "untrusted", "label": "agent"},
              "source": {"state": "changed"}}]
    (shown,) = with_why(items)
    assert shown["why"]["standing"] == "disputed"  # control: on, the block is there
    set_kill_switch(world.home, True)
    assert recall_notes.enabled(world.home) is False
    assert recall_notes.origin_enabled(world.home) is False
    assert recall_notes.annotate(world.db.db_path, "default", items, slm_home=world.home) == items
    assert with_why(items) == items  # no ``why``: no standing, no origin


def test_the_firewall_and_pins_ignore_taint_with_the_switch_on(world) -> None:
    from superlocalmemory.trust import firewall

    _all_on(world.home)
    assert firewall.active() and " AND " in firewall.clause(world.db)
    set_kill_switch(world.home, True)
    assert not firewall.active() and not firewall.observing()
    assert firewall.state() == "off"
    assert firewall.clause(world.db) == ""  # session start, pins and core blocks add no taint filter
    assert firewall.withheld(world.db, ["a", "b"]) == set()
    assert firewall.check_promotion(world.db, "a", firewall.PIN).allowed


def test_the_need_gate_and_the_harm_gate_stand_down(world) -> None:
    from superlocalmemory.core import need_gate
    from superlocalmemory.outcomes import memory_report

    _all_on(world.home)
    assert need_gate.flag_state(world.home) == "on"
    set_kill_switch(world.home, True)
    assert need_gate.flag_state(world.home) is None
    import sqlite3

    conn = sqlite3.connect(world.db.db_path)
    conn.row_factory = sqlite3.Row
    try:
        from datetime import UTC, datetime

        assert memory_report._harm_gate(conn, "default", datetime.now(UTC),
                                        world.home).get("state") in (None, "off")
    finally:
        conn.close()


def test_imports_media_and_captions_are_refused_with_the_switch_on(world) -> None:
    from superlocalmemory.portability import service

    _all_on(world.home)
    service.require_on()  # control: on, it passes
    set_kill_switch(world.home, True)
    with pytest.raises(service.ImportOff):
        service.require_on()
    flags = FeatureFlags(world.home)
    for flag_id in ("portability.import", "media.captions", "media.document_pack", "media.regions",
                    "trust.signing", "compiler.dream", "delivery.cues", "delivery.need_gate",
                    "delivery.harm_gate", "judge.ladder", "truth.slots", "truth.leases"):
        assert not flags.observing(flag_id), flag_id


def test_every_flag_including_the_default_on_ones_reads_off(world) -> None:
    set_kill_switch(world.home, True)
    flags = FeatureFlags(world.home)
    assert flags.killed
    assert {f: flags.state(f).value for f in FLAGS} == {f: "off" for f in FLAGS}
    # An environment override cannot pull a flag back on past the switch.
    env = {env_name("trust.taint"): "on", KILL_ENV: ""}
    assert not FeatureFlags(world.home, environ=env).enabled("trust.taint")


def test_the_session_start_names_no_dispute_and_no_changed_source_with_the_switch_on(
        truth, tmp_path, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.core import session_brief as sb

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "start-home"))
    (tmp_path / "start-home").mkdir()
    _db_, conn, _old, _new, _leased = truth
    ws = sb.Workspace(root=str(tmp_path), name="p", keys=("p",))
    shown = sb.notes(conn, "default", ws)
    assert any(line.startswith("[open dispute]") for line in shown)  # control: off, they show
    assert any(line.startswith("[source changed]") for line in shown)
    set_kill_switch(tmp_path / "start-home", True)
    assert sb.notes(conn, "default", ws) == ("[project] p",)
