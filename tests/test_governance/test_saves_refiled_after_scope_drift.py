# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saves folded onto another memory are re-filed (4.1.25 L-06), and attribution.

The owner's store holds 60 dead-lettered saves: 30 "scope mismatch", 29
"content mismatch", 1 "fact profile mismatch or missing fact", and 31 Codex
saves of 2026-10-10 whose operations say personal while the memory enrichment
found was global. Builds up to 2026-10-11 deduplicated a save's searchable fact
against ANY equal fact in the profile, so it was folded onto another memory's
fact; enrichment then judged the operation against that other memory.

``_corrupt_like_4124`` reproduces exactly that state on a fixture made with the
real writer, the repair is run dry and applied, and the re-filed saves then
materialize with the real materializer. A last test pins the writer: an equal
save now keeps its own memory.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionRequest, IngestionState
from superlocalmemory.storage import save_repair
from superlocalmemory.storage.save_repair_apply import apply

SCOPE = "queryable ingestion memory scope mismatch"
CONTENT = "queryable ingestion memory content mismatch"
MISSING = "queryable ingestion fact profile mismatch or missing fact"


def _submit(engine, command, text, *, scope="personal", agent=""):
    meta = {"agent_id": agent} if agent else {}
    return command.submit(IngestionRequest(
        content=text, profile_id=engine._profile_id, source_type="mcp",
        trusted_actor_id="test:owned", idempotency_key=uuid.uuid4().hex,
        scope=scope, metadata=meta))


def _corrupt_like_4124(engine, receipt, *, onto_fact: str, error: str) -> None:
    """The 4.1.24 state: the op points at another memory's fact, own memory factless."""
    db = engine._db
    own = receipt.queryable_fact_ids[0]
    db.execute("DELETE FROM atomic_facts WHERE fact_id=?", (own,))
    db.execute(
        "UPDATE ingestion_operations SET state='failed', attempt_count=10, last_error=?, "
        "queryable_fact_ids_json=? WHERE operation_id=?",
        (error, json.dumps([onto_fact]), receipt.operation_id))
    db.execute(
        "INSERT INTO dead_letter_operations (original_op_id, operation_type, content, error, "
        "attempt_count, profile_id, dead_lettered_at) VALUES (?,?,?,?,?,?,?)",
        (receipt.operation_id, "M018", "", error, 10, engine._profile_id, time.time()))


@pytest.fixture
def drifted(engine_with_mock_deps):
    from superlocalmemory.storage.migrations import M031_dead_letter_operations

    engine = engine_with_mock_deps
    with sqlite3.connect(engine._config.db_path) as conn:
        M031_dead_letter_operations.apply(conn)
    command = build_engine_ingestion_command(engine)
    text = "Ship the Atlas release on Friday after the security review."
    shared = _submit(engine, command, text, scope="global", agent="claude")
    other = _submit(engine, command, "Atlas notes: Ship it on Friday. Review first.")
    ops = {
        "scope": _submit(engine, command, text, agent="codex"),
        "content": _submit(engine, command, "Review first.", agent="codex"),
        "missing": _submit(engine, command, "A note whose folded fact was erased."),
        "edited": _submit(engine, command, "A note the owner later edited."),
    }
    _corrupt_like_4124(engine, ops["scope"], onto_fact=shared.queryable_fact_ids[0],
                       error=SCOPE)
    _corrupt_like_4124(engine, ops["content"], onto_fact=other.queryable_fact_ids[0],
                       error=CONTENT)
    _corrupt_like_4124(engine, ops["missing"], onto_fact="gone-fact", error=MISSING)
    db = engine._db
    db.execute("DELETE FROM memories WHERE json_extract(metadata_json,"
               "'$.ingestion_operation_id')=?", (ops["missing"].operation_id,))
    _corrupt_like_4124(engine, ops["edited"], onto_fact="gone-fact", error=CONTENT)
    db.execute("UPDATE memories SET content='edited by the owner' WHERE json_extract("
               "metadata_json,'$.ingestion_operation_id')=?", (ops["edited"].operation_id,))
    return engine, command, ops, shared


def _plan(engine):
    with sqlite3.connect(engine._config.db_path) as conn:
        return save_repair.plan(conn)


def test_the_dry_run_names_every_case_and_changes_nothing(drifted):
    engine, _, ops, _ = drifted
    before = engine._db.execute("SELECT COUNT(*) AS n FROM memories")[0]["n"]
    plan = _plan(engine)
    actions = {item.operation_id: item.action for item in plan.refile}
    assert actions == {ops["scope"].operation_id: "add_fact",
                       ops["content"].operation_id: "add_fact",
                       ops["missing"].operation_id: "add_memory_and_fact"}
    assert dict(plan.review) == {ops["edited"].operation_id: save_repair.NOT_RESTORED_DIFFERENT}
    assert engine._db.execute("SELECT COUNT(*) AS n FROM memories")[0]["n"] == before


def test_apply_refiles_backs_up_and_the_materializer_finishes_them(drifted, tmp_path):
    engine, command, ops, shared = drifted
    db_path = engine._config.db_path
    result = apply(db_path, _plan(engine))
    assert (result.refiled, result.skipped) == (3, 0)
    assert sqlite3.connect(result.backup).execute(
        "SELECT COUNT(*) FROM ingestion_operations").fetchone()[0] > 0
    for name in ("scope", "content", "missing"):
        op_id = ops[name].operation_id
        done = command.materialize(op_id)
        assert done.state is IngestionState.COMPLETE, (name, done.last_error)
        mem = engine._db.execute(
            "SELECT m.scope, m.content FROM atomic_facts f JOIN memories m ON "
            "m.memory_id=f.memory_id WHERE f.fact_id=?", (done.fact_ids[0],))[0]
        op = engine._db.execute("SELECT scope, raw_content FROM ingestion_operations "
                                "WHERE operation_id=?", (op_id,))[0]
        assert (mem["scope"], mem["content"]) == (op["scope"], op["raw_content"])
    left = engine._db.execute("SELECT COUNT(*) AS n FROM dead_letter_operations "
                              "WHERE original_op_id != ?", (ops["edited"].operation_id,))
    assert left[0]["n"] == 0
    # The global memory the save had been folded onto is untouched.
    still = engine._db.get_fact(shared.queryable_fact_ids[0])
    assert still is not None and still.scope == "global"


def test_a_second_apply_is_a_no_op(drifted):
    engine, *_ = drifted
    apply(engine._config.db_path, _plan(engine))
    again = _plan(engine)
    assert again.refile == ()


def test_agent_names_are_backfilled_from_the_save_record_only(drifted):
    engine, *_ = drifted
    db = engine._db
    db.execute("UPDATE memories SET metadata_json=json_remove(metadata_json,'$.agent_id')")
    plan = _plan(engine)
    named = {agent for _, agent in plan.attribution}
    assert named == {"claude", "codex"}
    assert plan.unattributed_by_source == {"mcp": len(db.execute(
        "SELECT 1 FROM memories")) - len(plan.attribution)}
    result = apply(engine._config.db_path, plan)
    assert result.agent_names_backfilled == len(plan.attribution)
    rows = db.execute("SELECT json_extract(metadata_json,'$.agent_id_source') AS s FROM memories "
                      "WHERE json_extract(metadata_json,'$.agent_id') IS NOT NULL")
    assert {r["s"] for r in rows} == {"backfill:operation"}


def test_the_cli_dry_run_reports_without_writing(drifted, capsys, monkeypatch):
    from argparse import Namespace

    from superlocalmemory.cli import ops_repair_cmd

    engine, *_ = drifted
    monkeypatch.setattr(ops_repair_cmd, "_db_path", lambda: engine._config.db_path)
    ops_repair_cmd.cmd_ops_repair_saves(Namespace(apply=False, json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["saves_to_refile"] == 3
    assert out["by_action"] == {"add_fact": 2, "add_memory_and_fact": 1}


def test_an_equal_save_now_keeps_its_own_memory(engine_with_mock_deps):
    """The writer: an equal fact in another memory no longer absorbs the save."""
    engine = engine_with_mock_deps
    command = build_engine_ingestion_command(engine)
    text = "The staging database moved to db-stage-2 on Monday."
    first = _submit(engine, command, text, scope="global")
    second = _submit(engine, command, text, scope="personal")
    assert first.queryable_fact_ids != second.queryable_fact_ids
    fact = engine._db.get_fact(second.queryable_fact_ids[0])
    mem = engine._db.execute("SELECT scope, content FROM memories WHERE memory_id=?",
                             (fact.memory_id,))[0]
    assert (mem["scope"], mem["content"]) == ("personal", text)
    assert command.materialize(second.operation_id).state is IngestionState.COMPLETE
