# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""#308: ``repair-saves`` restores a blanked copy of a save itself.

The reporter's 2,231 saves failed on "queryable ingestion memory content
mismatch" AFTER the one-time store repair had run (``upgrade-repair.json``
complete), so no tool reached them. The dry run said "blanked copy: the store
repair restores it". Now the save repair plans the restore, with the same
guarded statements the store repair uses, then requeues the save as it does for
every other re-file. ``save_repair`` never reads ``upgrade-repair.json``: the
repair does not depend on whether the one-time repair ran.

Fixtures reproduce the reporter's row shape on real writers: a save made
through the ingestion command, then its memory (and fact) overwritten with the
blanked text and the operation failed with ten attempts and a dead-letter row.
"""

# ruff: noqa: F811 (pytest fixture imported and redeclared as a parameter)

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from argparse import Namespace

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingest_gate import apply_ingest_gate
from superlocalmemory.core.ingestion_command import IngestionState
from superlocalmemory.storage import save_repair
from superlocalmemory.storage.save_repair_apply import apply
from tests.test_governance.test_saves_refiled_after_scope_drift import (
    CONTENT,
    _corrupt_like_4124,
    _plan,
    _submit,
)

PATH = "/Users/me/Documents/mail-anhang/report-2026.pdf"
ORIGINAL = f"[Dokument] {PATH} (geaendert 2026-10-04)\n\n# Quartalsbericht fuer Atlas"
BLANKED = ("[Dokument] [REDACTED:ENTROPY:.pdf] (geaendert 2026-10-04)"
           "\n\n# Quartalsbericht fuer Atlas")
SECRET = "AKIAIOSFODNN7EXAMPLE"          # AWS documentation example key shape


@pytest.fixture
def engine(engine_with_mock_deps):
    from superlocalmemory.storage.migrations import M031_dead_letter_operations

    engine = engine_with_mock_deps
    with sqlite3.connect(engine._config.db_path) as conn:
        M031_dead_letter_operations.apply(conn)
    return engine


def _blanked_save(engine, original=ORIGINAL, blanked=BLANKED, *, facts="blanked"):
    """A failed save whose own memory is ``blanked``; returns (operation_id, memory_id).

    ``facts``: "blanked" keeps the save's fact with the blanked gate text,
    "none" removes it (the reporter's zero-fact shape), "other" gives it text
    that is neither the blanked nor the restored form.
    """
    command = build_engine_ingestion_command(engine)
    receipt = _submit(engine, command, f"placeholder {uuid.uuid4().hex}")
    db = engine._db
    memory_id = db.execute(
        "SELECT memory_id FROM memories WHERE json_extract(metadata_json,"
        "'$.ingestion_operation_id')=?", (receipt.operation_id,))[0]["memory_id"]
    db.execute("UPDATE ingestion_operations SET raw_content=? WHERE operation_id=?",
               (original, receipt.operation_id))
    db.execute("UPDATE memories SET content=? WHERE memory_id=?", (blanked, memory_id))
    fact_id = receipt.queryable_fact_ids[0]
    if facts == "none":
        db.execute("DELETE FROM atomic_facts WHERE fact_id=?", (fact_id,))
    else:
        text = apply_ingest_gate(blanked).fact_content if facts == "blanked" else "extracted"
        db.execute("UPDATE atomic_facts SET content=? WHERE fact_id=?", (text, fact_id))
    if facts == "none":
        _corrupt_like_4124(engine, receipt, onto_fact="gone", error=CONTENT)
    else:       # the fact stays; the save fails and is dead-lettered like the reporter's
        db.execute("UPDATE ingestion_operations SET state='failed', attempt_count=10, "
                   "last_error=? WHERE operation_id=?", (CONTENT, receipt.operation_id))
        db.execute("INSERT INTO dead_letter_operations (original_op_id, operation_type, "
                   "content, error, attempt_count, profile_id, dead_lettered_at) "
                   "VALUES (?,?,?,?,?,?,?)", (receipt.operation_id, "M018", "", CONTENT, 10,
                                              engine._profile_id, time.time()))
    return receipt.operation_id, memory_id, fact_id


def _row(engine, sql, *params):
    return engine._db.execute(sql, params)[0]


def test_the_dry_run_lists_the_restore_and_changes_nothing(engine):
    zero, _, _ = _blanked_save(engine, facts="none")
    with_fact, mem, _ = _blanked_save(engine, ORIGINAL + " zwei", BLANKED + " zwei")
    plan = _plan(engine)
    actions = {item.operation_id: item.action for item in plan.refile}
    assert actions == {zero: "restore_then_add_fact", with_fact: "restore_then_repoint"}
    assert plan.review == ()
    assert plan.summary()["by_action"] == {"restore_then_add_fact": 1,
                                           "restore_then_repoint": 1}
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == BLANKED + " zwei"


def test_apply_restores_text_and_facts_and_requeues_with_attempts_zero(engine):
    op_id, mem, fact = _blanked_save(engine)
    result = apply(engine._config.db_path, _plan(engine))
    assert (result.refiled, result.skipped) == (1, 0)
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == apply_ingest_gate(ORIGINAL).full_content
    assert _row(engine, "SELECT content FROM atomic_facts WHERE fact_id=?", fact)["content"] \
        == apply_ingest_gate(ORIGINAL).fact_content
    op = _row(engine, "SELECT * FROM ingestion_operations WHERE operation_id=?", op_id)
    assert (op["state"], op["attempt_count"], op["last_error"]) == ("queryable", 0, "")
    assert json.loads(op["queryable_fact_ids_json"]) == [fact]
    assert (op["final_fact_ids_json"], op["derivation_state_json"]) == ("[]", "{}")
    assert not engine._db.execute(
        "SELECT 1 FROM dead_letter_operations WHERE original_op_id=?", (op_id,))


def test_the_materializer_finishes_a_restored_save(engine):
    op_id, mem, _ = _blanked_save(engine)
    apply(engine._config.db_path, _plan(engine))
    done = build_engine_ingestion_command(engine).materialize(op_id)
    assert done.state is IngestionState.COMPLETE, done.last_error
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == ORIGINAL
    assert "[REDACTED:" not in " ".join(
        r["content"] for r in engine._db.execute(
            "SELECT content FROM atomic_facts WHERE memory_id=?", (mem,)))


def test_a_memory_with_no_facts_gets_its_fact_from_the_original(engine):
    op_id, mem, _ = _blanked_save(engine, facts="none")
    result = apply(engine._config.db_path, _plan(engine))
    assert (result.refiled, result.skipped) == (1, 0)
    facts = engine._db.execute("SELECT fact_id, content FROM atomic_facts WHERE memory_id=?",
                               (mem,))
    assert [f["content"] for f in facts] == [apply_ingest_gate(ORIGINAL).fact_content]
    op = _row(engine, "SELECT * FROM ingestion_operations WHERE operation_id=?", op_id)
    assert (op["state"], op["attempt_count"]) == ("queryable", 0)
    assert json.loads(op["queryable_fact_ids_json"]) == [facts[0]["fact_id"]]
    done = build_engine_ingestion_command(engine).materialize(op_id)
    assert done.state is IngestionState.COMPLETE, done.last_error


def test_an_original_over_the_clamp_restores_the_clamped_fact(engine):
    filler = "lorem ipsum dolor sit amet " * 1200
    original = f"Build notes {PATH}\n{filler}"
    blanked = f"Build notes [REDACTED:ENTROPY:.pdf]\n{filler}"
    assert apply_ingest_gate(blanked).truncated
    _, mem, fact = _blanked_save(engine, original, blanked)
    assert {i.action for i in _plan(engine).refile} == {"restore_then_repoint"}
    assert apply(engine._config.db_path, _plan(engine)).refiled == 1
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == original
    restored = _row(engine, "SELECT content FROM atomic_facts WHERE fact_id=?", fact)["content"]
    assert restored == apply_ingest_gate(original).fact_content
    assert "[REDACTED:" not in restored


def test_a_clamped_original_with_no_facts_gets_a_clamped_fact(engine):
    filler = "lorem ipsum dolor sit amet " * 1200
    original, blanked = f"Notes {PATH}\n{filler}", f"Notes [REDACTED:ENTROPY:.pdf]\n{filler}"
    _, mem, _ = _blanked_save(engine, original, blanked, facts="none")
    assert apply(engine._config.db_path, _plan(engine)).refiled == 1
    facts = engine._db.execute("SELECT content FROM atomic_facts WHERE memory_id=?", (mem,))
    assert len(facts) == 1 and len(facts[0]["content"]) <= 24000 + 400


def test_an_original_holding_a_secret_stays_for_review_and_is_never_printed(engine, capsys,
                                                                            monkeypatch):
    from superlocalmemory.cli import ops_repair_cmd

    original = f"deploy key {SECRET} for the Atlas job"
    op_id, mem, _ = _blanked_save(engine, original, "deploy key [REDACTED:ENTROPY:x] for the "
                                  "Atlas job")
    plan = _plan(engine)
    assert plan.refile == ()
    assert dict(plan.review)[op_id] == "original holds what looks like a secret; review it"
    assert apply(engine._config.db_path, plan).refiled == 0
    assert "[REDACTED:" in _row(engine, "SELECT content FROM memories WHERE memory_id=?",
                                mem)["content"]
    monkeypatch.setattr(ops_repair_cmd, "_db_path", lambda: engine._config.db_path)
    ops_repair_cmd.cmd_ops_repair_saves(Namespace(apply=False, json=False))
    assert SECRET not in capsys.readouterr().out


def test_a_rejected_original_stays_for_review_with_the_gate_reason(engine):
    original = f"You are summarizing a Claude Code session about {PATH}"
    blanked = "You are summarizing a Claude Code session about [REDACTED:ENTROPY:.pdf]"
    op_id, mem, _ = _blanked_save(engine, original, blanked)
    plan = _plan(engine)
    assert plan.refile == ()
    why = dict(plan.review)[op_id]
    assert why.startswith("original is rejected by the ingest gate")
    assert apply(engine._config.db_path, plan).refiled == 0
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == blanked


def test_facts_that_are_neither_blanked_nor_restored_are_not_adopted(engine):
    op_id, _, _ = _blanked_save(engine, facts="other")
    assert dict(_plan(engine).review)[op_id] == \
        "its searchable facts were changed after saving"


def test_a_fact_already_restored_is_adopted_and_the_memory_restored(engine):
    op_id, mem, fact = _blanked_save(engine)
    engine._db.execute("UPDATE atomic_facts SET content=? WHERE fact_id=?",
                       (apply_ingest_gate(ORIGINAL).fact_content, fact))
    assert {i.action for i in _plan(engine).refile} == {"restore_then_repoint"}
    assert apply(engine._config.db_path, _plan(engine)).refiled == 1
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == ORIGINAL


def test_text_edited_after_saving_is_still_listed_with_its_old_reason(engine):
    op_id, mem, _ = _blanked_save(engine)
    engine._db.execute("UPDATE memories SET content='the owner rewrote this' "
                       "WHERE memory_id=?", (mem,))
    plan = _plan(engine)
    assert plan.refile == ()
    assert dict(plan.review)[op_id] == save_repair.NOT_RESTORED_DIFFERENT


def test_a_memory_edited_after_the_review_is_skipped_not_overwritten(engine):
    op_id, mem, _ = _blanked_save(engine)
    reviewed = _plan(engine)
    engine._db.execute("UPDATE memories SET content=? WHERE memory_id=?",
                       (BLANKED + " and then some", mem))
    result = apply(engine._config.db_path, reviewed)
    assert (result.refiled, result.skipped) == (0, 1)
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == BLANKED + " and then some"
    assert _row(engine, "SELECT state FROM ingestion_operations WHERE operation_id=?",
                op_id)["state"] == "failed"


def test_a_fact_edited_after_the_review_leaves_the_memory_blanked(engine):
    _, mem, fact = _blanked_save(engine)
    reviewed = _plan(engine)
    engine._db.execute("UPDATE atomic_facts SET content='drifted' WHERE fact_id=?", (fact,))
    result = apply(engine._config.db_path, reviewed)
    assert (result.refiled, result.skipped) == (0, 1)
    assert _row(engine, "SELECT content FROM memories WHERE memory_id=?", mem)["content"] \
        == BLANKED


def test_a_second_apply_is_a_no_op(engine):
    _blanked_save(engine)
    apply(engine._config.db_path, _plan(engine))
    assert _plan(engine).refile == ()


def test_the_cli_dry_run_names_the_actions_and_counts_the_review_reasons(engine, capsys,
                                                                         monkeypatch):
    from superlocalmemory.cli import ops_repair_cmd

    _blanked_save(engine)
    _blanked_save(engine, facts="none", original=ORIGINAL + " b", blanked=BLANKED + " b")
    _blanked_save(engine, f"key {SECRET} x", "key [REDACTED:ENTROPY:x] x")
    monkeypatch.setattr(ops_repair_cmd, "_db_path", lambda: engine._config.db_path)
    ops_repair_cmd.cmd_ops_repair_saves(Namespace(apply=False, json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["by_action"] == {"restore_then_add_fact": 1, "restore_then_repoint": 1}
    assert out["left_for_review_by_reason"] == {
        "original holds what looks like a secret; review it": 1}
    ops_repair_cmd.cmd_ops_repair_saves(Namespace(apply=False, json=False))
    text = capsys.readouterr().out
    assert "restore_then_add_fact" in text and "restore_then_repoint" in text
    assert "original holds what looks like a secret; review it" in text


def test_the_cli_apply_takes_the_backup_first(engine, capsys, monkeypatch):
    from pathlib import Path

    from superlocalmemory.cli import ops_repair_cmd

    _, mem, _ = _blanked_save(engine)
    monkeypatch.setattr(ops_repair_cmd, "_db_path", lambda: engine._config.db_path)
    ops_repair_cmd.cmd_ops_repair_saves(Namespace(apply=True, json=True))
    out = json.loads(capsys.readouterr().out)
    assert out["refiled"] == 1 and Path(out["backup"]).is_file()
    held = sqlite3.connect(out["backup"]).execute(
        "SELECT content FROM memories WHERE memory_id=?", (mem,)).fetchone()[0]
    assert held == BLANKED                      # the copy holds the state before the repair
    assert save_repair.REASONS                  # the planner module is unchanged in shape
