# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A GDPR erasure of a person erases them even when correction history names them.

Varun's decision (2026-10-07): "Erasure wins: erase the fact anyway, close its
correction cases as 'erased on request' and keep only a text-free audit row
(ids, when, who). The right to erasure isn't blocked by internal history."

Before 4.1.22 an entity erasure removed the search entries of every fact naming
the person and then deleted the facts one by one. The correction ledger refers
to both facts of a case ``ON DELETE RESTRICT``, so the first fact a person had
edited (or whose correction was applied) stopped the loop with an integrity
error: facts left stored but no longer findable, the entity still there, and the
HTTP route reporting an internal error.

An ordinary delete of such a fact still refuses (tests/core/test_delete_refusal_race.py
and the delete-protection tests); only the GDPR path closes the cases.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.helpers.env_capabilities import purge_keyword_index_on_old_sqlite

_NAME = "Quorvantel Brask"
_FIRST = f"{_NAME} stores the synthetic copper compass in the attic chest."
_SECOND = f"{_NAME} walks the synthetic hound along the canal at dawn."
_EDIT = f"{_NAME} stores the synthetic copper compass in the cellar chest."
_BYSTANDER = "Hollowmere keeps the synthetic brass lantern on the porch."
_REQUEST_RECORDS = {"compliance_audit.target_id", "erasure_receipts.subject_id",
                    "projection_obligations.subject_id"}


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _store(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    receipt = canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                              require_complete=True, return_receipt=True)
    return list(receipt.final_fact_ids)[0]


def _writer(engine, kind_name: str, payload: dict) -> dict:
    """Run one sole-writer command in its own transaction, as the writer does."""
    from superlocalmemory.core.remember_runtime import _execute_mutation
    from superlocalmemory.storage.write_coordinator import CommandKind

    db = engine._db
    with db.raw_connection() as conn:
        return _execute_mutation(db, CommandKind[kind_name], engine._profile_id,
                                 payload, connection=conn)


def _person_edits(engine, fact_id: str, successor: str, key: str) -> dict:
    return _writer(engine, "PROPOSE_CORRECTION", {
        "fact_id": fact_id, "successor_fact_id": successor, "content": _EDIT,
        "trusted_actor_id": "person-test", "idempotency_key": key})


def _n(engine, sql: str, args: tuple = ()) -> int:
    try:
        return int(dict(engine._db.execute(sql, args)[0])["n"])
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0
        raise


def _text_copies(engine, needle: str) -> dict[str, int]:
    """Every column of every table in memory.db that still holds the needle."""
    hits: dict[str, int] = {}
    conn = sqlite3.connect(f"file:{engine._db.db_path}?mode=ro", uri=True)
    try:
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
            try:
                cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            except sqlite3.Error:
                continue
            for col in cols:
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE instr(lower('
                                     f'CAST("{col}" AS TEXT)), ?) > 0',
                                     (needle.lower(),)).fetchone()[0]
                except sqlite3.Error:
                    continue
                if n:
                    hits[f"{table}.{col}"] = n
    finally:
        conn.close()
    return hits


def _footprint(engine, fact_ids: list[str]) -> dict[str, int]:
    ph = ",".join("?" * len(fact_ids))
    ids = tuple(fact_ids)
    return {
        "facts": _n(engine, f"SELECT COUNT(*) AS n FROM atomic_facts WHERE fact_id IN ({ph})", ids),
        "bm25": _n(engine, f"SELECT COUNT(*) AS n FROM bm25_tokens WHERE fact_id IN ({ph})", ids),
        "temporal": _n(engine, "SELECT COUNT(*) AS n FROM fact_temporal_validity "
                       f"WHERE fact_id IN ({ph})", ids),
        "cases": _n(engine, "SELECT COUNT(*) AS n FROM correction_cases WHERE "
                    f"predecessor_fact_id IN ({ph}) OR successor_fact_id IN ({ph})", ids * 2),
    }


def _live_index_has(engine, fact_id: str) -> bool:
    index = getattr(getattr(engine, "_retrieval_engine", None), "_bm25", None)
    return index is not None and fact_id in getattr(index, "_fact_id_set", ())


@pytest.mark.parametrize("decided", [False, True], ids=["proposed", "applied"])
def test_gdpr_entity_erasure_closes_the_cases_and_erases_everything(
    engine_with_mock_deps, decided,
):
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    engine = engine_with_mock_deps
    first, second = _store(engine, _FIRST), _store(engine, _SECOND)
    bystander = _store(engine, _BYSTANDER)
    edit = _person_edits(engine, first, "succ" + first[:12], "edit-1")
    assert edit["ok"] and edit["status"] == "proposed", edit
    successor = edit["successor_fact_id"]
    if decided:
        applied = _writer(engine, "APPLY_CORRECTION", {
            "case_id": edit["case_id"], "expected_version": edit["version"],
            "trusted_actor_id": "person-test", "idempotency_key": "apply-1"})
        assert applied["status"] == "applied", applied
    targets = [first, second, successor]
    assert _footprint(engine, targets)["facts"] == 3

    counts = GDPRCompliance(engine._db, engine=engine).forget_entity(_NAME, engine._profile_id)

    assert counts["facts"] == 3, counts
    assert counts["erasure_complete"] == 1, counts
    assert counts["correction_cases_erased"] == 1, counts
    assert _footprint(engine, targets) == {"facts": 0, "bm25": 0, "temporal": 0, "cases": 0}
    assert not any(_live_index_has(engine, f) for f in targets)
    assert _n(engine, "SELECT COUNT(*) AS n FROM canonical_entities WHERE profile_id = ? "
              "AND canonical_name = ?", (engine._profile_id, _NAME)) == 0
    purge_keyword_index_on_old_sqlite(engine._db.db_path)
    for needle in ("copper compass", "cellar chest", "canal at dawn", "attic chest"):
        assert _text_copies(engine, needle) == {}, needle
    # The erased person's name survives only where the erasure request itself is
    # recorded (its receipt, audit entry and obligations name the subject).
    assert set(_text_copies(engine, "quorvantel")) <= _REQUEST_RECORDS

    audit = [dict(r) for r in engine._db.execute(
        "SELECT * FROM correction_cases_erased WHERE case_id = ?", (edit["case_id"],))]
    assert len(audit) == 1, audit
    row = audit[0]
    assert row["closed_reason"] == "erased_on_request"
    assert row["prior_status"] == ("applied" if decided else "proposed")
    assert {row["predecessor_fact_id"], row["successor_fact_id"]} == {first, successor}
    assert row["profile_id"] == engine._profile_id and row["actor_id"] == "gdpr"
    assert row["erased_at"] and row["erasure_id"]
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_events WHERE case_id = ?",
              (edit["case_id"],)) == 0
    # The person who was not named is untouched.
    assert _footprint(engine, [bystander])["facts"] == 1
    assert _live_index_has(engine, bystander)


def test_an_ordinary_delete_still_refuses_a_person_edited_memory(engine_with_mock_deps):
    from superlocalmemory.core.mutations import delete_fact_authorized
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine = engine_with_mock_deps
    first = _store(engine, _FIRST)
    edit = _person_edits(engine, first, "succ" + first[:12], "edit-2")
    with pytest.raises(CanonicalMutationConflict, match="protected by correction history"):
        delete_fact_authorized(engine, first, trusted_actor_id=_actor(), source_agent_id="test")
    assert _footprint(engine, [first])["facts"] == 1
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases_erased") == 0
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases WHERE case_id = ?",
              (edit["case_id"],)) == 1


def test_an_entity_left_in_the_graph_is_not_reported_erased(engine_with_mock_deps, monkeypatch):
    """The graph keeps its own copy of the entity node, name included. When it
    refuses to drop it, the erasure is incomplete and every surface must say so."""
    from superlocalmemory.compliance import gdpr
    from superlocalmemory.server.routes.compliance import _erasure_succeeded

    engine = engine_with_mock_deps
    _store(engine, _SECOND)
    monkeypatch.setattr(gdpr, "_unproject_entity", lambda _entity_id: False)
    counts = gdpr.GDPRCompliance(engine._db, engine=engine).forget_entity(
        _NAME, engine._profile_id)
    assert counts["projection_failed"] == 1
    assert counts["erasure_complete"] == 0 and counts["erasure_provable"] == 0, counts
    assert _erasure_succeeded(counts) is False
