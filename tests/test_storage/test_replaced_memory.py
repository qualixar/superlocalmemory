# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A fact written for a memory its caller already replaced is retired in the
write's own transaction, and restored exactly when the replacement is undone."""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests.test_core.test_remember_replaces import (  # noqa: F401 - fixture
    ACTOR,
    _fact,
    _temporal,
    env,
)


def _memory(db, *contents) -> tuple[str, list[str]]:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=" ".join(contents)))
    return memory_id, [_fact(db, c, memory_id=memory_id) for c in contents]


def _late(db, memory_id, content) -> str:
    return db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id,
                                    content=content, fact_type=FactType.SEMANTIC))


def _rollback(runtime, case, key) -> None:
    runtime.transition_correction("default", case["case_id"], action="rollback",
                                  expected_version=case["version"], actor_id=ACTOR,
                                  idempotency_key=key)


def test_a_late_fact_is_retired_and_restored_exactly(env) -> None:
    db, runtime = env
    memory_id, _ = _memory(db, "Release day is Tuesday.")
    new = _fact(db, "Release day is Thursday.")
    [case] = runtime.replace_by_caller("default", memory_id, new, trusted_actor_id=ACTOR,
                                       idempotency_key="k1")["cases"]
    late = _late(db, memory_id, "The release owner is Alice.")
    retired = _temporal(db, late)
    assert retired["system_expired_at"] and retired["invalidated_by"] == new
    _rollback(runtime, case, "undo-1")
    restored = _temporal(db, late)
    assert restored["system_expired_at"] is None and restored["invalidated_by"] is None
    assert restored["system_created_at"] == retired["system_created_at"]


def test_late_facts_come_back_only_when_the_whole_replacement_is_undone(env) -> None:
    db, runtime = env
    memory_id, _ = _memory(db, "Fact one of two.", "Fact two of two.")
    new = _fact(db, "One fact replacing both.")
    cases = runtime.replace_by_caller("default", memory_id, new, trusted_actor_id=ACTOR,
                                      idempotency_key="k2")["cases"]
    assert len(cases) == 2
    late = _late(db, memory_id, "A fact enrichment wrote later.")
    _rollback(runtime, cases[0], "undo-2a")
    assert _temporal(db, late)["system_expired_at"]          # still partly replaced
    _rollback(runtime, cases[1], "undo-2b")
    assert _temporal(db, late)["system_expired_at"] is None


def test_a_single_fact_replacement_does_not_retire_later_facts(env) -> None:
    db, runtime = env
    memory_id, [one, _two] = _memory(db, "First claim.", "Second claim.")
    runtime.replace_by_caller("default", one, _fact(db, "Corrected claim."),
                              trusted_actor_id=ACTOR, idempotency_key="k3")
    late = _late(db, memory_id, "A later claim.")
    assert _temporal(db, late)["system_expired_at"] is None


def test_re_storing_a_retired_late_fact_keeps_its_one_case(env) -> None:
    db, runtime = env
    memory_id, _ = _memory(db, "Status: amber.")
    runtime.replace_by_caller("default", memory_id, _fact(db, "Status: green."),
                              trusted_actor_id=ACTOR, idempotency_key="k4")
    late = _late(db, memory_id, "Amber since Monday.")
    again = AtomicFact(fact_id=late, profile_id="default", memory_id=memory_id,
                       content="Amber since Monday morning.", fact_type=FactType.SEMANTIC)
    assert db.store_fact(again) == late
    assert len(db.execute("SELECT 1 FROM correction_cases WHERE predecessor_fact_id=?",
                          (late,))) == 1
    assert _temporal(db, late)["system_expired_at"]


def test_re_storing_a_fact_with_another_active_case_does_not_fail(env) -> None:
    """A fact already retired by some other correction keeps that case; a
    re-store must neither fail on the one-active-case rule nor add a second."""
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
        transition_on_connection,
    )

    db, runtime = env
    memory_id, [root, corrected] = _memory(db, "Root text.", "Corrected earlier.")
    fixed = _fact(db, "Corrected text.")
    reviewer = CorrectionActor(ACTOR, "host_authenticated", "trusted")
    with db.raw_connection() as conn:
        checks = dict(is_profile_active=lambda _p: True, is_actor_trusted=lambda _a: True)
        propose_on_connection(conn, case_id="direct1", profile_id="default", scope="personal",
                              predecessor_fact_id=corrected, successor_fact_id=fixed,
                              reason_code="direct_content_correction", actor=reviewer,
                              idempotency_key="direct1", **checks)
        transition_on_connection(conn, case_id="direct1", expected_version=0, actor=reviewer,
                                 operation_id="direct1-apply", from_status="proposed",
                                 to_status="applied", mutate_temporal=True, **checks)
    runtime.replace_by_caller("default", memory_id, _fact(db, "Whole new text."),
                              trusted_actor_id=ACTOR, idempotency_key="k5")
    before = _temporal(db, corrected)
    again = AtomicFact(fact_id=corrected, profile_id="default", memory_id=memory_id,
                       content="Corrected earlier, stored again.", fact_type=FactType.SEMANTIC)
    assert db.store_fact(again) == corrected
    assert len(db.execute("SELECT 1 FROM correction_cases WHERE predecessor_fact_id=?",
                          (corrected,))) == 1
    assert _temporal(db, corrected) == before


def test_an_ordinary_write_is_untouched(env) -> None:
    db, _ = env
    memory_id, [first] = _memory(db, "Nothing replaced here.")
    late = _late(db, memory_id, "Another plain fact.")
    assert _temporal(db, late)["system_expired_at"] is None
    assert db.execute("SELECT 1 FROM correction_cases") == []


def test_a_late_fact_is_retired_even_when_the_successor_was_rescoped(env) -> None:
    """L1-14: re-scoping the successor after the whole-memory replacement was
    applied makes the ledger refuse the late case (its scope check compares
    against the successor's CURRENT scope). The write must never fail for
    this -- the fact is retired by the safest available route instead, so a
    replaced memory's fact is never left current and visible in recall."""
    db, runtime = env
    memory_id, _ = _memory(db, "Checkpoint one: migration half done.")
    new = _fact(db, "Checkpoint two: migration done.")
    runtime.replace_by_caller("default", memory_id, new, trusted_actor_id=ACTOR,
                              idempotency_key="k-rescope")
    db.execute("UPDATE atomic_facts SET scope='global' WHERE fact_id=?", (new,))
    late = db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id,
                                    content="Enrichment: the migration touched 3 tables.",
                                    fact_type=FactType.SEMANTIC))
    retired = _temporal(db, late)
    assert retired["system_expired_at"], "late fact must never surface current"
    assert retired["invalidated_by"] == new
