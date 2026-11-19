# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A caller saying "this new memory replaces that one" is recorded through the
correction ledger: confined to the active profile, all-or-nothing, replayable
without a second mark, undoable exactly, and never undone by store repair."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
)
from superlocalmemory.storage.migrations import M052_memory_kinds as m052
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord


@pytest.fixture(autouse=True)
def _origin_firewall_off(monkeypatch):
    """These tests are about what session start selects, not the origin firewall: they save through agent
    doors (``agent_inferred``), which the firewall (on by default since 19 Nov) holds back until a promotion.
    tests/test_trust_firewall covers that rule."""
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "off")

ACTOR = "daemon-capability:" + "a" * 64
_TEMPORAL = ("valid_from", "valid_until", "system_created_at", "system_expired_at",
             "invalidated_by", "invalidation_reason")


@pytest.fixture()
def env(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                      M033_projection_transactions, M034_obligation_integrity):
        migration.apply(conn)
    conn.commit()
    conn.close()
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as raw:
        M042_correction_case_ledger.apply(raw)
        m052.apply(raw)
    db.execute("INSERT OR IGNORE INTO profiles(profile_id, name, description) "
               "VALUES ('other', 'other', 'test')")
    runtime = CanonicalRememberRuntime(db=db, profile_id="default",
                                       writer=lambda _r, _o: [],
                                       journal_path=tmp_path / "journal.db")
    runtime.start()
    yield db, runtime
    runtime.stop()


def _fact(db, content, *, profile="default", memory_id=None, scope="personal",
          shared_with=None, kind=None, source=None) -> str:
    if memory_id is None:
        memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=content,
                                                 scope=scope, shared_with=shared_with))
    fact = AtomicFact(profile_id=profile, memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC, scope=scope, shared_with=shared_with)
    fact.memory_kind, fact.memory_kind_source = kind, source
    return db.store_fact(fact)


def _temporal(db, fact_id) -> dict:
    rows = db.execute("SELECT " + ", ".join(_TEMPORAL)
                      + " FROM fact_temporal_validity WHERE fact_id=?", (fact_id,))
    return dict(rows[0]) if rows else {}


def _replace(runtime, replaces, successor, key="op-1"):
    return runtime.replace_by_caller("default", replaces, successor,
                                     trusted_actor_id=ACTOR, idempotency_key=key)


def _cases(db) -> list[dict]:
    return [dict(r) for r in db.execute(
        "SELECT case_id, predecessor_fact_id, successor_fact_id, reason_code, status, "
        "version FROM correction_cases ORDER BY predecessor_fact_id")]


# --- the input a caller sends --------------------------------------------------

@pytest.mark.parametrize("value", ["", "   ", 7, 1.5, ["abc"], {"id": "x"}, True,
                                   "has space", "semi;colon", "x" * 129, "-leading"])
def test_malformed_replaces_is_refused(value) -> None:
    from superlocalmemory.core.replaces_input import INVALID, ReplacesRejected, normalize_replaces

    with pytest.raises(ReplacesRejected) as refused:
        normalize_replaces(value)
    assert refused.value.code == INVALID
    assert refused.value.message


def test_a_well_formed_id_is_accepted_and_stripped() -> None:
    from superlocalmemory.core.replaces_input import normalize_replaces

    assert normalize_replaces("  3f2a9c0d11e84b7a ") == "3f2a9c0d11e84b7a"
    assert normalize_replaces("succ-1") == "succ-1"


# --- checks made before anything is saved ------------------------------------

def test_check_accepts_an_own_fact_and_an_own_memory(env) -> None:
    from superlocalmemory.core.remember_replaces import check_replaceable

    db, _ = env
    fid = _fact(db, "Recall ceiling is 2 seconds.")
    memory_id = dict(db.execute("SELECT memory_id FROM atomic_facts WHERE fact_id=?",
                                (fid,))[0])["memory_id"]
    for named in (fid, memory_id):
        assert check_replaceable(db, replaces=named, profile_id="default",
                                 scope="personal") == named


def test_check_refuses_an_unknown_id(env) -> None:
    from superlocalmemory.core.remember_replaces import check_replaceable
    from superlocalmemory.core.replaces_input import NOT_FOUND, ReplacesRejected

    db, _ = env
    with pytest.raises(ReplacesRejected) as refused:
        check_replaceable(db, replaces="0123456789abcdef", profile_id="default",
                          scope="personal")
    assert refused.value.code == NOT_FOUND


def test_check_refuses_another_profiles_fact_without_revealing_private_ones(env) -> None:
    from superlocalmemory.core.remember_replaces import check_replaceable
    from superlocalmemory.core.replaces_input import NOT_ALLOWED, NOT_FOUND, ReplacesRejected

    db, _ = env
    shown = _fact(db, "A global fact from the other profile.", profile="other", scope="global")
    shared = _fact(db, "Shared with default.", profile="other", scope="shared",
                   shared_with=["default"])
    private = _fact(db, "Private to the other profile.", profile="other")
    for named, code in ((shown, NOT_ALLOWED), (shared, NOT_ALLOWED), (private, NOT_FOUND)):
        with pytest.raises(ReplacesRejected) as refused:
            check_replaceable(db, replaces=named, profile_id="default", scope="global")
        assert refused.value.code == code, named
    with pytest.raises(ReplacesRejected) as refused:
        check_replaceable(db, replaces=shown, profile_id="default", scope="global")
    assert "another profile" in refused.value.message


def test_check_resolves_in_the_profile_written_and_refuses_a_scope_change(env) -> None:
    """A write routed to another profile may only name that profile's memories:
    this profile's private fact is then not found, exactly like a missing id."""
    from superlocalmemory.core.remember_replaces import check_replaceable
    from superlocalmemory.core.replaces_input import NOT_ALLOWED, NOT_FOUND, ReplacesRejected

    db, _ = env
    fid = _fact(db, "Personal decision.")
    theirs = _fact(db, "Their personal decision.", profile="other")
    assert check_replaceable(db, replaces=theirs, profile_id="other",
                             scope="personal") == theirs
    with pytest.raises(ReplacesRejected) as routed:
        check_replaceable(db, replaces=fid, profile_id="other", scope="personal")
    assert routed.value.code == NOT_FOUND
    with pytest.raises(ReplacesRejected) as scoped:
        check_replaceable(db, replaces=fid, profile_id="default", scope="global")
    assert scoped.value.code == NOT_ALLOWED and "personal" in scoped.value.message


# --- the replacement itself --------------------------------------------------

def test_replacing_a_fact_retires_it_through_the_ledger(env) -> None:
    from superlocalmemory.storage.correction_cases import CALLER_REPLACEMENT_REASON

    db, runtime = env
    old = _fact(db, "Recall ceiling is 2 seconds.")
    new = _fact(db, "Recall ceiling is 3 seconds.")
    receipt = _replace(runtime, old, new)
    assert receipt["ok"] is True and receipt["fact_ids"] == [old]
    row = _temporal(db, old)
    assert row["system_expired_at"] and row["invalidated_by"] == new
    assert row["invalidation_reason"] == CALLER_REPLACEMENT_REASON
    assert _temporal(db, new)["system_expired_at"] is None
    [case] = _cases(db)
    assert (case["predecessor_fact_id"], case["successor_fact_id"]) == (old, new)
    assert case["status"] == "applied" and case["reason_code"] == CALLER_REPLACEMENT_REASON
    assert receipt["cases"] == [{"case_id": case["case_id"], "version": case["version"]}]


def test_undo_through_the_correction_rollback_restores_the_row_exactly(env) -> None:
    db, runtime = env
    old = _fact(db, "Recall ceiling is 2 seconds.")
    new = _fact(db, "Recall ceiling is 3 seconds.")
    db.execute("UPDATE fact_temporal_validity SET valid_from='2026-01-01T00:00:00+00:00', "
               "valid_until='2027-01-01T00:00:00+00:00' WHERE fact_id=?", (old,))
    before = _temporal(db, old)
    receipt = _replace(runtime, old, new)
    assert _temporal(db, old) != before
    [case] = receipt["cases"]
    undone = runtime.transition_correction("default", case["case_id"], action="rollback",
                                           expected_version=case["version"], actor_id=ACTOR,
                                           idempotency_key="undo-1")
    assert undone["status"] == "rolled_back"
    assert _temporal(db, old) == before


def test_rollback_keeps_the_new_memory_findable(env) -> None:
    db, runtime = env
    old = _fact(db, "Status: amber.")
    new = _fact(db, "Status: green.")
    [case] = _replace(runtime, old, new)["cases"]
    assert db.get_correction_inadmissible_fact_ids([old, new], "default") == {old}
    runtime.transition_correction("default", case["case_id"], action="rollback",
                                  expected_version=case["version"], actor_id=ACTOR,
                                  idempotency_key="undo-2")
    assert db.get_correction_inadmissible_fact_ids([old, new], "default") == set()
    assert db.get_nonapplied_correction_successor_ids([old, new], "default") == set()


def test_the_context_cache_agrees_after_an_undo(env) -> None:
    from pathlib import Path

    from superlocalmemory.core.context_cache import _cache_fact_ids_are_current

    db, runtime = env
    home = Path(db.db_path).parent
    old = _fact(db, "Status: amber.")
    new = _fact(db, "Status: green.")
    [case] = _replace(runtime, old, new)["cases"]
    assert _cache_fact_ids_are_current(home, [new]) is True
    assert _cache_fact_ids_are_current(home, [old]) is False
    runtime.transition_correction("default", case["case_id"], action="rollback",
                                  expected_version=case["version"], actor_id=ACTOR,
                                  idempotency_key="undo-4")
    assert _cache_fact_ids_are_current(home, [old, new]) is True


def test_replaying_the_same_request_does_not_mark_twice(env) -> None:
    db, runtime = env
    old = _fact(db, "Owner: Alice.")
    new = _fact(db, "Owner: Bob.")
    first = _replace(runtime, old, new, key="same-op")
    stamped = _temporal(db, old)
    second = _replace(runtime, old, new, key="same-op")
    assert first == second
    assert len(_cases(db)) == 1
    assert len(db.execute("SELECT 1 FROM correction_events WHERE event_type='applied'")) == 1
    assert _temporal(db, old) == stamped


def test_a_memory_id_retires_every_current_fact_of_that_memory(env) -> None:
    db, runtime = env
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content="Two facts."))
    first = _fact(db, "Release day is Tuesday.", memory_id=memory_id)
    second = _fact(db, "Release owner is Alice.", memory_id=memory_id)
    earlier = _fact(db, "Release day was Monday.", memory_id=memory_id)
    db.execute("UPDATE fact_temporal_validity SET system_expired_at='2026-09-01T00:00:00+00:00', "
               "invalidation_reason='direct_content_correction' WHERE fact_id=?", (earlier,))
    untouched = _temporal(db, earlier)
    new = _fact(db, "Release day is Thursday and Bob owns it.")
    receipt = _replace(runtime, memory_id, new)
    assert sorted(receipt["fact_ids"]) == sorted([first, second])
    assert _temporal(db, first)["invalidated_by"] == new
    assert _temporal(db, second)["invalidated_by"] == new
    assert _temporal(db, earlier) == untouched


def test_an_already_replaced_fact_is_reported_not_marked_again(env) -> None:
    from superlocalmemory.core.remember_replaces import ReplacementRefused

    db, runtime = env
    old = _fact(db, "Plan A.")
    _replace(runtime, old, _fact(db, "Plan B."), key="op-b")
    stamped = _temporal(db, old)
    with pytest.raises(ReplacementRefused) as refused:
        _replace(runtime, old, _fact(db, "Plan C."), key="op-c")
    assert "already replaced" in str(refused.value)
    assert _temporal(db, old) == stamped and len(_cases(db)) == 1


def test_another_profiles_fact_is_refused_inside_the_writer_too(env) -> None:
    from superlocalmemory.core.remember_replaces import ReplacementRefused

    db, runtime = env
    theirs = _fact(db, "Their global rule.", profile="other", scope="global")
    before = _temporal(db, theirs)
    with pytest.raises(ReplacementRefused):
        _replace(runtime, theirs, _fact(db, "My rule.", scope="global"))
    assert _temporal(db, theirs) == before and _cases(db) == []


def test_a_failed_mark_changes_nothing(env, monkeypatch) -> None:
    from superlocalmemory.core import remember_replaces

    db, runtime = env
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content="Two facts."))
    first = _fact(db, "Fact one.", memory_id=memory_id)
    second = _fact(db, "Fact two.", memory_id=memory_id)
    before = (_temporal(db, first), _temporal(db, second))
    real = remember_replaces.transition_on_connection
    calls = []

    def fail_on_second(conn, **kwargs):
        calls.append(kwargs["case_id"])
        if len(calls) == 2:
            raise sqlite3.OperationalError("disk I/O error")
        return real(conn, **kwargs)

    monkeypatch.setattr(remember_replaces, "transition_on_connection", fail_on_second)
    from superlocalmemory.core.remember_runtime import CanonicalRememberUnavailable

    with pytest.raises(CanonicalRememberUnavailable):
        _replace(runtime, memory_id, _fact(db, "New."))
    # The first fact was marked inside the transaction before the second failed.
    assert len(calls) == 2
    assert (_temporal(db, first), _temporal(db, second)) == before
    assert _cases(db) == []


def test_a_long_actor_identity_still_records(env) -> None:
    from superlocalmemory.core.remember_replaces import ledger_actor_id

    db, runtime = env
    long_actor = "local-capability:dashboard:uid:501:" + "b" * 64 + ":" + "c" * 64
    assert ledger_actor_id(long_actor) == ledger_actor_id(long_actor)
    assert len(ledger_actor_id(long_actor).encode()) <= 128
    assert ledger_actor_id(ACTOR) == ACTOR
    old, new = _fact(db, "Old."), _fact(db, "New.")
    receipt = runtime.replace_by_caller("default", old, new, trusted_actor_id=long_actor,
                                        idempotency_key="long-actor")
    assert receipt["ok"] is True


# --- what the caller is told after the save ----------------------------------

def _engine(db):
    return SimpleNamespace(_db=db)


def test_after_save_reports_what_was_replaced(env) -> None:
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old, new = _fact(db, "Old status."), _fact(db, "New status.")
    out = replace_after_save(runtime, _engine(db), replaces=old, profile_id="default",
                             successor_fact_ids=[new], operation_id="op-x",
                             trusted_actor_id=ACTOR)
    assert out["ok"] is True and out["replaces"] == old and out["fact_ids"] == [old]
    assert out["cases"][0]["version"] == 1 and "rollback" in out["undo"]


def test_after_save_reports_a_failed_mark_and_never_raises(env, monkeypatch) -> None:
    from superlocalmemory.core import remember_replaces

    db, runtime = env
    old, new = _fact(db, "Old status."), _fact(db, "New status.")

    def broken(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(remember_replaces, "transition_on_connection", broken)
    out = remember_replaces.replace_after_save(
        runtime, _engine(db), replaces=old, profile_id="default", successor_fact_ids=[new],
        operation_id="op-y", trusted_actor_id=ACTOR)
    assert out["ok"] is False and out["replaces"] == old and out["reason"]
    assert _temporal(db, old)["system_expired_at"] is None


def test_after_save_with_nothing_saved_replaces_nothing(env) -> None:
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old = _fact(db, "Old status.")
    out = replace_after_save(runtime, _engine(db), replaces=old, profile_id="default",
                             successor_fact_ids=[], operation_id="op-z", trusted_actor_id=ACTOR)
    assert out["ok"] is False and "nothing" in out["reason"].lower()
    assert _temporal(db, old)["system_expired_at"] is None


def test_a_replay_after_undo_does_not_claim_the_replacement(env) -> None:
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old, new = _fact(db, "Old status."), _fact(db, "New status.")
    kwargs = dict(replaces=old, profile_id="default", successor_fact_ids=[new],
                  operation_id="op-undo", trusted_actor_id=ACTOR)
    first = replace_after_save(runtime, _engine(db), **kwargs)
    [case] = first["cases"]
    runtime.transition_correction("default", case["case_id"], action="rollback",
                                  expected_version=case["version"], actor_id=ACTOR,
                                  idempotency_key="undo-3")
    replay = replace_after_save(runtime, _engine(db), **kwargs)
    assert replay["ok"] is False and "undone" in replay["reason"]
    assert _temporal(db, old)["system_expired_at"] is None


def _pending_review(db, predecessor: str, successor: str, *, machine: bool = False) -> None:
    """A case waiting for review: a person's edit, or (``machine``) SLM's own guess."""
    from superlocalmemory.storage.correction_cases import CorrectionActor, propose_on_connection

    actor = (CorrectionActor("consolidator", "host_attested", "canonical_writer") if machine
             else CorrectionActor("a-person", "host_authenticated", "trusted"))
    with db.raw_connection() as conn:
        propose_on_connection(
            conn, case_id="pending-" + predecessor, profile_id="default", scope="personal",
            predecessor_fact_id=predecessor, successor_fact_id=successor,
            reason_code="consolidation_update" if machine else "direct_content_correction",
            actor=actor, idempotency_key="pending-" + predecessor,
            is_profile_active=lambda _p: True, is_actor_trusted=lambda _a: True)


def test_a_correction_waiting_for_review_is_reported_plainly(env) -> None:
    """A person's edit waiting for review still blocks a replace (4.1.22)."""
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old, proposed, new = _fact(db, "Old."), _fact(db, "A person's edit."), _fact(db, "New.")
    _pending_review(db, old, proposed)
    out = replace_after_save(runtime, _engine(db), replaces=old, profile_id="default",
                             successor_fact_ids=[new], operation_id="op-pending",
                             trusted_actor_id=ACTOR)
    assert out["ok"] is False and "waiting for review" in out["reason"]
    assert "try again" not in out["reason"]
    assert _temporal(db, old)["system_expired_at"] is None


def test_a_machine_guess_waiting_for_review_no_longer_blocks(env) -> None:
    """Varun, 2026-10-06: the user's explicit replace wins over SLM's own
    unreviewed proposal, which is closed as overtaken (core/overtaken_cases.py)."""
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old, proposed, new = _fact(db, "Old."), _fact(db, "Machine's guess."), _fact(db, "New.")
    _pending_review(db, old, proposed, machine=True)
    out = replace_after_save(runtime, _engine(db), replaces=old, profile_id="default",
                             successor_fact_ids=[new], operation_id="op-pending",
                             trusted_actor_id=ACTOR)
    assert out["ok"] is True, out
    assert _temporal(db, old)["system_expired_at"] is not None
    assert [dict(r)["user_action"] for r in db.execute(
        "SELECT user_action FROM correction_cases_overtaken WHERE case_id = ?",
        ("pending-" + old,))] == ["replace"]


def test_after_save_reports_an_already_replaced_target(env) -> None:
    from superlocalmemory.core.remember_replaces import replace_after_save

    db, runtime = env
    old = _fact(db, "Old status.")
    _replace(runtime, old, _fact(db, "Second status."), key="op-first")
    out = replace_after_save(runtime, _engine(db), replaces=old, profile_id="default",
                             successor_fact_ids=[_fact(db, "Third status.")],
                             operation_id="op-second", trusted_actor_id=ACTOR)
    assert out["ok"] is False and "already replaced" in out["reason"]


# --- what reads the replacement ----------------------------------------------

def test_a_replaced_rule_is_not_loaded_at_session_start(env) -> None:
    from superlocalmemory.core.standing_rules import standing_facts

    db, runtime = env
    old = _fact(db, "Always deploy on Fridays.", kind="rule", source="caller")
    new = _fact(db, "Never deploy on Fridays.", kind="rule", source="caller")
    assert {f.fact_id for f in standing_facts(db, "default")} == {old, new}
    _replace(runtime, old, new)
    assert [f.fact_id for f in standing_facts(db, "default")] == [new]


def test_store_repair_never_undoes_a_caller_replacement(env) -> None:
    from superlocalmemory.storage.store_repair import apply_repair, plan_repair

    db, runtime = env
    old, new = _fact(db, "Old plan."), _fact(db, "New plan.")
    _replace(runtime, old, new)
    stamped = _temporal(db, old)
    conn = sqlite3.connect(db.db_path, isolation_level=None)
    try:
        plan = plan_repair(conn)
        assert old not in plan.wrong_replacements
        apply_repair(conn, plan)
    finally:
        conn.close()
    assert _temporal(db, old) == stamped
