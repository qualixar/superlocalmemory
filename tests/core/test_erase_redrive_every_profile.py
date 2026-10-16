# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The background erase redrive covers every profile, not only the active one.

Before 4.1.21 the 30 s pass re-proved only the active profile's unfinished
erasures. Someone with two profiles who deleted a memory in "work", then
switched to "personal", left that erasure unproven (and reported) until they
switched back; a remote key deleting in its own profile hit the same gap.
Each erasure is still proven inside its own profile, with the same back-off.
"""

from __future__ import annotations

import time

import pytest

ACTIVE, OTHER = "personal", "work"


@pytest.fixture
def engine(request, mode_a_config):
    mode_a_config.active_profile = ACTIVE
    eng = request.getfixturevalue("engine_with_mock_deps")
    assert eng._profile_id == ACTIVE
    with eng._db.raw_connection() as conn:
        for profile in (ACTIVE, OTHER):
            conn.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                         (profile, profile.title()))
    return eng


def _redrive(engine, passes: int = 1) -> None:
    from superlocalmemory.server.unified_daemon import _reconcile_pending_projections

    for _ in range(passes):
        _reconcile_pending_projections(engine, force=True)


def _seed(engine, op_id: str, profile: str, subject: str, *, state: str = "failed",
          attempts: int = 10) -> None:
    """An erasure left unfinished in ``profile``: its tombstone and three
    obligations, as an interrupted delete (or the 4.1.17-4.1.20 redrive) left them."""
    now = time.time()
    with engine._db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, created_at) "
            "VALUES (?, ?, ?, ?)", (profile, subject, op_id, now))
        for owner in ("bm25", "temporal", "vector"):
            conn.execute(
                "INSERT INTO projection_obligations (operation_id, profile_id, owner, kind, "
                "subject_id, state, attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, 'erase', ?, ?, ?, ?, ?)",
                (op_id, profile, owner, subject, state, attempts, now, now))


def _store_in(engine, profile: str, text: str) -> str:
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    db = engine._db
    memory_id = db.store_memory(MemoryRecord(profile_id=profile, content=text))
    return db.store_fact(AtomicFact(profile_id=profile, memory_id=memory_id, content=text))


def _rows(engine, op_id: str) -> list[dict]:
    return [dict(r) for r in engine._db.execute(
        "SELECT profile_id, state, verify_attempts FROM projection_obligations "
        "WHERE operation_id = ? AND kind = 'erase'", (op_id,))]


def _exhausted(engine) -> int:
    from superlocalmemory.core.ops_remediation import get_failure_counts

    return get_failure_counts(engine._db.db_path)["exhausted_obligations"]


def test_an_erasure_left_over_in_another_profile_is_re_proven_and_closed(engine) -> None:
    _seed(engine, "erase-work-leftover", OTHER, "gone-work-fact")
    assert _exhausted(engine) == 1

    _redrive(engine)

    rows = _rows(engine, "erase-work-leftover")
    assert {(r["profile_id"], r["state"]) for r in rows} == {(OTHER, "erased")}, rows
    assert _exhausted(engine) == 0
    assert engine._profile_id == ACTIVE


def test_another_profiles_unconfirmed_erasure_stays_reported_and_backs_off(engine) -> None:
    from superlocalmemory.core.transactions.erase_redrive import MAX_REPROOFS

    fact_id = _store_in(engine, OTHER, "The work vendor contract renews on 2026-03-01.")
    _seed(engine, "erase-work-stored", OTHER, fact_id, attempts=1)

    _redrive(engine, passes=MAX_REPROOFS + 5)

    rows = _rows(engine, "erase-work-stored")
    assert {r["state"] for r in rows} == {"failed"}, rows
    assert {r["verify_attempts"] for r in rows} == {MAX_REPROOFS}, rows
    assert _exhausted(engine) == 1
    assert engine._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,))


def test_each_erasure_is_proven_in_its_own_profile(engine) -> None:
    """A "work" erasure is proven against "work" only: the same id still
    stored in the active profile neither blocks it nor is touched by it."""
    fact_id = _store_in(engine, ACTIVE, "The personal dentist visit is on Friday at noon.")
    _seed(engine, "erase-work-own-profile", OTHER, fact_id)
    _seed(engine, "erase-active-stored", ACTIVE, fact_id)

    _redrive(engine)

    assert {r["state"] for r in _rows(engine, "erase-work-own-profile")} == {"erased"}
    assert {r["state"] for r in _rows(engine, "erase-active-stored")} == {"failed"}
    assert engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ?", (fact_id, ACTIVE))


def test_erasures_of_every_profile_share_one_bounded_pass(engine) -> None:
    """The pass still re-proves at most ``limit`` erasures, oldest first, so a
    profile with many cannot make the pass unbounded."""
    from superlocalmemory.core.transactions.erase_redrive import reconcile_pending_erasures

    for n in range(3):
        _seed(engine, f"erase-work-{n}", OTHER, f"gone-work-{n}")
        _seed(engine, f"erase-personal-{n}", ACTIVE, f"gone-personal-{n}")

    assert reconcile_pending_erasures(engine, limit=4) == 4
    assert reconcile_pending_erasures(engine, limit=4) == 2
    states = {r["state"] for n in range(3) for op in (f"erase-work-{n}", f"erase-personal-{n}")
              for r in _rows(engine, op)}
    assert states == {"erased"}


def test_the_reconcile_action_proves_an_erasure_in_its_own_profile(engine) -> None:
    """``slm ops`` lists every profile's unfinished erasures, so Reconcile on
    one of "work"'s must prove it in "work", not report it as not this
    profile's while "personal" is active."""
    from superlocalmemory.core.ops_remediation import resolve_operation

    _seed(engine, "erase-work-reconcile", OTHER, "gone-work-fact")

    result = resolve_operation(engine._db.db_path, engine, "erase-work-reconcile",
                               "force_reconcile")

    assert result["success"] is True, result
    assert {r["state"] for r in _rows(engine, "erase-work-reconcile")} == {"erased"}
    assert engine._profile_id == ACTIVE


def test_the_reconcile_action_still_refuses_an_erasure_it_cannot_prove(engine) -> None:
    from superlocalmemory.core.ops_remediation import resolve_operation

    fact_id = _store_in(engine, OTHER, "The work vendor contract renews on 2026-03-01.")
    _seed(engine, "erase-work-still-stored", OTHER, fact_id)

    result = resolve_operation(engine._db.db_path, engine, "erase-work-still-stored",
                               "force_reconcile")

    assert result["success"] is False and "still stored" in result["reason"], result
    assert {r["state"] for r in _rows(engine, "erase-work-still-stored")} == {"failed"}


# -- 4.2.8: a removal left undone is finished, not only reported --------------

class _Index:
    """A projection owner whose removal can be made to fail, and holds a residue until it works."""

    name = "vector"

    def __init__(self, fail_times: int) -> None:
        self.residue, self.fail_times, self.erase_calls = True, fail_times, 0

    def erase(self, context):
        from superlocalmemory.core.transactions.owners import OwnerErasureProof

        self.erase_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            return OwnerErasureProof(owner="vector", erased=False)
        self.residue = False
        return OwnerErasureProof(owner="vector", erased=True, checksum="c")

    def prove_erased(self, context):
        from superlocalmemory.core.transactions.owners import OwnerErasureProof

        return OwnerErasureProof(owner="vector", erased=not self.residue, checksum="c")


def _with_index(engine, monkeypatch, index) -> None:
    from superlocalmemory.core.transactions import concrete_owners
    from superlocalmemory.core.transactions.erasure import ErasureService

    monkeypatch.setattr(concrete_owners, "build_erasure_service",
                        lambda eng: ErasureService({"vector": index}))


def _seed_vector_only(engine, op_id: str, subject: str, attempts: int) -> None:
    now = time.time()
    with engine._db.raw_connection() as conn:
        conn.execute("INSERT INTO projection_tombstones (profile_id, fact_id, erasure_id, "
                     "created_at) VALUES (?, ?, ?, ?)", (ACTIVE, subject, op_id, now))
        conn.execute(
            "INSERT INTO projection_obligations (operation_id, profile_id, owner, kind, "
            "subject_id, state, attempts, created_at, updated_at) "
            "VALUES (?, ?, 'vector', 'erase', ?, 'failed', ?, ?, ?)",
            (op_id, ACTIVE, subject, attempts, now, now))


def test_a_gone_memorys_leftover_index_rows_are_removed_by_the_redrive(engine, monkeypatch) -> None:
    index = _Index(fail_times=0)
    _with_index(engine, monkeypatch, index)
    _seed_vector_only(engine, "erase-vector-left", "gone-fact", attempts=1)

    _redrive(engine)

    assert index.erase_calls == 1 and index.residue is False
    assert {r["state"] for r in _rows(engine, "erase-vector-left")} == {"erased"}


def test_a_removal_that_keeps_failing_stops_at_the_attempt_limit(engine, monkeypatch) -> None:
    from superlocalmemory.core.transactions.erasure import MAX_ERASE_ATTEMPTS

    index = _Index(fail_times=10**6)
    _with_index(engine, monkeypatch, index)
    _seed_vector_only(engine, "erase-vector-stuck", "gone-fact-2", attempts=1)

    _redrive(engine, passes=MAX_ERASE_ATTEMPTS + 8)

    assert index.erase_calls == MAX_ERASE_ATTEMPTS - 1  # attempts 1 -> limit, then reported only
    assert {r["state"] for r in _rows(engine, "erase-vector-stuck")} == {"failed"}
    assert _exhausted(engine) == 1


def test_a_stored_memory_is_never_erased_by_the_redrive(engine, monkeypatch) -> None:
    index = _Index(fail_times=0)
    _with_index(engine, monkeypatch, index)
    fact_id = _store_in(engine, ACTIVE, "The personal dentist visit is on Friday at noon.")
    _seed_vector_only(engine, "erase-vector-stored", fact_id, attempts=1)

    _redrive(engine, passes=3)

    assert index.erase_calls == 0 and index.residue is True
    assert engine._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,))
