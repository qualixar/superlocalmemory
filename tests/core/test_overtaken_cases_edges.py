# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Edge cases of "the user's action wins over a machine proposal" (4.1.22).

The fact on the other side of a case, a fact named by several cases at once,
a restore that would collide with what the user did since, a case that changed
while the action ran, and an action that fails halfway: none may lose or
corrupt correction history.
"""

from __future__ import annotations

import sqlite3

import pytest
from tests.core.test_overtaken_cases import _delete, _fact, _propose, _replace, _rows


@pytest.fixture
def three(engine_with_mock_deps):
    engine = engine_with_mock_deps
    return (engine,
            _fact(engine, "Synthetic plover count at the west spit was four."),
            _fact(engine, "Synthetic plover count at the west spit was six."),
            _fact(engine, "Synthetic plover count at the west spit was seven."))


def _ledger(engine) -> list[str]:
    return [r["case_id"] for r in _rows(engine, "SELECT case_id FROM correction_cases "
                                                "ORDER BY case_id")]


def _overtaken(engine) -> list[str]:
    try:
        return [r["case_id"] for r in _rows(engine, "SELECT case_id FROM "
                                                    "correction_cases_overtaken")]
    except sqlite3.OperationalError:
        return []


def test_the_table_is_part_of_the_schema(engine_with_mock_deps):
    assert _rows(engine_with_mock_deps, "SELECT name FROM sqlite_master WHERE name = "
                                        "'correction_cases_overtaken'")


def test_deleting_the_successor_overtakes_the_case_and_keeps_the_predecessor(three):
    from superlocalmemory.core import overtaken_cases as ot

    engine, a, b, _c = three
    _propose(engine, "case-s", a, b, machine=True)

    assert _delete(engine, b).get("ok") is True

    assert _ledger(engine) == [] and _overtaken(engine) == ["case-s"]
    assert _rows(engine, "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (a,))
    [listed] = ot.listing(engine._db, engine._profile_id)
    assert listed["restorable"] is False and "no longer exists" in listed[
        "not_restorable_because"]


def test_one_person_case_among_machine_cases_refuses_and_moves_nothing(three):
    from superlocalmemory.core.remember_runtime import CanonicalMutationConflict

    engine, a, b, c = three
    d = _fact(engine, "Synthetic plover count at the west spit was eight.")
    _propose(engine, "case-m1", a, b, machine=True)   # b is the successor
    _propose(engine, "case-m2", c, b, machine=True)   # b is the successor again
    _propose(engine, "case-h", b, d, machine=False)   # a person's edit of b

    with pytest.raises(CanonicalMutationConflict, match="case-h"):
        _delete(engine, b)

    assert _ledger(engine) == ["case-h", "case-m1", "case-m2"]
    assert _overtaken(engine) == []
    assert _rows(engine, "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (b,))


def test_a_failed_delete_closes_no_case(three, monkeypatch):
    engine, a, b, _c = three
    _propose(engine, "case-f", a, b, machine=True)

    def boom(*_a, **_k):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(engine._db, "delete_fact", boom)
    with pytest.raises(RuntimeError, match="disk went away"):
        _delete(engine, a)

    assert _ledger(engine) == ["case-f"] and _overtaken(engine) == []


def test_restore_after_a_replace_is_refused_with_the_reason(three):
    from superlocalmemory.core import overtaken_cases as ot

    engine, p, s, x = three
    _propose(engine, "case-x", p, x, machine=True)
    assert _replace(engine, p, s)["ok"] is True  # applied p -> s now holds p

    with pytest.raises(ot.RestoreRefused, match="another open correction"):
        with engine._db.transaction():
            ot.restore(engine._db, "case-x")
    [listed] = ot.listing(engine._db, engine._profile_id)
    assert listed["restorable"] is False
    assert _rows(engine, "SELECT status FROM correction_cases WHERE predecessor_fact_id = ?",
                 (p,)) == [{"status": "applied"}]


def test_restore_refuses_a_case_proposed_again_since(three):
    from superlocalmemory.core import overtaken_cases as ot

    engine, a, b, _c = three
    _propose(engine, "case-r", a, b, machine=True)
    with engine._db.transaction():
        ot.overtake(engine._db, ot.cases_naming(engine._db, [a]), user_action="update",
                    actor_id="u", operation_id="op")
    _propose(engine, "case-r", a, b, machine=True)  # the same detector, the same key

    with pytest.raises(ot.RestoreRefused, match="proposed again"):
        with engine._db.transaction():
            ot.restore(engine._db, "case-r")
    assert _ledger(engine) == ["case-r"]


def test_a_case_reviewed_meanwhile_aborts_the_action(three):
    from superlocalmemory.core import overtaken_cases as ot

    engine, a, b, _c = three
    _propose(engine, "case-v", a, b, machine=True)
    stale = ot.cases_naming(engine._db, [a])
    engine._db.execute("UPDATE correction_cases SET version = 1 WHERE case_id = 'case-v'")

    with pytest.raises(ot.OvertakeRaced):
        with engine._db.transaction():
            ot.overtake(engine._db, stale, user_action="delete", actor_id="u",
                        operation_id="op")
    assert _ledger(engine) == ["case-v"] and _overtaken(engine) == []


def test_restore_and_listing_stay_inside_the_profile(three):
    from superlocalmemory.core import overtaken_cases as ot

    engine, a, b, _c = three
    _propose(engine, "case-p", a, b, machine=True)
    with engine._db.transaction():
        ot.overtake(engine._db, ot.cases_naming(engine._db, [a]), user_action="update",
                    actor_id="u", operation_id="op")

    assert ot.listing(engine._db, "some-other-profile") == []
    with pytest.raises(ot.RestoreRefused, match="no overtaken case"):
        with engine._db.transaction():
            ot.restore(engine._db, "case-p", profile_id="some-other-profile")
    with engine._db.transaction():
        assert ot.restore(engine._db, "case-p", profile_id=engine._profile_id,
                          actor_id="me")["restored"] == "case-p"
    [listed] = ot.listing(engine._db, engine._profile_id)
    assert listed["restored_by"] == "me" and listed["not_restorable_because"] == (
        "already restored")


def test_a_deleted_workspace_s_overtaken_cases_move_with_its_memories(tmp_path, monkeypatch):
    """A workspace made later under the same name must not list them; since
    4.1.22 they move to 'default' with the memories they close (still
    restorable there) instead of being dropped."""
    from superlocalmemory.core import overtaken_cases as ot
    from superlocalmemory.server.routes import helpers
    from superlocalmemory.storage import schema as real_schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(real_schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('alice', 'Alice')")
    for profile in ("alice", "bob"):
        db.execute("INSERT INTO correction_cases_overtaken (case_id, profile_id, "
                   "predecessor_fact_id, successor_fact_id, reason_code, case_json, "
                   "events_json, user_action, actor_id, operation_id, closed_reason, "
                   "overtaken_at) VALUES (?, ?, 'p', 's', 'consolidation_update', '{}', "
                   "'[]', 'delete', 'u', 'op', ?, 'now')",
                   (f"case-{profile}", profile, ot.OVERTAKEN))
    monkeypatch.setattr(helpers, "DB_PATH", tmp_path / "memory.db")

    helpers.delete_profile_from_db("alice")

    assert sorted((r["case_id"], r["profile_id"]) for r in db.execute(
        "SELECT case_id, profile_id FROM correction_cases_overtaken")) == [
        ("case-alice", "default"), ("case-bob", "bob")]
