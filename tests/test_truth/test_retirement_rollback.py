# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-23 / D-49: "Bring back memories that 4.2.30 retired", the separate ledger command.

The kill switch never undoes a retirement. This command does, for the cases the truth layer
filed and nobody else's; it says how many first (``dry_run``), is recorded in the decision log
like any other decision, and can be undone: the same cases are applied again if the value of
each has not moved since, otherwise it refuses in words and changes nothing.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import decision_undo, decisions
from superlocalmemory.truth import retirement_rollback as rr
from tests.test_truth.test_slot_pass import (
    _consolidator_proposal,
    _fact,
    _reconcile,
    _retired,
    _settled_by_a_person,
    store,  # noqa: F401 - the fixture
)

WHO = decisions.Decider("agent:claude", "mcp")


@pytest.fixture
def retired(store):  # noqa: F811
    """A store where the truth layer retired ``old`` for ``new``, and a consolidator proposal waits."""
    db, old, new, _tmp = store
    _settled_by_a_person(db, old, new)
    _reconcile(db)
    assert _retired(db, old)[2] == new
    other = _fact(db, "Some unrelated memory about a plant", "2026-05-01T00:00:00+00:00")
    third = _fact(db, "Another memory about the plant", "2026-06-01T00:00:00+00:00")
    _consolidator_proposal(db, "keep-me", other, third)
    return db, old, new


def _log(db) -> list[tuple]:
    return [(r["kind"], r["action"], r["actor"], r["channel"], r["reverted_at"] is not None)
            for r in db.execute("SELECT * FROM decision_log ORDER BY rowid")]


def test_a_dry_run_counts_and_changes_nothing(retired) -> None:
    db, old, _new = retired
    with db.raw_connection() as conn:
        got = rr.run(conn, profile_id="default", decider=WHO, dry_run=True)
    assert got == {"dry_run": True, "would_come_back": 1, "came_back": 0, "refused": 0,
                   "decision_id": ""}
    assert _retired(db, old)[2] is not None
    assert _log(db) == []


def test_the_command_brings_back_only_the_truth_layers_retirements_and_records_it(retired) -> None:
    db, old, _new = retired
    with db.raw_connection() as conn:
        got = rr.run(conn, profile_id="default", decider=WHO)
    assert (got["came_back"], got["refused"], got["dry_run"]) == (1, 0, False)
    assert _retired(db, old) == (None, None, None)
    statuses = {r["case_id"]: r["status"] for r in db.execute(
        "SELECT case_id, status FROM correction_cases")}
    assert statuses["keep-me"] == "proposed"  # a consolidator's case is not ours to touch
    assert _log(db) == [("rollback_retirements", "run", "agent:claude", "mcp", False)]
    entry = decisions.recent(db_conn(db), "default")[0]
    assert entry["can_undo"] and "Brought back 1 memory" in entry["summary"]


def db_conn(db):
    import sqlite3

    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def test_with_nothing_retired_it_says_so_and_records_nothing(store) -> None:  # noqa: F811
    db = store[0]
    with db.raw_connection() as conn:
        got = rr.run(conn, profile_id="default", decider=WHO)
    assert got["came_back"] == 0 and got["decision_id"] == ""
    assert _log(db) == []


def test_it_works_whatever_the_kill_switch_says(retired, monkeypatch) -> None:
    """The command is separate from the switch: it must run with the switch on, which is when it is wanted."""
    db, old, _new = retired
    monkeypatch.setenv("SLM_4230_OFF", "1")
    with db.raw_connection() as conn:
        assert rr.run(conn, profile_id="default", decider=WHO)["came_back"] == 1
    assert _retired(db, old) == (None, None, None)


def test_another_profiles_retirements_are_left_alone(retired) -> None:
    db, old, _new = retired
    with db.raw_connection() as conn:
        got = rr.run(conn, profile_id="someone-else", decider=WHO)
    assert got["came_back"] == 0 and _retired(db, old)[2] is not None


def test_undo_applies_the_same_cases_again_and_the_log_shows_both(retired) -> None:
    db, old, new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
        done = decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                                  decider=decisions.Decider("dashboard", "dashboard"))
    assert done["undone"] and done["kind"] == "rollback_retirements" and done["reapplied"] == 1
    assert _retired(db, old)[2] == new  # retired again, for the same successor
    assert _log(db) == [("rollback_retirements", "run", "agent:claude", "mcp", True),
                        ("rollback_retirements", "undo", "dashboard", "dashboard", False)]
    with db.raw_connection() as conn:  # and the command can run again on the re-applied case
        assert rr.run(conn, profile_id="default", decider=WHO)["came_back"] == 1


def test_undo_refuses_in_words_when_a_value_moved_and_changes_nothing(retired) -> None:
    db, old, new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
    # The older value was retired again by someone else after the roll back.
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-11-18T00:00:00+00:00' "
               "WHERE fact_id = ?", (old,))
    before = [dict(r) for r in db.execute("SELECT * FROM correction_cases ORDER BY case_id")]
    with db.raw_connection() as conn, pytest.raises(decision_undo.UndoRefused) as caught:
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)
    assert "changed since" in str(caught.value) and new not in str(caught.value)
    assert [dict(r) for r in db.execute("SELECT * FROM correction_cases ORDER BY case_id")] == before
    assert _log(db) == [("rollback_retirements", "run", "agent:claude", "mcp", False)]


def test_undo_refuses_when_the_newer_value_is_gone(retired) -> None:
    db, _old, new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-11-18T00:00:00+00:00' "
               "WHERE fact_id = ?", (new,))
    with db.raw_connection() as conn, pytest.raises(decision_undo.UndoRefused):
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)


def test_undo_is_all_or_nothing(retired) -> None:
    """Two retirements came back; one value moved: neither is applied again."""
    db, old, _new = retired
    a = _fact(db, "The deploy runs on Fridays", "2026-02-01T00:00:00+00:00")
    b = _fact(db, "The deploy runs on Mondays", "2026-03-01T00:00:00+00:00")
    from superlocalmemory.truth.slot_resolver import retire_superseded

    with db.raw_connection() as conn:
        retire_superseded(conn, profile_id="default", scope="personal", predecessor=a,
                          successor=b)
        first = rr.run(conn, profile_id="default", decider=WHO)
    assert first["came_back"] == 2
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = '2026-11-18T00:00:00+00:00' "
               "WHERE fact_id = ?", (a,))
    with db.raw_connection() as conn, pytest.raises(decision_undo.UndoRefused):
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)
    assert _retired(db, old) == (None, None, None)  # the first pair stays as rolled back


def test_a_second_undo_is_refused(retired) -> None:
    db, _old, _new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)
        with pytest.raises(decision_undo.UndoRefused):
            decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                               decider=WHO)


def test_undo_refuses_when_something_else_is_now_deciding_about_the_older_memory(retired) -> None:
    db, old, new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
    _consolidator_proposal(db, "rival", old, _fact(db, "A third wording of the cadence",
                                                    "2026-11-15T00:00:00+00:00"))
    with db.raw_connection() as conn, pytest.raises(decision_undo.UndoRefused):
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)
    assert _retired(db, old) == (None, None, None) and new


def test_a_ledger_refusal_part_way_through_undo_leaves_every_case_as_it_was(
        retired, monkeypatch) -> None:
    from superlocalmemory.storage.correction_cases import CorrectionCaseError
    from superlocalmemory.truth import slot_resolver

    db, old, _new = retired
    with db.raw_connection() as conn:
        first = rr.run(conn, profile_id="default", decider=WHO)
    before = [dict(r) for r in db.execute("SELECT * FROM correction_cases ORDER BY case_id")]

    def refuse(*_a, **_k):
        raise CorrectionCaseError("changed")

    monkeypatch.setattr(slot_resolver, "_apply", refuse)
    with db.raw_connection() as conn, pytest.raises(decision_undo.UndoRefused) as caught:
        decision_undo.undo(conn, profile_id="default", decision_id=first["decision_id"],
                           decider=WHO)
    assert "changed since" in str(caught.value)
    assert [dict(r) for r in db.execute("SELECT * FROM correction_cases ORDER BY case_id")] == before
    assert _retired(db, old) == (None, None, None)
