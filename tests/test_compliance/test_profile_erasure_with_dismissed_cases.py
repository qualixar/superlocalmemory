# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Erasing a profile also erases the cases a user's action (or a bulk dismissal)
moved out of the ledger into ``correction_cases_overtaken`` (issue #309).

Those rows hold fact ids and the case's lifecycle metadata, not memory text, but
they name the profile's data and must not outlive its erasure.
"""

from __future__ import annotations

import sqlite3

_PROFILE = "erasable"
_OTHER = "keeper"


def _put(db, case_id: str, profile: str, action: str = "dismissed") -> None:
    db.execute(
        "INSERT INTO correction_cases_overtaken (case_id, profile_id, predecessor_fact_id, "
        "successor_fact_id, reason_code, case_json, events_json, user_action, actor_id, "
        "operation_id, closed_reason, overtaken_at) VALUES (?, ?, 'p', 's', "
        "'temporal_contradiction', '{}', '[]', ?, 'cli', 'op', 'r', '2026-11-01T00:00:00+00:00')",
        (case_id, profile, action))


def _count(db, profile: str) -> int:
    return int(dict(db.execute("SELECT COUNT(*) AS n FROM correction_cases_overtaken "
                               "WHERE profile_id = ?", (profile,))[0])["n"])


def test_erase_profile_cases_removes_the_profiles_overtaken_rows_only(engine_with_mock_deps):
    from superlocalmemory.compliance.erased_cases import erase_profile_cases
    from superlocalmemory.core.overtaken_cases import ensure_table

    db = engine_with_mock_deps._db
    ensure_table(db)
    for i in range(3):
        _put(db, f"gone-{i}", _PROFILE, "dismissed" if i else "delete")
    _put(db, "kept-1", _OTHER)

    erase_profile_cases(db, _PROFILE, erasure_id="e1")

    assert _count(db, _PROFILE) == 0
    assert _count(db, _OTHER) == 1


def test_a_store_with_no_overtaken_table_still_erases(engine_with_mock_deps):
    from superlocalmemory.compliance.erased_cases import erase_profile_cases

    db = engine_with_mock_deps._db
    db.execute("DROP TABLE IF EXISTS correction_cases_overtaken")
    assert erase_profile_cases(db, _PROFILE, erasure_id="e2") == 0


def test_gdpr_forget_profile_leaves_no_overtaken_row(engine_with_mock_deps):
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.core.overtaken_cases import ensure_table

    engine = engine_with_mock_deps
    engine._db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                       (_PROFILE, _PROFILE))
    ensure_table(engine._db)
    _put(engine._db, "gone-a", _PROFILE)
    _put(engine._db, "gone-b", _PROFILE)

    counts = GDPRCompliance(engine._db, engine=engine).forget_profile(_PROFILE)

    assert not counts.get("erasure_aborted"), counts
    try:
        assert _count(engine._db, _PROFILE) == 0
    except sqlite3.OperationalError as exc:  # the table going away is also "no rows left"
        assert "no such table" in str(exc)


_SECRET = "Brindlemoor keeps the synthetic pewter kettle on the third shelf."


def test_another_profiles_overtaken_row_naming_an_erased_fact_holds_no_text(engine_with_mock_deps):
    """A case filed under ``keeper`` may name a fact of the erased profile (the ledger
    has no foreign key to the profile). The row is ids and lifecycle metadata only:
    after the erasure no table of the store holds the erased text, and the row can
    no longer be put back because its fact is gone."""
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.core import overtaken_cases as ot
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    engine = engine_with_mock_deps
    db = engine._db
    for pid in (_PROFILE, _OTHER):
        db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
    keeper = canonical_store(engine, "Keeper note that stays.", source_type="python-api",
                             trusted_actor_id=local_trusted_actor_id("python-api"),
                             require_complete=True, return_receipt=True, profile_id=_OTHER)
    doomed = canonical_store(engine, _SECRET, source_type="python-api",
                             trusted_actor_id=local_trusted_actor_id("python-api"),
                             require_complete=True, return_receipt=True, profile_id=_PROFILE)
    keep_id, doomed_id = list(keeper.final_fact_ids)[0], list(doomed.final_fact_ids)[0]
    _case_across_profiles(db, keep_id, doomed_id)
    stored = db.execute("SELECT COUNT(*) AS n FROM atomic_facts WHERE content LIKE '%pewter%'")
    assert dict(stored[0])["n"] == 1      # the text really is there before the erasure

    counts = GDPRCompliance(db, engine=engine).forget_profile(_PROFILE)

    assert not counts.get("erasure_aborted"), counts
    rows = [dict(r) for r in db.execute("SELECT * FROM correction_cases_overtaken")]
    assert [r["case_id"] for r in rows] == ["cross-1"]          # ids only: it may stay
    assert "pewter" not in " ".join(str(v) for r in rows for v in r.values())
    [listed] = ot.listing(db, _OTHER)
    assert listed["restorable"] is False and "no longer exists" in listed["not_restorable_because"]
    assert doomed_id in listed["not_restorable_because"]
    for table in ("memories", "atomic_facts"):
        left = db.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE content LIKE '%pewter%'")  # noqa: S608
        assert dict(left[0])["n"] == 0, table


def _case_across_profiles(db, keep_id: str, doomed_id: str) -> None:
    import json

    from superlocalmemory.core.overtaken_cases import ensure_table

    ensure_table(db)
    case = {"case_id": "cross-1", "profile_id": _OTHER, "predecessor_fact_id": keep_id,
            "successor_fact_id": doomed_id, "reason_code": "temporal_contradiction",
            "idempotency_key": "k-cross-1", "status": "proposed"}
    db.execute(
        "INSERT INTO correction_cases_overtaken (case_id, profile_id, predecessor_fact_id, "
        "successor_fact_id, reason_code, case_json, events_json, user_action, actor_id, "
        "operation_id, closed_reason, overtaken_at) VALUES ('cross-1', ?, ?, ?, "
        "'temporal_contradiction', ?, '[]', 'dismissed', 'cli', 'op', 'r', "
        "'2026-11-01T00:00:00+00:00')", (_OTHER, keep_id, doomed_id, json.dumps(case)))
