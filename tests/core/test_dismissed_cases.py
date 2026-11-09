# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Bulk dismissal of machine-proposed correction cases (issue #309, request 2).

``slm corrections dismiss`` closes the pile of cases SLM filed by itself and
nobody reviewed. It never rejects: a rejected case counts as decided and would
block deleting its facts forever. Each case is moved whole to
``correction_cases_overtaken`` (restorable), only while it is still a pending
machine proposal, and is not filed again when its save is re-materialized.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


def _actor() -> str:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id

    return local_trusted_actor_id("python-api")


def _fact(engine, text: str) -> str:
    from superlocalmemory.core.engine_ingestion import canonical_store

    return list(canonical_store(engine, text, source_type="python-api", trusted_actor_id=_actor(),
                                require_complete=True, return_receipt=True).final_fact_ids)[0]


def _facts(engine, count: int) -> list[str]:
    words = ["heron", "otter", "badger", "falcon", "lynx", "marten", "weasel", "plover",
             "ibis", "stoat", "curlew", "vole"]
    return [_fact(engine, f"Synthetic {words[i]} count at pond number {i} was {i + 31}.")
            for i in range(count)]


def _propose(engine, case_id: str, pred: str, succ: str, *, reason="temporal_contradiction",
             kind="host_attested", status="proposed", profile=None) -> None:
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
        transition_on_connection,
    )

    actor = CorrectionActor(actor_id="canonical-writer", actor_kind=kind,
                            trust_tier="canonical_writer" if kind == "host_attested" else "trusted")
    conn = sqlite3.connect(engine._db.db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("BEGIN IMMEDIATE")
    propose_on_connection(conn, case_id=case_id, profile_id=profile or engine._profile_id,
                          scope="personal", predecessor_fact_id=pred, successor_fact_id=succ,
                          reason_code=reason, actor=actor, idempotency_key=f"k-{case_id}",
                          is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    if status != "proposed":
        transition_on_connection(conn, case_id=case_id, expected_version=0, actor=actor,
                                 operation_id=f"t-{case_id}", from_status="proposed",
                                 to_status=status, mutate_temporal=False,
                                 is_profile_active=lambda p: True,
                                 is_actor_trusted=lambda a: True)
    conn.execute("COMMIT")
    conn.close()


def _rows(engine, sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in engine._db.execute(sql, args)]


def _age(engine, case_id: str, days: int) -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    engine._db.execute("UPDATE correction_cases SET created_at = ? WHERE case_id = ?",
                       (stamp, case_id))


def _filter(engine, **kw):
    from superlocalmemory.core.dismissed_cases import DismissFilter

    return DismissFilter(profile_id=engine._profile_id, **kw)


@pytest.fixture
def mixed(engine_with_mock_deps):
    """Machine and person cases of every state, over disjoint predecessors."""
    engine = engine_with_mock_deps
    f = _facts(engine, 10)
    _propose(engine, "m-temporal", f[0], f[1], reason="temporal_contradiction")
    _propose(engine, "m-update", f[1 + 1], f[3], reason="consolidation_update")
    _propose(engine, "m-super", f[4], f[5], reason="consolidation_supersede")
    _propose(engine, "p-edit", f[6], f[7], reason="direct_content_correction",
             kind="host_authenticated")
    _propose(engine, "m-applied", f[8], f[9], reason="consolidation_update", status="applied")
    return engine, f


def test_preview_counts_by_reason_and_changes_nothing(mixed):
    from superlocalmemory.core import dismissed_cases as dc

    engine, _ = mixed
    _age(engine, "m-temporal", 40)
    before = _rows(engine, "SELECT * FROM correction_cases ORDER BY case_id")

    summary = dc.preview(engine._db, _filter(engine))

    assert summary["eligible"] == 3
    assert summary["by_reason"] == {"temporal_contradiction": 1, "consolidation_update": 1,
                                    "consolidation_supersede": 1}
    assert summary["oldest"] < summary["newest"]
    assert "case_id" not in json.dumps(summary)  # counts and dates, never memory text or ids
    assert _rows(engine, "SELECT * FROM correction_cases ORDER BY case_id") == before


def test_dismiss_moves_only_pending_machine_cases(mixed):
    from superlocalmemory.core import dismissed_cases as dc

    engine, _ = mixed
    result = dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")

    assert result == {"dismissed": 3, "skipped_raced": 0, "skipped_changed": 0}
    left = {r["case_id"] for r in _rows(engine, "SELECT case_id FROM correction_cases")}
    assert left == {"p-edit", "m-applied"}
    moved = _rows(engine, "SELECT * FROM correction_cases_overtaken ORDER BY case_id")
    assert [m["case_id"] for m in moved] == ["m-super", "m-temporal", "m-update"]
    for m in moved:
        assert m["user_action"] == "dismissed"
        assert m["closed_reason"] == "dismissed in bulk by the user"
        assert (m["actor_id"], m["operation_id"]) == ("cli", "op-1")
        assert [e["event_type"] for e in json.loads(m["events_json"])] == ["proposed"]
    assert _rows(engine, "SELECT 1 FROM correction_events WHERE case_id IN "
                         "('m-temporal', 'm-update', 'm-super')") == []
    # Nothing was rejected: no case gained a decided state that would protect its facts.
    assert _rows(engine, "SELECT 1 FROM correction_cases WHERE status = 'rejected'") == []


def test_a_dismissed_case_comes_back_with_restore(mixed):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core import overtaken_cases as ot

    engine, _ = mixed
    before = _rows(engine, "SELECT * FROM correction_cases WHERE case_id = 'm-update'")
    events = _rows(engine, "SELECT * FROM correction_events WHERE case_id = 'm-update'")
    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")

    with engine._db.transaction():
        assert ot.restore(engine._db, "m-update") == {"restored": "m-update", "events": 1}

    assert _rows(engine, "SELECT * FROM correction_cases WHERE case_id = 'm-update'") == before
    assert _rows(engine, "SELECT * FROM correction_events WHERE case_id = 'm-update'") == events


def test_a_second_run_finds_nothing(mixed):
    from superlocalmemory.core import dismissed_cases as dc

    engine, _ = mixed
    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")
    again = dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-2")
    assert again == {"dismissed": 0, "skipped_raced": 0, "skipped_changed": 0}


def test_another_profiles_cases_are_left_alone(mixed):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core.dismissed_cases import DismissFilter

    engine, _ = mixed
    result = dc.dismiss(engine._db, DismissFilter(profile_id="someone-else"),
                        actor_id="cli", operation_id="op-1")
    assert result["dismissed"] == 0
    assert len(_rows(engine, "SELECT 1 FROM correction_cases")) == 5


def test_reason_before_and_older_than_filters(mixed):
    from superlocalmemory.core import dismissed_cases as dc

    engine, _ = mixed
    _age(engine, "m-temporal", 40)
    _age(engine, "m-update", 10)

    by_reason = dc.preview(engine._db, _filter(engine, reasons=("consolidation_update",)))
    assert by_reason["by_reason"] == {"consolidation_update": 1}

    cutoff = dc.cutoff_for(older_than_days=30, before=None)
    assert dc.preview(engine._db, _filter(engine, before=cutoff))["eligible"] == 1
    cutoff = dc.cutoff_for(older_than_days=5, before=None)
    assert dc.preview(engine._db, _filter(engine, before=cutoff))["eligible"] == 2
    day = (datetime.now(timezone.utc) - timedelta(days=20)).date().isoformat()
    cutoff = dc.cutoff_for(older_than_days=None, before=day)
    assert dc.preview(engine._db, _filter(engine, before=cutoff))["eligible"] == 1
    # both given: the earlier cutoff wins (a case must satisfy both)
    both = dc.cutoff_for(older_than_days=5, before=day)
    assert dc.preview(engine._db, _filter(engine, before=both))["eligible"] == 1


def test_bad_filters_are_refused_in_plain_words():
    from superlocalmemory.core import dismissed_cases as dc

    with pytest.raises(ValueError, match="--older-than"):
        dc.cutoff_for(older_than_days=-1, before=None)
    with pytest.raises(ValueError, match="--before"):
        dc.cutoff_for(older_than_days=None, before="not-a-date")
    with pytest.raises(ValueError, match="unknown reason"):
        dc.DismissFilter(profile_id="p", reasons=("replaced_by_caller",))
    with pytest.raises(ValueError, match="unknown reason"):
        dc.DismissFilter(profile_id="p", reasons=("direct_content_correction",))


def test_work_is_committed_in_chunks_and_a_raced_case_is_skipped(mixed, monkeypatch):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core import overtaken_cases as ot

    engine, _ = mixed
    real = ot.overtake

    def racing(target, cases, **kw):
        if cases[0]["case_id"] == "m-update":  # reviewed by someone while we ran
            target.execute("UPDATE correction_cases SET version = version + 1 "
                           "WHERE case_id = 'm-update'")
        return real(target, cases, **kw)

    monkeypatch.setattr(ot, "overtake", racing)
    result = dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1",
                        chunk_size=2)

    assert result == {"dismissed": 2, "skipped_raced": 1, "skipped_changed": 0}
    # the raced case is exactly as it was: still pending, event log intact, no audit row
    assert [r["status"] for r in _rows(
        engine, "SELECT status FROM correction_cases WHERE case_id = 'm-update'")] == ["proposed"]
    assert len(_rows(engine, "SELECT 1 FROM correction_events WHERE case_id = 'm-update'")) == 1
    assert _rows(engine, "SELECT 1 FROM correction_cases_overtaken "
                         "WHERE case_id = 'm-update'") == []


def test_a_case_reviewed_after_selection_is_skipped_not_dismissed(mixed, monkeypatch):
    from superlocalmemory.core import dismissed_cases as dc

    engine, _ = mixed
    real = dc._eligible_ids

    def select_then_review(db, flt):
        ids = real(db, flt)
        db.execute("UPDATE correction_cases SET status = 'rejected' WHERE case_id = 'm-super'")
        return ids

    monkeypatch.setattr(dc, "_eligible_ids", select_then_review)
    result = dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")

    assert result == {"dismissed": 2, "skipped_raced": 0, "skipped_changed": 1}
    assert _rows(engine, "SELECT status FROM correction_cases WHERE case_id = 'm-super'") == [
        {"status": "rejected"}]


def test_scale_in_chunks(engine_with_mock_deps):
    """Thousands of cases: ids only, so build the rows straight into the ledger."""
    from superlocalmemory.core import dismissed_cases as dc

    engine = engine_with_mock_deps
    a, b = _facts(engine, 2)
    conn = sqlite3.connect(engine._db.db_path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN")
    cols = ("case_id, profile_id, scope, predecessor_fact_id, successor_fact_id, reason_code, "
            "status, version, idempotency_key, proposed_by_actor_id, proposed_by_actor_kind, "
            "proposed_by_trust_tier, created_at, updated_at")
    for i in range(1200):
        conn.execute(f"INSERT INTO correction_cases ({cols}) VALUES (?, ?, 'personal', ?, ?, "
                     "'consolidation_update', 'proposed', 0, ?, 'w', 'host_attested', "
                     "'canonical_writer', '2026-01-01T00:00:00+00:00', "
                     "'2026-01-01T00:00:00+00:00')",
                     (f"bulk-{i}", engine._profile_id, f"{a}-{i}", b, f"bulk-key-{i}"))
    conn.execute("COMMIT")
    conn.close()

    result = dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-big")

    assert result["dismissed"] == 1200
    assert _rows(engine, "SELECT COUNT(*) AS n FROM correction_cases_overtaken")[0]["n"] == 1200


# -- the tombstone: a dismissed case is not filed again -----------------------------------


def test_a_dismissed_proposal_is_not_filed_again(mixed):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        CorrectionDismissedError,
        propose_on_connection,
    )

    engine, f = mixed
    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")

    actor = CorrectionActor(actor_id="canonical-writer", actor_kind="host_attested",
                            trust_tier="canonical_writer")
    with engine._db.raw_connection() as conn:
        with pytest.raises(CorrectionDismissedError):
            propose_on_connection(
                conn, case_id="m-temporal", profile_id=engine._profile_id, scope="personal",
                predecessor_fact_id=f[0], successor_fact_id=f[1],
                reason_code="temporal_contradiction", actor=actor, idempotency_key="k-m-temporal",
                is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    assert _rows(engine, "SELECT 1 FROM correction_cases WHERE case_id = 'm-temporal'") == []


def test_the_save_path_skips_a_dismissed_candidate_quietly(engine_with_mock_deps, caplog):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core.store_pipeline import _record_correction_candidate

    engine = engine_with_mock_deps
    a, b = _facts(engine, 2)
    args = dict(operation_id="op-save-1", profile_id=engine._profile_id, scope="personal",
                predecessor_fact_id=a, successor_fact_id=b,
                reason_code="consolidation_update", trusted_actor_id=_actor())
    _record_correction_candidate(engine._db, **args)
    [case] = _rows(engine, "SELECT case_id, status FROM correction_cases")
    assert case["status"] == "proposed"

    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")
    with caplog.at_level("WARNING"):
        _record_correction_candidate(engine._db, **args)   # the save is materialized again

    assert _rows(engine, "SELECT 1 FROM correction_cases") == []
    assert [r for r in caplog.records if "not recorded" in r.getMessage()] == []


def test_restoring_a_dismissed_case_lifts_the_tombstone(mixed):
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core import overtaken_cases as ot
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
    )

    engine, f = mixed
    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")
    with engine._db.transaction():
        ot.restore(engine._db, "m-temporal")

    actor = CorrectionActor(actor_id="canonical-writer", actor_kind="host_attested",
                            trust_tier="canonical_writer")
    with engine._db.raw_connection() as conn:   # same key: an idempotent replay of the live case
        case = propose_on_connection(
            conn, case_id="m-temporal", profile_id=engine._profile_id, scope="personal",
            predecessor_fact_id=f[0], successor_fact_id=f[1],
            reason_code="temporal_contradiction", actor=actor, idempotency_key="k-m-temporal",
            is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    assert case.status == "proposed"


def test_a_new_proposal_for_the_same_pair_is_filed_after_the_dismissal_was_put_back(
        engine_with_mock_deps):
    """Dismiss, put back, then the case leaves the ledger by a person's edit: the
    dismissal row is restored, so it is no tombstone, and the save path's next
    proposal for the same pair is filed (not skipped) as a new open case."""
    from superlocalmemory.core import dismissed_cases as dc
    from superlocalmemory.core import overtaken_cases as ot
    from superlocalmemory.core.store_pipeline import _record_correction_candidate

    engine = engine_with_mock_deps
    a, b = _facts(engine, 2)
    args = dict(operation_id="op-save-2", profile_id=engine._profile_id, scope="personal",
                predecessor_fact_id=a, successor_fact_id=b,
                reason_code="consolidation_update", trusted_actor_id=_actor())
    _record_correction_candidate(engine._db, **args)
    dc.dismiss(engine._db, _filter(engine), actor_id="cli", operation_id="op-1")
    _record_correction_candidate(engine._db, **args)
    assert _rows(engine, "SELECT 1 FROM correction_cases") == []      # tombstone holds

    with engine._db.transaction():
        [case_id] = [r["case_id"] for r in _rows(engine, "SELECT case_id FROM "
                                                 "correction_cases_overtaken")]
        ot.restore(engine._db, case_id)
    with engine._db.transaction():
        ot.overtake(engine._db, ot.cases_naming(engine._db, [a]), user_action="edit",
                    actor_id="u", operation_id="op-edit")
    assert _rows(engine, "SELECT 1 FROM correction_cases") == []

    _record_correction_candidate(engine._db, **args)                  # lifted: filed again

    [filed] = _rows(engine, "SELECT case_id, status FROM correction_cases")
    assert filed["case_id"] == case_id and filed["status"] == "proposed"


def test_other_overtaken_rows_do_not_block_a_new_proposal(mixed):
    """4.2.13 behaviour: only a bulk dismissal is remembered, a user's overtake is not."""
    from superlocalmemory.core import overtaken_cases as ot
    from superlocalmemory.storage.correction_cases import (
        CorrectionActor,
        propose_on_connection,
    )

    engine, f = mixed
    with engine._db.transaction():
        ot.overtake(engine._db, ot.cases_naming(engine._db, [f[0]]), user_action="edit",
                    actor_id="u", operation_id="op-edit")
    assert _rows(engine, "SELECT 1 FROM correction_cases WHERE case_id = 'm-temporal'") == []

    actor = CorrectionActor(actor_id="canonical-writer", actor_kind="host_attested",
                            trust_tier="canonical_writer")
    with engine._db.raw_connection() as conn:
        case = propose_on_connection(
            conn, case_id="m-temporal", profile_id=engine._profile_id, scope="personal",
            predecessor_fact_id=f[0], successor_fact_id=f[1],
            reason_code="temporal_contradiction", actor=actor, idempotency_key="k-m-temporal",
            is_profile_active=lambda p: True, is_actor_trusted=lambda a: True)
    assert case.status == "proposed"


def test_overtake_keeps_its_default_reason(mixed):
    from superlocalmemory.core import overtaken_cases as ot

    engine, f = mixed
    with engine._db.transaction():
        ot.overtake(engine._db, ot.cases_naming(engine._db, [f[0]]), user_action="delete",
                    actor_id="u", operation_id="op-d")
    [row] = _rows(engine, "SELECT closed_reason, user_action FROM correction_cases_overtaken")
    assert row == {"closed_reason": "overtaken by a user action", "user_action": "delete"}
