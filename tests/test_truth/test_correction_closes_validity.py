# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30: applying a correction closes the predecessor's valid time; rollback is exact."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.correction_cases import CorrectionActor, CorrectionCaseStore
from superlocalmemory.storage.migrations import M042_correction_case_ledger as m042

_ACTOR = CorrectionActor(actor_id="reviewer-1", actor_kind="human", trust_tier="operator_verified")


def _store(tmp_path: Path, *, successor_tv_from: str | None,
           successor_created: str | None) -> CorrectionCaseStore:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as conn:
        m042.apply(conn)
        conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT, "
                     "scope TEXT, content TEXT, valid_from TEXT, created_at TEXT)")
        conn.execute("INSERT INTO atomic_facts VALUES "
                     "('old', 'alpha', 'project', 'x', NULL, '2026-01-01T00:00:00+00:00'), "
                     "('new', 'alpha', 'project', 'y', NULL, ?)", (successor_created,))
        conn.execute("CREATE TABLE fact_temporal_validity (fact_id TEXT PRIMARY KEY, "
                     "profile_id TEXT, valid_from TEXT, valid_until TEXT, system_created_at "
                     "TEXT, system_expired_at TEXT, invalidated_by TEXT, invalidation_reason "
                     "TEXT)")
        conn.execute("INSERT INTO fact_temporal_validity VALUES ('old', 'alpha', "
                     "'2026-01-01T00:00:00+00:00', NULL, '2026-01-01T00:00:00+00:00', NULL, "
                     "NULL, NULL)")
        if successor_tv_from is not None:
            conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
                         "VALUES ('new', 'alpha', ?)", (successor_tv_from,))
    store = CorrectionCaseStore(path, is_profile_active=lambda p: p == "alpha",
                                is_actor_trusted=lambda a: a.actor_id == "reviewer-1")
    store.propose(case_id="c1", profile_id="alpha", scope="project", predecessor_fact_id="old",
                  successor_fact_id="new", reason_code="release_state_replaced", actor=_ACTOR,
                  idempotency_key="k1")
    return store


def _until(store: CorrectionCaseStore) -> str | None:
    with sqlite3.connect(store.path) as conn:
        return conn.execute("SELECT valid_until FROM fact_temporal_validity "
                            "WHERE fact_id='old'").fetchone()[0]


@pytest.mark.parametrize("tv_from,created,expected", [
    ("2026-03-01", "2026-04-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00"),
    (None, "2026-04-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
])
def test_apply_closes_from_the_successors_takeover(tmp_path, tv_from, created, expected) -> None:
    store = _store(tmp_path, successor_tv_from=tv_from, successor_created=created)
    store.apply("c1", expected_version=0, actor=_ACTOR, operation_id="a1")
    assert _until(store) == expected


def test_reviewer_time_still_wins(tmp_path) -> None:
    store = _store(tmp_path, successor_tv_from="2026-03-01", successor_created=None)
    store.apply("c1", expected_version=0, actor=_ACTOR, operation_id="a1",
                event_valid_until="2026-02-15T00:00:00+00:00")
    assert _until(store) == "2026-02-15T00:00:00+00:00"


def test_no_successor_time_falls_back_to_when_it_was_applied(tmp_path) -> None:
    store = _store(tmp_path, successor_tv_from=None, successor_created=None)
    case = store.apply("c1", expected_version=0, actor=_ACTOR, operation_id="a1")
    assert _until(store) is not None and _until(store) == case.applied_at


def test_a_takeover_before_the_predecessor_began_falls_back_to_the_apply_time(tmp_path) -> None:
    store = _store(tmp_path, successor_tv_from="2025-06-01", successor_created=None)
    case = store.apply("c1", expected_version=0, actor=_ACTOR, operation_id="a1")
    assert _until(store) == case.applied_at


def test_rollback_restores_the_open_interval_exactly(tmp_path) -> None:
    store = _store(tmp_path, successor_tv_from="2026-03-01", successor_created=None)
    store.apply("c1", expected_version=0, actor=_ACTOR, operation_id="a1")
    store.rollback("c1", expected_version=1, actor=_ACTOR, operation_id="r1")
    assert _until(store) is None
