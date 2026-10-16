# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M066: memories the hours-clock ELC step archived come back at the first 4.2.8 start.

Only an archive with no recorded decision behind it is reconsidered, and only
with the decay pass's own store-scaled function. Temporary stores only.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.storage import data_repair_undo as undo_log
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import (
    M066_wrongly_archived_memories_are_restored as M066,
)
from tests.test_migrations._repair_store import (
    add_fact,
    first_start,
    forget_migration,
    migrated_store,
)


def _ago(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _archived(conn, fact_id: str, *, days: float = 10, zone: str = "archive", **cols) -> None:
    add_fact(conn, fact_id, f"memory {fact_id}", created_at=_ago(days),
             lifecycle="archived", **cols)
    conn.execute("INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone, "
                 "memory_strength, retention_score) VALUES (?, 'default', ?, 0, 0)",
                 (fact_id, zone))


#: fact id -> what the first start must leave it as (atomic lifecycle, retention zone).
EXPECTED = {
    "a_warm": ("warm", "warm"),            # 60 days unused: the store clock says warm
    "a_cold": ("cold", "cold"),            # 150 days unused: cold, still not archived
    "a_recent": ("active", "active"),      # used yesterday
    "a_old": ("archived", "archive"),      # 400 days, never used: still archived
    "a_user": ("archived", "archive"),     # the user forgot it
    "a_reason": ("archived", "archive"),   # a recorded archive reason
    "a_merged": ("archived", "archive"),   # merged into another memory
    "a_forgotten": ("archived", "forgotten"),
    "a_pinned": ("archived", "archive"),
    "a_vault": ("archived", "archive"),    # in memory_archive
    "a_erasing": ("archived", "archive"),  # an erasure is in flight
    "a_superseded": ("archived", "archive"),
    "a_expired": ("archived", "archive"),  # invalidated by a newer fact
    "a_consolidated": ("archived", "archive"),
    "a_core": ("archived", "archive"),
    "a_langevin": ("archived", "archive"),  # the Langevin tier archives it too
    "live": ("warm", "warm"),
}


def _seed(memory) -> None:
    conn = sqlite3.connect(memory)
    _archived(conn, "a_warm", days=60)
    _archived(conn, "a_cold", days=150)
    _archived(conn, "a_recent", days=60)
    conn.execute("INSERT INTO fact_access_log (log_id, fact_id, profile_id, accessed_at, "
                 "access_type, session_id) VALUES ('l1', 'a_recent', 'default', ?, 'recall', 's')",
                 (_ago(1),))
    _archived(conn, "a_old", days=400)
    _archived(conn, "a_user", archive_status="archived")
    _archived(conn, "a_reason", archive_reason="consolidated")
    _archived(conn, "a_merged", archive_status="merged", merged_into="a_warm")
    _archived(conn, "a_forgotten", zone="forgotten")
    _archived(conn, "a_pinned", pinned=1)
    _archived(conn, "a_vault")
    conn.execute("INSERT INTO memory_archive (archive_id, fact_id, profile_id, payload_json, "
                 "archived_at, reason) VALUES ('ar1', 'a_vault', 'default', '{}', ?, 'manual')",
                 (_ago(5),))
    _archived(conn, "a_erasing")
    conn.execute("INSERT INTO projection_tombstones (fact_id, profile_id, memory_id, erasure_id, "
                 "created_at) "
                 "VALUES ('a_erasing', 'default', 'm-a_erasing', 'er1', 0)")
    _archived(conn, "a_superseded")
    conn.execute("INSERT INTO consolidation_log (action_id, profile_id, action_type, "
                 "new_fact_id, existing_fact_id) VALUES ('c1', 'default', 'supersede', "
                 "'a_warm', 'a_superseded')")
    _archived(conn, "a_expired")
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, system_expired_at) "
                 "VALUES ('a_expired', 'default', ?)", (_ago(3),))
    _archived(conn, "a_consolidated")
    conn.execute("INSERT INTO ccq_consolidated_blocks (block_id, profile_id, content, "
                 "source_fact_ids, char_count, cluster_id, created_at) VALUES "
                 "('b1', 'default', 'gist', '[\"a_consolidated\"]', 4, 'k', ?)", (_ago(4),))
    _archived(conn, "a_core")
    conn.execute("INSERT INTO core_memory_blocks (block_id, profile_id, block_type, content, "
                 "source_fact_ids) VALUES ('k1', 'default', 'user_profile', 'x', "
                 "'[\"a_core\"]')")
    _archived(conn, "a_langevin", days=60, importance=0.0)
    add_fact(conn, "live", "memory live", created_at=_ago(10), lifecycle="warm")
    conn.execute("INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
                 "VALUES ('live', 'default', 'warm')")
    conn.commit()
    conn.close()


@pytest.fixture()
def store(tmp_path):
    memory, learning = migrated_store(tmp_path)
    _seed(memory)
    forget_migration(memory, M066.NAME)
    return memory, learning


def _states(memory) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(memory)
    rows = conn.execute("SELECT f.fact_id, f.lifecycle, r.lifecycle_zone FROM atomic_facts f "
                        "LEFT JOIN fact_retention r USING (fact_id)").fetchall()
    conn.close()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_registered_deferred_with_a_floor() -> None:
    assert _MODULES[M066.NAME] is M066
    assert any(m.name == M066.NAME for m in DEFERRED_MIGRATIONS)
    assert M066.DB_TARGET == "memory" and M066.DOWNGRADE_FLOOR == 51
    assert M066.REPAIR_NOT_APPLICABLE.strip()


def test_first_start_restores_only_what_the_store_clock_calls_live(store) -> None:
    memory, learning = store
    result = first_start(memory, learning)
    assert M066.NAME in result["applied"], result
    assert _states(memory) == EXPECTED


def test_restored_memories_are_queued_for_the_projections(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    queued = {r[0] for r in conn.execute("SELECT fact_id FROM projection_outbox")}
    assert {"a_warm", "a_cold", "a_recent"} <= queued and "a_old" not in queued


def test_nothing_qualifies_is_a_no_op(tmp_path) -> None:
    memory, learning = migrated_store(tmp_path)
    conn = sqlite3.connect(memory)
    add_fact(conn, "live", "memory live", created_at=_ago(10), lifecycle="warm")
    conn.commit()
    conn.close()
    forget_migration(memory, M066.NAME)
    before = _states(memory)
    assert M066.NAME in first_start(memory, learning)["applied"]
    assert _states(memory) == before
    assert undo_log.summary(sqlite3.connect(memory), M066.NAME)["facts"] == 0


def test_it_runs_once_and_apply_is_idempotent(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    again = first_start(memory, learning)
    assert M066.NAME in again["skipped"] and M066.NAME not in again["applied"]
    conn = sqlite3.connect(memory, isolation_level=None)
    rows = conn.execute("SELECT COUNT(*) FROM data_repair_undo").fetchone()[0]
    M066.apply(conn)
    assert conn.execute("SELECT COUNT(*) FROM data_repair_undo").fetchone()[0] == rows
    conn.close()
    assert _states(memory) == EXPECTED


def test_the_undo_log_archives_them_again(store) -> None:
    memory, learning = store
    before = _states(memory)
    first_start(memory, learning)
    conn = sqlite3.connect(memory, isolation_level=None)
    assert undo_log.undo(conn, M066.NAME) == 6  # three facts, mirror + authority each
    conn.close()
    assert _states(memory) == before


def test_a_crash_mid_run_changes_nothing_and_the_next_start_finishes(store, monkeypatch) -> None:
    memory, learning = store
    before = _states(memory)
    from superlocalmemory.storage import archive_repair

    real = archive_repair._restore

    def dies_on_the_second(conn, writer, fact, zone, _calls={"n": 0}):
        _calls["n"] += 1
        if _calls["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error (simulated crash)")
        return real(conn, writer, fact, zone)

    monkeypatch.setattr(archive_repair, "_restore", dies_on_the_second)
    assert M066.NAME in first_start(memory, learning)["failed"]
    assert _states(memory) == before
    monkeypatch.setattr(archive_repair, "_restore", real)
    assert M066.NAME in first_start(memory, learning)["applied"]
    assert _states(memory) == EXPECTED


def test_summary_reports_what_the_repair_did(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    summary = undo_log.summary(sqlite3.connect(memory), M066.NAME)
    assert summary["facts"] == 3 and summary["lifecycle_restored"] == 3


def test_a_memory_that_changed_after_the_scan_is_left_alone(store, monkeypatch) -> None:
    """The scan runs without the write lock; the write re-checks each memory."""
    memory, learning = store
    from superlocalmemory.storage import archive_repair

    real = archive_repair._plan

    def user_acts_meanwhile(conn, settings):
        planned = real(conn, settings)
        other = sqlite3.connect(memory)
        other.execute("UPDATE atomic_facts SET archive_status = 'archived', lifecycle = 'cold' "
                      "WHERE fact_id = 'a_warm'")
        other.commit()
        other.close()
        return planned

    monkeypatch.setattr(archive_repair, "_plan", user_acts_meanwhile)
    assert M066.NAME in first_start(memory, learning)["applied"]
    states = _states(memory)
    assert states["a_warm"] == ("cold", "archive")  # what the user left, untouched
    assert states["a_cold"] == ("cold", "cold")
    conn = sqlite3.connect(memory)
    assert not conn.execute("SELECT 1 FROM data_repair_undo WHERE fact_id = 'a_warm'").fetchone()


def test_a_compliance_rule_that_archives_keeps_its_archive(store) -> None:
    """A retention rule archives by age and records nothing per memory: any
    memory old enough for an archiving rule stays archived."""
    memory, learning = store
    from superlocalmemory.compliance.retention import _RETENTION_RULES_TABLE

    conn = sqlite3.connect(memory)
    conn.executescript(_RETENTION_RULES_TABLE)
    conn.execute("INSERT INTO retention_rules (profile_id, rule_name, days, action) "
                 "VALUES ('default', 'gdpr-90', 90, 'archive')")
    conn.commit()
    conn.close()
    first_start(memory, learning)
    states = _states(memory)
    assert states["a_cold"] == ("archived", "archive")    # 150 days: the rule covers it
    assert states["a_warm"] == ("warm", "warm")          # 60 days: the rule does not


def test_what_the_default_settings_archived_comes_back_on_the_persons_own(
        tmp_path, monkeypatch) -> None:
    """Before 4.2.8 the Langevin pass decided tiers on the default forgetting
    settings. A person whose own settings keep a 140-day-old memory gets it back;
    one they pinned-out, forgot or erased stays where they put it."""
    from superlocalmemory.core.config import ForgettingConfig
    from superlocalmemory.storage import archive_repair

    personal = ForgettingConfig(alpha=50.0, beta=100.0, gamma=30.0, delta=10.0,
                                min_strength=720.0, max_strength=87600.0,
                                archive_threshold=0.05, forget_threshold=0.01)
    monkeypatch.setattr(archive_repair, "_settings",
                        lambda conn: archive_repair._Settings(personal, True))
    memory, learning = migrated_store(tmp_path)
    conn = sqlite3.connect(memory)
    _archived(conn, "kept_by_settings", days=140, importance=0.0)
    _archived(conn, "user_forgot", days=140, importance=0.0, archive_status="archived")
    _archived(conn, "user_pinned", days=140, importance=0.0, pinned=1)
    conn.commit()
    conn.close()
    forget_migration(memory, M066.NAME)
    first_start(memory, learning)
    states = _states(memory)
    assert states["kept_by_settings"][0] != "archived"
    assert states["kept_by_settings"][0] == states["kept_by_settings"][1].replace(
        "archive", "archived")
    assert states["user_forgot"] == ("archived", "archive")
    assert states["user_pinned"] == ("archived", "archive")
    rows = sqlite3.connect(memory).execute(
        "SELECT COUNT(*) FROM data_repair_undo WHERE fact_id = 'kept_by_settings'").fetchone()
    assert rows[0] == 2  # mirror + authority, each undoable


def test_junk_is_never_brought_back(tmp_path) -> None:
    """A memory the keep-worthiness predicate calls junk stays archived even when
    every clock calls it live: restoring it would put noise back in recall."""
    memory, learning = migrated_store(tmp_path)
    conn = sqlite3.connect(memory)
    junk = {"j_session": "The session ended at 20:52 on August 3, 2026",
            "j_file": "The file ISSUE-011-DRAFT.md was closed on 2026-07-30",
            "j_date": "The file was modified on January 15, 2024."}
    for fact_id, text in junk.items():
        add_fact(conn, fact_id, text, created_at=_ago(60), lifecycle="archived")
        conn.execute("INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
                     "VALUES (?, 'default', 'archive')", (fact_id,))
    _archived(conn, "real", days=60)
    conn.commit()
    conn.close()
    forget_migration(memory, M066.NAME)
    first_start(memory, learning)
    states = _states(memory)
    assert all(states[f] == ("archived", "archive") for f in junk), states
    assert states["real"] == ("warm", "warm")
