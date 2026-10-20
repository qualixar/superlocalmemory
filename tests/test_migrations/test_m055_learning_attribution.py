# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M055: durable learning attribution, additive, learning.db only.

Run on a store shaped like a 4.1.24 one that has real learning history
(settled plays, a trained model without ``metadata_json``) — the upgrade must
keep every row, copy only learning.db, be idempotent, and leave the new
unique index enforcing "one outcome settles one play".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import superlocalmemory.storage._schema_version as sv
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migrations import M055_learning_attribution as M055
from tests.test_storage._upgrade_store import clear_snapshots, current_store


def _as_4124(learning: Path) -> None:
    """Take M055 back out, keeping history rows: a 4.1.24 learning.db."""
    with sqlite3.connect(learning) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_plays_outcome_once")
        conn.execute("DROP TABLE IF EXISTS shadow_assignments")
        conn.execute("ALTER TABLE bandit_plays DROP COLUMN outcome_id")
        conn.execute("ALTER TABLE learning_model_state DROP COLUMN metadata_json")
        conn.execute("DELETE FROM migration_log WHERE name = ?", (M055.NAME,))
        for i in range(500):
            conn.execute(
                "INSERT INTO bandit_plays (profile_id, query_id, stratum, arm_id,"
                " played_at, reward, settled_at, settlement_type) VALUES "
                "('default', ?, 's', 'a', '2026-09-01T00:00:00+00:00', 0.5, "
                "'2026-09-01T00:15:00+00:00', 'default')", (f"q{i}",))
        conn.execute(
            "INSERT INTO learning_model_state (profile_id, state_bytes, "
            "bytes_sha256, is_active, trained_at, updated_at) VALUES "
            "('default', x'00', '', 1, 't', 't')")


def _snapshots(root: Path) -> list[str]:
    snaps = root / "pre-migration-snapshots"
    if not snaps.is_dir():
        return []
    return sorted(p.name.split("-")[0] for p in snaps.iterdir()
                  if p.name.endswith("-pre-migration.db"))


def test_registered_eager_on_learning_and_additive() -> None:
    m = next(m for m in mr.MIGRATIONS if m.name == M055.NAME)
    assert m.db_target == "learning" == M055.DB_TARGET
    assert _MODULES[M055.NAME] is M055
    assert sv.SUPPORTED_SCHEMA_VERSION == 67  # M055, WP-4b recall, M059 erasure visibility, M060 memory description, M061 hub baselines, M062 hook bookkeeping, M063 blank temporal dates, M064 orphan index rows, M065 fabricated dates, M066 wrongly archived, M067 projection change trigger
    assert M055.BREAKING_VERSION == 0 and M055.DOWNGRADE_FLOOR == 51


def test_a_store_with_history_upgrades_and_keeps_every_row(tmp_path) -> None:
    learning, memory = current_store(tmp_path)
    _as_4124(learning)
    result = mr.apply_all(learning, memory)
    assert M055.NAME in result["applied"] and result["failed"] == []
    assert _snapshots(tmp_path) == ["learning"], "memory.db must not be copied"
    with sqlite3.connect(learning) as conn:
        assert M055.verify(conn)
        assert conn.execute("SELECT COUNT(*) FROM bandit_plays").fetchone() == (500,)
        assert conn.execute(
            "SELECT COUNT(*) FROM bandit_plays WHERE outcome_id IS NOT NULL"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT metadata_json FROM learning_model_state").fetchone() == (None,)
    clear_snapshots(tmp_path)


def test_running_twice_is_a_no_op(tmp_path) -> None:
    learning, memory = current_store(tmp_path)
    again = mr.apply_all(learning, memory)
    assert M055.NAME not in again["applied"] and again["failed"] == []
    with sqlite3.connect(learning) as conn:
        M055.apply(conn)  # apply itself is idempotent
        assert M055.verify(conn)
    assert _snapshots(tmp_path) == []


def test_one_outcome_can_settle_only_one_play(tmp_path) -> None:
    learning, _memory = current_store(tmp_path)
    with sqlite3.connect(learning) as conn:
        for q in ("a", "b"):
            conn.execute(
                "INSERT INTO bandit_plays (profile_id, query_id, stratum, arm_id,"
                " played_at) VALUES ('default', ?, 's', 'x', 't')", (q,))
        conn.execute("UPDATE bandit_plays SET outcome_id='oc1' WHERE query_id='a'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE bandit_plays SET outcome_id='oc1' WHERE query_id='b'")


def test_repair_restores_a_dropped_index(tmp_path) -> None:
    learning, _memory = current_store(tmp_path)
    with sqlite3.connect(learning) as conn:
        conn.execute("DROP INDEX idx_plays_outcome_once")
        assert M055.verify(conn) is False
        M055.repair(conn)
        assert M055.verify(conn) is True
