# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-07

"""Tests for superlocalmemory.storage.migration_runner.

Covers LLD-07 §8.1 — apply_all, idempotency, failure recovery,
DDL drift detection, dry-run, status.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage import migration_runner as mr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_dbs(tmp_path: Path) -> tuple[Path, Path]:
    """Create fresh learning.db + memory.db with the pre-3.4.22 schema.

    Matches the live schema in LLD-07 §1 — learning_signals (7 cols),
    learning_features (7 cols), learning_model_state (UNIQUE profile_id),
    action_outcomes.
    """
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"

    with sqlite3.connect(learning_db) as conn:
        conn.executescript(
            """
            CREATE TABLE learning_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                query TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                value REAL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX idx_signals_profile ON learning_signals(profile_id);
            CREATE INDEX idx_signals_fact ON learning_signals(fact_id);

            CREATE TABLE learning_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                features_json TEXT NOT NULL,
                label REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX idx_features_profile ON learning_features(profile_id);

            CREATE TABLE learning_model_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL UNIQUE,
                state_bytes BLOB NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE channel_credits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                query_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                hits INTEGER DEFAULT 0,
                total INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX idx_channel_credit_unique
                ON channel_credits(profile_id, query_type, channel);
            """
        )

    with sqlite3.connect(memory_db) as conn:
        conn.executescript(
            """
            CREATE TABLE profiles (
                profile_id TEXT PRIMARY KEY
            );
            CREATE TABLE action_outcomes (
                outcome_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default',
                query TEXT NOT NULL DEFAULT '',
                fact_ids_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT NOT NULL DEFAULT '',
                context_json TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            -- atomic_facts is bootstrapped by MemoryEngine.initialize in
            -- production; the tests need it so DEFERRED M011 can apply.
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL DEFAULT ''
            );
            """
        )

    return learning_db, memory_db


def _table_cols(db: Path, table: str) -> list[str]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in rows]


def _log_rows(db: Path) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        try:
            return conn.execute(
                "SELECT name, status FROM migration_log ORDER BY name"
            ).fetchall()
        except sqlite3.OperationalError:
            return []


# ---------------------------------------------------------------------------
# apply_all — happy paths
# ---------------------------------------------------------------------------


def test_apply_all_on_fresh_db_applies_all(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_all(learning_db, memory_db)
    assert stats["applied"]
    assert stats["failed"] == [] or stats["failed"] == 0 or not stats["failed"]
    # 8 migrations in MIGRATIONS: M003, M001, M002, M005, M009, M010, M004, M007.
    assert len(stats["applied"]) == 9
    # migration_log should contain each as complete.
    log_learning = _log_rows(learning_db)
    log_memory = _log_rows(memory_db)
    names_learning = [r[0] for r in log_learning]
    names_memory = [r[0] for r in log_memory]
    assert "M003_migration_log" in names_learning
    assert "M001_add_signal_features_columns" in names_learning
    assert "M002_model_state_history" in names_learning
    assert "M005_bandit_tables" in names_learning
    assert "M009_model_lineage" in names_learning
    assert "M010_evolution_config" in names_learning
    assert "M004_cross_platform_sync_log" in names_memory
    assert "M007_pending_outcomes" in names_memory


def test_apply_all_creates_all_new_columns(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    sig_cols = set(_table_cols(learning_db, "learning_signals"))
    assert {"query_id", "query_text_hash", "position", "channel_scores",
            "cross_encoder"} <= sig_cols
    feat_cols = set(_table_cols(learning_db, "learning_features"))
    assert {"signal_id", "is_synthetic"} <= feat_cols
    # model_state now has extra cols (via table rebuild)
    model_cols = set(_table_cols(learning_db, "learning_model_state"))
    assert {"model_version", "bytes_sha256", "is_active", "trained_at",
            "metrics_json", "feature_names", "trained_on_count"} <= model_cols


def test_apply_all_creates_bandit_tables(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    with sqlite3.connect(learning_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "bandit_arms" in tables
    assert "bandit_plays" in tables


def test_apply_all_creates_sync_log_in_memory_db(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    with sqlite3.connect(memory_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "cross_platform_sync_log" in tables


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_apply_all_idempotent(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    first = mr.apply_all(learning_db, memory_db)
    second = mr.apply_all(learning_db, memory_db)
    # 8 migrations: M003, M001, M002, M005, M009, M010, M004, M007.
    assert len(first["applied"]) == 9
    # Second pass: everything skipped.
    assert len(second["applied"]) == 0
    assert len(second["skipped"]) == 9


def test_apply_all_preserves_model_state_rows(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, _ = fresh_dbs
    # Seed a row in legacy learning_model_state
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "INSERT INTO learning_model_state (profile_id, state_bytes, updated_at) "
            "VALUES (?, ?, ?)",
            ("default", b"\x00\x01\x02", "2026-04-17T00:00:00"),
        )
        conn.commit()
    mr.apply_all(learning_db, fresh_dbs[1])
    with sqlite3.connect(learning_db) as conn:
        rows = conn.execute(
            "SELECT profile_id, state_bytes, is_active FROM learning_model_state"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "default"
    assert rows[0][1] == b"\x00\x01\x02"
    assert rows[0][2] == 1  # marked active


# ---------------------------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------------------------


def test_apply_all_recovers_from_failed_status(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    # First run completes.
    mr.apply_all(learning_db, memory_db)
    # Simulate a migration recorded as failed.
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "UPDATE migration_log SET status = 'failed' "
            "WHERE name = 'M001_add_signal_features_columns'"
        )
        conn.commit()
    stats = mr.apply_all(learning_db, memory_db)
    # Should retry and mark complete — but columns already exist. The runner
    # must recognise that the migration is idempotent and succeed without error.
    with sqlite3.connect(learning_db) as conn:
        row = conn.execute(
            "SELECT status FROM migration_log "
            "WHERE name = 'M001_add_signal_features_columns'"
        ).fetchone()
    assert row[0] == "complete"
    # Reported under "applied" (we re-ran it).
    assert "M001_add_signal_features_columns" in stats["applied"]


def test_apply_all_ddl_drift_detected(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    # Tamper: set a bogus ddl_sha256 for an applied migration.
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "UPDATE migration_log SET ddl_sha256 = 'deadbeef' "
            "WHERE name = 'M001_add_signal_features_columns'"
        )
        conn.commit()
    stats = mr.apply_all(learning_db, memory_db)
    # Drift must be surfaced (not silently skipped).
    details = stats.get("details", {})
    m1_detail = details.get("M001_add_signal_features_columns", "")
    assert "drift" in m1_detail.lower()


def test_apply_all_non_fatal_on_broken_migration(
    fresh_dbs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learning_db, memory_db = fresh_dbs
    # Inject an extra broken migration via monkeypatch.
    broken = mr.Migration(
        name="MX_broken",
        db_target="learning",
        ddl="BEGIN IMMEDIATE;\nSELECT nonsense_column FROM nonsense_table;\nCOMMIT;",
    )
    monkeypatch.setattr(mr, "MIGRATIONS", mr.MIGRATIONS + [broken])
    stats = mr.apply_all(learning_db, memory_db)
    assert "MX_broken" in stats["failed"]
    # Other migrations still applied.
    assert len(stats["applied"]) >= 5


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------


def test_apply_all_dry_run_applies_nothing(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_all(learning_db, memory_db, dry_run=True)
    assert len(stats["applied"]) == 0
    # Columns should NOT exist yet.
    sig_cols = set(_table_cols(learning_db, "learning_signals"))
    assert "query_id" not in sig_cols
    # Planned migrations surfaced in details.
    assert stats.get("details")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_before_any_migration(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    s = mr.status(learning_db, memory_db)
    # All should be "missing"
    assert s["M001_add_signal_features_columns"] == "missing"
    assert s["M002_model_state_history"] == "missing"
    assert s["M003_migration_log"] == "missing"
    assert s["M004_cross_platform_sync_log"] == "missing"
    assert s["M005_bandit_tables"] == "missing"


def test_status_after_apply(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    s = mr.status(learning_db, memory_db)
    for name in (
        "M001_add_signal_features_columns",
        "M002_model_state_history",
        "M003_migration_log",
        "M004_cross_platform_sync_log",
        "M005_bandit_tables",
    ):
        assert s[name] == "complete", f"{name} not complete"


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def test_dry_run_with_in_progress_row(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "UPDATE migration_log SET status='in_progress' WHERE name=?",
            ("M001_add_signal_features_columns",),
        )
        conn.commit()
    stats = mr.apply_all(learning_db, memory_db, dry_run=True)
    detail = stats["details"]["M001_add_signal_features_columns"]
    assert "dry-run" in detail.lower()


def test_dry_run_with_log_but_no_row(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    # Pre-bootstrap migration_log so dry-run reaches the per-migration branch.
    with sqlite3.connect(learning_db) as conn:
        conn.executescript(
            """
            CREATE TABLE migration_log (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                ddl_sha256 TEXT NOT NULL,
                rows_affected INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL
            );
            """
        )
    stats = mr.apply_all(learning_db, memory_db, dry_run=True)
    # M001/M002/M005 have no row in migration_log → dry-run: would apply.
    detail = stats["details"]["M001_add_signal_features_columns"]
    assert "would apply" in detail.lower()


def test_dry_run_on_fully_fresh_dbs(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_all(learning_db, memory_db, dry_run=True)
    assert stats["applied"] == []
    # Dry-run must not create tables on either DB.
    with sqlite3.connect(learning_db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
    assert "migration_log" not in tables
    assert "bandit_arms" not in tables


def test_status_partial_complete(fresh_dbs: tuple[Path, Path]) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    # Force one to 'failed' and check status reflects it.
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            "UPDATE migration_log SET status='failed' WHERE name=?",
            ("M005_bandit_tables",),
        )
        conn.commit()
    s = mr.status(learning_db, memory_db)
    assert s["M005_bandit_tables"] == "failed"


def test_power_loss_during_migration_leaves_clean_state(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    # Put the migration_log into 'in_progress' as if process died mid-write.
    with sqlite3.connect(learning_db) as conn:
        conn.executescript(
            """
            CREATE TABLE migration_log (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                ddl_sha256 TEXT NOT NULL,
                rows_affected INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO migration_log (name, applied_at, ddl_sha256, status) "
            "VALUES ('M001_add_signal_features_columns', '2026-04-17', 'x', "
            "'in_progress')"
        )
        conn.commit()
    # Running apply_all should treat in_progress like failed → retry.
    stats = mr.apply_all(learning_db, memory_db)
    with sqlite3.connect(learning_db) as conn:
        row = conn.execute(
            "SELECT status FROM migration_log "
            "WHERE name = 'M001_add_signal_features_columns'"
        ).fetchone()
    assert row[0] == "complete"


# ---------------------------------------------------------------------------
# apply_deferred — D.1-apply (v3.4.22): behavioral-post-boot M006 hook
# ---------------------------------------------------------------------------


def test_apply_all_does_not_include_m006(fresh_dbs: tuple[Path, Path]) -> None:
    """apply_all must never touch M006 — it's deferred until after engine init.

    Guarantees the reward column is absent on a vanilla daemon-startup flow
    where the deferred phase hasn't run yet.
    """
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_all(learning_db, memory_db)
    assert "M006_action_outcomes_reward" not in stats["applied"]
    assert "M006_action_outcomes_reward" not in stats["skipped"]
    assert "M006_action_outcomes_reward" not in stats["failed"]
    cols = _table_cols(memory_db, "action_outcomes")
    assert "reward" not in cols


def test_apply_deferred_adds_reward_column(fresh_dbs: tuple[Path, Path]) -> None:
    """apply_deferred wires the M006 reward column onto action_outcomes."""
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)  # bootstrap migration_log first
    stats = mr.apply_deferred(learning_db, memory_db)
    assert "M006_action_outcomes_reward" in stats["applied"]
    assert stats["failed"] == []
    cols = _table_cols(memory_db, "action_outcomes")
    assert "reward" in cols
    # Partial index from the M006 DDL should also exist.
    with sqlite3.connect(memory_db) as conn:
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
    assert "idx_action_outcomes_settled_reward" in idx


def test_apply_deferred_is_idempotent(fresh_dbs: tuple[Path, Path]) -> None:
    """Second apply_deferred run is a no-op; column stays, status stays complete."""
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    mr.apply_deferred(learning_db, memory_db)
    stats2 = mr.apply_deferred(learning_db, memory_db)
    assert "M006_action_outcomes_reward" not in stats2["applied"]
    assert "M006_action_outcomes_reward" in stats2["skipped"]
    assert stats2["failed"] == []
    # Column must still be present — no rebuild, no loss.
    assert "reward" in _table_cols(memory_db, "action_outcomes")


def test_apply_deferred_without_prior_apply_all_refuses_split_brain(
    fresh_dbs: tuple[Path, Path],
) -> None:
    """S9-W1 C3: apply_deferred must NOT independently bootstrap migration_log.

    Prior contract let apply_deferred create the log on its own. That
    created split-brain state when apply_all had crashed before reaching
    the memory DB — the log would then record only deferred migrations as
    applied, with no trace of the sync set ever being attempted. Memory
    DB is sacred (18k+ atomic facts). The new contract: apply_deferred
    fails loudly if the log is missing, forcing the operator to run
    apply_all first.
    """
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_deferred(learning_db, memory_db)
    assert "M006_action_outcomes_reward" in stats["failed"]
    assert "migration_log missing" in stats["details"][
        "M006_action_outcomes_reward"
    ]
    # Reward column NOT added because the migration refused to run.
    assert "reward" not in _table_cols(memory_db, "action_outcomes")

    # After apply_all bootstraps both logs, apply_deferred succeeds.
    mr.apply_all(learning_db, memory_db)
    stats2 = mr.apply_deferred(learning_db, memory_db)
    assert "M006_action_outcomes_reward" in stats2["applied"]
    assert "reward" in _table_cols(memory_db, "action_outcomes")


def test_apply_deferred_missing_action_outcomes_is_non_fatal(
    tmp_path: Path,
) -> None:
    """If action_outcomes doesn't exist, apply_deferred records failed, never raises."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    # Touch both DBs; memory_db is empty (no action_outcomes table).
    sqlite3.connect(learning_db).close()
    sqlite3.connect(memory_db).close()
    stats = mr.apply_deferred(learning_db, memory_db)
    # Must not raise and must report M006 — either failed or skipped, depending
    # on whether verify() returns False before DDL runs.
    assert (
        "M006_action_outcomes_reward" in stats["failed"]
        or "M006_action_outcomes_reward" in stats["skipped"]
    )


def test_apply_deferred_dry_run_does_not_modify_db(
    fresh_dbs: tuple[Path, Path],
) -> None:
    """dry_run must leave the schema untouched."""
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    stats = mr.apply_deferred(learning_db, memory_db, dry_run=True)
    # Nothing gets applied in dry-run.
    assert "M006_action_outcomes_reward" not in stats["applied"]
    assert "reward" not in _table_cols(memory_db, "action_outcomes")


def test_status_reports_m006(fresh_dbs: tuple[Path, Path]) -> None:
    """status() must include M006 alongside the regular migrations."""
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    before = mr.status(learning_db, memory_db)
    assert "M006_action_outcomes_reward" in before
    assert before["M006_action_outcomes_reward"] == "missing"
    mr.apply_deferred(learning_db, memory_db)
    after = mr.status(learning_db, memory_db)
    assert after["M006_action_outcomes_reward"] == "complete"


# ---------------------------------------------------------------------------
# P0.6 — M007, M009, M010, M011 registration + behaviour
# ---------------------------------------------------------------------------


def _table_names(db: Path) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }


def _index_names(db: Path) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }


def test_apply_all_registers_m007_m009_m010(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    stats = mr.apply_all(learning_db, memory_db)
    for name in ("M007_pending_outcomes",
                 "M009_model_lineage",
                 "M010_evolution_config"):
        assert name in stats["applied"], f"{name} not applied"


def test_apply_all_creates_pending_outcomes_table(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    assert "pending_outcomes" in _table_names(memory_db)
    cols = set(_table_cols(memory_db, "pending_outcomes"))
    required = {
        "outcome_id", "profile_id", "session_id", "recall_query_id",
        "fact_ids_json", "query_text_hash", "created_at_ms",
        "expires_at_ms", "signals_json", "status",
    }
    assert required <= cols
    indexes = _index_names(memory_db)
    assert "idx_pending_profile_expires" in indexes
    assert "idx_pending_status" in indexes


def test_apply_all_extends_model_state_columns(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    cols = set(_table_cols(learning_db, "learning_model_state"))
    required = {
        "is_previous", "is_rollback", "is_candidate",
        "shadow_results_json", "promoted_at", "rollback_reason",
    }
    assert required <= cols
    indexes = _index_names(learning_db)
    assert "idx_model_active_one" in indexes
    assert "idx_model_candidate_one" in indexes


def test_apply_all_creates_evolution_config_tables(
    fresh_dbs: tuple[Path, Path],
) -> None:
    learning_db, memory_db = fresh_dbs
    mr.apply_all(learning_db, memory_db)
    tables = _table_names(learning_db)
    assert {"evolution_config", "evolution_llm_cost_log"} <= tables
    # Defaults: evolution disabled, Haiku backend.
    with sqlite3.connect(learning_db) as conn:
        cols = {
            r[1]: (r[3], r[4])  # (notnull, default_value)
            for r in conn.execute(
                "PRAGMA table_info(evolution_config)"
            ).fetchall()
        }
    assert cols["enabled"] == (1, "0")
    assert cols["llm_model"][1] == "'claude-haiku-4-5'"


def test_apply_deferred_registers_m011(
    fresh_dbs: tuple[Path, Path],
) -> None:
    """M011 lives in DEFERRED_MIGRATIONS alongside M006."""
    names = [m.name for m in mr.DEFERRED_MIGRATIONS]
    assert "M011_archive_and_merge" in names


def test_apply_deferred_creates_archive_and_merge_log(
    tmp_path: Path,
) -> None:
    """Deferred apply against a memory.db that includes atomic_facts."""
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    sqlite3.connect(learning_db).close()
    # Pre-seed atomic_facts to mimic engine-init bootstrap.
    with sqlite3.connect(memory_db) as conn:
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL DEFAULT ''
            );
            """
        )
    # S9-W1 C3: apply_all must run first to bootstrap migration_log on
    # both DBs. apply_deferred no longer independently creates the log.
    mr.apply_all(learning_db, memory_db)
    stats = mr.apply_deferred(learning_db, memory_db)
    assert "M011_archive_and_merge" in stats["applied"]
    tables = _table_names(memory_db)
    assert {"memory_archive", "memory_merge_log"} <= tables


def test_apply_deferred_extends_atomic_facts(tmp_path: Path) -> None:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    sqlite3.connect(learning_db).close()
    with sqlite3.connect(memory_db) as conn:
        conn.executescript(
            """
            CREATE TABLE atomic_facts (
                fact_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL DEFAULT 'default'
            );
            """
        )
    # S9-W1 C3: apply_all bootstraps both DBs' migration_log first.
    mr.apply_all(learning_db, memory_db)
    mr.apply_deferred(learning_db, memory_db)
    cols = set(_table_cols(memory_db, "atomic_facts"))
    assert {"archive_status", "archive_reason", "merged_into",
            "retrieval_prior"} <= cols


def test_all_migrations_idempotent_on_second_run(
    fresh_dbs: tuple[Path, Path],
) -> None:
    """Running apply_all + apply_deferred twice is a no-op on the second run."""
    learning_db, memory_db = fresh_dbs
    # atomic_facts is already in the fixture; DEFERRED M011 can apply directly.
    mr.apply_all(learning_db, memory_db)
    mr.apply_deferred(learning_db, memory_db)

    stats2 = mr.apply_all(learning_db, memory_db)
    stats2d = mr.apply_deferred(learning_db, memory_db)
    assert stats2["failed"] == []
    assert stats2d["failed"] == []
    for name in ("M007_pending_outcomes", "M009_model_lineage",
                 "M010_evolution_config"):
        assert name in stats2["skipped"]
    for name in ("M006_action_outcomes_reward", "M011_archive_and_merge"):
        assert name in stats2d["skipped"]
