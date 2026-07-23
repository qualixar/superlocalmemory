"""Reward-to-provenance source-quality wiring stays bounded and idempotent."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from superlocalmemory.learning import source_quality
from superlocalmemory.learning.reward import EngagementRewardModel
from superlocalmemory.learning.source_quality import (
    SourceQualityRepairUnavailable,
    SourceQualityScorer,
    enumerate_source_quality_repair_profiles,
    repair_historical_source_quality,
    update_source_quality_for_reward,
)


def _memory_schema(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE provenance (
                profile_id TEXT, fact_id TEXT, source_type TEXT,
                source_id TEXT, created_by TEXT
            );
            CREATE TABLE action_outcomes (
                outcome_id TEXT, profile_id TEXT, fact_ids_json TEXT,
                outcome TEXT, reward REAL, settled INTEGER, settled_at TEXT
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_finalized_reward_updates_real_provenance_once(tmp_path: Path) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    _memory_schema(memory_db)
    conn = sqlite3.connect(memory_db)
    try:
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
            [
                ("p1", "f1", "mcp", "claude-code", ""),
                ("p1", "f2", "http", "", "codex"),
                ("p2", "f1", "mcp", "excluded", ""),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    first = update_source_quality_for_reward(
        memory_db_path=memory_db,
        learning_db_path=learning_db,
        profile_id="p1",
        outcome_id="o1",
        fact_ids=["f1", "f2"],
        reward=0.8,
    )
    second = update_source_quality_for_reward(
        memory_db_path=memory_db,
        learning_db_path=learning_db,
        profile_id="p1",
        outcome_id="o1",
        fact_ids=["f1", "f2"],
        reward=0.8,
    )

    scorer = SourceQualityScorer(learning_db)
    assert first == 2
    assert second == 0
    assert scorer.get_detailed("p1", "mcp:claude-code") == pytest.approx({
        "alpha": 1.8, "beta": 1.2, "quality": 0.6,
        "updated_at": scorer.get_detailed("p1", "mcp:claude-code")["updated_at"],
    })
    assert scorer.get_quality("p1", "http:codex") == pytest.approx(0.6)
    assert "mcp:excluded" not in scorer.get_all_qualities("p1")


def test_legacy_repair_cursor_upgrade_is_serialized_across_scorers(
    tmp_path: Path,
) -> None:
    learning_db = tmp_path / "learning.db"
    with sqlite3.connect(learning_db) as conn:
        conn.execute(
            """
            CREATE TABLE source_quality_repair_state (
                profile_id TEXT PRIMARY KEY,
                last_rowid INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )

    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def initialize() -> None:
        start.wait()
        try:
            SourceQualityScorer(learning_db)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [threading.Thread(target=initialize) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert errors == []
    with sqlite3.connect(learning_db) as conn:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(source_quality_repair_state)"
            )
        }
    assert {"last_settled_at", "last_outcome_id"} <= columns


def test_operation_uuid_provenance_aggregates_by_stable_trusted_actor(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    _memory_schema(memory_db)
    conn = sqlite3.connect(memory_db)
    try:
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
            [
                ("p1", "f1", "http", "operation-uuid-1", "trusted:codex"),
                ("p1", "f2", "http", "operation-uuid-2", "trusted:codex"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    inserted = update_source_quality_for_reward(
        memory_db_path=memory_db,
        learning_db_path=learning_db,
        profile_id="p1",
        outcome_id="outcome-1",
        fact_ids=["f1", "f2"],
        reward=1.0,
    )

    scorer = SourceQualityScorer(learning_db)
    assert inserted == 1
    assert set(scorer.get_all_qualities("p1")) == {"http:trusted:codex"}
    assert scorer.get_quality("p1", "http:trusted:codex") == pytest.approx(2 / 3)


def test_historical_repair_is_bounded_resumable_and_idempotent(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    _memory_schema(memory_db)
    conn = sqlite3.connect(memory_db)
    try:
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
            [
                ("p1", "f1", "cli", "manual", ""),
                ("p1", "f2", "cli", "manual", ""),
            ],
        )
        conn.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("o1", "p1", json.dumps(["f1"]), "settled", 1.0, 1, "2026-07-20"),
                ("o2", "p1", json.dumps(["f2"]), "settled", 0.0, 1, "2026-07-21"),
                ("o3", "p2", json.dumps(["f1"]), "settled", 1.0, 1, "2026-07-22"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    first = repair_historical_source_quality(
        memory_db, learning_db, "p1", batch_size=1, max_batches=1,
    )
    second = repair_historical_source_quality(
        memory_db, learning_db, "p1", batch_size=1, max_batches=1,
    )
    third = repair_historical_source_quality(
        memory_db, learning_db, "p1", batch_size=1, max_batches=1,
    )

    assert first == {"scanned": 1, "observations": 1, "complete": False}
    assert second == {"scanned": 1, "observations": 1, "complete": False}
    assert third == {"scanned": 0, "observations": 0, "complete": True}
    scorer = SourceQualityScorer(learning_db)
    assert scorer.get_quality("p1", "cli:manual") == pytest.approx(0.5)


def test_historical_repair_sees_an_older_row_after_late_settlement(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    _memory_schema(memory_db)
    with sqlite3.connect(memory_db) as conn:
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
            [
                ("p1", "f1", "mcp", "late", ""),
                ("p1", "f2", "mcp", "early", ""),
            ],
        )
        conn.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("o-old", "p1", '["f1"]', "pending", None, 0, None),
                (
                    "o-new",
                    "p1",
                    '["f2"]',
                    "settled",
                    1.0,
                    1,
                    "2026-07-20T00:00:00+00:00",
                ),
            ],
        )
        conn.commit()

    first = repair_historical_source_quality(
        memory_db, learning_db, "p1", batch_size=10, max_batches=1,
    )
    with sqlite3.connect(memory_db) as conn:
        conn.execute(
            "UPDATE action_outcomes SET outcome='settled',reward=0.0,"
            "settled=1,settled_at=? WHERE outcome_id='o-old'",
            ("2026-07-23T00:00:00+00:00",),
        )
        conn.commit()
    second = repair_historical_source_quality(
        memory_db, learning_db, "p1", batch_size=10, max_batches=1,
    )

    scorer = SourceQualityScorer(learning_db)
    assert first["observations"] == 1
    assert second["observations"] == 1
    assert scorer.get_quality("p1", "mcp:early") > 0.5
    assert scorer.get_quality("p1", "mcp:late") < 0.5


def test_repair_profile_enumeration_uses_only_settled_numeric_outcomes(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    _memory_schema(memory_db)
    conn = sqlite3.connect(memory_db)
    try:
        conn.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("o1", "work", "[]", "settled", 1.0, 1, "2026-07-20"),
                ("o2", "personal", "[]", "pending", None, 0, None),
                ("o3", "other", "[]", "settled", "bad", 1, "2026-07-21"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    assert enumerate_source_quality_repair_profiles(memory_db) == ["work"]


def test_repair_profile_enumeration_does_not_treat_sqlite_error_as_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    memory_db.touch()
    monkeypatch.setattr(
        source_quality.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked"),
        ),
    )

    with pytest.raises(SourceQualityRepairUnavailable):
        enumerate_source_quality_repair_profiles(memory_db)


def test_historical_repair_does_not_treat_batch_error_as_complete(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    memory_db.touch()
    monkeypatch.setattr(
        source_quality,
        "_load_reward_repair_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SourceQualityRepairUnavailable("temporary read failure"),
        ),
    )

    with pytest.raises(SourceQualityRepairUnavailable):
        repair_historical_source_quality(
            memory_db, learning_db, "work", batch_size=1, max_batches=1,
        )
    assert SourceQualityScorer(learning_db).get_repair_cursor("work") == 0


def test_reward_finalizer_feeds_source_quality_after_commit(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    conn = sqlite3.connect(memory_db)
    try:
        conn.executescript(
            """
            CREATE TABLE pending_outcomes (
                outcome_id TEXT PRIMARY KEY, profile_id TEXT, session_id TEXT,
                recall_query_id TEXT, fact_ids_json TEXT, query_text_hash TEXT,
                created_at_ms INTEGER, expires_at_ms INTEGER,
                signals_json TEXT, status TEXT
            );
            CREATE TABLE action_outcomes (
                outcome_id TEXT PRIMARY KEY, profile_id TEXT, query TEXT,
                fact_ids_json TEXT, outcome TEXT, context_json TEXT,
                timestamp TEXT, reward REAL, settled INTEGER,
                settled_at TEXT, recall_query_id TEXT
            );
            CREATE TABLE provenance (
                profile_id TEXT, fact_id TEXT, source_type TEXT,
                source_id TEXT, created_by TEXT
            );
            INSERT INTO provenance VALUES
                ('p1', 'f1', 'mcp', 'codex', '');
            """
        )
        conn.commit()
    finally:
        conn.close()
    model = EngagementRewardModel(memory_db, clock_ms=lambda: 1_000)
    outcome_id = model.record_recall(
        profile_id="p1",
        session_id="s1",
        recall_query_id="q1",
        fact_ids=["f1"],
        query_text="where",
    )
    assert model.register_signal(
        outcome_id=outcome_id, signal_name="cite", signal_value=True,
    )

    reward = model.finalize_outcome(outcome_id=outcome_id)
    model.close()

    assert reward == pytest.approx(0.9)
    scorer = SourceQualityScorer(tmp_path / "learning.db")
    assert scorer.get_quality("p1", "mcp:codex") > 0.5
