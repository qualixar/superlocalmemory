"""Upgrade startup repair stays post-readiness, bounded, and event-loop safe.

Research witness: upgraded databases can already contain settled
``action_outcomes`` plus provenance while ``learning.db:source_quality`` is
empty.  The repair must consume that durable history without putting a
full-database pass back on FastAPI's readiness boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.learning.source_quality import (
    SourceQualityRepairUnavailable,
    SourceQualityScorer,
)
from superlocalmemory.server import unified_daemon


def _seed_upgrade_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE action_outcomes (
                outcome_id TEXT, profile_id TEXT, fact_ids_json TEXT,
                outcome TEXT, reward REAL, settled INTEGER, settled_at TEXT
            );
            CREATE TABLE provenance (
                profile_id TEXT, fact_id TEXT, source_type TEXT,
                source_id TEXT, created_by TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO provenance VALUES (?, ?, ?, ?, ?)",
            [
                ("work", "f1", "mcp", "codex", ""),
                ("work", "f2", "mcp", "codex", ""),
                ("personal", "f3", "cli", "manual", ""),
            ],
        )
        conn.executemany(
            "INSERT INTO action_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("o1", "work", json.dumps(["f1"]), "success", 1.0, 1, "2026-07-20"),
                ("o2", "work", json.dumps(["f2"]), "failure", 0.0, 1, "2026-07-21"),
                (
                    "o3", "personal", json.dumps(["f3"]), "success",
                    1.0, 1, "2026-07-22",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_post_readiness_repair_recovers_historical_provenance(
    tmp_path: Path,
) -> None:
    memory_db = tmp_path / "memory.db"
    learning_db = tmp_path / "learning.db"
    _seed_upgrade_database(memory_db)
    app = SimpleNamespace(state=SimpleNamespace())

    async def scenario() -> None:
        task = unified_daemon._schedule_source_quality_repair(
            app,
            memory_db,
            learning_db,
            batch_size=1,
            tick_seconds=0,
        )

        assert app.state.source_quality_repair_task is task
        assert app.state.source_quality_repair_status["state"] == "scheduled"
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())

    scorer = SourceQualityScorer(learning_db)
    assert scorer.get_quality("work", "mcp:codex") == pytest.approx(0.5)
    assert scorer.get_quality("personal", "cli:manual") > 0.5
    assert app.state.source_quality_repair_status["state"] == "complete"
    assert app.state.source_quality_repair_status["completed_profiles"] == [
        "personal",
        "work",
    ]


def test_lifespan_schedules_repair_only_after_ready_publication() -> None:
    source = inspect.getsource(unified_daemon.lifespan)

    assert source.index('_publish_process_descriptor(') < source.index(
        '_schedule_source_quality_repair('
    )
    assert source.index('_schedule_source_quality_repair(') < source.index(
        'yield'
    )
    assert 'await _cancel_source_quality_repair(application)' in source


def test_repair_batch_runs_off_event_loop_and_cancels_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setattr(
        unified_daemon,
        "enumerate_source_quality_repair_profiles",
        lambda _path: ["work"],
    )

    def blocking_repair(*_args, **_kwargs):
        started.set()
        release.wait(timeout=2)
        return {"scanned": 1, "observations": 1, "complete": False}

    monkeypatch.setattr(
        unified_daemon,
        "repair_historical_source_quality",
        blocking_repair,
    )

    async def scenario() -> None:
        task = unified_daemon._schedule_source_quality_repair(
            app,
            tmp_path / "memory.db",
            tmp_path / "learning.db",
            tick_seconds=60,
        )
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.1)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert app.state.source_quality_repair_status["state"] == "cancelled"


def test_transient_profile_enumeration_failure_retries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0
    app = SimpleNamespace(state=SimpleNamespace())

    def enumerate_profiles(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SourceQualityRepairUnavailable("database is busy")
        return []

    monkeypatch.setattr(
        unified_daemon,
        "enumerate_source_quality_repair_profiles",
        enumerate_profiles,
    )

    async def scenario() -> None:
        task = unified_daemon._schedule_source_quality_repair(
            app,
            tmp_path / "memory.db",
            tmp_path / "learning.db",
            tick_seconds=0,
        )
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())
    assert calls == 2
    assert app.state.source_quality_repair_status["state"] == "complete"


def test_large_repair_caches_profiles_and_adapts_batch_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    historical_rows = 10_000
    remaining = historical_rows
    enumeration_calls = 0
    batch_sizes = []
    app = SimpleNamespace(state=SimpleNamespace())

    def enumerate_profiles(_path):
        nonlocal enumeration_calls
        enumeration_calls += 1
        return ["work"]

    def repair_batch(*_args, batch_size, **_kwargs):
        nonlocal remaining
        batch_sizes.append(batch_size)
        scanned = min(remaining, batch_size)
        remaining -= scanned
        return {
            "scanned": scanned,
            "observations": scanned,
            "complete": remaining == 0,
        }

    monkeypatch.setattr(
        unified_daemon,
        "enumerate_source_quality_repair_profiles",
        enumerate_profiles,
    )
    monkeypatch.setattr(
        unified_daemon,
        "repair_historical_source_quality",
        repair_batch,
    )

    async def scenario() -> None:
        task = unified_daemon._schedule_source_quality_repair(
            app,
            tmp_path / "memory.db",
            tmp_path / "learning.db",
            batch_size=25,
            tick_seconds=0,
        )
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(scenario())

    assert enumeration_calls == 1
    assert remaining == 0
    assert batch_sizes[:4] == [25, 50, 100, 200]
    assert max(batch_sizes) == 250
    assert len(batch_sizes) < 50
