"""Dashboard ingestion-source metrics must come from stored provenance."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

from superlocalmemory.server.routes import stats
from superlocalmemory.server.routes.stats import (
    _load_ingestion_sources,
    _load_ingestion_sources_with_status,
)


def test_ingestion_sources_are_profile_scoped_and_deduplicate_facts() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE provenance ("
        "profile_id TEXT, fact_id TEXT, source_type TEXT)"
    )
    conn.executemany(
        "INSERT INTO provenance VALUES (?, ?, ?)",
        [
            ("default", "f1", "http"),
            ("default", "f1", "http"),
            ("default", "f2", "cli-sync"),
            ("other", "f3", "http"),
        ],
    )

    assert _load_ingestion_sources(conn.cursor(), "default") == [
        {"source_type": "cli-sync", "count": 1},
        {"source_type": "http", "count": 1},
    ]


def test_ingestion_sources_degrade_to_empty_for_legacy_schema() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    assert _load_ingestion_sources(conn.cursor(), "default") == []


def test_ingestion_source_lock_failure_is_not_reported_as_empty() -> None:
    class LockedCursor:
        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

    rows, status = _load_ingestion_sources_with_status(LockedCursor(), "default")

    assert rows == []
    assert status == {
        "available": False,
        "state": "temporarily_unavailable",
        "source": "memory.db:provenance",
    }


def test_stats_routes_are_sync_so_sqlite_runs_in_fastapi_threadpool() -> None:
    assert not inspect.iscoroutinefunction(stats.get_stats)
    assert not inspect.iscoroutinefunction(stats.get_timeline)


def test_dashboard_timeline_skips_unneeded_category_scans(
    monkeypatch, tmp_path: Path,
) -> None:
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE atomic_facts ("
        "profile_id TEXT, created_at TEXT, fact_type TEXT)"
    )
    conn.executemany(
        "INSERT INTO atomic_facts VALUES (?, ?, ?)",
        [
            ("default", "2026-07-22T10:00:00", "semantic"),
            ("default", "2026-07-22T11:00:00", "episodic"),
            ("other", "2026-07-22T12:00:00", "semantic"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(stats, "get_db_connection", lambda: sqlite3.connect(db_path))
    monkeypatch.setattr(stats, "get_active_profile", lambda: "default")

    result = stats.get_timeline(
        days=365,
        group_by="day",
        include_categories=False,
    )

    assert result["timeline"] == [{
        "period": "2026-07-22",
        "count": 2,
        "categories": "semantic,episodic",
    }]
    assert result["category_trend"] == []
    assert result["period_stats"] == {
        "total_memories": 2,
        "categories_used": None,
    }


def test_timeline_buckets_by_the_callers_local_day(monkeypatch, tmp_path: Path) -> None:
    """Audit 4.1.20 M5: a fact saved at 20:00 UTC is the NEXT day in India.

    The dashboard keys its chart by local day and sends its offset; the server
    must bucket by that same local day or the two never agree east of UTC.
    """
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE atomic_facts (profile_id TEXT, created_at TEXT, fact_type TEXT)"
    )
    conn.executemany(
        "INSERT INTO atomic_facts VALUES (?, ?, ?)",
        [
            ("default", "2026-07-22T20:00:00.123456+00:00", "semantic"),
            ("default", "2026-07-22T10:00:00+00:00", "semantic"),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(stats, "get_db_connection", lambda: sqlite3.connect(db_path))
    monkeypatch.setattr(stats, "get_active_profile", lambda: "default")

    utc = stats.get_timeline(days=365, group_by="day", include_categories=False)
    india = stats.get_timeline(
        days=365, group_by="day", include_categories=False, tz_offset_minutes=330,
    )
    west = stats.get_timeline(
        days=365, group_by="day", include_categories=False, tz_offset_minutes=-660,
    )

    assert {r["period"]: r["count"] for r in utc["timeline"]} == {"2026-07-22": 2}
    assert {r["period"]: r["count"] for r in india["timeline"]} == {
        "2026-07-23": 1, "2026-07-22": 1,
    }
    assert {r["period"]: r["count"] for r in west["timeline"]} == {
        "2026-07-22": 1, "2026-07-21": 1,
    }


def test_timeline_offset_is_bounded() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(stats.router)
    client = TestClient(app)
    assert client.get("/api/timeline?tz_offset_minutes=99999").status_code == 422
    assert client.get("/api/timeline?tz_offset_minutes=1;DROP").status_code == 422
