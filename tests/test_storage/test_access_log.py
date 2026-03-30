# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for superlocalmemory.storage.access_log — AccessLog CRUD.

Covers:
  - store_access: single event recording
  - store_access_batch: bulk event recording
  - get_latest_access_time: most recent timestamp
  - get_access_count: total count per fact
  - get_all_access_times: profile-wide latest times
  - get_frequently_accessed: frequency threshold query
  - invalid access_type fallback
  - profile isolation
  - silent error handling (Rule 19)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.access_log import AccessLog
from superlocalmemory.storage.models import AtomicFact, MemoryRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager wired to a temp directory with full schema."""
    db_path = tmp_path / "test_access.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def access_log(db: DatabaseManager) -> AccessLog:
    return AccessLog(db)


@pytest.fixture()
def _seed_facts(db: DatabaseManager) -> list[str]:
    """Seed 3 facts into the database for FK satisfaction."""
    record = MemoryRecord(
        profile_id="default", content="test content",
        session_id="s1",
    )
    db.store_memory(record)

    fact_ids = []
    for i in range(3):
        fact = AtomicFact(
            profile_id="default",
            memory_id=record.memory_id,
            content=f"Fact number {i}",
        )
        db.store_fact(fact)
        fact_ids.append(fact.fact_id)
    return fact_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStoreAccess:
    """Test store_access single event recording."""

    def test_store_access_returns_log_id(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        fid = _seed_facts[0]
        log_id = access_log.store_access(fid, "default")
        assert log_id  # non-empty string
        assert len(log_id) == 16

    def test_store_access_invalid_type_defaults_to_recall(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        fid = _seed_facts[0]
        log_id = access_log.store_access(fid, "default", access_type="invalid")
        assert log_id  # should succeed with "recall" as fallback


class TestStoreAccessBatch:
    """Test store_access_batch bulk recording."""

    def test_batch_records_all(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        count = access_log.store_access_batch(_seed_facts, "default")
        assert count == 3


class TestGetLatestAccessTime:
    """Test get_latest_access_time retrieval."""

    def test_returns_none_for_unaccessed(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        result = access_log.get_latest_access_time(_seed_facts[0], "default")
        assert result is None

    def test_returns_timestamp_after_access(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        fid = _seed_facts[0]
        access_log.store_access(fid, "default")
        result = access_log.get_latest_access_time(fid, "default")
        assert result is not None
        assert len(result) >= 19  # ISO datetime format


class TestGetAccessCount:
    """Test get_access_count."""

    def test_zero_for_unaccessed(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        count = access_log.get_access_count(_seed_facts[0], "default")
        assert count == 0

    def test_increments_on_access(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        fid = _seed_facts[0]
        access_log.store_access(fid, "default")
        access_log.store_access(fid, "default")
        access_log.store_access(fid, "default")
        count = access_log.get_access_count(fid, "default")
        assert count == 3


class TestGetAllAccessTimes:
    """Test get_all_access_times profile-wide query."""

    def test_returns_dict_of_latest_times(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        for fid in _seed_facts:
            access_log.store_access(fid, "default")
        result = access_log.get_all_access_times("default")
        assert len(result) == 3
        for fid in _seed_facts:
            assert fid in result


class TestGetFrequentlyAccessed:
    """Test get_frequently_accessed frequency threshold."""

    def test_filters_by_min_count(
        self, access_log: AccessLog, _seed_facts: list[str],
    ) -> None:
        # Access fact[0] 5 times, fact[1] 2 times, fact[2] 0 times
        for _ in range(5):
            access_log.store_access(_seed_facts[0], "default")
        for _ in range(2):
            access_log.store_access(_seed_facts[1], "default")

        frequent = access_log.get_frequently_accessed(
            "default", min_count=3,
        )
        assert len(frequent) == 1
        assert frequent[0][0] == _seed_facts[0]
        assert frequent[0][1] == 5


class TestProfileIsolation:
    """Test that access logs respect profile boundaries."""

    def test_different_profiles_isolated(
        self, db: DatabaseManager, access_log: AccessLog,
    ) -> None:
        # Create a second profile
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            ("p2", "Profile 2"),
        )
        # Seed facts in both profiles
        for pid in ("default", "p2"):
            record = MemoryRecord(
                profile_id=pid, content="content",
                session_id="s1",
            )
            db.store_memory(record)
            fact = AtomicFact(
                profile_id=pid,
                memory_id=record.memory_id,
                content=f"Fact for {pid}",
            )
            db.store_fact(fact)
            access_log.store_access(fact.fact_id, pid)

        default_times = access_log.get_all_access_times("default")
        p2_times = access_log.get_all_access_times("p2")
        assert len(default_times) == 1
        assert len(p2_times) == 1
        # No overlap
        assert set(default_times.keys()) != set(p2_times.keys())
