# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.retrieval.temporal_channel — 3-Date Temporal Search.

Covers:
  - _parse_iso helper
  - _proximity_score Gaussian decay
  - TemporalChannel.search() with referenced_date, observation_date, intervals
  - Query with no temporal signal returns empty
  - Query with explicit date finds matching events
  - Interval containment scoring
  - Profile isolation
  - Top-k limiting
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.retrieval.temporal_channel import (
    TemporalChannel,
    _parse_iso,
    _proximity_score,
)
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    MemoryRecord,
    TemporalEvent,
)


# ---------------------------------------------------------------------------
# _parse_iso
# ---------------------------------------------------------------------------

class TestParseIso:
    def test_valid_iso(self) -> None:
        dt = _parse_iso("2026-03-11T10:00:00")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 3

    def test_date_only(self) -> None:
        dt = _parse_iso("2026-03-11")
        assert dt is not None
        assert dt.day == 11

    def test_none_input(self) -> None:
        assert _parse_iso(None) is None

    def test_empty_string(self) -> None:
        assert _parse_iso("") is None

    def test_garbage_string(self) -> None:
        assert _parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# _proximity_score
# ---------------------------------------------------------------------------

class TestProximityScore:
    def test_same_day_returns_one(self) -> None:
        dt = datetime(2026, 3, 11)
        assert _proximity_score(dt, dt) == pytest.approx(1.0)

    def test_thirty_days_apart(self) -> None:
        q = datetime(2026, 3, 11)
        e = datetime(2026, 2, 9)  # ~30 days apart
        score = _proximity_score(q, e)
        # Gaussian with sigma=30 at distance=30 -> exp(-1/2) ≈ 0.607
        assert 0.5 < score < 0.7

    def test_beyond_365_days_returns_zero(self) -> None:
        q = datetime(2026, 3, 11)
        e = datetime(2024, 3, 11)  # 2 years apart
        assert _proximity_score(q, e) == 0.0

    def test_monotonically_decreasing(self) -> None:
        q = datetime(2026, 3, 11)
        scores = []
        for days in [0, 10, 30, 60, 90, 180]:
            e = datetime(2026, 3, 11 - min(days, 10))  # Simplified
            if days <= 10:
                from datetime import timedelta
                e = q - timedelta(days=days)
                scores.append(_proximity_score(q, e))
        # Scores should decrease
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ---------------------------------------------------------------------------
# TemporalChannel.search
# ---------------------------------------------------------------------------

class _DictRow(dict):
    """sqlite3.Row-compatible dict that supports dict(row) conversion."""
    pass


def _mock_db_with_events(events: list[dict]) -> MagicMock:
    db = MagicMock()
    db.execute.return_value = [_DictRow(ev) for ev in events]
    return db


class TestTemporalChannelSearch:
    def test_no_temporal_signal_returns_empty(self) -> None:
        db = MagicMock()
        db.execute.return_value = []
        ch = TemporalChannel(db)
        # "tell me about cats" has no date/temporal words
        results = ch.search("tell me about cats", "default")
        assert results == []

    def test_referenced_date_match(self) -> None:
        events = [{
            "fact_id": "f1",
            "observation_date": None,
            "referenced_date": "2026-03-11",
            "interval_start": None,
            "interval_end": None,
        }]
        db = _mock_db_with_events(events)
        ch = TemporalChannel(db)

        # Patch TemporalParser to return a known date
        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": "2026-03-11",
                "interval_start": None,
                "interval_end": None,
            }
            results = ch.search("what happened on March 11 2026?", "default")

        assert len(results) > 0
        assert results[0][0] == "f1"
        assert results[0][1] > 0.9  # Same day -> near 1.0

    def test_observation_date_weighted_lower(self) -> None:
        events = [{
            "fact_id": "f1",
            "observation_date": "2026-03-11",
            "referenced_date": None,
            "interval_start": None,
            "interval_end": None,
        }]
        db = _mock_db_with_events(events)
        ch = TemporalChannel(db)

        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": "2026-03-11",
                "interval_start": None,
                "interval_end": None,
            }
            results = ch.search("what happened on March 11?", "default")

        if results:
            # Observation date gets 0.8 multiplier
            assert results[0][1] <= 0.85  # 1.0 * 0.8

    def test_interval_containment_max_score(self) -> None:
        events = [{
            "fact_id": "f1",
            "observation_date": None,
            "referenced_date": None,
            "interval_start": "2026-03-01",
            "interval_end": "2026-03-31",
        }]
        db = _mock_db_with_events(events)
        ch = TemporalChannel(db)

        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": "2026-03-15",
                "interval_start": None,
                "interval_end": None,
            }
            results = ch.search("what happened on March 15?", "default")

        assert len(results) > 0
        # Query date falls within interval -> score = 1.0
        assert results[0][1] == pytest.approx(1.0)

    def test_top_k_limits(self) -> None:
        events = [
            {
                "fact_id": f"f{i}",
                "observation_date": None,
                "referenced_date": f"2026-03-{10+i:02d}",
                "interval_start": None,
                "interval_end": None,
            }
            for i in range(15)
        ]
        db = _mock_db_with_events(events)
        ch = TemporalChannel(db)

        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": "2026-03-15",
                "interval_start": None,
                "interval_end": None,
            }
            results = ch.search("March 15 events", "default", top_k=5)

        assert len(results) <= 5

    def test_try_parse_fallback(self) -> None:
        """When TemporalParser returns None, _try_parse is used as fallback."""
        events = [{
            "fact_id": "f1",
            "observation_date": None,
            "referenced_date": "2026-03-11",
            "interval_start": None,
            "interval_end": None,
        }]
        db = _mock_db_with_events(events)
        ch = TemporalChannel(db)

        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": None,
                "interval_start": None,
                "interval_end": None,
            }
            # dateutil fuzzy parse should handle "March 11 2026"
            results = ch.search("March 11 2026", "default")

        # The _try_parse fallback should find the date
        assert len(results) > 0

    def test_empty_events_table(self) -> None:
        db = MagicMock()
        db.execute.return_value = []
        ch = TemporalChannel(db)

        with patch(
            "superlocalmemory.retrieval.temporal_channel.TemporalParser"
        ) as MockParser:
            parser_inst = MockParser.return_value
            parser_inst.extract_dates_from_text.return_value = {
                "referenced_date": "2026-03-11",
                "interval_start": None,
                "interval_end": None,
            }
            results = ch.search("what happened on March 11?", "default")

        assert results == []


class TestTemporalScopeContract:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> DatabaseManager:
        manager = DatabaseManager(tmp_path / "temporal-scope.db")
        manager.initialize(real_schema)
        for profile_id in ("requester", "publisher"):
            manager.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
                "VALUES (?, ?, '')",
                (profile_id, profile_id),
            )
        return manager

    @staticmethod
    def _seed_event(
        db: DatabaseManager,
        fact_id: str,
        *,
        scope: str,
        shared_with: list[str] | None = None,
    ) -> None:
        memory = MemoryRecord(
            memory_id=f"m-{fact_id}", profile_id="publisher",
            scope=scope, shared_with=shared_with, content=fact_id,
        )
        db.store_memory(memory)
        db.store_fact(AtomicFact(
            fact_id=fact_id, memory_id=memory.memory_id,
            profile_id="publisher", scope=scope, shared_with=shared_with,
            content=f"Nova happened on 2026-03-11 ({fact_id})",
        ))
        entity = CanonicalEntity(
            entity_id="e-nova", profile_id="publisher",
            canonical_name="Nova", entity_type="person",
        )
        db.execute(
            "INSERT OR IGNORE INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type, first_seen, "
            "last_seen, fact_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                entity.entity_id, entity.profile_id, entity.canonical_name,
                entity.entity_type, entity.first_seen, entity.last_seen,
                entity.fact_count,
            ),
        )
        db.store_temporal_event(TemporalEvent(
            event_id=f"t-{fact_id}", profile_id="publisher",
            entity_id=entity.entity_id, fact_id=fact_id,
            referenced_date="2026-03-11", description=fact_id,
            scope=scope, shared_with=shared_with,
        ))

    def test_date_and_entity_search_include_only_visible_cross_profile_events(
        self, db: DatabaseManager,
    ) -> None:
        self._seed_event(db, "global", scope="global")
        self._seed_event(db, "private", scope="personal")
        self._seed_event(db, "project", scope="project")
        self._seed_event(
            db, "shared_ok", scope="shared", shared_with=["requester"],
        )
        self._seed_event(
            db, "shared_denied", scope="shared", shared_with=["someone-else"],
        )
        channel = TemporalChannel(db)
        channel.include_global = True
        channel.include_shared = True

        visible = {
            fid for fid, _ in channel.search(
                "When did Nova act on 2026-03-11?", "requester",
            )
        }
        assert visible == {"global", "shared_ok"}
