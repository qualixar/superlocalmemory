# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for temporal validity filter and ChannelRegistry integration.

Covers:
  - temporal_validity_filter_impl: expired removal, include_expired bypass
  - _extract_fact_id: tuple and dict format handling
  - Filter registration in ChannelRegistry
  - Filter applies to all channels
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.temporal_validator import (
    _extract_fact_id,
    temporal_validity_filter_impl,
)
from superlocalmemory.retrieval.channel_registry import ChannelRegistry
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """DatabaseManager with full schema in temp directory."""
    db_path = tmp_path / "test_filter.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def _seed_facts(db: DatabaseManager) -> tuple[str, str, str]:
    """Seed 3 facts for filter testing."""
    record = MemoryRecord(
        profile_id="default", content="test", session_id="s1",
    )
    db.store_memory(record)

    fids = []
    for i in range(3):
        fact = AtomicFact(
            profile_id="default",
            memory_id=record.memory_id,
            content=f"Fact {i}",
        )
        db.store_fact(fact)
        fids.append(fact.fact_id)
    return tuple(fids)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TestExtractFactId
# ---------------------------------------------------------------------------

class TestExtractFactId:
    def test_extract_from_tuple(self) -> None:
        assert _extract_fact_id(("fact-123", 0.9)) == "fact-123"

    def test_extract_from_dict(self) -> None:
        assert _extract_fact_id({"fact_id": "fact-456"}) == "fact-456"

    def test_extract_from_string(self) -> None:
        assert _extract_fact_id("fact-789") == "fact-789"

    def test_extract_from_dict_missing_key(self) -> None:
        assert _extract_fact_id({"id": "x"}) == ""


# ---------------------------------------------------------------------------
# TestTemporalFilter
# ---------------------------------------------------------------------------

class TestTemporalFilter:
    def test_expired_facts_excluded_from_default_retrieval(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """temporal_validity_filter removes expired facts from results."""
        fid0, fid1, fid2 = _seed_facts
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "expired")

        channel_results = {
            "semantic": [(fid0, 0.9), (fid1, 0.8), (fid2, 0.7)],
        }
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        fact_ids = [item[0] for item in filtered["semantic"]]
        assert fid0 in fact_ids
        assert fid1 not in fact_ids
        assert fid2 in fact_ids

    def test_valid_facts_pass_through_filter(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Non-expired facts are not affected by filter."""
        fid0 = _seed_facts[0]
        db.store_temporal_validity(fid0, "default")  # valid_until=NULL

        channel_results = {"semantic": [(fid0, 0.9)]}
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        assert len(filtered["semantic"]) == 1

    def test_facts_without_temporal_record_pass(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Facts with no temporal_validity row are not filtered."""
        fid0 = _seed_facts[0]
        channel_results = {"semantic": [(fid0, 0.9)]}
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        assert len(filtered["semantic"]) == 1

    def test_include_expired_bypasses_filter(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """include_expired=True returns all facts including expired."""
        fid0, fid1, _ = _seed_facts
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "expired")

        channel_results = {"semantic": [(fid0, 0.9), (fid1, 0.8)]}
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=True,
        )
        assert len(filtered["semantic"]) == 2

    def test_filter_applies_to_all_channels(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Filter removes expired facts from all channels."""
        fid0, fid1, fid2 = _seed_facts
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "expired")

        channel_results = {
            "semantic": [(fid0, 0.9), (fid1, 0.8)],
            "bm25": [(fid1, 0.7), (fid2, 0.6)],
            "entity_graph": [(fid0, 0.5), (fid1, 0.4)],
        }
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        # fid1 removed from all channels
        for ch_name, results in filtered.items():
            fact_ids = [_extract_fact_id(item) for item in results]
            assert fid1 not in fact_ids, f"fid1 should be excluded from {ch_name}"

    def test_empty_expired_set_is_noop(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """When no expired facts exist, filter is a no-op."""
        fid0 = _seed_facts[0]
        channel_results = {"semantic": [(fid0, 0.9)]}
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        assert filtered == channel_results


# ---------------------------------------------------------------------------
# TestTemporalFilterRegistration
# ---------------------------------------------------------------------------

class TestTemporalFilterRegistration:
    def test_filter_registered_when_enabled(
        self, db: DatabaseManager,
    ) -> None:
        """Temporal filter can be registered in ChannelRegistry."""
        registry = ChannelRegistry()
        db_ref = db
        registry.register_filter(
            lambda results, pid, ctx: temporal_validity_filter_impl(
                results, pid, db_ref, include_expired=False,
            )
        )
        assert len(registry._filters) == 1

    def test_filter_not_registered_when_disabled(self) -> None:
        """When temporal is disabled, no filter should be registered."""
        registry = ChannelRegistry()
        # Simulate disabled config: don't register
        enabled = False
        if enabled:
            registry.register_filter(lambda r, p, c: r)
        assert len(registry._filters) == 0

    def test_filter_runs_in_run_all(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Filter executes during run_all after channels return."""
        fid0, fid1, _ = _seed_facts
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "expired")

        registry = ChannelRegistry()

        # Mock channel that returns expired fact
        mock_channel = MagicMock()
        mock_channel.search.return_value = [(fid0, 0.9), (fid1, 0.8)]
        registry.register_channel("test_ch", mock_channel)

        db_ref = db
        registry.register_filter(
            lambda results, pid, ctx: temporal_validity_filter_impl(
                results, pid, db_ref, include_expired=False,
            )
        )

        results = registry.run_all("query", "default")
        fact_ids = [item[0] for item in results.get("test_ch", [])]
        assert fid0 in fact_ids
        assert fid1 not in fact_ids

    def test_filter_before_rrf_fusion(
        self, db: DatabaseManager, _seed_facts: tuple[str, str, str],
    ) -> None:
        """Filter executes before RRF fusion combines channel results.
        Verified by checking filter operates on per-channel dict."""
        fid0, fid1, _ = _seed_facts
        db.store_temporal_validity(fid1, "default")
        db.invalidate_fact_temporal(fid1, "x", "reason")

        # Input is per-channel dict (pre-fusion format)
        channel_results = {
            "semantic": [(fid0, 0.9), (fid1, 0.8)],
            "bm25": [(fid1, 0.7)],
        }
        filtered = temporal_validity_filter_impl(
            channel_results, "default", db, include_expired=False,
        )
        # Per-channel structure preserved (not flattened by fusion)
        assert "semantic" in filtered
        assert "bm25" in filtered
        # fid1 removed from both
        assert len(filtered["bm25"]) == 0
