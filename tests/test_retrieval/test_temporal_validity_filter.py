# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for the bi-temporal validity filter — Phase 4 (T1).

Covers: superseded-fact removal across channels, valid/no-record passthrough,
fail-open on DB error, empty results, and register gating.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import TemporalValidatorConfig
from superlocalmemory.retrieval.temporal_validity_filter import (
    TemporalValidityFilter,
    register_temporal_validity_filter,
)


@pytest.fixture
def config() -> TemporalValidatorConfig:
    return TemporalValidatorConfig()


@pytest.fixture
def disabled_config() -> TemporalValidatorConfig:
    return TemporalValidatorConfig(enabled=False)


def _make_mock_db(invalid_ids: set[str]) -> MagicMock:
    """Mock DB whose get_invalidated_fact_ids returns the candidates that
    intersect ``invalid_ids`` (mirrors the real bounded query)."""
    db = MagicMock()

    def get_invalidated_fact_ids(fact_ids: list[str], profile_id: str) -> set[str]:
        return {fid for fid in fact_ids if fid in invalid_ids}

    db.get_invalidated_fact_ids = MagicMock(side_effect=get_invalidated_fact_ids)
    return db


# ---- T1-1: superseded facts are removed, valid ones kept ----

def test_filter_removes_superseded() -> None:
    db = _make_mock_db({"fact_superseded"})
    filt = TemporalValidityFilter(db)

    all_results = {
        "semantic": [("fact_valid", 0.9), ("fact_superseded", 0.8)],
    }
    filtered = filt.filter(all_results, "default", None)

    ids = [fid for fid, _ in filtered["semantic"]]
    assert "fact_valid" in ids
    assert "fact_superseded" not in ids
    # Score of the surviving fact is untouched.
    assert filtered["semantic"] == [("fact_valid", 0.9)]


# ---- T1-2: removal applies across every channel ----

def test_filter_removes_across_channels() -> None:
    db = _make_mock_db({"gone"})
    filt = TemporalValidityFilter(db)

    all_results = {
        "semantic": [("keep1", 0.9), ("gone", 0.8)],
        "bm25": [("gone", 0.7), ("keep2", 0.6)],
        "temporal": [("keep3", 0.5)],
    }
    filtered = filt.filter(all_results, "default", None)

    assert filtered["semantic"] == [("keep1", 0.9)]
    assert filtered["bm25"] == [("keep2", 0.6)]
    assert filtered["temporal"] == [("keep3", 0.5)]


# ---- T1-3: nothing invalidated -> unchanged (and same object, cheap path) ----

def test_filter_none_invalid_unchanged() -> None:
    db = _make_mock_db(set())
    filt = TemporalValidityFilter(db)

    all_results = {
        "semantic": [("a", 0.9), ("b", 0.7)],
        "bm25": [("c", 0.6)],
    }
    filtered = filt.filter(all_results, "default", None)
    assert filtered == all_results


# ---- T1-4: facts with no temporal record are kept (mock returns none) ----

def test_filter_no_record_keeps_facts() -> None:
    db = _make_mock_db(set())  # nothing is invalidated
    filt = TemporalValidityFilter(db)

    all_results = {"semantic": [("brand_new_fact", 0.95)]}
    filtered = filt.filter(all_results, "default", None)
    assert filtered["semantic"] == [("brand_new_fact", 0.95)]


# ---- T1-5: empty results returned unchanged, DB not queried ----

def test_filter_empty_results() -> None:
    db = MagicMock()
    filt = TemporalValidityFilter(db)

    filtered = filt.filter({}, "default", None)
    assert filtered == {}
    db.get_invalidated_fact_ids.assert_not_called()


# ---- T1-6: fail-open when the validity lookup raises ----

def test_filter_fail_open_on_db_error() -> None:
    db = MagicMock()
    db.get_invalidated_fact_ids = MagicMock(side_effect=RuntimeError("boom"))
    filt = TemporalValidityFilter(db)

    all_results = {"semantic": [("a", 0.9), ("b", 0.7)]}
    filtered = filt.filter(all_results, "default", None)
    # Retrieval must never break because validity lookup failed.
    assert filtered == all_results


# ---- T1-7: inputs are not mutated (immutability) ----

def test_filter_does_not_mutate_input() -> None:
    db = _make_mock_db({"gone"})
    filt = TemporalValidityFilter(db)

    original = {"semantic": [("keep", 0.9), ("gone", 0.8)]}
    snapshot = {"semantic": [("keep", 0.9), ("gone", 0.8)]}
    filt.filter(original, "default", None)
    assert original == snapshot  # original untouched


# ---- T1-8: register gating ----

def test_register_temporal_validity_filter(config: TemporalValidatorConfig) -> None:
    registry = MagicMock()
    db = MagicMock()
    register_temporal_validity_filter(registry, db, config)
    registry.register_filter.assert_called_once()


def test_register_temporal_validity_filter_disabled(
    disabled_config: TemporalValidatorConfig,
) -> None:
    registry = MagicMock()
    db = MagicMock()
    register_temporal_validity_filter(registry, db, disabled_config)
    registry.register_filter.assert_not_called()
