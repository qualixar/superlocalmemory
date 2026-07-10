# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Langevin position initialization in store pipeline and maintenance.

Covers:
  - Strategy A: _init_langevin_position() near-origin seeding for new facts
  - Strategy B: _compute_equilibrium_radius() metadata-aware radius
  - Strategy B: _seed_langevin_position() metadata-aware position seeding
  - Strategy C: burn-in integration in maintenance backfill
  - enrich_fact() sets langevin_position on new facts
  - run_maintenance() backfills facts with None positions
  - run_maintenance() does NOT re-backfill already-positioned facts
  - Backward compatibility: existing maintenance batch_step still works
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from superlocalmemory.core.store_pipeline import _init_langevin_position
from superlocalmemory.core.maintenance import (
    _compute_equilibrium_radius,
    _seed_langevin_position,
    run_maintenance,
)


# ---------------------------------------------------------------------------
# Strategy A: _init_langevin_position
# ---------------------------------------------------------------------------

class TestInitLangevinPosition:
    def test_returns_list_of_correct_dim(self) -> None:
        pos = _init_langevin_position(dim=8)
        assert isinstance(pos, list)
        assert len(pos) == 8

    def test_custom_dim(self) -> None:
        pos = _init_langevin_position(dim=4)
        assert len(pos) == 4

    def test_near_origin(self) -> None:
        """Position should be in the ACTIVE zone (radius < 0.3)."""
        for _ in range(50):
            pos = _init_langevin_position(dim=8)
            radius = float(np.linalg.norm(pos))
            assert radius < 0.3, f"Initial position radius {radius} >= 0.3"

    def test_consistent_radius(self) -> None:
        """All positions should have radius ~0.05 (the init radius)."""
        radii = []
        for _ in range(100):
            pos = _init_langevin_position(dim=8)
            radii.append(float(np.linalg.norm(pos)))
        np.testing.assert_allclose(radii, 0.05, atol=0.001)

    def test_positions_are_unique(self) -> None:
        """Each call should produce a different position (random)."""
        p1 = _init_langevin_position(dim=8)
        p2 = _init_langevin_position(dim=8)
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Strategy B: _compute_equilibrium_radius
# ---------------------------------------------------------------------------

class TestComputeEquilibriumRadius:
    def test_high_access_smaller_radius(self) -> None:
        """Frequently accessed facts should have smaller equilibrium radius."""
        r_high = _compute_equilibrium_radius(access_count=100, age_days=1.0, importance=0.8)
        r_low = _compute_equilibrium_radius(access_count=0, age_days=1.0, importance=0.8)
        assert r_high < r_low

    def test_high_importance_smaller_radius(self) -> None:
        """Important facts should have smaller equilibrium radius."""
        r_high = _compute_equilibrium_radius(access_count=5, age_days=10.0, importance=1.0)
        r_low = _compute_equilibrium_radius(access_count=5, age_days=10.0, importance=0.0)
        assert r_high < r_low

    def test_old_age_larger_radius(self) -> None:
        """Old facts should have slightly larger equilibrium radius."""
        r_old = _compute_equilibrium_radius(access_count=0, age_days=365.0, importance=0.5)
        r_new = _compute_equilibrium_radius(access_count=0, age_days=0.0, importance=0.5)
        assert r_old > r_new

    def test_radius_always_positive(self) -> None:
        r = _compute_equilibrium_radius(access_count=0, age_days=0.0, importance=0.0)
        assert r > 0.0

    def test_radius_below_max_norm(self) -> None:
        """Radius should never exceed MAX_NORM * 0.95."""
        r = _compute_equilibrium_radius(
            access_count=0, age_days=365.0, importance=0.0, temperature=100.0,
        )
        assert r <= 0.99 * 0.95


# ---------------------------------------------------------------------------
# Strategy B: _seed_langevin_position
# ---------------------------------------------------------------------------

class TestSeedLangevinPosition:
    def test_returns_list_of_correct_dim(self) -> None:
        pos = _seed_langevin_position(
            access_count=5, age_days=10.0, importance=0.5, dim=8,
        )
        assert isinstance(pos, list)
        assert len(pos) == 8

    def test_radius_matches_equilibrium(self) -> None:
        """Seeded position radius should equal the computed equilibrium radius."""
        for _ in range(20):
            pos = _seed_langevin_position(
                access_count=10, age_days=30.0, importance=0.7, dim=8,
            )
            expected_r = _compute_equilibrium_radius(
                access_count=10, age_days=30.0, importance=0.7,
            )
            actual_r = float(np.linalg.norm(pos))
            np.testing.assert_allclose(actual_r, expected_r, atol=1e-6)

    def test_inside_unit_ball(self) -> None:
        pos = _seed_langevin_position(
            access_count=0, age_days=365.0, importance=0.0, dim=8,
        )
        assert float(np.linalg.norm(pos)) < 1.0


# ---------------------------------------------------------------------------
# Strategy A in enrich_fact: langevin_position is set on new facts
# ---------------------------------------------------------------------------

class TestEnrichFactSetsLangevinPosition:
    def test_enriched_fact_has_langevin_position(self) -> None:
        """enrich_fact should set langevin_position on every new fact."""
        from superlocalmemory.storage.models import (
            AtomicFact, FactType, MemoryRecord,
        )
        from superlocalmemory.core.store_pipeline import enrich_fact

        fact = AtomicFact(
            fact_id="test-f1", memory_id="m0", content="Test fact",
            entities=[], fact_type=FactType.SEMANTIC, confidence=0.9,
        )
        record = MemoryRecord(
            profile_id="default", content="Test fact",
            session_id="s1",
        )
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 64
        mock_embedder.compute_fisher_params.return_value = ([0.1] * 64, [0.01] * 64)

        enriched = enrich_fact(
            fact, record, "default",
            embedder=mock_embedder,
            entity_resolver=None,
            temporal_parser=None,
        )
        assert enriched.langevin_position is not None
        assert len(enriched.langevin_position) == 8
        radius = float(np.linalg.norm(enriched.langevin_position))
        assert radius < 0.3  # In ACTIVE zone


# ---------------------------------------------------------------------------
# Strategy B+C in maintenance: backfill None positions
# ---------------------------------------------------------------------------

class TestMaintenanceBackfill:
    def _make_config(self) -> MagicMock:
        config = MagicMock()
        config.math.langevin_persist_positions = True
        config.math.langevin_dt = 0.005
        config.math.langevin_temperature = 0.3
        config.math.sheaf_at_encoding = False
        return config

    def _make_fact(
        self,
        fact_id: str = "f1",
        langevin_position: list[float] | None = None,
        access_count: int = 5,
        importance: float = 0.5,
        age_days: float = 10.0,
    ) -> MagicMock:
        fact = MagicMock()
        fact.fact_id = fact_id
        fact.langevin_position = langevin_position
        fact.access_count = access_count
        fact.importance = importance
        fact.fisher_variance = None
        created = datetime.now(UTC)
        if age_days > 0:
            from datetime import timedelta
            created = created - timedelta(days=age_days)
        fact.created_at = created.isoformat()
        return fact

    def test_backfills_none_positions(self) -> None:
        """Facts with None langevin_position should get initialized."""
        config = self._make_config()
        db = MagicMock()
        f1 = self._make_fact("f1", langevin_position=None, access_count=10, age_days=30.0)
        f2 = self._make_fact("f2", langevin_position=None, access_count=0, age_days=90.0)
        db.get_all_facts.return_value = [f1, f2]

        counts = run_maintenance(db, config, "default")

        assert counts["langevin_backfilled"] == 2
        assert db.update_fact.call_count >= 2  # backfill + batch step

    def test_skips_already_positioned_facts_in_backfill(self) -> None:
        """Facts with existing langevin_position should NOT be re-backfilled."""
        config = self._make_config()
        db = MagicMock()
        existing_pos = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        f1 = self._make_fact("f1", langevin_position=existing_pos)
        db.get_all_facts.return_value = [f1]

        counts = run_maintenance(db, config, "default")

        assert counts["langevin_backfilled"] == 0
        assert counts["langevin_updated"] == 1  # normal batch step

    def test_backfilled_positions_inside_unit_ball(self) -> None:
        """All backfilled positions must be inside the unit ball."""
        config = self._make_config()
        db = MagicMock()
        facts = [
            self._make_fact(f"f{i}", langevin_position=None, access_count=i * 5, age_days=float(i * 30))
            for i in range(10)
        ]
        db.get_all_facts.return_value = facts

        run_maintenance(db, config, "default")

        for c in db.update_fact.call_args_list:
            updates = c[0][1]
            if "langevin_position" in updates:
                pos = updates["langevin_position"]
                radius = float(np.linalg.norm(pos))
                assert radius < 1.0, f"Backfilled position outside unit ball: radius={radius}"

    def test_mixed_facts_only_backfills_none(self) -> None:
        """Only facts with None position should be backfilled."""
        config = self._make_config()
        db = MagicMock()
        positioned = self._make_fact("f1", langevin_position=[0.1] * 8)
        unpositioned = self._make_fact("f2", langevin_position=None, age_days=60.0)
        db.get_all_facts.return_value = [positioned, unpositioned]

        counts = run_maintenance(db, config, "default")

        assert counts["langevin_backfilled"] == 1

    def test_backfill_sets_lifecycle(self) -> None:
        """Backfilled facts should have a valid lifecycle value."""
        config = self._make_config()
        db = MagicMock()
        f1 = self._make_fact("f1", langevin_position=None, age_days=5.0)
        db.get_all_facts.return_value = [f1]

        run_maintenance(db, config, "default")

        # First update call is from backfill
        first_call = db.update_fact.call_args_list[0]
        updates = first_call[0][1]
        assert "lifecycle" in updates
        assert updates["lifecycle"] in {"active", "warm", "cold", "archived"}

    def test_returns_backfill_count_key(self) -> None:
        """Return dict should include langevin_backfilled key."""
        config = self._make_config()
        db = MagicMock()
        db.get_all_facts.return_value = []

        counts = run_maintenance(db, config, "default")

        assert "langevin_backfilled" in counts
        assert counts["langevin_backfilled"] == 0

    def test_backfills_facts_with_naive_created_at(self) -> None:
        """Naive (tz-less) created_at strings must not abort the backfill.

        Regression: some store paths persist created_at without timezone
        info; subtracting the parsed naive datetime from datetime.now(UTC)
        raised TypeError, which aborted the whole backfill loop and left
        every remaining fact unpositioned.
        """
        config = self._make_config()
        db = MagicMock()
        naive = self._make_fact("f1", langevin_position=None)
        naive.created_at = datetime.now().isoformat()  # no tzinfo, no Z
        aware = self._make_fact("f2", langevin_position=None, age_days=30.0)
        db.get_all_facts.return_value = [naive, aware]

        counts = run_maintenance(db, config, "default")

        assert counts["langevin_backfilled"] == 2

    def test_disabled_langevin_skips_backfill(self) -> None:
        """When langevin_persist_positions is False, no backfill occurs."""
        config = self._make_config()
        config.math.langevin_persist_positions = False
        db = MagicMock()
        f1 = self._make_fact("f1", langevin_position=None)
        db.get_all_facts.return_value = [f1]

        counts = run_maintenance(db, config, "default")

        assert counts["langevin_backfilled"] == 0
        assert db.update_fact.call_count == 0
