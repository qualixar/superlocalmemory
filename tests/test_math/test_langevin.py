# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.math.langevin — Riemannian Langevin dynamics.

Covers:
  - LangevinDynamics construction and parameter validation
  - step(): position update, ball containment, weight range
  - compute_lifecycle_weight: origin->max, boundary->min
  - get_lifecycle_state: zone classification from weight
  - batch_step: multiple facts processed correctly
  - Edge cases: zero position, wrong dimensionality, seed reproducibility
  - Mathematical invariants: positions inside unit ball, weights in range
"""

from __future__ import annotations

import numpy as np
import pytest

from superlocalmemory.math.langevin import (
    _MAX_NORM,
    LangevinDynamics,
    _project_to_ball,
    _resize_position,
)
from superlocalmemory.storage.models import MemoryLifecycle

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self) -> None:
        ld = LangevinDynamics()
        assert ld.dt == 0.01
        assert ld.temperature == 1.0
        assert ld.weight_range == (0.0, 1.0)
        assert ld.dim == 8

    def test_custom_params(self) -> None:
        ld = LangevinDynamics(dt=0.05, temperature=2.0, dim=16)
        assert ld.dt == 0.05
        assert ld.temperature == 2.0
        assert ld.dim == 16

    def test_zero_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            LangevinDynamics(dt=0.0)

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            LangevinDynamics(dt=-0.01)

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be positive"):
            LangevinDynamics(temperature=0.0)

    def test_inverted_weight_range_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_range min > max"):
            LangevinDynamics(weight_range=(1.0, 0.0))


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_position_and_weight(self) -> None:
        ld = LangevinDynamics(dim=4)
        pos = [0.0, 0.0, 0.0, 0.0]
        new_pos, weight = ld.step(pos, access_count=5, age_days=1.0, importance=0.5, seed=42)
        assert isinstance(new_pos, list)
        assert isinstance(weight, float)
        assert len(new_pos) == 4

    def test_position_inside_unit_ball(self) -> None:
        """All returned positions must have norm < 1."""
        ld = LangevinDynamics(dim=8, temperature=5.0)
        rng = np.random.default_rng(99)
        for i in range(50):
            pos = (rng.standard_normal(8) * 0.5).tolist()
            new_pos, _ = ld.step(pos, access_count=i, age_days=float(i), importance=0.5, seed=i)
            norm = np.linalg.norm(new_pos)
            assert norm < 1.0, f"Position norm {norm} >= 1.0 at iteration {i}"

    def test_weight_in_range(self) -> None:
        ld = LangevinDynamics(weight_range=(0.2, 0.9))
        pos = [0.3] * 8
        _, weight = ld.step(pos, access_count=10, age_days=5.0, importance=0.7, seed=1)
        assert 0.2 <= weight <= 0.9

    def test_seed_reproducibility(self) -> None:
        ld = LangevinDynamics()
        pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        p1, w1 = ld.step(pos, 5, 2.0, 0.5, seed=123)
        p2, w2 = ld.step(pos, 5, 2.0, 0.5, seed=123)
        np.testing.assert_allclose(p1, p2)
        np.testing.assert_allclose(w1, w2)

    def test_different_seeds_give_different_results(self) -> None:
        ld = LangevinDynamics()
        pos = [0.1] * 8
        p1, _ = ld.step(pos, 5, 2.0, 0.5, seed=1)
        p2, _ = ld.step(pos, 5, 2.0, 0.5, seed=2)
        assert not np.allclose(p1, p2)

    def test_zero_position_evolves(self) -> None:
        """From origin, diffusion should push the position outward."""
        ld = LangevinDynamics(dim=4, temperature=1.0)
        pos = [0.0, 0.0, 0.0, 0.0]
        new_pos, _ = ld.step(pos, 0, 0.0, 0.0, seed=42)
        # Should not stay exactly at origin due to stochastic noise
        assert np.linalg.norm(new_pos) > 0.0

    def test_wrong_dim_resized(self) -> None:
        """Position with wrong dimensionality gets resized gracefully."""
        ld = LangevinDynamics(dim=4)
        # Provide 6-dim position for a 4-dim system
        pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        new_pos, weight = ld.step(pos, 1, 1.0, 0.5, seed=10)
        assert len(new_pos) == 4
        assert 0.0 <= weight <= 1.0

    def test_short_dim_padded(self) -> None:
        """Position shorter than dim gets zero-padded."""
        ld = LangevinDynamics(dim=8)
        pos = [0.1, 0.2]
        new_pos, _ = ld.step(pos, 1, 1.0, 0.5, seed=10)
        assert len(new_pos) == 8


# ---------------------------------------------------------------------------
# compute_lifecycle_weight
# ---------------------------------------------------------------------------

class TestLifecycleWeight:
    def test_origin_gives_max_weight(self) -> None:
        ld = LangevinDynamics(weight_range=(0.0, 1.0))
        w = ld.compute_lifecycle_weight([0.0] * 8)
        np.testing.assert_allclose(w, 1.0, atol=1e-10)

    def test_boundary_gives_min_weight(self) -> None:
        ld = LangevinDynamics(weight_range=(0.0, 1.0))
        # Position very close to boundary
        pos = [_MAX_NORM / np.sqrt(8)] * 8
        w = ld.compute_lifecycle_weight(pos)
        assert w <= 0.05  # Very close to 0

    def test_intermediate_radius(self) -> None:
        ld = LangevinDynamics(weight_range=(0.0, 1.0))
        # Radius ~0.5
        pos = [0.5 / np.sqrt(8)] * 8
        w = ld.compute_lifecycle_weight(pos)
        # Linear: weight = 1 - (radius / MAX_NORM)
        expected_radius = np.linalg.norm(pos)
        expected_weight = 1.0 - expected_radius / _MAX_NORM
        np.testing.assert_allclose(w, expected_weight, atol=1e-6)

    def test_custom_weight_range(self) -> None:
        ld = LangevinDynamics(weight_range=(0.3, 0.8))
        w = ld.compute_lifecycle_weight([0.0] * 8)
        np.testing.assert_allclose(w, 0.8, atol=1e-10)


# ---------------------------------------------------------------------------
# get_lifecycle_state
# ---------------------------------------------------------------------------

class TestLifecycleState:
    def test_high_weight_is_active(self) -> None:
        ld = LangevinDynamics()
        assert ld.get_lifecycle_state(0.95) == MemoryLifecycle.ACTIVE

    def test_medium_weight_is_warm(self) -> None:
        ld = LangevinDynamics()
        state = ld.get_lifecycle_state(0.6)
        assert state in (MemoryLifecycle.ACTIVE, MemoryLifecycle.WARM)

    def test_low_weight_is_cold_or_archived(self) -> None:
        ld = LangevinDynamics()
        state = ld.get_lifecycle_state(0.1)
        assert state in (MemoryLifecycle.COLD, MemoryLifecycle.ARCHIVED)

    def test_zero_weight_is_archived(self) -> None:
        ld = LangevinDynamics()
        assert ld.get_lifecycle_state(0.0) == MemoryLifecycle.ARCHIVED

    def test_max_weight_is_active(self) -> None:
        ld = LangevinDynamics()
        assert ld.get_lifecycle_state(1.0) == MemoryLifecycle.ACTIVE

    def test_collapsed_weight_range(self) -> None:
        """When weight_range has zero span, should default to ACTIVE."""
        ld = LangevinDynamics(weight_range=(0.5, 0.5))
        assert ld.get_lifecycle_state(0.5) == MemoryLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# batch_step
# ---------------------------------------------------------------------------

class TestBatchStep:
    def test_processes_all_facts(self) -> None:
        ld = LangevinDynamics(dim=4)
        facts = [
            {"fact_id": "f1", "position": [0.0] * 4, "access_count": 10, "age_days": 1.0, "importance": 0.8},
            {"fact_id": "f2", "position": [0.2] * 4, "access_count": 0, "age_days": 30.0, "importance": 0.2},
            {
                "fact_id": "f3",
                "position": [0.1, -0.1, 0.05, 0.0],
                "access_count": 5,
                "age_days": 7.0,
                "importance": 0.5,
            },
        ]
        results = ld.batch_step(facts, seed=42)
        assert len(results) == 3
        for r in results:
            assert "fact_id" in r
            assert "position" in r
            assert "weight" in r
            assert "lifecycle" in r
            assert len(r["position"]) == 4

    def test_fact_ids_preserved(self) -> None:
        ld = LangevinDynamics(dim=4)
        facts = [
            {"fact_id": "abc", "position": [0.0] * 4, "access_count": 1, "age_days": 1.0, "importance": 0.5},
        ]
        results = ld.batch_step(facts, seed=1)
        assert results[0]["fact_id"] == "abc"

    def test_lifecycle_values_are_valid_strings(self) -> None:
        ld = LangevinDynamics(dim=4)
        facts = [
            {"fact_id": "f1", "position": [0.0] * 4, "access_count": 0, "age_days": 0.0, "importance": 0.0},
        ]
        results = ld.batch_step(facts, seed=1)
        assert results[0]["lifecycle"] in {"active", "warm", "cold", "archived"}

    def test_empty_input(self) -> None:
        ld = LangevinDynamics()
        results = ld.batch_step([], seed=1)
        assert results == []

    def test_seed_consistency_in_batch(self) -> None:
        ld = LangevinDynamics(dim=4)
        facts = [
            {"fact_id": f"f{i}", "position": [0.1 * i] * 4, "access_count": i, "age_days": float(i), "importance": 0.5}
            for i in range(5)
        ]
        r1 = ld.batch_step(facts, seed=100)
        r2 = ld.batch_step(facts, seed=100)
        for a, b in zip(r1, r2):
            np.testing.assert_allclose(a["position"], b["position"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestProjectToBall:
    def test_inside_ball_unchanged(self) -> None:
        pos = np.array([0.1, 0.2, 0.3])
        result = _project_to_ball(pos)
        np.testing.assert_allclose(result, pos)

    def test_outside_ball_projected(self) -> None:
        pos = np.array([5.0, 5.0, 5.0])
        result = _project_to_ball(pos)
        assert np.linalg.norm(result) < 1.0

    def test_at_boundary(self) -> None:
        pos = np.array([_MAX_NORM, 0.0])
        result = _project_to_ball(pos)
        assert np.linalg.norm(result) <= _MAX_NORM + 1e-12


class TestResizePosition:
    def test_same_dim_unchanged(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        result = _resize_position(pos, 3)
        np.testing.assert_allclose(result, pos)

    def test_pad_smaller(self) -> None:
        pos = np.array([1.0, 2.0])
        result = _resize_position(pos, 5)
        assert result.shape[0] == 5
        np.testing.assert_allclose(result[:2], pos)
        np.testing.assert_allclose(result[2:], [0.0, 0.0, 0.0])

    def test_truncate_larger(self) -> None:
        pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _resize_position(pos, 3)
        assert result.shape[0] == 3
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Integration: step evolves over time
# ---------------------------------------------------------------------------

class TestEvolution:
    def test_frequently_accessed_stays_near_origin(self) -> None:
        """High access count + high importance should keep memory near origin."""
        ld = LangevinDynamics(dim=4, dt=0.01, temperature=0.5)
        pos = [0.0, 0.0, 0.0, 0.0]
        for i in range(20):
            pos, w = ld.step(pos, access_count=100, age_days=1.0, importance=1.0, seed=i)
        norm = np.linalg.norm(pos)
        # Should stay relatively close to origin
        assert norm < 0.8, f"Heavily accessed memory drifted to norm={norm}"

    def test_positions_always_valid_over_many_steps(self) -> None:
        ld = LangevinDynamics(dim=8, dt=0.01, temperature=2.0)
        pos = [0.5 / np.sqrt(8)] * 8
        for i in range(100):
            pos, w = ld.step(pos, access_count=0, age_days=100.0, importance=0.0, seed=i)
            norm = np.linalg.norm(pos)
            assert norm < 1.0, f"Position escaped ball at step {i}: norm={norm}"
            assert 0.0 <= w <= 1.0, f"Weight out of range at step {i}: {w}"
