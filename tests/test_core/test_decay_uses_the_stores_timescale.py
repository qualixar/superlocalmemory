# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Forgetting is measured on the store's clock, not a working-memory clock.

``EbbinghausCurve`` models working memory: ``max_strength`` is 100, about four
days. The decay cycle fed it real elapsed time from a persistent store, so
anything older than a few days computed ``R ~ 0`` and was filed ``forgotten``.

MEASURED on a real store: **5,546 of 5,561 memories in `forgotten`**, which the
mirror reported as 93.6% archived — and every subsystem that reads only active
memories saw almost nothing.

4.1.15 rescaled the SEED path and left the DECAY path alone, so the decay cycle
overwrote every corrected tier minutes later. The conversion now lives in ONE
place, ``EbbinghausCurve.store_scaled_strength``, which both call — the two
cannot drift apart again.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.core.tier_manager import (
    ARCHIVE_AFTER_DAYS,
    COLD_AFTER_DAYS,
)
from superlocalmemory.math.ebbinghaus import EbbinghausCurve


@pytest.fixture()
def curve() -> EbbinghausCurve:
    return EbbinghausCurve(ForgettingConfig())


def _zone_at(curve: EbbinghausCurve, days: float, access: int = 0) -> str:
    s = curve.store_scaled_strength(
        curve.memory_strength(access, 0.5, 0, 0.0))
    return curve.lifecycle_zone(curve.retention(days * 24.0, s))


class TestTheLadderMatchesWhatTheProductDocuments:
    def test_a_fresh_memory_is_active(self, curve) -> None:
        assert _zone_at(curve, 0) == "active"

    def test_a_month_old_memory_is_still_active(self, curve) -> None:
        """It used to be `forgotten` after about four days."""
        assert _zone_at(curve, 30) == "active"

    def test_it_reaches_the_archive_threshold_at_the_documented_day(
        self, curve,
    ) -> None:
        """The constant is DERIVED from ARCHIVE_AFTER_DAYS, not chosen."""
        cfg = ForgettingConfig()
        s = curve.store_scaled_strength(curve.memory_strength(0, 0.5, 0, 0.0))
        r = curve.retention(ARCHIVE_AFTER_DAYS * 24.0, s)
        assert r == pytest.approx(cfg.archive_threshold, abs=1e-6)

    def test_the_bands_are_ordered_over_the_stores_lifetime(self, curve) -> None:
        order = ["active", "active", "warm", "cold", "archive", "forgotten"]
        got = [_zone_at(curve, d) for d in (0, 30, 90, COLD_AFTER_DAYS,
                                            ARCHIVE_AFTER_DAYS, 730)]
        assert got == order, got

    def test_nothing_lands_forgotten_inside_a_working_week(self, curve) -> None:
        """The failure users actually saw."""
        assert all(_zone_at(curve, d) == "active" for d in (1, 3, 7))


class TestBeingUsedPullsAMemoryBack:
    def test_access_moves_it_up_the_ladder(self, curve) -> None:
        zones = [_zone_at(curve, 180, access=a) for a in (0, 1, 5)]
        assert zones == ["cold", "warm", "active"], zones

    def test_retention_is_monotonic_in_access_count(self, curve) -> None:
        rs = []
        for a in (0, 1, 5, 25, 100):
            s = curve.store_scaled_strength(curve.memory_strength(a, 0.5, 0, 0.0))
            rs.append(curve.retention(180 * 24.0, s))
        assert rs == sorted(rs)


class TestOneDerivationNotTwo:
    def test_the_seed_path_calls_the_same_helper(self) -> None:
        """4.1.15 had two copies and they diverged within one release; 4.2.8
        found a third (the Langevin pass on default settings). Every tier path
        now goes through ``EbbinghausCurve.retention_and_zone``."""
        import ast
        import inspect

        from superlocalmemory.core import langevin_tier

        src = inspect.getsource(langevin_tier._retention_and_zone)
        assert ".retention_and_zone(" in src
        for fn in (langevin_tier._retention_radius, langevin_tier.langevin_lifecycle):
            body = ast.parse(inspect.getsource(fn).lstrip()).body[0]
            names = {n.id for n in ast.walk(body) if isinstance(n, ast.Name)}
            assert "_retention_and_zone" in names, fn.__name__
            assert "ARCHIVE_AFTER_DAYS" not in names, (
                "the seed path is deriving the constant itself again"
            )

    def test_the_decay_path_calls_the_same_helper(self) -> None:
        import inspect

        from superlocalmemory.math.ebbinghaus import EbbinghausCurve as C

        assert "self.retention_and_zone(" in inspect.getsource(C.batch_compute_retention)
        assert "store_scaled_strength" in inspect.getsource(C.retention_and_zone)

    def test_trust_weighting_still_applies(self, curve) -> None:
        """Rescaling must not quietly drop trust-weighted forgetting."""
        s = curve.store_scaled_strength(curve.memory_strength(0, 0.5, 0, 0.0))
        trusted = curve.trust_modulated_retention(180 * 24.0, s, 1.0)
        untrusted = curve.trust_modulated_retention(180 * 24.0, s, 0.0)
        assert untrusted < trusted
