# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Background Math Maintenance.

Periodic batch processing for mathematical layers:
1. Langevin batch_step on all active facts (self-organization)
   1a. Backfill: seed uninitialized facts with metadata-aware positions (B+C)
2. Sheaf batch consistency check on recent facts
3. Fisher adaptive temperature recalculation

Frequency: every 6-24h or after 100 stores.
~100 Langevin steps to stationarity.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""
from __future__ import annotations

import logging
import math as _math
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# Backfill constants
_BACKFILL_BURN_IN_STEPS = 50
_LANGEVIN_DIM = 8
_MAX_NORM = 0.99


def _age_days(created_at: str | None) -> float:
    """Age in days from an ISO timestamp.

    Naive timestamps (no offset, no Z) are assumed UTC — some store paths
    persist created_at without timezone info, and subtracting a naive
    datetime from datetime.now(UTC) raises TypeError, which previously
    aborted the whole backfill loop.
    """
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - created).total_seconds() / 86400.0)


def _compute_equilibrium_radius(
    access_count: int,
    age_days: float,
    importance: float,
    temperature: float = 0.3,
    dim: int = 8,
) -> float:
    """Compute metadata-aware equilibrium radius (Strategy B).

    Uses the Langevin potential coefficients to estimate where a fact
    would settle if it had been in the dynamics from the start.

    r_eq ≈ sqrt(T * dim / (2 * effective_alpha))
    """
    alpha, beta, gamma, delta = 3.0, 0.8, 0.005, 0.5
    effective_alpha = (
        alpha
        + beta * _math.log(access_count + 1) / 10.0
        - gamma * min(age_days, 365.0) / 365.0
        + delta * importance
    )
    effective_alpha = max(0.1, effective_alpha)
    r_eq = _math.sqrt(temperature * dim / (2.0 * effective_alpha))
    return min(r_eq, _MAX_NORM * 0.95)


def _seed_langevin_position(
    access_count: int,
    age_days: float,
    importance: float,
    temperature: float = 0.3,
    dim: int = 8,
) -> list[float]:
    """Create a metadata-aware initial position (Strategy B).

    Places the fact at the equilibrium radius with a random direction.
    """
    r_eq = _compute_equilibrium_radius(
        access_count, age_days, importance, temperature, dim,
    )
    rng = np.random.default_rng()
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.ones(dim)
        norm = float(np.linalg.norm(direction))
    return (direction / norm * r_eq).tolist()


def run_maintenance(
    db: DatabaseManager,
    config: SLMConfig,
    profile_id: str = "default",
) -> dict[str, int]:
    """Run background maintenance on mathematical layers.

    Args:
        db: Database manager.
        config: Full SLM configuration.
        profile_id: Scope to this profile.

    Returns:
        Dict of counts: langevin_updated, sheaf_checked, etc.
    """
    counts: dict[str, int] = {
        "langevin_backfilled": 0,
        "langevin_updated": 0,
        "fisher_coupled": 0,
        "sheaf_checked": 0,
        "entity_summaries_consolidated": 0,  # V3.4.40
        "orphan_metadata_gc": 0,             # v3.6.4 (P1-3)
    }

    # P1-3 (embeddings-vector-02): sweep orphaned embedding_metadata left by
    # any FK-off delete path, so the semantic channel never maps to dead facts.
    # Runs before the early-return so it sweeps even for empty profiles.
    try:
        counts["orphan_metadata_gc"] = db.gc_orphaned_embedding_metadata()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("orphan metadata GC skipped: %s", exc)

    facts = db.get_all_facts(profile_id)
    if not facts:
        return counts

    # 1a. Backfill: seed uninitialized facts with metadata-aware positions (B+C)
    if config.math.langevin_persist_positions:
        try:
            from superlocalmemory.math.langevin import LangevinDynamics

            ld = LangevinDynamics(
                dim=_LANGEVIN_DIM,
                dt=config.math.langevin_dt,
                temperature=config.math.langevin_temperature,
            )

            backfilled = 0
            for f in facts:
                if f.langevin_position is not None:
                    continue
                age_days = _age_days(f.created_at)
                # Strategy B: metadata-aware seed position
                position = _seed_langevin_position(
                    f.access_count, age_days, f.importance,
                    config.math.langevin_temperature, _LANGEVIN_DIM,
                )
                # Strategy C: burn-in from the seeded position
                for step_i in range(_BACKFILL_BURN_IN_STEPS):
                    position, _ = ld.step(
                        position, f.access_count, age_days, f.importance,
                    )
                weight = ld.compute_lifecycle_weight(position)
                lifecycle = ld.get_lifecycle_state(weight).value
                db.update_fact(f.fact_id, {
                    "langevin_position": position,
                    "lifecycle": lifecycle,
                })
                f.langevin_position = position  # update in-memory for step 1b
                backfilled += 1

            counts["langevin_backfilled"] = backfilled
            if backfilled:
                logger.info("Langevin backfill: %d facts initialized", backfilled)
        except Exception as exc:
            logger.warning("Langevin backfill failed: %s", exc)

    # 1b. Langevin batch step on all positioned facts
    if config.math.langevin_persist_positions:
        try:
            from superlocalmemory.math.langevin import LangevinDynamics

            ld = LangevinDynamics(
                dim=_LANGEVIN_DIM,
                dt=config.math.langevin_dt,
                temperature=config.math.langevin_temperature,
            )
            fact_dicts = []
            for f in facts:
                if f.langevin_position is None:
                    continue
                age_days = _age_days(f.created_at)
                fact_dicts.append({
                    "fact_id": f.fact_id,
                    "position": f.langevin_position,
                    "access_count": f.access_count,
                    "age_days": age_days,
                    "importance": f.importance,
                })

            if fact_dicts:
                results = ld.batch_step(fact_dicts)
                for r in results:
                    db.update_fact(r["fact_id"], {
                        "langevin_position": r["position"],
                        "lifecycle": r["lifecycle"],
                    })
                counts["langevin_updated"] = len(results)
        except Exception as exc:
            logger.warning("Langevin maintenance failed: %s", exc)

    # 1b. Fisher-Langevin coupling: modulate temperature per-fact
    # High Fisher confidence (low variance) -> low temperature -> memory stabilizes
    # Low Fisher confidence (high variance) -> high temperature -> memory fades
    if config.math.langevin_persist_positions and counts["langevin_updated"] > 0:
        try:
            from superlocalmemory.dynamics.fisher_langevin_coupling import (
                FisherLangevinCoupling,
            )

            coupling = FisherLangevinCoupling(
                base_temperature=config.math.langevin_temperature,
            )
            coupled_count = 0

            for f in facts:
                if f.langevin_position is None or f.fisher_variance is None:
                    continue
                eff_temp = coupling.get_effective_temperature(
                    f.fisher_variance, f.access_count,
                )
                # Re-run Langevin step with Fisher-coupled temperature
                # only if it differs meaningfully from the base temperature
                if abs(eff_temp - config.math.langevin_temperature) > 0.01:
                    from superlocalmemory.math.langevin import LangevinDynamics

                    coupled_ld = LangevinDynamics(
                        dim=8,
                        dt=config.math.langevin_dt,
                        temperature=eff_temp,
                    )
                    age_days = _age_days(f.created_at)
                    new_pos, weight = coupled_ld.step(
                        position=f.langevin_position,
                        access_count=f.access_count,
                        age_days=age_days,
                        importance=f.importance,
                    )
                    lifecycle = coupled_ld.get_lifecycle_state(weight).value
                    db.update_fact(f.fact_id, {
                        "langevin_position": new_pos,
                        "lifecycle": lifecycle,
                    })
                    coupled_count += 1

            counts["fisher_coupled"] = coupled_count
        except Exception as exc:
            logger.warning("Fisher-Langevin coupling failed: %s", exc)

    # 2. Sheaf batch consistency on recent facts (last 24h)
    if config.math.sheaf_at_encoding:
        try:
            from superlocalmemory.math.sheaf import SheafConsistencyChecker

            checker = SheafConsistencyChecker(
                db, config.math.sheaf_contradiction_threshold,
            )
            cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
            recent = [f for f in facts if f.created_at and f.created_at >= cutoff]
            for f in recent:
                if f.embedding and f.canonical_entities:
                    checker.check_consistency(f, profile_id)
                    counts["sheaf_checked"] += 1
        except Exception as exc:
            logger.warning("Sheaf maintenance failed: %s", exc)

    # 3. V3.4.40: Entity summary consolidation
    # Re-bound any entity_profiles whose knowledge_summary exceeded the cap
    # (e.g. created before V3.4.40, or via a code path that bypassed the
    # bounded _build_summary). Truncates in-place — keeps entity identity,
    # drops bloat. Future writes go through ObservationBuilder.SUMMARY_*
    # bounds and stay clean.
    try:
        consolidated = db.execute(
            """
            UPDATE entity_profiles
               SET knowledge_summary = SUBSTR(knowledge_summary, 1, 2047) || '…',
                   last_updated = datetime('now')
             WHERE LENGTH(knowledge_summary) > 2048
               AND profile_id = ?
            """,
            (profile_id,),
        )
        # SQLite doesn't return rowcount via execute() wrapper consistently.
        # Re-count instead — fast on the small subset.
        rows = db.execute(
            "SELECT COUNT(*) AS c FROM entity_profiles "
            "WHERE LENGTH(knowledge_summary) > 2048 AND profile_id = ?",
            (profile_id,),
        )
        # If any remain >2048 after the UPDATE, log it. Otherwise count
        # how many were truncated by diffing against the prior pass.
        # (Best-effort; non-fatal.)
        if rows:
            remaining = dict(rows[0]).get("c", 0)
            counts["entity_summaries_consolidated"] = max(
                0, counts.get("entity_summaries_consolidated", 0)
            ) - remaining
    except Exception as exc:
        logger.warning("Entity summary consolidation failed: %s", exc)

    logger.info(
        "Maintenance complete: %d backfilled, %d Langevin, %d Fisher-coupled, "
        "%d Sheaf, %d entity-summaries",
        counts["langevin_backfilled"], counts["langevin_updated"],
        counts["fisher_coupled"], counts["sheaf_checked"],
        counts["entity_summaries_consolidated"],
    )
    return counts
