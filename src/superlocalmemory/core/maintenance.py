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
License: AGPL-3.0-or-later
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

# ELC zone vocabulary: EbbinghausCurve returns 'archive'/'forgotten' but
# atomic_facts.lifecycle CHECK only allows 'active|warm|cold|archived'.
# Remap at write boundary so ELC never triggers IntegrityError.
_VALID_LIFECYCLE_ZONES: frozenset[str] = frozenset({"active", "warm", "cold", "archived"})
_ELC_ZONE_REMAP: dict[str, str] = {"archive": "archived", "forgotten": "archived"}


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
    embedder: object | None = None,
) -> dict[str, int]:
    """Run background maintenance on mathematical layers.

    Args:
        db: Database manager.
        config: Full SLM configuration.
        profile_id: Scope to this profile.
        embedder: Optional embedder for self-healing NULL-embedding backfill.
            When provided and NULL embeddings exist, up to 100 facts are
            embedded per maintenance pass so the DB converges over time.
            Pass ``None`` (default) to skip the backfill — existing callers
            are unaffected.

    Returns:
        Dict of counts: langevin_updated, sheaf_checked, etc.
    """
    counts: dict[str, int] = {
        "langevin_backfilled": 0,
        "langevin_updated": 0,
        "fisher_coupled": 0,
        "fisher_posterior_updated": 0,       # P1-9: Fisher bayesian_update on access
        "ebbinghaus_coupled": 0,             # Phase 5: Ebbinghaus-Langevin coupling
        "sheaf_checked": 0,
        "entity_summaries_consolidated": 0,  # V3.4.40
        "orphan_metadata_gc": 0,             # v3.6.4 (P1-3)
        "expansion_backfilled": 0,           # T3b
        "embeddings_backfilled": 0,          # v3.8.x NULL-embedding self-heal
    }

    # P1-3 (embeddings-vector-02): sweep orphaned embedding_metadata left by
    # any FK-off delete path, so the semantic channel never maps to dead facts.
    # Runs before the early-return so it sweeps even for empty profiles.
    try:
        counts["orphan_metadata_gc"] = db.gc_orphaned_embedding_metadata()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("orphan metadata GC skipped: %s", exc)

    # v3.8.x: self-healing NULL-embedding backfill.  Facts stored while the
    # embedder was unavailable end up with NULL embedding and are invisible to
    # semantic recall.  When an embedder is available, embed up to 100 facts
    # per maintenance pass so the DB converges without blocking the caller.
    if embedder is not None:
        try:
            from superlocalmemory.storage.embedding_migrator import (
                backfill_missing_embeddings,
            )

            # Guard: skip entirely when nothing needs backfilling.
            null_rows = db.execute(
                "SELECT count(*) AS c FROM atomic_facts "
                "WHERE embedding IS NULL AND profile_id = ?",
                (profile_id,),
            )
            null_count = int(null_rows[0]["c"]) if null_rows else 0
            if null_count > 0:
                result = backfill_missing_embeddings(
                    config,
                    db,
                    embedder,
                    batch_size=50,
                    limit=100,
                )
                counts["embeddings_backfilled"] = result["embedded"]
                if result["embedded"] > 0:
                    logger.info(
                        "Maintenance embedding backfill: %d facts embedded, "
                        "%d remaining.",
                        result["embedded"],
                        result["remaining_null"],
                    )
        except Exception as exc:
            logger.debug("embedding backfill skipped during maintenance: %s", exc)

    facts = db.get_all_facts(profile_id)
    if not facts:
        return counts

    # T3b: backfill fact-expansion alt-keys (Mode A, entity-alias based) for
    # facts stored before expansion existed. Bounded per run + skips already-
    # populated and entity-less facts, so it converges without re-work churn.
    try:
        from superlocalmemory.core.key_expander import KeyExpander
        populated = {
            dict(r)["fact_id"]
            for r in db.execute("SELECT DISTINCT fact_id FROM fact_expansion_fts")
        }
        expander = KeyExpander(db)
        for f in facts:
            if counts["expansion_backfilled"] >= 500:
                break
            if f.fact_id in populated or not f.canonical_entities:
                continue
            alt = expander.expand(f, profile_id, mode="a")
            if alt:
                db.upsert_fact_expansion(f.fact_id, alt)
                counts["expansion_backfilled"] += 1
    except Exception as exc:  # pragma: no cover — legacy DB / missing FTS
        logger.debug("expansion backfill skipped: %s", exc)

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

    # 1c. Fisher posterior update (P1-9): tighten variance per new access event.
    # Access-delta semantics: apply one Bayesian update per net-new access since
    # the last maintenance run.  Zero new accesses → variance unchanged.
    # This prevents idle-corpus drift that tick-based updates would cause.
    # Inline schema migration: adds fisher_last_applied_access when absent.
    # Gate: config.math.fisher_bayesian_update (default True).
    if config.math.fisher_bayesian_update:
        try:
            import json as _json
            from superlocalmemory.math.fisher import FisherRaoMetric

            # Inline migration — harmless no-op if column already exists.
            try:
                db.execute(
                    "ALTER TABLE atomic_facts "
                    "ADD COLUMN fisher_last_applied_access INTEGER NOT NULL DEFAULT 0"
                )
            except Exception:
                pass  # already migrated on a previous run

            frm = FisherRaoMetric(temperature=config.math.fisher_temperature)
            posterior_count = 0
            for f in facts:
                if f.fisher_variance is None:
                    continue
                rows = db.execute(
                    "SELECT access_count, fisher_last_applied_access "
                    "FROM atomic_facts WHERE fact_id = ?",
                    (f.fact_id,),
                )
                if not rows:
                    continue
                r = dict(rows[0])
                acc = r.get("access_count") or 0
                last_applied = r.get("fisher_last_applied_access") or 0
                delta = acc - last_applied
                if delta <= 0:
                    continue  # no new accesses — variance unchanged this run
                # Apply min(delta, 100) unit-information Bayesian updates.
                # One update per access: 1/v_new = 1/v_old + 1 (unit obs_var).
                current_var = list(f.fisher_variance)
                dim = len(current_var)
                obs_var = [1.0] * dim
                applied = min(delta, 100)
                for _ in range(applied):
                    current_var = frm.bayesian_update(current_var, obs_var)
                # Single atomic write: variance + watermark together. Advance the
                # watermark only by the number of updates ACTUALLY applied (not to
                # acc), so accesses beyond the per-run cap are applied on subsequent
                # runs instead of being silently dropped.
                db.execute(
                    "UPDATE atomic_facts "
                    "SET fisher_variance = ?, fisher_last_applied_access = ? "
                    "WHERE fact_id = ?",
                    (_json.dumps(current_var), last_applied + applied, f.fact_id),
                )
                # Refresh in-memory so step 1d ELC sees the updated variance.
                f.fisher_variance = current_var
                posterior_count += 1
            counts["fisher_posterior_updated"] = posterior_count
        except Exception as exc:
            logger.warning("Fisher posterior update failed: %s", exc)

    # 1d. Ebbinghaus-Langevin coupling (Phase 5 — P1-ELC): combine forgetting
    # drift with Fisher-Langevin dynamics to produce a unified lifecycle state.
    # Updates the lifecycle zone of each fact based on Ebbinghaus retention.
    # NOTE: this step overwrites the Langevin-only lifecycle set in step 1b.
    #       The Ebbinghaus zone is intentionally authoritative when ELC is ON.
    # Gate: config.math.ebbinghaus_langevin_coupling_enabled (default False).
    if config.math.ebbinghaus_langevin_coupling_enabled:
        try:
            from superlocalmemory.dynamics.ebbinghaus_langevin_coupling import (
                EbbinghausLangevinCoupling,
            )
            from superlocalmemory.dynamics.fisher_langevin_coupling import (
                FisherLangevinCoupling,
            )
            from superlocalmemory.math.ebbinghaus import EbbinghausCurve
            from superlocalmemory.math.langevin import LangevinDynamics

            ebbinghaus = EbbinghausCurve(config.forgetting)
            langevin = LangevinDynamics(
                dim=_LANGEVIN_DIM,
                dt=config.math.langevin_dt,
                temperature=config.math.langevin_temperature,
            )
            fisher_coupling = FisherLangevinCoupling(
                base_temperature=config.math.langevin_temperature,
            )
            coupling = EbbinghausLangevinCoupling(
                ebbinghaus, langevin, fisher_coupling, config.forgetting,
            )
            import numpy as np

            # Build fact_id → last_accessed_at lookup from fact_retention.
            # Using real last-access time (not created_at) so hot facts are not
            # mis-classified as forgotten due to old creation timestamps.
            if facts:
                retention_rows = db.execute(
                    "SELECT fact_id, last_accessed_at FROM fact_retention "
                    "WHERE fact_id IN ({})".format(",".join("?" * len(facts))),
                    tuple(f.fact_id for f in facts),
                )
            else:
                retention_rows = []
            last_accessed_map: dict[str, str | None] = {
                dict(r)["fact_id"]: dict(r)["last_accessed_at"]
                for r in retention_rows
            }

            elc_count = 0
            for f in facts:
                if f.fisher_variance is None or f.langevin_position is None:
                    continue
                # Prefer real last-access timestamp; fall back to created_at.
                raw_ts = last_accessed_map.get(f.fact_id) or f.created_at
                hours_since = _age_days(raw_ts) * 24.0
                state = coupling.compute_coupled_state(
                    fact_id=f.fact_id,
                    fisher_variance=np.asarray(f.fisher_variance, dtype=np.float64),
                    langevin_radius=float(np.linalg.norm(f.langevin_position)),
                    access_count=f.access_count,
                    importance=f.importance,
                    confirmation_count=f.evidence_count,
                    emotional_salience=0.0,
                    hours_since_last_access=hours_since,
                )
                # Remap ELC zone vocabulary to atomic_facts CHECK constraint.
                # EbbinghausCurve returns 'archive'/'forgotten'; schema only allows
                # 'active|warm|cold|archived'.
                zone = _ELC_ZONE_REMAP.get(state.lifecycle_zone, state.lifecycle_zone)
                if zone not in _VALID_LIFECYCLE_ZONES:
                    logger.warning(
                        "ELC returned unknown lifecycle zone %r for fact %s — skipping",
                        state.lifecycle_zone, f.fact_id,
                    )
                    continue
                # Count fact as processed regardless of whether we write.
                elc_count += 1
                # Skip write when zone hasn't changed — avoids O(N) UPDATEs per tick.
                current_zone = (
                    f.lifecycle.value
                    if hasattr(f.lifecycle, "value")
                    else str(f.lifecycle)
                )
                if zone == current_zone:
                    continue
                db.update_fact(f.fact_id, {"lifecycle": zone})
            counts["ebbinghaus_coupled"] = elc_count
        except Exception as exc:
            logger.warning("Ebbinghaus-Langevin coupling failed: %s", exc)

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
