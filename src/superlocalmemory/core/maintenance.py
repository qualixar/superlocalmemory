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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from superlocalmemory.core.langevin_tier import (  # noqa: F401 - re-exported
    _LANGEVIN_DIM,
    _MAX_NORM,
    _age_days,
    _anchor_to_retention,
    _compute_equilibrium_radius,
    _direction_seed,
    _langevin_pass,
    _retention_radius,
    _seed_langevin_position,
    langevin_lifecycle,
)

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class _ConsolidationDisabled(Exception):
    """Internal signal: consolidation is switched off, so skip its block.

    A private exception rather than restructuring the surrounding try/except:
    the block's job is to keep one optional maintenance step from taking the
    whole pass down with it, and that guarantee should not be weakened to
    express "deliberately skipped". Caught immediately below, never propagated.
    """

# Backfill constants
_BACKFILL_BURN_IN_STEPS = 50  # retained for compatibility; no longer applied
                              # at seed time -- see _seed_langevin_position

# ELC zone vocabulary: EbbinghausCurve returns 'archive'/'forgotten' but
# atomic_facts.lifecycle CHECK only allows 'active|warm|cold|archived'.
# Remap at write boundary so ELC never triggers IntegrityError.
_VALID_LIFECYCLE_ZONES: frozenset[str] = frozenset({"active", "warm", "cold", "archived"})
_ELC_ZONE_REMAP: dict[str, str] = {"archive": "archived", "forgotten": "archived"}


def _persist_lifecycle(
    db: object,
    profile_id: str,
    updates: "list[tuple[str, str, object]]",
) -> int:
    """Write a computed tier to the authority AND the mirror, together.

    WHY THIS EXISTS. Every lifecycle write in this module used to be
    ``db.update_fact(fact_id, {"lifecycle": ...})`` -- the mirror only.
    ``fact_retention.lifecycle_zone`` is the authority, and
    ``reconcile_profile_lifecycle`` runs later in the SAME maintenance tick and
    syncs authority -> mirror. So the pass computed a tier, wrote it to the
    losing column, and reconcile faithfully threw it away. Observed on a real
    store: the backfill moved ``active`` from 111 to 2,631, and one tick later
    it was 111 again.

    ``set_fact_lifecycle_zone`` writes both in one transaction and handles the
    spelling difference -- ``atomic_facts`` says ``archived``,
    ``fact_retention`` says ``archive``.

    ``updates`` is ``(fact_id, lifecycle, position_or_None)``. The position has
    no mirror, so it still goes through ``update_fact``; without it the backfill
    would recompute the same facts on every pass and never converge.

    Returns the number of facts whose tier was written.
    """
    if not updates:
        return 0
    from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone

    for fact_id, _lifecycle, position in updates:
        if position is not None:
            db.update_fact(fact_id, {"langevin_position": position})

    by_zone: dict[str, list[str]] = {}
    for fact_id, lifecycle, _position in updates:
        by_zone.setdefault(str(lifecycle), []).append(fact_id)

    written = 0
    for zone, fact_ids in by_zone.items():
        try:
            written += set_fact_lifecycle_zone(
                db, fact_ids, zone, profile_id=profile_id,
            )
        except Exception as exc:  # noqa: BLE001 -- maintenance is best-effort
            logger.warning("lifecycle persist failed for %s: %s", zone, exc)
    return written


def close_stale_sessions(
    db: DatabaseManager,
    profile_id: str = "default",
    *,
    idle_hours: float = 24.0,
    max_per_pass: int = 50,
) -> int:
    """Close application sessions idle longer than ``idle_hours``.

    Nothing auto-calls ``close_session`` except the MCP tool — un-closed
    sessions never get temporal summaries. This maintenance pass finds
    sessions whose newest fact is older than the idle window and closes
    them via ``run_close_session``.

    Properties:
      - Idempotent: already-summarised sessions are skipped (no double write).
      - Bounded: at most ``max_per_pass`` sessions closed per call.
      - Only sessions with entity-linked facts (summarisable) are selected.

    Returns:
        Number of sessions successfully summarised in this pass.
    """
    if idle_hours <= 0 or max_per_pass <= 0:
        return 0

    from superlocalmemory.core.store_pipeline import (
        _session_already_summarised,
        run_close_session,
    )

    cutoff = (datetime.now(UTC) - timedelta(hours=float(idle_hours))).isoformat()
    # Over-fetch slightly so already-closed rows in the window do not starve
    # the bounded close budget.
    fetch_limit = max(int(max_per_pass) * 3, int(max_per_pass))
    try:
        rows = db.execute(
            """
            SELECT session_id, MAX(created_at) AS last_at
              FROM atomic_facts
             WHERE profile_id = ?
               AND session_id IS NOT NULL
               AND session_id != ''
               AND canonical_entities_json IS NOT NULL
               AND canonical_entities_json != '[]'
             GROUP BY session_id
            HAVING MAX(created_at) < ?
             ORDER BY last_at ASC
             LIMIT ?
            """,
            (profile_id, cutoff, fetch_limit),
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("stale session query failed: %s", exc)
        return 0

    closed = 0
    for row in rows:
        if closed >= int(max_per_pass):
            break
        d = dict(row)
        sid = str(d.get("session_id") or "")
        if not sid:
            continue
        if _session_already_summarised(db, profile_id, sid):
            continue
        try:
            n = run_close_session(sid, profile_id, db=db)
        except Exception as exc:
            logger.warning("stale session close failed for %s: %s", sid, exc)
            continue
        if n > 0:
            closed += 1
    if closed:
        logger.info(
            "Closed %d stale session(s) (idle > %.1fh, profile=%s)",
            closed, idle_hours, profile_id,
        )
    return closed


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
        "stale_sessions_closed": 0,          # v4: orphaned application sessions
    }

    # Close idle application sessions so temporal summaries exist even when
    # clients never call close_session. Bounded + idempotent; fail-soft.
    try:
        idle_hours = float(getattr(config, "session_idle_close_hours", 24.0) or 24.0)
        max_close = int(getattr(config, "session_idle_close_max_per_pass", 50) or 50)
        counts["stale_sessions_closed"] = close_stale_sessions(
            db,
            profile_id,
            idle_hours=idle_hours,
            max_per_pass=max_close,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("stale session close skipped: %s", exc)

    # P1-3 (embeddings-vector-02): sweep orphaned embedding_metadata left by
    # any FK-off delete path, so the semantic channel never maps to dead facts.
    # Runs before the early-return so it sweeps even for empty profiles.
    try:
        counts["orphan_metadata_gc"] = db.gc_orphaned_embedding_metadata()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("orphan metadata GC skipped: %s", exc)
    # 4.1.22: ...and the vectors that sweep (and every fact delete) leaves with
    # no reference. They cannot be results but still take nearest-neighbour slots.
    from superlocalmemory.storage.vector_residue import sweep_unreferenced_vectors
    counts["unreferenced_vectors_gc"] = sweep_unreferenced_vectors(db.db_path)
    try:  # deleted words leave the keyword index at once from now on
        from superlocalmemory.storage.fts_residue import ensure_secure_delete
        ensure_secure_delete(db)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("keyword index secure delete skipped: %s", exc)

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

    # The person's forgetting settings, the ones the decay pass reads: every
    # Langevin tier below is decided by the same function with the same
    # settings (EbbinghausCurve.retention_and_zone), never by the defaults.
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve

    tier_curve = EbbinghausCurve(config.forgetting)

    # 1a. Backfill: seed uninitialized facts with metadata-aware positions (B+C)
    if config.math.langevin_persist_positions:
        try:
            backfilled = 0
            for f in facts:
                if f.langevin_position is not None:
                    continue
                age_days = _age_days(f.created_at)
                # The seed is now a measurement of this fact's retention, not
                # a guess to be annealed. The 50-step burn-in that used to
                # follow was there to anneal a seed that carried almost no
                # information -- it added sd 0.1225 of thermal noise against
                # 0.1120 of total signal, and moved the mean radius from
                # 0.6076 to 0.7515, i.e. it aged every memory on creation.
                # Fifty steps of simulated neglect applied to a fact four
                # seconds old is not what the README describes. GitHub #136.
                position = _seed_langevin_position(
                    f.access_count, age_days, f.importance,
                    config.math.langevin_temperature, _LANGEVIN_DIM,
                    fact_id=f.fact_id, curve=tier_curve,
                )
                lifecycle = langevin_lifecycle(
                    f.access_count, age_days, f.importance, tier_curve,
                )
                _persist_lifecycle(
                    db, profile_id, [(f.fact_id, lifecycle, position)],
                )
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
                results = _langevin_pass(ld, fact_dicts, tier_curve)
                _persist_lifecycle(db, profile_id, [
                    (r["fact_id"], r["lifecycle"], r["position"])
                    for r in results
                ])
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
                    (coupled,) = _langevin_pass(coupled_ld, [{
                        "fact_id": f.fact_id,
                        "position": f.langevin_position,
                        "access_count": f.access_count,
                        "age_days": age_days,
                        "importance": f.importance,
                    }], tier_curve)
                    new_pos, lifecycle = coupled["position"], coupled["lifecycle"]
                    _persist_lifecycle(
                        db, profile_id, [(f.fact_id, lifecycle, new_pos)],
                    )
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
            from superlocalmemory.math.fisher import FisherRaoMetric
            from superlocalmemory.storage.embedding_codec import encode_float_vector

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
                # The stored form is float32, like every other vector on a fact.
                # Writing JSON text here undid the conversion one accessed fact
                # per pass, and the next start converted it back again.
                db.execute(
                    "UPDATE atomic_facts "
                    "SET fisher_variance = ?, fisher_last_applied_access = ? "
                    "WHERE fact_id = ?",
                    (encode_float_vector(current_var), last_applied + applied,
                     f.fact_id),
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
                _persist_lifecycle(db, profile_id, [(f.fact_id, zone, None)])
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

    # 4. Fact consolidation (v3.8.4 concurrency-safe path via DatabaseManager).
    # Groups warm/cold atomic facts that share an entity and writes ONE
    # DISPLAY summary per cluster into consolidated_summaries, with provenance
    # in fact_consolidations. It does not write to atomic_facts and does not
    # archive the source facts — until 4.0.10 it did both, which put 1,195
    # model-written rows into the retrieval corpus and left 528 genuine
    # memories archived out of normal recall.
    #
    # Uses the DatabaseManager path so LLM calls happen OUTSIDE the write lock:
    #   - Discover clusters in a short memory_read() (no write lock held).
    #   - Generate summary OUTSIDE any lock (Ollama/Cloud may take 30s).
    #   - Write per-cluster inside a short memory_write() (lock held for SQL only).
    #
    # NOT on the recall/store hot path.  Runs as part of background maintenance.
    counts["facts_consolidated"] = 0
    try:
        from superlocalmemory.core.fact_consolidator import consolidate_facts

        # The documented off-switch has to actually switch something off.
        # ConsolidationConfig.enabled has existed since Phase 5 and this call
        # site never read it, so a user who ran `slm config` to turn
        # consolidation off got consolidation anyway — for four months, on
        # every maintenance pass. -2 is a third distinguishable value, kept
        # apart from 0 (nothing to merge) and -1 (the step failed), so a
        # deliberately disabled step is never mistaken for either.
        _consolidation = getattr(config, "consolidation", None)
        if _consolidation is not None and not getattr(_consolidation, "enabled", True):
            counts["facts_consolidated"] = -2
            logger.debug("Fact consolidation disabled by configuration")
            raise _ConsolidationDisabled

        fc_stats = consolidate_facts(
            db,
            profile_id=profile_id,
            # Read from ConsolidationConfig, with the old SLMConfig-level name
            # as the fallback. `getattr(config, "max_consolidation_clusters")`
            # alone never resolved — SLMConfig has no such attribute — so the
            # default was the only value this had ever used.
            max_clusters=int(
                getattr(_consolidation, "max_consolidation_clusters", None)
                or getattr(config, "max_consolidation_clusters", None)
                or 20
            ),
            dry_run=False,
            config=config,
        )
        counts["facts_consolidated"] = fc_stats.get("consolidated", 0)
        if fc_stats.get("consolidated", 0) > 0:
            logger.info(
                "Fact consolidation: %d display summaries over %d facts "
                "(%d clusters refused)",
                fc_stats.get("consolidated", 0),
                fc_stats.get("facts_summarized", 0),
                fc_stats.get("rejected", 0),
            )
    except _ConsolidationDisabled:
        pass
    except Exception as exc:
        # WARNING, not debug, and a distinguishable count. Leaving this at debug
        # with facts_consolidated=0 made a failing consolidation report exactly
        # the same numbers as a healthy run with nothing to merge, so a step that
        # never worked would look like a step with no work to do — and nobody
        # would ever see it in normal logs.
        counts["facts_consolidated"] = -1
        logger.warning(
            "Fact consolidation FAILED during maintenance (reported as -1, "
            "which is distinct from 0 = nothing to merge): %s", exc,
        )

    # 5. Code↔memory bridge (4.0.7).
    # Resolves code entity mentions in new facts against the code graph and
    # stores the links, plus derived enrichment text, in code_graph.db.
    #
    # RUNS HERE, NOT ON THE WRITE PATH. The bridge was authored to fire from
    # BridgeEventListeners.on_memory_stored, and EventBus._notify_listeners
    # dispatches synchronously on the emitting thread — which would have put
    # entity resolution and enrichment inside every remember. The owner's
    # constraint for 4.0.7 is that remember/recall timing must not move, so the
    # memory-stored subscription is gone and the work happens in this pass.
    # tests/test_code_graph/test_bridge_off_write_path.py fails if it comes back.
    #
    # Writes only code_graph.db, which no recall path opens, so this step cannot
    # affect recall latency or results. Hebbian edges are the one exception and
    # are NOT run here — they land in association_edges, which spreading
    # activation reads; they are generated on explicit request instead.
    counts["bridge_links"] = 0
    counts["bridge_enriched"] = 0
    try:
        from superlocalmemory.code_graph.config import CodeGraphConfig

        cg_cfg = CodeGraphConfig.load()
        if cg_cfg.enabled and cg_cfg.bridge_enabled:
            from superlocalmemory.code_graph.bridge.maintenance import run_bridge_pass
            from superlocalmemory.code_graph.database import CodeGraphDatabase

            cg_path = cg_cfg.get_db_path()
            if cg_path.exists():
                bridge_stats = run_bridge_pass(
                    db, CodeGraphDatabase(cg_path), profile_id,
                )
                counts["bridge_links"] = bridge_stats.get("links_created", 0)
                counts["bridge_enriched"] = bridge_stats.get("enriched", 0)
    except Exception as exc:
            # -1 rather than 0, for the same reason fact consolidation uses it:
            # a step that never worked must not report the same numbers as a
            # healthy step with no work to do.
            counts["bridge_links"] = -1
            logger.warning(
                "Code bridge pass FAILED during maintenance (reported as -1, "
                "which is distinct from 0 = nothing to link): %s", exc,
            )

    logger.info(
        "Maintenance complete: %d backfilled, %d Langevin, %d Fisher-coupled, "
        "%d Sheaf, %d entity-summaries, %d facts-consolidated, %d code-links",
        counts["langevin_backfilled"], counts["langevin_updated"],
        counts["fisher_coupled"], counts["sheaf_checked"],
        counts["entity_summaries_consolidated"],
        counts["facts_consolidated"],
        counts["bridge_links"],
    )
    return counts


# The bridge gate reads code_graph_config.json via CodeGraphConfig.load(), not
# SLMConfig: SLMConfig has no code_graph block, so there is nothing to read
# there. cli/setup_wizard.py writes that file; before 4.0.7 no loader existed
# and every call site hardcoded CodeGraphConfig(enabled=True) instead.
