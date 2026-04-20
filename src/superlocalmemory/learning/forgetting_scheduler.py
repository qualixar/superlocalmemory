# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Forgetting scheduler — periodic retention decay and lifecycle management.

Runs the Ebbinghaus decay cycle on all facts for a profile:
1. Fetches all facts with access counts, importance, and emotion data
2. Excludes core memory facts (HR-01: Core Memory NEVER forgets)
3. Computes retention scores via EbbinghausCurve
4. UPSERTs results into fact_retention table
5. Soft-deletes facts that fall below forget_threshold

Also handles real-time spaced repetition updates on access events.

HR-04: Soft-delete ONLY. Never physically deletes.
HR-08: Runs synchronously in main thread.
HR-09: Access events are idempotent.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.math.ebbinghaus import EbbinghausCurve

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class ForgettingScheduler:
    """Periodic forgetting scheduler for memory lifecycle management.

    Computes Ebbinghaus retention scores and manages memory lifecycle
    transitions. Runs synchronously (HR-08).
    """

    __slots__ = ("_db", "_ebbinghaus", "_config", "_last_run_times")

    def __init__(
        self,
        db: DatabaseManager,
        ebbinghaus: EbbinghausCurve,
        config: ForgettingConfig,
    ) -> None:
        self._db = db
        self._ebbinghaus = ebbinghaus
        self._config = config
        # Track last run time per profile for interval enforcement
        self._last_run_times: dict[str, float] = {}

    def run_decay_cycle(
        self, profile_id: str, *, force: bool = False,
    ) -> dict:
        """Run a full decay cycle for all facts in a profile.

        Fetches facts, computes retention, upserts results, and
        soft-deletes forgotten facts.

        Args:
            profile_id: Profile to process.
            force: If True, bypass interval check.

        Returns:
            Stats dict: {total, active, warm, cold, archive, forgotten,
                         transitions, skipped}.
        """
        # Interval check (Test 21)
        now = time.monotonic()
        last_run = self._last_run_times.get(profile_id)
        if not force and last_run is not None:
            elapsed_minutes = (now - last_run) / 60.0
            if elapsed_minutes < self._config.scheduler_interval_minutes:
                return {"skipped": True, "reason": "within_interval"}

        # Step 1: Fetch all facts with metadata
        facts_data = self._fetch_facts_with_metadata(profile_id)

        if not facts_data:
            self._last_run_times[profile_id] = now
            return {
                "total": 0, "active": 0, "warm": 0, "cold": 0,
                "archive": 0, "forgotten": 0, "transitions": 0,
            }

        # Step 2: Get existing retention data for transition tracking
        existing_zones = self._get_existing_zones(profile_id, facts_data)

        # Step 3: Compute retention scores via EbbinghausCurve
        retention_results = self._ebbinghaus.batch_compute_retention(facts_data)

        # Step 4: Batch UPSERT into fact_retention
        self._db.batch_upsert_retention(retention_results, profile_id)

        # Step 5: Count zones and transitions
        zone_counts = {"active": 0, "warm": 0, "cold": 0, "archive": 0, "forgotten": 0}
        transitions = 0
        forgotten_fact_ids: list[str] = []

        for result in retention_results:
            zone = result["zone"]
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

            # Track transitions
            old_zone = existing_zones.get(result["fact_id"])
            if old_zone is not None and old_zone != zone:
                transitions += 1

            # Collect forgotten facts for soft-delete
            if zone == "forgotten":
                forgotten_fact_ids.append(result["fact_id"])

        # Step 6: Soft-delete forgotten facts (HR-04)
        for fact_id in forgotten_fact_ids:
            self._soft_delete_with_audit(fact_id, profile_id)

        # Update last run time
        self._last_run_times[profile_id] = now

        return {
            "total": len(retention_results),
            "active": zone_counts["active"],
            "warm": zone_counts["warm"],
            "cold": zone_counts["cold"],
            "archive": zone_counts["archive"],
            "forgotten": zone_counts["forgotten"],
            "transitions": transitions,
        }

    def on_access_event(self, fact_id: str, profile_id: str) -> None:
        """Handle real-time access event with spaced repetition update.

        Called from access_log.store_access() hook.
        HR-09: Idempotent — reads current state, computes from scratch.

        Args:
            fact_id: Accessed fact ID.
            profile_id: Profile ID.
        """
        # Fetch current retention data
        current = self._db.get_retention(fact_id, profile_id)
        if current is None:
            # No retention data yet — nothing to update
            logger.debug("on_access_event: no retention for %s, skipping", fact_id)
            return

        current_strength = float(current["memory_strength"])
        last_accessed = current.get("last_accessed_at", "")

        # Compute hours since last access
        hours_since = 0.0
        if last_accessed:
            try:
                last_dt = datetime.fromisoformat(last_accessed)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                now_dt = datetime.now(UTC)
                hours_since = max(0.0, (now_dt - last_dt).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                hours_since = 0.0

        # Spaced repetition update (HR-07: only increases strength)
        new_strength = self._ebbinghaus.spaced_repetition_update(
            current_strength, hours_since,
        )

        # Recompute retention with new strength
        new_retention = self._ebbinghaus.retention(0.0, new_strength)
        new_zone = self._ebbinghaus.lifecycle_zone(new_retention)

        now_iso = datetime.now(UTC).isoformat()

        # UPSERT updated data
        self._db.upsert_retention(
            fact_id=fact_id,
            profile_id=profile_id,
            retention_score=new_retention,
            memory_strength=new_strength,
            access_count=int(current["access_count"]) + 1,
            last_accessed_at=now_iso,
            lifecycle_zone=new_zone,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_facts_with_metadata(self, profile_id: str) -> list[dict]:
        """Fetch all non-core-memory facts with access, importance, and emotion data.

        Uses the query from LLD Section 2.4 with A-HIGH-02/03 fixes:
        - confirmation_count mapped from atomic_facts.evidence_count
        - emotional_salience from atomic_facts.emotional_valence
        """
        # V3.3.26: Trust-weighted forgetting — look up trust score for
        # the agent that created each fact. Falls back to 1.0 if trust_scores
        # table or created_by column is unavailable.
        trust_available = self._has_trust_tables()
        if trust_available:
            sql = (
                "SELECT f.fact_id, "
                "  COALESCE(al.access_count, 0) as access_count, "
                "  COALESCE(fi.pagerank_score, 0.0) as importance, "
                "  COALESCE(f.evidence_count, 0) as confirmation_count, "
                "  f.created_at, "
                "  COALESCE(r.last_accessed_at, f.created_at) as last_accessed_at, "
                "  COALESCE(f.emotional_valence, 0.0) as emotional_salience, "
                "  COALESCE(ts.trust_score, 1.0) as trust_score "
                "FROM atomic_facts f "
                "LEFT JOIN ("
                "  SELECT fact_id, COUNT(*) as access_count "
                "  FROM fact_access_log WHERE profile_id = ? GROUP BY fact_id"
                ") al ON f.fact_id = al.fact_id "
                "LEFT JOIN fact_importance fi "
                "  ON f.fact_id = fi.fact_id AND fi.profile_id = ? "
                "LEFT JOIN fact_retention r "
                "  ON f.fact_id = r.fact_id AND r.profile_id = ? "
                "LEFT JOIN trust_scores ts "
                "  ON ts.target_id = f.created_by "
                "  AND ts.target_type = 'agent' "
                "  AND ts.profile_id = ? "
                "WHERE f.profile_id = ? "
                "AND f.fact_id NOT IN ("
                "  SELECT json_each.value "
                "  FROM core_memory_blocks, json_each(core_memory_blocks.source_fact_ids) "
                "  WHERE core_memory_blocks.profile_id = ?"
                ")"
            )
            params = (profile_id,) * 6
        else:
            sql = (
                "SELECT f.fact_id, "
                "  COALESCE(al.access_count, 0) as access_count, "
                "  COALESCE(fi.pagerank_score, 0.0) as importance, "
                "  COALESCE(f.evidence_count, 0) as confirmation_count, "
                "  f.created_at, "
                "  COALESCE(r.last_accessed_at, f.created_at) as last_accessed_at, "
                "  COALESCE(f.emotional_valence, 0.0) as emotional_salience "
                "FROM atomic_facts f "
                "LEFT JOIN ("
                "  SELECT fact_id, COUNT(*) as access_count "
                "  FROM fact_access_log WHERE profile_id = ? GROUP BY fact_id"
                ") al ON f.fact_id = al.fact_id "
                "LEFT JOIN fact_importance fi "
                "  ON f.fact_id = fi.fact_id AND fi.profile_id = ? "
                "LEFT JOIN fact_retention r "
                "  ON f.fact_id = r.fact_id AND r.profile_id = ? "
                "WHERE f.profile_id = ? "
                "AND f.fact_id NOT IN ("
                "  SELECT json_each.value "
                "  FROM core_memory_blocks, json_each(core_memory_blocks.source_fact_ids) "
                "  WHERE core_memory_blocks.profile_id = ?"
                ")"
            )
            params = (profile_id,) * 5

        rows = self._db.execute(sql, params)

        facts: list[dict] = []
        for row in rows:
            d = dict(row)
            facts.append({
                "fact_id": d["fact_id"],
                "access_count": int(d["access_count"]),
                "importance": float(d["importance"]),
                "confirmation_count": int(d["confirmation_count"]),
                "emotional_salience": float(d["emotional_salience"]),
                "last_accessed_at": str(d["last_accessed_at"]),
                "trust_score": float(d.get("trust_score", 1.0)),
            })
        return facts

    def _get_existing_zones(
        self, profile_id: str, facts_data: list[dict],
    ) -> dict[str, str]:
        """Get existing lifecycle zones for transition tracking."""
        fact_ids = [f["fact_id"] for f in facts_data]
        if not fact_ids:
            return {}
        retention_rows = self._db.batch_get_retention(fact_ids, profile_id)
        return {r["fact_id"]: r["lifecycle_zone"] for r in retention_rows}

    def _has_trust_tables(self) -> bool:
        """Check if trust_scores table and created_by column exist."""
        try:
            self._db.execute(
                "SELECT 1 FROM trust_scores LIMIT 0", (),
            )
            self._db.execute(
                "SELECT created_by FROM atomic_facts LIMIT 0", (),
            )
            return True
        except Exception:
            return False

    def _soft_delete_with_audit(self, fact_id: str, profile_id: str) -> None:
        """Soft-delete a forgotten fact with compliance audit trail.

        v3.4.22 (LLD-12 §4): reward-gated. If the fact has any positive
        reward (>0.3) in the last 60 days, it is considered "still
        useful" and kept live — consolidation will retry next cycle.

        HR-04: Never physically deletes.
        """
        if self._has_recent_positive_reward(fact_id, profile_id):
            logger.debug(
                "forgetting_scheduler: fact_id=%s kept live (recent reward)",
                fact_id,
            )
            return
        logger.info(
            "Soft-deleting forgotten fact: fact_id=%s, profile_id=%s",
            fact_id, profile_id,
        )
        self._db.soft_delete_fact(fact_id, profile_id)

    def _has_recent_positive_reward(
        self, fact_id: str, profile_id: str,
    ) -> bool:
        """True if fact has an outcome_reward > 0.3 in the last 60 days.

        v3.4.22 (Stage 8 H-06): routes through the JSON1-backed
        ``fact_outcome_joins.has_recent_positive_reward`` helper —
        eliminates the substring-LIKE false-positive class.

        Resilient to schema drift: if ``action_outcomes`` or its columns
        are unavailable we return False (no gating), preserving legacy
        behaviour.
        """
        try:
            # ``DatabaseManager`` is the owner of a persistent sqlite
            # connection; the JSON1 helper needs a raw connection. We
            # fall through to the legacy execute-path if the DB wrapper
            # does not expose a ``.conn`` handle.
            raw_conn = getattr(self._db, "conn", None) or getattr(
                self._db, "_conn", None,
            )
            if raw_conn is not None:
                from superlocalmemory.learning.fact_outcome_joins import (
                    has_recent_positive_reward,
                )
                return has_recent_positive_reward(
                    raw_conn, profile_id, fact_id,
                    min_reward=0.3, window_days=60,
                )
            # Fallback: use the DB wrapper with JSON1 SQL inline.
            rows = self._db.execute(
                "SELECT 1 FROM action_outcomes "
                "WHERE profile_id = ? "
                "  AND reward IS NOT NULL AND reward > 0.3 "
                "  AND EXISTS ("
                "    SELECT 1 FROM json_each(fact_ids_json) WHERE value = ?"
                "  ) "
                "  AND COALESCE(settled_at, '') >= datetime('now', '-60 days') "
                "LIMIT 1",
                (profile_id, fact_id),
            )
            return bool(rows)
        except Exception:
            return False
