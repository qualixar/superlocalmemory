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
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from superlocalmemory.core.config import ForgettingConfig
from superlocalmemory.math.ebbinghaus import EbbinghausCurve

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


#: Access rows written by SLM's own maintenance, not by anyone using a memory.
#: Consolidation logs one per source it archives; counting it as a use raised
#: the source's strength and the next decay pass reactivated every source the
#: consolidation had just archived.
_INTERNAL_ACCESS_SQL = "('consolidation')"

#: A source a consolidation archived stays out of the decay pass — its
#: archive is a decision, not a score — until a GENUINE access after the
#: consolidation (a recall or search that reached it) brings it back.
_CONSOLIDATED_SOURCE_SQL = (
    " AND NOT EXISTS ("
    "  SELECT 1 FROM ccq_consolidated_blocks ccb, "
    "       json_each(ccb.source_fact_ids) je"
    "  WHERE ccb.profile_id = ? AND je.value = f.fact_id"
    "    AND NOT EXISTS ("
    "      SELECT 1 FROM fact_access_log a"
    "      WHERE a.fact_id = f.fact_id AND a.profile_id = ?"
    f"       AND a.access_type NOT IN {_INTERNAL_ACCESS_SQL}"
    "        AND a.accessed_at > ccb.created_at))"
)


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

    def run_decay_cycle(self, profile_id: str, *, force=False, dry_run=False) -> dict:
        """Plan decay from canonical inputs; persist only outside dry-run."""
        from superlocalmemory.learning.lifecycle_policy import apply_configured_policy

        now = time.monotonic()
        last_run = self._last_run_times.get(profile_id)
        if (
            not force
            and last_run is not None
            and (now - last_run) / 60.0 < self._config.scheduler_interval_minutes
        ):
            return {"skipped": True, "reason": "within_interval"}
        facts = self._fetch_facts_with_metadata(profile_id)
        previous = self._get_existing_zones(profile_id, facts)
        results = apply_configured_policy(
            self._db, self._ebbinghaus.batch_compute_retention(facts), profile_id
        )
        stats, forgotten = _retention_stats(results, previous)
        if not dry_run:
            self._db.batch_upsert_retention(results, profile_id)
            for fact_id in forgotten:
                self._soft_delete_with_audit(fact_id, profile_id)
            self._last_run_times[profile_id] = now
        return stats

    def on_access_event(self, fact_id: str, profile_id: str) -> None:
        """Refresh strength and the access clock without losing repetition on decay."""
        # Fetch current retention data
        current = self._db.get_retention(fact_id, profile_id)
        if current is None:
            # No retention data yet — nothing to update
            logger.debug("on_access_event: no retention for %s, skipping", fact_id)
            return

        current_strength = float(current["memory_strength"])
        last_accessed = current.get("last_accessed_at", "")

        hours_since = _hours_since(last_accessed)

        # Spaced repetition update (HR-07: only increases strength)
        new_strength = self._ebbinghaus.spaced_repetition_update(
            current_strength,
            hours_since,
        )

        # Recompute retention with new strength; junk is held at the keep
        # threshold however often it is touched (core/keep_worthiness.py).
        from superlocalmemory.core.keep_worthiness import cap_retention

        rows = self._db.execute(
            "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        content = dict(rows[0])["content"] if rows else None
        new_retention = cap_retention(
            self._ebbinghaus.retention(0.0, new_strength), content,
        )
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
        # Legacy UPSERT does not update its timestamp on conflict.
        self._db.execute(
            "UPDATE fact_retention SET last_accessed_at=? WHERE fact_id=? AND profile_id=?",
            (now_iso, fact_id, profile_id),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_facts_with_metadata(self, profile_id: str) -> list[dict]:
        """Read profile-scoped use, protection and trust before any lifecycle write."""
        from superlocalmemory.learning.forgetting_metadata import (
            attach_canonical_trust,
            metadata_query,
        )

        consolidated = (
            _CONSOLIDATED_SOURCE_SQL if self._has_table("ccq_consolidated_blocks") else ""
        )
        sql = metadata_query(self._db, _INTERNAL_ACCESS_SQL, consolidated)
        rows = [
            dict(row) for row in self._db.execute(sql, (profile_id,) * (7 if consolidated else 5))
        ]
        return attach_canonical_trust(self._db, rows, profile_id)

    def _get_existing_zones(
        self,
        profile_id: str,
        facts_data: list[dict],
    ) -> dict[str, str]:
        """Get existing lifecycle zones for transition tracking."""
        fact_ids = [f["fact_id"] for f in facts_data]
        if not fact_ids:
            return {}
        retention_rows = self._db.batch_get_retention(fact_ids, profile_id)
        return {r["fact_id"]: r["lifecycle_zone"] for r in retention_rows}

    def _has_table(self, name: str) -> bool:
        """Whether ``name`` exists (an old store may predate it)."""
        try:
            return bool(
                self._db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
                )
            )
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
            fact_id,
            profile_id,
        )
        self._db.soft_delete_fact(fact_id, profile_id)

    def _has_recent_positive_reward(
        self,
        fact_id: str,
        profile_id: str,
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
                self._db,
                "_conn",
                None,
            )
            if raw_conn is not None:
                from superlocalmemory.learning.fact_outcome_joins import (
                    has_recent_positive_reward,
                )

                return has_recent_positive_reward(
                    raw_conn,
                    profile_id,
                    fact_id,
                    min_reward=0.3,
                    window_days=60,
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


def _retention_stats(results, previous):
    counts = {zone: 0 for zone in ("active", "warm", "cold", "archive", "forgotten")}
    transitions, forgotten = 0, []
    for result in results:
        counts[result["zone"]] += 1
        old = previous.get(result["fact_id"])
        transitions += int(old is not None and old != result["zone"])
        if result["zone"] == "forgotten":
            forgotten.append(result["fact_id"])
    return {"total": len(results), **counts, "transitions": transitions}, forgotten


def _hours_since(value):
    if not value:
        return 0.0
    try:
        last = datetime.fromisoformat(value)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - last).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return 0.0
