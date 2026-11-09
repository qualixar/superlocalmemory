# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Memory Lifecycle Management.

Implements Active → Warm → Cold → Archived state machine.
Coupled with Langevin dynamics: positions naturally create lifecycle states.

Ported from V2.8 with Langevin coupling (Innovation's unique feature).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from superlocalmemory.storage.models import AtomicFact, MemoryLifecycle

logger = logging.getLogger(__name__)

# Lifecycle transition thresholds (days since last access)
_ACTIVE_MAX_DAYS = 7        # Active for 7 days after last access
_WARM_MAX_DAYS = 30         # Warm for 30 days
_COLD_MAX_DAYS = 90         # Cold for 90 days, then archived

# Langevin weight thresholds (if Langevin is available)
_ACTIVE_WEIGHT_MIN = 0.7
_WARM_WEIGHT_MIN = 0.4
_COLD_WEIGHT_MIN = 0.1


class LifecycleManager:
    """Manage memory lifecycle states.

    Two complementary strategies:
    1. Time-based: days since last access → state transition
    2. Langevin-based: position on Poincaré ball → lifecycle weight → state

    When Langevin is available, it takes precedence (more nuanced).
    Time-based is the fallback for Mode A (no dynamics).
    """

    def __init__(self, db, langevin=None) -> None:
        self._db = db
        self._langevin = langevin

    def get_lifecycle_state(self, fact: AtomicFact) -> MemoryLifecycle:
        """Determine current lifecycle state for a fact."""
        # Strategy 1: Langevin-based (if available and position exists)
        if self._langevin is not None and fact.langevin_position:
            weight = self._langevin.compute_lifecycle_weight(fact.langevin_position)
            return self._langevin.get_lifecycle_state(weight)

        # Strategy 2: Time-based fallback
        return self._time_based_state(fact)

    def update_lifecycle(self, fact_id: str, profile_id: str) -> MemoryLifecycle:
        """Recompute and persist lifecycle state for a fact."""
        rows = self._db.execute(
            "SELECT lifecycle, access_count, created_at FROM atomic_facts "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        if not rows:
            return MemoryLifecycle.ACTIVE

        d = dict(rows[0])
        days = self._days_since(d.get("created_at", ""))
        access = d.get("access_count", 0)

        if access > 10 or days < _ACTIVE_MAX_DAYS:
            state = MemoryLifecycle.ACTIVE
        elif days < _WARM_MAX_DAYS:
            state = MemoryLifecycle.WARM
        elif days < _COLD_MAX_DAYS:
            state = MemoryLifecycle.COLD
        else:
            state = MemoryLifecycle.ARCHIVED

        current = d.get("lifecycle", "active")
        if current != state.value:
            self._db.update_fact(fact_id, {"lifecycle": state})
            logger.debug("Fact %s: %s → %s", fact_id, current, state.value)

        return state

    def run_maintenance(self, profile_id: str) -> dict[str, int]:
        """Run lifecycle maintenance on all facts in a profile.

        If Langevin is available, evolve positions and update states.
        Otherwise, use time-based transitions.
        """
        facts = self._db.get_all_facts(profile_id)
        counts: dict[str, int] = {s.value: 0 for s in MemoryLifecycle}
        transitions = 0

        for fact in facts:
            old_state = fact.lifecycle

            if self._langevin is not None and fact.langevin_position:
                # Langevin step
                age = self._days_since(fact.created_at)
                new_pos, weight = self._langevin.step(
                    fact.langevin_position, fact.access_count, age, fact.importance
                )
                new_state = self._langevin.get_lifecycle_state(weight)
                # Persist position + state
                self._db.update_fact(fact.fact_id, {
                    "langevin_position": new_pos,
                    "lifecycle": new_state,
                })
            else:
                new_state = self._time_based_state(fact)
                if new_state != old_state:
                    self._db.update_fact(fact.fact_id, {"lifecycle": new_state})

            counts[new_state.value] = counts.get(new_state.value, 0) + 1
            if new_state != old_state:
                transitions += 1

        counts["transitions"] = transitions
        logger.info("Lifecycle maintenance for '%s': %s", profile_id, counts)
        return counts

    def get_archived_facts(self, profile_id: str) -> list[AtomicFact]:
        """Get all archived facts (candidates for deletion/export)."""
        rows = self._db.execute(
            "SELECT * FROM atomic_facts WHERE profile_id = ? AND lifecycle = 'archived'",
            (profile_id,),
        )
        return [self._db._row_to_fact(r) for r in rows]

    # -- Internal ----------------------------------------------------------

    def _time_based_state(self, fact: AtomicFact) -> MemoryLifecycle:
        """Determine lifecycle state from time since creation + access count."""
        days = self._days_since(fact.created_at)
        if fact.access_count > 10 or days < _ACTIVE_MAX_DAYS:
            return MemoryLifecycle.ACTIVE
        if days < _WARM_MAX_DAYS:
            return MemoryLifecycle.WARM
        if days < _COLD_MAX_DAYS:
            return MemoryLifecycle.COLD
        return MemoryLifecycle.ARCHIVED

    @staticmethod
    def _days_since(iso_date: str) -> float:
        """Days since an ISO date string."""
        if not iso_date:
            return 0.0
        try:
            dt = datetime.fromisoformat(iso_date)
            return (datetime.now(UTC) - dt).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 0.0
