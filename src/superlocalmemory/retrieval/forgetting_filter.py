# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Forgetting filter for retrieval pipeline.

Post-retrieval filter that adjusts scores based on Ebbinghaus retention:
  - Active/warm facts: score weighted by lifecycle_weight
  - Cold facts: score reduced (weight = 0.3)
  - Archive/forgotten facts: REMOVED from results entirely

Integrates with ChannelRegistry.register_filter() using the FilterFn
signature: (all_results, profile_id, context) -> filtered_results.

HR-06: When config.enabled=False, returns results unchanged.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from superlocalmemory.core.config import ForgettingConfig

if TYPE_CHECKING:
    from superlocalmemory.retrieval.channel_registry import ChannelRegistry
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


# Lifecycle zone weights — same as in ebbinghaus.py
_ZONE_WEIGHTS: dict[str, float] = {
    "active": 1.0,
    "warm": 0.7,
    "cold": 0.3,
    "archive": 0.0,
    "forgotten": 0.0,
}

# V3.4.11: Deep recall weights — includes cold/archive with reduced scores
_DEEP_ZONE_WEIGHTS: dict[str, float] = {
    "active": 1.0,
    "warm": 0.7,
    "cold": 0.3,
    "archive": 0.15,
    "forgotten": 0.05,
}

# Zones where facts are excluded from results (default recall)
_EXCLUDED_ZONES: frozenset[str] = frozenset({"archive", "forgotten"})

# Deep recall excludes nothing — every fact is searchable
_DEEP_EXCLUDED_ZONES: frozenset[str] = frozenset()


class ForgettingFilter:
    """Post-retrieval filter that applies Ebbinghaus retention weighting.

    Removes archived/forgotten facts and adjusts scores for other zones.
    """

    __slots__ = ("_db", "_config", "_deep_recall")

    def __init__(self, db: DatabaseManager, config: ForgettingConfig, deep_recall: bool = False) -> None:
        self._db = db
        self._config = config
        self._deep_recall = deep_recall

    def filter(
        self,
        all_results: dict[str, list[tuple[str, float]]],
        profile_id: str,
        context: Any,
    ) -> dict[str, list[tuple[str, float]]]:
        """Apply forgetting filter to retrieval results.

        Matches FilterFn signature from channel_registry.py.

        Args:
            all_results: Channel name -> [(fact_id, score)] dict.
            profile_id: Current profile.
            context: Optional context (unused).

        Returns:
            Filtered results dict with scores adjusted by retention weight.
        """
        # HR-06: If disabled, return unchanged
        if not self._config.enabled:
            return all_results

        # Collect all unique fact_ids across all channels
        all_fact_ids: set[str] = set()
        for channel_results in all_results.values():
            for fact_id, _ in channel_results:
                all_fact_ids.add(fact_id)

        if not all_fact_ids:
            return all_results

        # Batch query retention data
        retention_rows = self._db.batch_get_retention(
            list(all_fact_ids), profile_id,
        )

        # Build lookup: fact_id -> retention row
        retention_map: dict[str, dict] = {}
        for row in retention_rows:
            retention_map[row["fact_id"]] = row

        # Filter and weight each channel's results
        filtered: dict[str, list[tuple[str, float]]] = {}
        for channel_name, channel_results in all_results.items():
            new_results: list[tuple[str, float]] = []
            for fact_id, score in channel_results:
                ret_data = retention_map.get(fact_id)

                if ret_data is None:
                    # No retention data yet -> new memory, keep as-is
                    new_results.append((fact_id, score))
                    continue

                zone = ret_data.get("lifecycle_zone", "active")

                # V3.4.11: Deep recall mode includes all tiers
                excluded = _DEEP_EXCLUDED_ZONES if self._deep_recall else _EXCLUDED_ZONES
                weights = _DEEP_ZONE_WEIGHTS if self._deep_recall else _ZONE_WEIGHTS

                if zone in excluded:
                    continue

                weight = weights.get(zone, 1.0)
                new_results.append((fact_id, score * weight))

            filtered[channel_name] = new_results

        return filtered


def register_forgetting_filter(
    registry: ChannelRegistry,
    db: DatabaseManager,
    config: ForgettingConfig,
) -> None:
    """Register the forgetting filter with the channel registry.

    Does nothing if config.enabled is False.

    Args:
        registry: Channel registry to register with.
        db: Database manager for retention queries.
        config: Forgetting configuration.
    """
    if not config.enabled:
        return
    f = ForgettingFilter(db, config)
    registry.register_filter(f.filter)
