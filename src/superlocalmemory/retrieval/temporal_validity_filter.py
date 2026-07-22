# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Bi-temporal validity filter for the retrieval pipeline (Phase 4, T1).

Post-retrieval admission filter that removes *system-invalidated* facts — a
fact whose temporal record has ``system_expired_at`` set was superseded or
contradicted by a newer fact (see ``invalidate_fact_temporal`` /
conflict-resolution supersession). Such a fact is outdated and must never
surface in default retrieval.

The filter runs on the per-channel candidate dict BEFORE RRF fusion, so fused
ranks reflect only currently-valid facts. It queries validity only for the
bounded candidate set (never the full ``get_valid_facts`` set), so it adds an
indexed, O(candidates) lookup to the hot path — no full-table scan.

Pure SQL, no LLM → safe in every mode including Mode A. A no-op when nothing is
invalidated (the bounded query returns empty) and when config.enabled is False.

Integrates with ChannelRegistry.register_filter() using the FilterFn signature:
    (all_channel_results, profile_id, context) -> filtered_results

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import TemporalValidatorConfig
    from superlocalmemory.retrieval.channel_registry import ChannelRegistry
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class TemporalValidityFilter:
    """Drops system-invalidated (superseded) facts from retrieval candidates."""

    __slots__ = ("_db",)

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def filter(
        self,
        all_results: dict[str, list[tuple[str, float]]],
        profile_id: str,
        context: Any,
    ) -> dict[str, list[tuple[str, float]]]:
        """Remove superseded fact_ids from every channel's candidate list.

        Matches FilterFn signature from channel_registry.py.

        Args:
            all_results: Channel name -> [(fact_id, score)] dict.
            profile_id: Current profile.
            context: Optional context (unused).

        Returns:
            A new dict with system-invalidated facts removed. Inputs are never
            mutated (immutability). Unchanged when nothing is invalidated.
        """
        # Collect all unique candidate fact_ids across every channel.
        all_fact_ids: set[str] = set()
        for channel_results in all_results.values():
            for fact_id, _ in channel_results:
                all_fact_ids.add(fact_id)

        if not all_fact_ids:
            return all_results

        try:
            invalid = self._db.get_invalidated_fact_ids(
                list(all_fact_ids), profile_id,
            )
        except Exception as exc:
            # Fail-open: a validity-lookup error must never break retrieval.
            logger.warning("Temporal validity lookup failed: %s", exc)
            return all_results

        if not invalid:
            return all_results

        filtered: dict[str, list[tuple[str, float]]] = {}
        for channel_name, channel_results in all_results.items():
            filtered[channel_name] = [
                (fact_id, score)
                for fact_id, score in channel_results
                if fact_id not in invalid
            ]
        return filtered


def register_temporal_validity_filter(
    registry: ChannelRegistry,
    db: DatabaseManager,
    config: TemporalValidatorConfig,
) -> None:
    """Register the bi-temporal validity filter into the channel registry.

    Does nothing if config.enabled is False.

    Args:
        registry: Channel registry to register with.
        db: Database manager for validity queries.
        config: Temporal-validator configuration.
    """
    if not getattr(config, "enabled", True):
        return
    f = TemporalValidityFilter(db)
    registry.register_filter(f.filter)
