# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Bi-temporal validity filter for the retrieval pipeline (Phase 4, T1).

Post-retrieval filter for *system-invalidated* facts — a fact whose temporal
record has ``system_expired_at`` set was superseded/contradicted by a newer
fact (see ``invalidate_fact_temporal`` / conflict-resolution supersession).

P5-INT-01 (non-destructive supersession): such a fact is DEMOTED, not hidden.
Its per-channel score is multiplied by ``superseded_demotion_factor`` (default
0.25) and the channel lists are re-sorted, so currently-valid facts rank above
it — but nothing valid silently vanishes. This is the Mem0-2026 design that
wins long-term-memory benchmarks: keep every fact recallable and let
retrieval-time recency resolve conflicts, rather than destructively deleting on
a write-time contradiction guess (which over-fires: two complementary facts
about the same entity diverge past the coboundary threshold and one would be
wrongly hidden). A factor of 0.0 restores the legacy hide behaviour (a demoted
score of 0 is gated out by the evidence floor).

The filter runs on the per-channel candidate dict BEFORE RRF fusion, so fused
ranks reflect the demotion. It queries validity only for the bounded candidate
set (never the full ``get_valid_facts`` set) — an indexed, O(candidates) lookup
on the hot path, no full-table scan.

Pure SQL, no LLM → safe in every mode including Mode A. A no-op when nothing is
invalidated and when config.enabled is False.

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
    """Demotes system-invalidated (superseded) facts in retrieval candidates."""

    __slots__ = ("_db", "_demotion_factor")

    def __init__(self, db: DatabaseManager, demotion_factor: float = 0.25) -> None:
        self._db = db
        # Clamp to [0, 1]. 0.0 = legacy hide (evidence floor drops zero-score
        # facts); 1.0 = no demotion.
        self._demotion_factor = max(0.0, min(1.0, float(demotion_factor)))

    def filter(
        self,
        all_results: dict[str, list[tuple[str, float]]],
        profile_id: str,
        context: Any,
    ) -> dict[str, list[tuple[str, float]]]:
        """Demote superseded fact_ids in every channel's candidate list.

        Matches FilterFn signature from channel_registry.py.

        Args:
            all_results: Channel name -> [(fact_id, score)] dict.
            profile_id: Current profile.
            context: Optional context (unused).

        Returns:
            A new dict where system-invalidated facts keep their channel
            presence but have their score scaled by the demotion factor and the
            channel lists re-sorted (so valid facts rank above them). Inputs are
            never mutated (immutability). Unchanged when nothing is invalidated.
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

        factor = self._demotion_factor
        demoted: dict[str, list[tuple[str, float]]] = {}
        for channel_name, channel_results in all_results.items():
            new_list = [
                (fact_id, score * factor if fact_id in invalid else score)
                for fact_id, score in channel_results
            ]
            # Re-sort descending so demoted (superseded) facts fall below
            # currently-valid facts in this channel's rank order.
            new_list.sort(key=lambda pair: pair[1], reverse=True)
            demoted[channel_name] = new_list
        return demoted


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
    factor = getattr(config, "superseded_demotion_factor", 0.25)
    f = TemporalValidityFilter(db, demotion_factor=factor)
    registry.register_filter(f.filter)
