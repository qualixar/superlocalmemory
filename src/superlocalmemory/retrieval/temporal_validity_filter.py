# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Bi-temporal validity filter for the retrieval pipeline (Phase 4, T1 + T1b).

Post-retrieval filter that demotes facts whose bi-temporal validity window
makes them ineligible for the current recall point in time. Two independent
demotion axes:

**Axis 1 — Transaction-time supersession (P5-INT-01, existing):**
A fact whose temporal record has ``system_expired_at`` set was superseded or
contradicted by a newer fact (see ``invalidate_fact_temporal`` /
conflict-resolution supersession). Its per-channel score is multiplied by
``superseded_demotion_factor`` (default 0.25).

**Axis 2 — Event-time expiry (Phase 4 T1b, new):**
A fact whose ``valid_until`` has passed (in wall-clock time) was true in the
real world only up to that date. Its per-channel score is multiplied by
``event_time_demotion_factor`` (default 0.5, softer than supersession). A
separate "not-yet-valid" demotion applies when ``as_of`` is set for
point-in-time time-travel recall: facts whose ``valid_from > as_of`` are
also demoted.

**Zero-regression guarantee (default path):**
Almost all existing facts have ``valid_until = NULL`` (open-ended / still
valid). The event-time expiry lookup therefore returns an empty set on the
default path, so no scores change and default recall output is identical.
Additionally, when ``include_expired_in_history = True`` (the config default)
AND no explicit ``as_of`` is requested, event-time demotion is skipped
entirely — the guard ensures historical recall modes that intentionally want
to surface expired facts are unaffected.

**As-of time-travel:**
When ``as_of`` is packed into the filter context dict (``{"as_of": "..."}``),
event-time demotion is always applied regardless of
``include_expired_in_history``, because the caller explicitly requested a
point-in-time view. Facts not yet valid at ``as_of`` (``valid_from > as_of``)
and facts already expired at ``as_of`` (``valid_until <= as_of``, half-open) are
both demoted.

**Demotion priority:**
- System-invalidated facts: score × ``superseded_demotion_factor`` (0.25).
- Event-time-expired only (not system-invalidated): score × ``event_time_demotion_factor`` (0.5).
- Both: system-invalidated wins (the harder demotion, 0.25, is applied; event-time
  demotion is not stacked to avoid excessive double-penalty).

All demotions are non-destructive (P5-INT-01): facts stay in the candidate
list but rank below valid facts. A factor of 0.0 restores the legacy hide
behaviour (a score of zero is gated out by the evidence floor).

Both lookups are bounded (candidate ids only), chunked, indexed, and
fail-open: a DB error returns results unchanged.

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

# Module-level fallback for event-time demotion factor.
# Used when the config object does not yet carry ``event_time_demotion_factor``
# (e.g. older serialised configs loaded from disk). Softer than the supersession
# factor (0.25) because event-time expiry is a softer signal — the boundary may
# be approximate or the fact may still be conceptually relevant.
_EVENT_TIME_DEMOTION_FACTOR: float = 0.5


class TemporalValidityFilter:
    """Demotes bi-temporally invalid facts in retrieval candidates.

    Two demotion axes:
    - Axis 1: system-invalidated (superseded) facts — existing behaviour.
    - Axis 2: event-time-expired facts — new in Phase 4 T1b.

    Both are non-destructive: facts stay recallable but rank below valid ones.
    """

    __slots__ = (
        "_db",
        "_demotion_factor",
        "_event_time_factor",
        "_include_expired_in_history",
    )

    def __init__(
        self,
        db: DatabaseManager,
        demotion_factor: float = 0.25,
        event_time_factor: float = _EVENT_TIME_DEMOTION_FACTOR,
        include_expired_in_history: bool = True,
    ) -> None:
        self._db = db
        # Clamp to [0, 1]. 0.0 = legacy hide; 1.0 = no demotion.
        self._demotion_factor = max(0.0, min(1.0, float(demotion_factor)))
        self._event_time_factor = max(0.0, min(1.0, float(event_time_factor)))
        self._include_expired_in_history = bool(include_expired_in_history)

    def filter(
        self,
        all_results: dict[str, list[tuple[str, float]]],
        profile_id: str,
        context: Any,
    ) -> dict[str, list[tuple[str, float]]]:
        """Demote bi-temporally invalid fact_ids in every channel's candidate list.

        Matches FilterFn signature from channel_registry.py.

        Args:
            all_results: Channel name -> [(fact_id, score)] dict.
            profile_id: Current profile.
            context: Optional dict that may carry ``as_of`` (ISO 8601 string)
                for point-in-time time-travel recall. None or any non-dict value
                means "current time, no time-travel" — the default path.

        Returns:
            A new dict where invalid facts keep their channel presence but have
            their score scaled by the appropriate demotion factor and the channel
            lists re-sorted so valid facts rank above them. Inputs are never
            mutated (immutability contract). Returns the input unchanged when
            nothing is invalid (fast path).
        """
        # Collect all unique candidate fact_ids across every channel.
        all_fact_ids: set[str] = set()
        for channel_results in all_results.values():
            for fact_id, _ in channel_results:
                all_fact_ids.add(fact_id)

        if not all_fact_ids:
            return all_results

        # Extract as_of FIRST — needed for both Axis 1 and Axis 2. Must happen
        # before any DB call so as_of is available for transaction-time gating.
        # None context = legacy/default; non-dict = upstream caller without
        # as_of support. Both treated as "no time-travel, current time".
        as_of: str | None = (
            context.get("as_of") if isinstance(context, dict) else None
        )
        # UTC-normalize at the filter boundary to guarantee format consistency
        # before forwarding to DB comparisons.
        if as_of is not None:
            from superlocalmemory.retrieval.temporal_utils import normalize_as_of
            as_of = normalize_as_of(as_of)
            # normalize_as_of returns None on invalid input; treat as no as_of.

        # --- Axis 1: Transaction-time supersession ---
        # When as_of is set: only supersessions that occurred AT OR BEFORE
        # as_of contribute (Phase 4b bi-temporal fix). Supersessions after
        # as_of are invisible — the fact was still valid at the query point.
        try:
            invalid = self._db.get_invalidated_fact_ids(
                list(all_fact_ids), profile_id, as_of=as_of,
            )
        except Exception as exc:
            # Fail-open: a validity-lookup error must never break retrieval.
            logger.warning("Temporal validity lookup failed: %s", exc)
            return all_results

        # --- Axis 2: Event-time expiry (Phase 4 T1b) ---
        # Guard: skip event-time demotion when the caller signals it wants
        # historical / all-expired-included mode AND no explicit as_of point
        # is requested.  Config default is include_expired_in_history=True,
        # so the default recall path always takes this skip branch — the two
        # new DB calls are never made, and no scores change.
        apply_event_time = (as_of is not None) or (not self._include_expired_in_history)

        event_expired: set[str] = set()
        if apply_event_time:
            try:
                event_expired = self._db.get_event_time_expired_fact_ids(
                    list(all_fact_ids), profile_id, as_of=as_of,
                )
            except Exception as exc:
                # Fail-open: continue with empty event_expired set.
                logger.warning("Event-time expiry lookup failed: %s", exc)

        # Fast path: nothing to demote — return input unchanged.
        if not invalid and not event_expired:
            return all_results

        system_factor = self._demotion_factor
        event_factor = self._event_time_factor
        # event_time_only: expired by event-time but NOT system-invalidated.
        # System-invalid already carries the harder demotion (0.25 < 0.5).
        # We do not stack both factors to avoid excessive double-penalty.
        event_time_only = event_expired - invalid

        demoted: dict[str, list[tuple[str, float]]] = {}
        for channel_name, channel_results in all_results.items():
            new_list = [
                (fact_id,
                 score * system_factor if fact_id in invalid
                 else score * event_factor if fact_id in event_time_only
                 else score)
                for fact_id, score in channel_results
            ]
            # Re-sort descending so demoted facts fall below currently-valid
            # facts in this channel's rank order.
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

    Reads ``event_time_demotion_factor`` from the config with a fallback to the
    module constant ``_EVENT_TIME_DEMOTION_FACTOR`` (0.5) so older serialised
    configs without this field work without a migration.

    Args:
        registry: Channel registry to register with.
        db: Database manager for validity queries.
        config: Temporal-validator configuration.
    """
    if not getattr(config, "enabled", True):
        return
    factor = getattr(config, "superseded_demotion_factor", 0.25)
    event_time_factor = getattr(
        config, "event_time_demotion_factor", _EVENT_TIME_DEMOTION_FACTOR,
    )
    include_expired_in_history = getattr(config, "include_expired_in_history", True)
    f = TemporalValidityFilter(
        db,
        demotion_factor=factor,
        event_time_factor=event_time_factor,
        include_expired_in_history=include_expired_in_history,
    )
    registry.register_filter(f.filter)
