# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-03/H-17/H-18 shim

"""HNSW dedup + reward-gated archive + strong-memory boost — shim.

As of v3.4.22 (Stage 8 H-03/H-17/H-18 fixes), the 535-LOC god-module
was split into three cohesive files:

  - ``dedup_hnsw.py``     — :class:`HnswDeduplicator` + fallback counter.
  - ``reward_archive.py`` — :func:`run_reward_gated_archive`.
  - ``reward_boost.py``   — :func:`apply_strong_memory_boost`,
                            :func:`select_high_reward_fact_ids`.

Outcome lookups that used to issue ``fact_ids_json LIKE`` now go
through :mod:`superlocalmemory.learning.fact_outcome_joins` which wraps
SQLite JSON1 so overlapping fact_id prefixes cannot collide (H-06).

This shim re-exports the original surface so that existing imports
continue to work unchanged.
"""

from __future__ import annotations

import logging

from superlocalmemory.core.ram_lock import ram_reservation  # noqa: F401

from superlocalmemory.learning.dedup_hnsw import (  # noqa: F401
    HnswDeduplicator,
    get_hnsw_degraded_count,
    reset_hnsw_degraded_count,
    _cosine,
    _jaccard,
    _parse_embedding,
    _pick_canonical,
)
from superlocalmemory.learning.reward_archive import (  # noqa: F401
    ARCHIVE_REWARD_THRESHOLD,
    REWARD_WINDOW_DAYS,
    run_reward_gated_archive,
)
from superlocalmemory.learning.reward_boost import (  # noqa: F401
    STRONG_BOOST_CAP,
    STRONG_BOOST_INCREMENT,
    STRONG_BOOST_MIN_MEAN,
    STRONG_BOOST_MIN_OUTCOMES,
    apply_strong_memory_boost,
    select_high_reward_fact_ids,
)

logger = logging.getLogger(__name__)


__all__ = (
    "HnswDeduplicator",
    "run_reward_gated_archive",
    "apply_strong_memory_boost",
    "select_high_reward_fact_ids",
    "get_hnsw_degraded_count",
    "reset_hnsw_degraded_count",
    "REWARD_WINDOW_DAYS",
    "ARCHIVE_REWARD_THRESHOLD",
    "STRONG_BOOST_INCREMENT",
    "STRONG_BOOST_CAP",
    "STRONG_BOOST_MIN_OUTCOMES",
    "STRONG_BOOST_MIN_MEAN",
    "ram_reservation",
)
