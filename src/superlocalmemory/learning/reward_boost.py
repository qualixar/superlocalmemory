# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-06/H-18 fix

"""Strong-memory boost + reward-aware fact selection.

Nudges ``atomic_facts.retrieval_prior`` upward for facts with recurring
high reward, capped at 0.5 (LLD-12 §5). Also exposes
``select_high_reward_fact_ids`` for the soft-prompt generator.

H-06 regression fix: outcome lookups now use the JSON1-backed
``fact_outcome_joins`` helper instead of the fragile
``fact_ids_json LIKE '%"<fid>"%'`` pattern that leaked substring matches
across overlapping fact_id prefixes.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from superlocalmemory.learning.fact_outcome_joins import (
    aggregate_reward_for_fact,
)


# H-12/H-P-01: single-pass JSON1 aggregation across ALL facts for a profile.
# Returns ``{fact_id: (count, mean_reward)}`` for outcomes with reward NOT
# NULL. Replaces the per-fact O(F) loop of ``aggregate_reward_for_fact``
# with one GROUP BY scan — O(F+O) instead of O(F·O). JSON1 has been
# mandatory since v3.4.22 (see ``fact_outcome_joins._json1_available``
# contract in the module docstring); if it is missing at runtime the
# caller falls back to the per-fact helper which retains its own
# LIKE-based shim.


# S9-W3 H-PERF-03: consolidation invokes ``apply_strong_memory_boost``
# AND ``select_high_reward_fact_ids`` in the same cycle. Both call
# ``_bulk_fact_reward_stats`` which is a full GROUP BY scan — at 100k
# outcomes that's 1-3 s × 2 = 2-6 s wasted inside the 5-min cap. We
# memoise the result with a short TTL so consecutive calls within the
# same consolidation cycle share the stats. Key is (id(conn),
# profile_id) so different conns / profiles get independent caches.
# TTL expires quickly so live recall updates are reflected within one
# cycle of the consolidation loop.
_BULK_STATS_TTL_SEC: float = 30.0
_bulk_stats_cache: dict[tuple[int, str], tuple[float, dict[str, tuple[int, float]]]] = {}


# S9-W3 M-PERF-07: module-level MISS constant so ``stats.get(fid, _MISS)``
# does not allocate a fresh ``(0, 0.0)`` tuple per call. At 100k facts
# that saved ~2 MB of short-lived garbage per consolidation cycle.
_MISS: tuple[int, float] = (0, 0.0)


def _bulk_fact_reward_stats(
    conn: sqlite3.Connection, profile_id: str,
) -> dict[str, tuple[int, float]]:
    import time as _time
    now = _time.monotonic()
    key = (id(conn), profile_id)
    cached = _bulk_stats_cache.get(key)
    if cached is not None:
        ts, result = cached
        if now - ts < _BULK_STATS_TTL_SEC:
            return result
    try:
        rows = conn.execute(
            "SELECT j.value AS fact_id, "
            "       COUNT(*) AS c, "
            "       AVG(reward) AS m "
            "FROM action_outcomes a, json_each(a.fact_ids_json) j "
            "WHERE a.profile_id = ? AND a.reward IS NOT NULL "
            "GROUP BY j.value",
            (profile_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        # JSON1 missing — signal to caller to fall back. Do NOT cache
        # this (empty) result: a subsequent call on the same conn may
        # fall through to the per-fact loop intentionally.
        return {}
    out: dict[str, tuple[int, float]] = {}
    for fid, c, m in rows:
        if fid is None:
            continue
        out[str(fid)] = (int(c or 0), float(m or 0.0))
    _bulk_stats_cache[key] = (now, out)
    # Prune the cache if it grows beyond 64 entries (multi-profile envs).
    if len(_bulk_stats_cache) > 64:
        stale = [k for k, (t, _) in _bulk_stats_cache.items()
                 if now - t >= _BULK_STATS_TTL_SEC]
        for k in stale:
            _bulk_stats_cache.pop(k, None)
    return out

logger = logging.getLogger(__name__)


STRONG_BOOST_INCREMENT: float = 0.1
STRONG_BOOST_CAP: float = 0.5
STRONG_BOOST_MIN_OUTCOMES: int = 3
STRONG_BOOST_MIN_MEAN: float = 0.7


__all__ = (
    "apply_strong_memory_boost",
    "select_high_reward_fact_ids",
    "STRONG_BOOST_INCREMENT",
    "STRONG_BOOST_CAP",
    "STRONG_BOOST_MIN_OUTCOMES",
    "STRONG_BOOST_MIN_MEAN",
)


def apply_strong_memory_boost(
    memory_db_path: str | Path, profile_id: str,
) -> int:
    """Nudge retrieval_prior up for high-reward facts, capped at 0.5.

    Eligibility: ≥ MIN_OUTCOMES outcomes with mean reward > MIN_MEAN.
    Effect: retrieval_prior = MIN(retrieval_prior + INCREMENT, CAP).

    Returns number of rows boosted.
    """
    conn = sqlite3.connect(str(memory_db_path), timeout=10.0)
    conn.execute("PRAGMA busy_timeout=2000")
    boosted = 0
    try:
        rows = conn.execute(
            "SELECT fact_id FROM atomic_facts WHERE profile_id=? "
            "  AND (archive_status IS NULL OR archive_status='live')",
            (profile_id,),
        ).fetchall()
        if not rows:
            return 0

        # H-12/H-P-01: single JSON1 GROUP BY replaces the per-fact loop.
        # Fallback to per-fact helper preserves legacy behaviour on
        # SQLite without JSON1.
        stats = _bulk_fact_reward_stats(conn, profile_id)
        conn.execute("BEGIN IMMEDIATE")
        for (fid,) in rows:
            if stats:
                count, mean = stats.get(fid, _MISS)
            else:
                count, mean = aggregate_reward_for_fact(conn, profile_id, fid)
            if count < STRONG_BOOST_MIN_OUTCOMES:
                continue
            if mean <= STRONG_BOOST_MIN_MEAN:
                continue
            conn.execute(
                "UPDATE atomic_facts "
                "SET retrieval_prior = MIN(COALESCE(retrieval_prior, 0) + ?, ?) "
                "WHERE fact_id=?",
                (STRONG_BOOST_INCREMENT, STRONG_BOOST_CAP, fid),
            )
            boosted += 1
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.warning("apply_strong_memory_boost rollback: %s", exc)
    finally:
        conn.close()
    return boosted


def select_high_reward_fact_ids(
    memory_db_path: str | Path,
    profile_id: str,
    *,
    min_reward: float = 0.6,
    min_outcomes: int = 1,
) -> list[str]:
    """Return fact_ids whose mean outcome reward ≥ ``min_reward``.

    Used by ``soft_prompt_generator`` to mine only high-reward facts
    (LLD-12 §6). JSON1-backed — no substring false positives.
    """
    conn = sqlite3.connect(str(memory_db_path), timeout=10.0)
    try:
        fact_rows = conn.execute(
            "SELECT fact_id FROM atomic_facts WHERE profile_id=? "
            "  AND (archive_status IS NULL OR archive_status='live')",
            (profile_id,),
        ).fetchall()
        # H-12/H-P-01: bulk aggregate replaces per-fact loop.
        stats = _bulk_fact_reward_stats(conn, profile_id)
        out: list[str] = []
        for (fid,) in fact_rows:
            if stats:
                count, mean = stats.get(fid, _MISS)
            else:
                count, mean = aggregate_reward_for_fact(conn, profile_id, fid)
            if count < min_outcomes:
                continue
            if mean >= min_reward:
                out.append(fid)
        return out
    finally:
        conn.close()
