# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-03/H-06 fix

"""Parameterised JSON1-backed join helpers for ``action_outcomes``.

Replaces the fragile ``fact_ids_json LIKE '%"<fid>"%'`` pattern that five
call sites depended on. Substring matching on serialised JSON leaks
false positives across overlapping fact_id prefixes — see Stage-8
skeptic-H06 for the exact failure mode.

This module centralises the correct lookup:

    SELECT outcome_id, ... FROM action_outcomes
    WHERE profile_id = ?
      AND EXISTS (
          SELECT 1 FROM json_each(fact_ids_json) WHERE value = ?
      )

SQLite ships JSON1 enabled by default since 3.38 (February 2022) and the
minimum Python supported by SLM is 3.9 — which ships SQLite ≥ 3.31. We
defensively fall back to a ``LIKE`` probe only when JSON1 is missing at
runtime, with a one-off warning.

Callers:
  - ``learning/hnsw_dedup.py`` — ``apply_strong_memory_boost``,
    ``select_high_reward_fact_ids``, ``run_reward_gated_archive``.
  - ``learning/forgetting_scheduler.py`` — ``_has_recent_positive_reward``.

Contract refs:
  - Stage 8 H-03 (architect-H3) + H-06 (skeptic-H06).
  - LLD-12 §5 — reward-gated archive.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Iterable

logger = logging.getLogger(__name__)

__all__ = (
    "iter_outcomes_for_fact",
    "has_recent_positive_reward",
    "aggregate_reward_for_fact",
)


# Columns returned — mirror what the legacy LIKE callers read.
# NB: callers that need extra columns can pass ``columns=``.
_DEFAULT_COLUMNS = "outcome_id, profile_id, fact_ids_json, reward, settled_at"


def _json1_available(conn: sqlite3.Connection) -> bool:
    """Return True iff SQLite ``json_each`` is usable on ``conn``.

    Result is intentionally not cached across connections — JSON1 is a
    compile-time flag and runtime-swapping SQLite libraries is a rare
    edge case but we stay defensive.
    """
    try:
        conn.execute("SELECT value FROM json_each('[\"x\"]') LIMIT 1").fetchall()
        return True
    except sqlite3.OperationalError:
        return False


def iter_outcomes_for_fact(
    conn: sqlite3.Connection,
    profile_id: str,
    fact_id: str,
    *,
    columns: str = _DEFAULT_COLUMNS,
    extra_where: str = "",
    extra_params: tuple = (),
) -> Iterable[tuple]:
    """Yield action_outcomes rows whose fact_ids_json contains ``fact_id``.

    Scoped strictly to ``profile_id``; SQL parameters are always bound,
    never string-interpolated. Returns a materialised list so the caller
    can close the connection immediately.

    Args:
        conn: SQLite connection pointing at the database holding the
            ``action_outcomes`` table (usually ``memory.db``).
        profile_id: Profile scope.
        fact_id: Exact fact_id to find.
        columns: Comma-separated column list projected into the
            SELECT. Defaults to (outcome_id, profile_id, fact_ids_json,
            reward, settled_at).
        extra_where: Optional extra predicate — must start with 'AND'
            and use '?' placeholders. E.g.
            ``"AND reward IS NOT NULL AND reward > ?"``.
        extra_params: Bound parameters for ``extra_where``.

    Returns:
        List of sqlite3.Row-compatible tuples (or sqlite3.Row objects if
        the caller set ``conn.row_factory = sqlite3.Row``).
    """
    if not profile_id or not fact_id:
        return []

    if _json1_available(conn):
        sql = (
            f"SELECT {columns} FROM action_outcomes "
            f"WHERE profile_id = ? "
            f"  AND EXISTS ("
            f"    SELECT 1 FROM json_each(fact_ids_json) WHERE value = ?"
            f"  ) "
            f"{extra_where}"
        )
        params = (profile_id, fact_id, *extra_params)
    else:
        # Fallback: prefix-LIKE. Accurate ONLY for simple ids.
        # Logged once per process to flag that JSON1 is missing.
        _warn_fallback_once()
        sql = (
            f"SELECT {columns} FROM action_outcomes "
            f"WHERE profile_id = ? AND fact_ids_json LIKE ? "
            f"{extra_where}"
        )
        params = (profile_id, f'%"{fact_id}"%', *extra_params)

    cursor = conn.execute(sql, params)
    return cursor.fetchall()


def has_recent_positive_reward(
    conn: sqlite3.Connection,
    profile_id: str,
    fact_id: str,
    *,
    min_reward: float = 0.3,
    window_days: int = 60,
) -> bool:
    """True if ``fact_id`` has any outcome with reward > ``min_reward``
    settled in the last ``window_days`` days.
    """
    extra = (
        "AND reward IS NOT NULL AND reward > ? "
        f"AND COALESCE(settled_at, '') >= datetime('now', '-{int(window_days)} days') "
        "LIMIT 1"
    )
    rows = iter_outcomes_for_fact(
        conn, profile_id, fact_id,
        columns="1",
        extra_where=extra,
        extra_params=(float(min_reward),),
    )
    return bool(rows)


def aggregate_reward_for_fact(
    conn: sqlite3.Connection,
    profile_id: str,
    fact_id: str,
) -> tuple[int, float]:
    """Return ``(count, mean_reward)`` for a single fact_id.

    Count is the number of outcomes with reward IS NOT NULL; mean is
    ``AVG(reward)`` across that same subset. Returns ``(0, 0.0)`` when
    the fact has no outcomes.
    """
    if not profile_id or not fact_id:
        return 0, 0.0

    if _json1_available(conn):
        sql = (
            "SELECT COUNT(*), AVG(reward) FROM action_outcomes "
            "WHERE profile_id = ? "
            "  AND reward IS NOT NULL "
            "  AND EXISTS ("
            "    SELECT 1 FROM json_each(fact_ids_json) WHERE value = ?"
            "  )"
        )
        row = conn.execute(sql, (profile_id, fact_id)).fetchone()
    else:
        _warn_fallback_once()
        sql = (
            "SELECT COUNT(*), AVG(reward) FROM action_outcomes "
            "WHERE profile_id = ? "
            "  AND reward IS NOT NULL "
            "  AND fact_ids_json LIKE ?"
        )
        row = conn.execute(sql, (profile_id, f'%"{fact_id}"%')).fetchone()

    if row is None:
        return 0, 0.0
    count, mean = row
    return int(count or 0), float(mean or 0.0)


_FALLBACK_WARNED = False


def _warn_fallback_once() -> None:
    """Log the JSON1-missing fallback exactly once per process."""
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    logger.warning(
        "fact_outcome_joins: SQLite JSON1 unavailable — falling back to "
        "prefix-LIKE. Expect substring false positives on overlapping "
        "fact_id prefixes. Upgrade SQLite to ≥3.38 for correct matches.",
    )
