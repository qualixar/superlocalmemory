"""Canonical logical-edge projection shared by scale backends.

Legacy databases can contain multiple physical ``graph_edges`` rows for one
logical relationship. Current writes define identity as profile, source,
target, and edge type, retaining the strongest weight. Derived projections
must use that same contract without rewriting canonical SQLite history.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

_LOGICAL_EDGE_SELECT = """
    SELECT
        source_id,
        target_id,
        COALESCE(edge_type, 'related') AS edge_type,
        MAX(COALESCE(weight, 1.0)) AS weight,
        profile_id
    FROM graph_edges
    WHERE profile_id = ?
    GROUP BY profile_id, source_id, target_id, COALESCE(edge_type, 'related')
"""


def iter_logical_edges(
    conn: sqlite3.Connection, profile_id: str
) -> Iterator[tuple[Any, ...]]:
    """Yield normalized graph edges in deterministic fingerprint order."""
    return iter(
        conn.execute(
            _LOGICAL_EDGE_SELECT + " ORDER BY source_id, target_id, edge_type",
            (profile_id,),
        )
    )


def count_logical_edges(conn: sqlite3.Connection, profile_id: str) -> int:
    """Count relationships using the canonical logical identity."""
    row = conn.execute(
        "SELECT COUNT(*) FROM (" + _LOGICAL_EDGE_SELECT + ")",
        (profile_id,),
    ).fetchone()
    return int(row[0] if row else 0)
