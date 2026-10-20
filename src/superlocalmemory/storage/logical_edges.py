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
    WHERE profile_id = ?{scope}
      AND NOT EXISTS (
            SELECT 1 FROM atomic_facts f
            WHERE f.fact_id IN (graph_edges.source_id, graph_edges.target_id)
              AND NOT ({visible})
      )
    GROUP BY profile_id, source_id, target_id, COALESCE(edge_type, 'related')
"""


def _edge_select(conn: sqlite3.Connection) -> str:
    """The logical-edge query, with the withheld-endpoint exclusion resolved.

    WHY AN EDGE WITH A WITHHELD ENDPOINT IS NOT A LOGICAL EDGE
    ----------------------------------------------------------
    The retrieval channel this projection stands in for does not traverse them.
    It loads ``graph_edges`` by scope and then prunes the result: "Edge scope
    alone cannot authorize an endpoint. Prune both endpoints against the visible
    fact corpus so denied facts cannot influence an allowed candidate indirectly
    through propagation." Its entity map is filtered the same way, for a reason
    it spells out — a withheld row carries its whole cluster's pooled entity
    list, so it out-ranks real memories and then gets discarded at hydration,
    spending the channel's budget on nothing.

    The export predated that fix and kept the withheld endpoints. Measured on a
    copy of the author's store: Cozo's bridge held 1,257 facts the store may not
    return and its edges touched 805, and the graph search diverged from SQLite
    on **every** query — three shadow checks, three mismatches. One query
    returned 9 results against SQLite's 20, because withheld facts had taken the
    top-k budget. The projection failed closed every time, so recall was correct
    and the projection was dead weight.

    The predicate is resolved against the passed connection because
    ``archive_status`` and ``quarantined`` each arrive with a migration and may
    be absent on a store the engine has not opened.
    """
    from superlocalmemory.storage.database import visible_fact_clause_for_connection

    # The helper returns a leading-AND clause for appending; here it is needed as
    # a standalone predicate, so strip the connective and default to "always
    # visible" on a store that has neither column yet.
    return _LOGICAL_EDGE_SELECT.format(
        visible=_standalone(visible_fact_clause_for_connection(conn, prefix="f")), scope="",
    )


def _standalone(clause: str) -> str:
    """A leading-AND visibility clause as a standalone predicate ("always visible" if empty)."""
    clause = clause.strip()
    predicate = clause[4:].strip() if clause.upper().startswith("AND ") else clause
    return predicate or "1=1"


def fact_edge_select(visible_clause: str) -> str:
    """The logical edges touching one fact, by the same definition as ``iter_logical_edges``.

    Params: ``(profile_id, fact_id, fact_id)``. ``visible_clause`` is
    ``DatabaseManager.visible_fact_clause(prefix="f")``. The projection drain re-projects a fact
    from this, so an incrementally kept graph holds exactly what a fresh bulk import would: an
    edge to a withheld fact stays out even when its visible neighbour is re-projected (#296).
    """
    return _LOGICAL_EDGE_SELECT.format(
        visible=_standalone(visible_clause), scope=" AND (source_id = ? OR target_id = ?)",
    )


def iter_logical_edges(
    conn: sqlite3.Connection, profile_id: str
) -> Iterator[tuple[Any, ...]]:
    """Yield normalized graph edges in deterministic fingerprint order."""
    return iter(
        conn.execute(
            _edge_select(conn) + " ORDER BY source_id, target_id, edge_type",
            (profile_id,),
        )
    )


def count_logical_edges(conn: sqlite3.Connection, profile_id: str) -> int:
    """Count relationships using the canonical logical identity."""
    row = conn.execute(
        "SELECT COUNT(*) FROM (" + _edge_select(conn) + ")",
        (profile_id,),
    ).fetchone()
    return int(row[0] if row else 0)
