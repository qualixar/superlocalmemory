# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tell the owner, in their own words, whether their memory works.

Until now the only way to learn that 43.7% of a store could not be found by
asking a question was to write the SQL yourself. One machine sat in exactly
that state for months while every status line it showed said the system was
healthy, because nothing measured reachability and nothing reported it.

So this module answers four questions a non-engineer can act on:

  * How many memories do I have?
  * How many can actually be found by asking a question?
  * How many were withheld because a model wrote them, not me?
  * Is anything still being repaired?

Read-only, and every query is bounded. Fail-soft by construction: a missing
table or column yields ``None`` for that line rather than an exception, because
a health report that crashes on an old store is worse than one that says "not
known yet".

Consumed by ``slm doctor``, ``GET /api/v3/memory-health``, and the dashboard.
One implementation so the three cannot disagree with each other.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryHealth", "RefileReport", "measure", "describe", "classify",
    "refile_hidden",
]


@dataclass(frozen=True)
class MemoryHealth:
    """A store's answer-ability, counted rather than assumed."""

    #: Memories that recall is allowed to return.
    live_facts: int = 0
    #: Of those, how many have a vector projection, i.e. can be found by
    #: meaning rather than only by matching words.
    findable_by_meaning: int = 0
    #: Memories with no vector at all. These are reachable by keyword only.
    missing_vector: int = 0
    #: Machine-written summaries withheld from recall and kept for display.
    withheld_summaries: int = 0
    #: Summaries preserved in the display table.
    display_summaries: int = 0
    #: Memories hidden by the retention system, excluding the withheld ones.
    hidden_by_forgetting: int = 0
    #: Rows whose retention zone contradicts their retention score, i.e. hidden
    #: while the maths says to keep them. Should be zero after repair.
    inconsistently_hidden: int = 0
    #: Present only when a table or column was absent.
    unavailable: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reachability(self) -> float:
        """Share of live memories findable by meaning, 0.0-1.0."""
        if self.live_facts <= 0:
            return 1.0
        return self.findable_by_meaning / self.live_facts

    @property
    def healthy(self) -> bool:
        """Whether anything here warrants telling the owner about."""
        return (
            self.reachability >= 0.99
            and self.missing_vector == 0
            and self.inconsistently_hidden == 0
        )


def measure(db_path: str | Path) -> MemoryHealth:
    """Count the store's answer-ability. Read-only; never raises."""
    unavailable: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.debug("memory health: cannot open %s: %s", db_path, exc)
        return MemoryHealth(unavailable=("database",))

    try:
        if not _table_exists(conn, "atomic_facts"):
            return MemoryHealth(unavailable=("atomic_facts",))

        # Quarantine came in 4.0.10. On an older store every fact is "live",
        # which is the honest reading of a store that has no withheld rows.
        has_q = _has_column(conn, "atomic_facts", "quarantined")
        if not has_q:
            unavailable.append("quarantined")
        live_clause = "COALESCE(quarantined, 0) = 0" if has_q else "1=1"

        live = _count(conn, f"SELECT COUNT(*) FROM atomic_facts WHERE {live_clause}")
        withheld = (
            _count(conn, "SELECT COUNT(*) FROM atomic_facts WHERE quarantined = 1")
            if has_q else 0
        )
        missing_vec = _count(
            conn,
            f"SELECT COUNT(*) FROM atomic_facts "
            f"WHERE embedding IS NULL AND {live_clause}",
        )

        if _table_exists(conn, "embedding_metadata"):
            findable = _count(
                conn,
                "SELECT COUNT(*) FROM embedding_metadata em "
                "JOIN atomic_facts af ON af.fact_id = em.fact_id "
                f"WHERE {_prefixed(live_clause, 'af')}",
            )
        else:
            unavailable.append("embedding_metadata")
            findable = 0

        display = (
            _count(conn, "SELECT COUNT(*) FROM consolidated_summaries")
            if _table_exists(conn, "consolidated_summaries") else 0
        )
        if not _table_exists(conn, "consolidated_summaries"):
            unavailable.append("consolidated_summaries")

        hidden = inconsistent = 0
        if _table_exists(conn, "fact_retention"):
            hidden = _count(
                conn,
                "SELECT COUNT(*) FROM fact_retention r "
                "JOIN atomic_facts af ON af.fact_id = r.fact_id "
                "WHERE r.lifecycle_zone IN ('archive', 'forgotten') "
                f"  AND {_prefixed(live_clause, 'af')}",
            )
            # The contradiction M043 repairs: hidden, yet scored to keep.
            inconsistent = _count(
                conn,
                "SELECT COUNT(*) " + _inconsistent_from(live_clause),
            )
        else:
            unavailable.append("fact_retention")

        return MemoryHealth(
            live_facts=live,
            findable_by_meaning=findable,
            missing_vector=missing_vec,
            withheld_summaries=withheld,
            display_summaries=display,
            hidden_by_forgetting=hidden,
            inconsistently_hidden=inconsistent,
            unavailable=tuple(unavailable),
        )
    except sqlite3.Error as exc:
        logger.debug("memory health measurement failed: %s", exc)
        return MemoryHealth(unavailable=(*unavailable, "query_failed"))
    finally:
        conn.close()


def describe(health: MemoryHealth) -> list[str]:
    """Plain-language lines for a reader who does not write SQL.

    No percentages without the counts behind them, and no jargon: "findable by
    asking a question" rather than "vector coverage", because the person who
    needs this line is the one who would not know what a vector is.
    """
    lines: list[str] = []
    if "atomic_facts" in health.unavailable or "database" in health.unavailable:
        return ["Memory store not readable yet."]

    lines.append(f"You have {health.live_facts:,} memories.")

    if "embedding_metadata" in health.unavailable:
        lines.append(
            "Whether they can be found by asking a question is not known yet — "
            "the search index has not been built."
        )
    elif health.live_facts:
        pct = 100.0 * health.reachability
        if health.findable_by_meaning >= health.live_facts:
            # "All" only when the counts actually agree. The threshold used to
            # be reachability >= 0.99, which printed "All of them can be found
            # by asking a question (5,199 indexed)" on a store of 5,205 — a
            # claim of all, contradicted by the number beside it. This module
            # exists to be believed; it cannot round in its own favour.
            lines.append(
                f"All {health.live_facts:,} of them can be found by asking a "
                f"question."
            )
        elif health.reachability >= 0.99:
            gap = health.live_facts - health.findable_by_meaning
            lines.append(
                f"{health.findable_by_meaning:,} of them can be found by asking "
                f"a question. The other {gap:,} can only be found by matching "
                f"words. That is a small enough share to be normal — a memory "
                f"written moments ago, or one the model could not read."
            )
        else:
            gap = health.live_facts - health.findable_by_meaning
            lines.append(
                f"{health.findable_by_meaning:,} of them ({pct:.0f}%) can be "
                f"found by asking a question. The other {gap:,} can only be "
                f"found by matching words, so a question phrased differently "
                f"will miss them. This repairs itself as the service runs; if "
                f"it does not, the embedding model is unavailable."
            )

    if health.withheld_summaries:
        lines.append(
            f"{health.withheld_summaries:,} machine-written summaries are kept "
            f"out of your answers and shown on the dashboard instead. They were "
            f"written by the summarizer, not by you, and they used to be "
            f"returned as if they were your own notes."
        )

    if health.inconsistently_hidden:
        lines.append(
            f"{health.inconsistently_hidden:,} memories are hidden even though "
            f"they are marked worth keeping. `slm doctor --refile-hidden` "
            f"files exactly those again; nothing else moves."
        )

    if health.hidden_by_forgetting:
        lines.append(
            f"{health.hidden_by_forgetting:,} older memories are set aside by "
            f"the forgetting curve. They are not deleted and a deep search "
            f"still reaches them."
        )

    return lines


def classify(health: MemoryHealth) -> tuple[str, str]:
    """Doctor status and the one command that fixes it: ``(status, fix)``.

    A store-wide decay pass moves every memory whose score changed, thousands
    on a large store, so it is never offered for an answer-ability finding.
    """
    if health.healthy:
        return "PASS", ""
    if health.reachability < 0.9:
        return "FAIL", "slm restart"
    if health.inconsistently_hidden:
        return "WARN", "slm doctor --refile-hidden"
    return "WARN", "slm db reembed --missing-only"


@dataclass(frozen=True)
class RefileReport:
    """Memories flagged as wrongly hidden, and how many were moved."""

    fact_ids: tuple[str, ...] = ()
    #: First 80 characters of each flagged memory, in the same order.
    previews: tuple[str, ...] = ()
    moved: int = 0


def _inconsistent_from(live_clause: str) -> str:
    """FROM/WHERE shared by the count and the re-file, so they cannot differ."""
    return (
        "FROM fact_retention r "
        "JOIN atomic_facts af ON af.fact_id = r.fact_id "
        "WHERE r.lifecycle_zone IN ('archive', 'forgotten') "
        "  AND r.retention_score > 0.8 "
        f"  AND {_prefixed(live_clause, 'af')}"
    )


def refile_hidden(db_path: str | Path, *, dry_run: bool = True) -> RefileReport:
    """Re-file exactly the memories ``measure`` counts as inconsistently hidden.

    Each goes to the zone its score implies (above 0.8 is ``active``) through
    the canonical lifecycle writer, which keeps the ``atomic_facts`` mirror in
    step. No other row is touched. No snapshot is taken: the change is one
    reversible zone per memory and the ids are logged.
    """
    from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone

    conn = sqlite3.connect(str(Path(db_path)), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        has_q = _has_column(conn, "atomic_facts", "quarantined")
        live = "COALESCE(quarantined, 0) = 0" if has_q else "1=1"
        rows = conn.execute(
            "SELECT af.fact_id AS fact_id, af.content AS content "
            + _inconsistent_from(live) + " ORDER BY af.fact_id"
        ).fetchall()
        ids = tuple(r["fact_id"] for r in rows)
        previews = tuple((r["content"] or "")[:80] for r in rows)
        if dry_run or not ids:
            return RefileReport(ids, previews, 0)
        moved = set_fact_lifecycle_zone(conn, ids, "active")
        conn.commit()
        logger.info("memory health: re-filed %d hidden memories: %s",
                    moved, ", ".join(ids))
        return RefileReport(ids, previews, moved)
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(
        row[1] == column for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _prefixed(clause: str, prefix: str) -> str:
    """Qualify a bare column reference for use in a joined query.

    Word-bounded, so a future column named ``quarantined_at`` is not silently
    rewritten to ``af.quarantined_at`` by a substring match. No such column
    exists today; the point is that the failure would be a wrong count rather
    than an error, and a wrong count in a health report is the one thing this
    module must not produce.
    """
    return re.sub(r"\bquarantined\b", f"{prefix}.quarantined", clause)


def _count(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0
