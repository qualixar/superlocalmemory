# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com | https://varunpratap.com

"""Can a schema-guarded code path run against this store, and has it had the data?

A note on what this proves, because the honest scope is narrower than the
convenient phrasing. The check reads the schema: it establishes whether the
guard's requirements are present *now*, and therefore whether the path is
reachable at all. It does not read an execution history, so a satisfied guard
means "this would execute", not "this has executed". The inference to "never
executed" is sound only in the other direction -- if a required column is absent
from a store that has been in service, the guarded path cannot have run against
it -- and that is the direction the findings rely on.

Some features are wired behind a guard that asks the schema a question before
doing any work — "is this table here, does that column exist" — and fall back
silently when the answer is no. The fallback is correct behaviour: it is what
keeps an old store openable. But it means a feature can be implemented, called
on the hot path, and covered by tests, while never once executing against real
data. Static analysis passes. A call-graph trace passes. Coverage passes. The
guard returns ``False`` and the feature is arithmetically absent.

Two things make that failure hard to see from inside the code:

1. **The guard is doing its job.** There is no error to raise. Falling back is
   the designed response to a missing column.
2. **The fallback is often a neutral value**, which composes into an identity.
   A decay-rate multiplier that falls back to a trust of 1.0 collapses
   ``lambda * (1 + kappa * (1 - trust))`` to ``lambda`` — the feature is on, and
   it computes exactly what having no feature would compute.

So this check does two things a plain schema assertion does not. It records
whether each named guard **passes right now**, and when a guard fails it looks
for the required data **elsewhere in the store** — because the common case is
not that the data is missing, it is that the guard is asking the wrong table.

A guard reported as ``SATISFIED_ELSEWHERE`` is the most actionable outcome
available: the feature is one re-keyed join away from working, and no backfill
or migration is required.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("superlocalmemory.reliability.join_liveness")


@dataclass(frozen=True)
class Requirement:
    """One schema object a guard needs: a table, optionally a column on it."""

    table: str
    column: str | None = None

    def __str__(self) -> str:
        return f"{self.table}.{self.column}" if self.column else self.table


@dataclass(frozen=True)
class Guard:
    """A named conditional path and the schema it requires to execute."""

    name: str
    describes: str
    requires: tuple[Requirement, ...]
    #: What the feature computes when the guard fails, in words. Recording this
    #: is the difference between "a feature is off" and "a feature is off and
    #: indistinguishable from not having it".
    fallback_behaviour: str


@dataclass(frozen=True)
class GuardVerdict:
    name: str
    describes: str
    #: LIVE = requirements present, so the path is reachable now. DEAD /     #: SATISFIED_ELSEWHERE = a requirement is absent, so the path cannot have run
    #: against this store. LIVE is NOT evidence of past execution.
    verdict: str
    missing: tuple[str, ...] = field(default=())
    found_elsewhere: tuple[tuple[str, str, int, float], ...] = field(default=())
    detail: str = ""

    @property
    def is_live(self) -> bool:
        return self.verdict == "LIVE"


#: Guards worth reporting on. Each entry names a real conditional in the code,
#: so that a reader can go from this list to the line that asks the question.
GUARDS: tuple[Guard, ...] = (
    Guard(
        name="trust_weighted_forgetting",
        describes=(
            "learning/forgetting_scheduler.py::_has_trust_tables — gates the "
            "per-fact trust lookup that modulates the decay rate"
        ),
        requires=(
            Requirement("trust_scores"),
            Requirement("atomic_facts", "created_by"),
        ),
        fallback_behaviour=(
            "every fact takes trust = 1.0, so lambda_eff collapses to "
            "lambda_base and the decay rate is identical to no trust weighting"
        ),
    ),
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.Error:
        return set()


def _satisfied(conn: sqlite3.Connection, requirement: Requirement) -> bool:
    if not _table_exists(conn, requirement.table):
        return False
    if requirement.column is None:
        return True
    return requirement.column in _columns(conn, requirement.table)


def _look_elsewhere(
    conn: sqlite3.Connection,
    column: str,
    *,
    join_target: str | None = None,
    join_key: str = "fact_id",
) -> list[tuple[str, str, int]]:
    """Find other tables carrying ``column``, with a populated-row count.

    This is the part that turns a failed guard into a fix. A column the guard
    could not find on its own table is often present on a neighbouring one,
    already populated.

    **A populated count is not coverage, and reporting it alone overstates the
    remedy.** A provenance-style table can carry a value on every one of its own
    rows while describing only part of the set the join needs: rows can be
    missing for older entities entirely. When ``join_target`` is given, the
    coverage fraction over that table is measured and returned, because the
    honest question is not "does this column exist somewhere" but "how much of
    what the guard needs would the re-keyed join actually resolve".
    """
    out: list[tuple[str, str, int]] = []
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'",
        )
    ]
    for table in tables:
        if column not in _columns(conn, table):
            continue
        try:
            populated = conn.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND TRIM("{column}") <> \'\'',
            ).fetchone()[0]
        except sqlite3.Error:
            populated = 0
        covered = -1
        if join_target and join_target != table:
            try:
                total = conn.execute(
                    f'SELECT COUNT(*) FROM "{join_target}"',
                ).fetchone()[0]
                if total:
                    hit = conn.execute(
                        f'SELECT COUNT(*) FROM "{join_target}" t WHERE EXISTS ('
                        f'  SELECT 1 FROM "{table}" s WHERE s."{join_key}" = t."{join_key}"'
                        f'  AND s."{column}" IS NOT NULL AND TRIM(s."{column}") <> \'\')',
                    ).fetchone()[0]
                    covered = round(100.0 * hit / total, 1)
            except sqlite3.Error:
                covered = -1
        out.append((table, column, int(populated), covered))
    return out


def _evaluate(conn: sqlite3.Connection, guard: Guard) -> GuardVerdict:
    missing = [str(r) for r in guard.requires if not _satisfied(conn, r)]
    if not missing:
        return GuardVerdict(
            name=guard.name,
            describes=guard.describes,
            verdict="LIVE",
            detail=(
                "Every requirement is present, so the guarded path is reachable "
                "on this store. This is schema evidence, not execution history: "
                "it does not establish that the path has ever actually run."
            ),
        )

    elsewhere: list[tuple[str, str, int, float]] = []
    for requirement in guard.requires:
        if requirement.column and not _satisfied(conn, requirement):
            for found in _look_elsewhere(
                conn, requirement.column, join_target=requirement.table,
            ):
                if found[0] != requirement.table:
                    elsewhere.append(found)

    populated = [e for e in elsewhere if e[2] > 0]
    if populated:
        # Rank by coverage of the guarded table, falling back to row count only
        # when coverage could not be measured. Picking the largest table instead
        # would recommend a 4,000-row partial source over a smaller complete one.
        best = max(populated, key=lambda e: (e[3], e[2]))
        cov = best[3]
        if cov < 0:
            remedy = (
                "Re-keying the join onto that table would make the path "
                "executable; coverage over the guarded table could not be "
                "measured here, so confirm it before relying on the remedy."
            )
        elif cov >= 99.5:
            remedy = (
                f"That table covers {cov}% of the rows the guard needs, so "
                f"re-keying the join makes the path live without a backfill."
            )
        else:
            remedy = (
                f"That table covers only {cov}% of the rows the guard needs. "
                f"Re-keying the join makes the path executable for those rows "
                f"and leaves the remainder on the same fallback, so this is a "
                f"partial remedy and a backfill decision, not a free fix."
            )
        return GuardVerdict(
            name=guard.name,
            describes=guard.describes,
            verdict="SATISFIED_ELSEWHERE",
            missing=tuple(missing),
            found_elsewhere=tuple(elsewhere),
            detail=(
                f"The guard requires {', '.join(missing)}, which is absent, so "
                f"the path has never executed against this store — "
                f"{guard.fallback_behaviour}. The data it needs is present on "
                f"{best[0]}.{best[1]} with {best[2]} populated rows. {remedy}"
            ),
        )

    return GuardVerdict(
        name=guard.name,
        describes=guard.describes,
        verdict="DEAD",
        missing=tuple(missing),
        detail=(
            f"The guard requires {', '.join(missing)}, which is absent from "
            f"this store and not carried by any other table, so the path has "
            f"never executed — {guard.fallback_behaviour}."
        ),
    )


def check_schema_guards(
    memory_db: Any, *, guards: tuple[Guard, ...] = GUARDS,
) -> list[GuardVerdict]:
    """Report, per registered guard, whether its path can run on this store.

    ``memory_db`` may be a path or an open connection. Read-only, and fail-soft:
    an error yields an empty list, because a diagnostic must never be the reason
    something breaks.
    """
    owns_connection = not isinstance(memory_db, sqlite3.Connection)
    conn: sqlite3.Connection | None = None
    try:
        conn = (
            sqlite3.connect(f"file:{memory_db}?mode=ro", uri=True)
            if owns_connection
            else memory_db
        )
        return [_evaluate(conn, guard) for guard in guards]
    except Exception:
        logger.debug("join-liveness check skipped", exc_info=True)
        return []
    finally:
        if owns_connection and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


__all__ = ["GUARDS", "Guard", "GuardVerdict", "Requirement", "check_schema_guards"]
