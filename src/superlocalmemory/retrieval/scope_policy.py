# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fail-closed authorization helpers for retrieval candidate paths.

Candidate generators may use caches, approximate indexes, or graph stores that
are not the authorization source of truth.  Every such path must therefore
re-authorize fact IDs through ``DatabaseManager.get_facts_by_ids()``, whose SQL
is built by the canonical ``_scope_where`` predicate.
"""

from __future__ import annotations

from typing import Any, Iterable

from superlocalmemory.storage.database import _scope_where


def authorized_fact_ids(
    db: Any,
    fact_ids: Iterable[str],
    profile_id: str,
    *,
    include_global: bool = False,
    include_shared: bool = False,
) -> set[str]:
    """Return only IDs visible under the canonical scope predicate.

    Authorization errors fail closed.  The stable de-duplication avoids SQLite
    parameter waste without changing candidate order at the caller boundary.
    """
    unique_ids = list(dict.fromkeys(fact_ids))
    if not unique_ids:
        return set()
    try:
        facts = db.get_facts_by_ids(
            unique_ids,
            profile_id,
            include_global=bool(include_global),
            include_shared=bool(include_shared),
        )
        if isinstance(facts, list):
            return {fact.fact_id for fact in facts}
    except Exception:
        pass

    # Lightweight DB wrappers used by maintenance paths may expose execute()
    # without the higher-level method.  Keep the same canonical predicate.
    try:
        where, params = _scope_where(
            profile_id,
            include_global=include_global,
            include_shared=include_shared,
        )
        placeholders = ",".join("?" for _ in unique_ids)
        rows = db.execute(
            f"SELECT fact_id FROM atomic_facts WHERE fact_id IN ({placeholders}) "
            f"AND {where}",
            (*unique_ids, *params),
        )
        if not isinstance(rows, list):
            rows = list(rows)
        return {dict(row)["fact_id"] for row in rows}
    except Exception:
        return set()


def filter_authorized_results(
    db: Any,
    results: Iterable[tuple[str, float]],
    profile_id: str,
    *,
    include_global: bool = False,
    include_shared: bool = False,
) -> list[tuple[str, float]]:
    """Preserve result order/scores while removing unauthorized fact IDs."""
    materialized = list(results)
    allowed = authorized_fact_ids(
        db,
        (fact_id for fact_id, _score in materialized),
        profile_id,
        include_global=include_global,
        include_shared=include_shared,
    )
    return [item for item in materialized if item[0] in allowed]
