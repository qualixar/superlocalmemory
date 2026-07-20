# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Canonical lifecycle mutations.

``fact_retention.lifecycle_zone`` is the source of truth used by retrieval.
``atomic_facts.lifecycle`` is a materialized compatibility mirror used by
older queries.  All non-retention lifecycle writers must pass through this
module so the two representations cannot drift during a successful commit.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable

_RETENTION_ZONES = frozenset({"active", "warm", "cold", "archive", "forgotten"})


def _materialize_rows(result: Any) -> list[Any]:
    """Normalize DatabaseManager lists and raw sqlite3 cursors."""
    if hasattr(result, "fetchall"):
        return list(result.fetchall())
    return list(result)


def normalize_retention_zone(zone: str) -> str:
    """Return the canonical retention spelling or reject an invalid state."""
    canonical = "archive" if zone == "archived" else zone
    if canonical not in _RETENTION_ZONES:
        raise ValueError(f"invalid lifecycle zone: {zone!r}")
    return canonical


def atomic_lifecycle_for(zone: str) -> str:
    """Map canonical retention state to the legacy atomic-fact mirror."""
    canonical = normalize_retention_zone(zone)
    return "archived" if canonical in {"archive", "forgotten"} else canonical


def set_fact_lifecycle_zone(
    db: Any,
    fact_ids: Iterable[str],
    zone: str,
    *,
    profile_id: str | None = None,
    from_atomic: Iterable[str] | None = None,
) -> int:
    """Atomically update canonical lifecycle and its materialized mirror.

    Missing retention rows are created from the owning ``atomic_facts`` row.
    ``from_atomic`` provides compare-and-set behavior for tier transitions.
    The helper accepts the small ``execute``-compatible test DB as well as
    ``DatabaseManager``; production writes use its transaction context.
    """
    ids = tuple(dict.fromkeys(str(fid) for fid in fact_ids if str(fid)))
    if not ids:
        return 0

    canonical = normalize_retention_zone(zone)
    atomic = atomic_lifecycle_for(canonical)
    id_marks = ",".join("?" for _ in ids)
    filters = [f"fact_id IN ({id_marks})"]
    filter_params: list[Any] = list(ids)
    if profile_id is not None:
        filters.append("profile_id = ?")
        filter_params.append(profile_id)
    source_states = tuple(dict.fromkeys(from_atomic or ()))
    if source_states:
        state_marks = ",".join("?" for _ in source_states)
        filters.append(f"lifecycle IN ({state_marks})")
        filter_params.extend(source_states)
    where = " AND ".join(filters)

    transaction = getattr(db, "transaction", None)
    txn_state = getattr(db, "_txn_state", None)
    already_in_transaction = getattr(txn_state, "conn", None) is not None
    context = (
        transaction()
        if callable(transaction) and not already_in_transaction
        else nullcontext()
    )
    with context:
        candidates = _materialize_rows(db.execute(
            f"SELECT fact_id FROM atomic_facts WHERE {where}",
            tuple(filter_params),
        ))
        selected = tuple(row["fact_id"] for row in candidates)
        if not selected:
            return 0
        selected_marks = ",".join("?" for _ in selected)
        db.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
            f"SELECT fact_id, profile_id, ? FROM atomic_facts "
            f"WHERE fact_id IN ({selected_marks}) "
            "ON CONFLICT(fact_id) DO UPDATE SET "
            "profile_id = excluded.profile_id, "
            "lifecycle_zone = excluded.lifecycle_zone, "
            "last_computed_at = datetime('now')",
            (canonical, *selected),
        )
        db.execute(
            f"UPDATE atomic_facts SET lifecycle = ? "
            f"WHERE fact_id IN ({selected_marks})",
            (atomic, *selected),
        )
    return len(selected)


def reconcile_profile_lifecycle(db: Any, profile_id: str) -> int:
    """Repair historical mirror drift using retention state as authority.

    Legacy archived atomic facts without a retention row are first imported as
    ``archive`` so they remain excluded from ordinary recall.  Existing
    retention rows then overwrite the compatibility mirror in one transaction.
    """
    transaction = getattr(db, "transaction", None)
    txn_state = getattr(db, "_txn_state", None)
    already_in_transaction = getattr(txn_state, "conn", None) is not None
    context = (
        transaction()
        if callable(transaction) and not already_in_transaction
        else nullcontext()
    )
    with context:
        db.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone) "
            "SELECT fact_id, profile_id, "
            "CASE WHEN lifecycle = 'archived' THEN 'archive' ELSE lifecycle END "
            "FROM atomic_facts af WHERE profile_id = ? AND lifecycle != 'active' "
            "AND NOT EXISTS (SELECT 1 FROM fact_retention fr "
            "WHERE fr.fact_id = af.fact_id)",
            (profile_id,),
        )
        rows = _materialize_rows(db.execute(
            "SELECT COUNT(*) AS count FROM atomic_facts af "
            "JOIN fact_retention fr ON fr.fact_id = af.fact_id "
            "WHERE af.profile_id = ? AND af.lifecycle != "
            "CASE WHEN fr.lifecycle_zone IN ('archive', 'forgotten') "
            "THEN 'archived' ELSE fr.lifecycle_zone END",
            (profile_id,),
        ))
        changed = int(rows[0]["count"]) if rows else 0
        db.execute(
            "UPDATE atomic_facts SET lifecycle = ("
            "SELECT CASE WHEN fr.lifecycle_zone IN ('archive', 'forgotten') "
            "THEN 'archived' ELSE fr.lifecycle_zone END "
            "FROM fact_retention fr WHERE fr.fact_id = atomic_facts.fact_id) "
            "WHERE profile_id = ? AND EXISTS ("
            "SELECT 1 FROM fact_retention fr WHERE fr.fact_id = atomic_facts.fact_id)",
            (profile_id,),
        )
    return changed
