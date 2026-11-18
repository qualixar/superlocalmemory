# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""One memory's place in the truth layer, for the dashboard's memory detail (§8.10.3 b).

For a fact: each slot it fills (what it states a value of, or "a correction"
for a private legacy slot), that slot's values over time newest first with
when each began and whether it is the value shown now, its open disputes and
the state of the file it came from, and where the memory came from (its origin class,
WS-5). Read-only, and another value's text passes
the caller's visibility filter like any recall result.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from superlocalmemory.trust import fact_trust
from superlocalmemory.truth import disputes, leases
from superlocalmemory.truth.legacy_cases import KEY_SOURCE as LEGACY

TEXT_CHARS = 400
MAX_VALUES = 20

_SLOTS = """
SELECT DISTINCT s.slot_id, s.subject_key, s.relation_key, s.qualifier, s.key_source, s.state,
       s.head_revision_id
  FROM slot_revisions r JOIN memory_slots s ON s.slot_id = r.slot_id
 WHERE r.fact_id = ? AND s.profile_id = ?
 ORDER BY s.created_at, s.slot_id
"""

_VALUES = """
SELECT r.revision_id, r.fact_id, r.state, COALESCE(r.valid_from, r.recorded_at), f.content
  FROM slot_revisions r LEFT JOIN atomic_facts f ON f.fact_id = r.fact_id
 WHERE r.slot_id = ?
"""


def _slot_view(conn: sqlite3.Connection, row: tuple, fact_id: str,
               hidden: Callable[[list[str]], set[str]]) -> dict:
    slot_id, subject, relation, qualifier, source, state, head = row
    values = conn.execute(_VALUES, (slot_id,)).fetchall()
    values.sort(key=lambda v: v[3] or "", reverse=True)
    values = values[:MAX_VALUES]
    others = [v[1] for v in values if v[1] != fact_id]
    blocked = hidden(others) if others else set()
    items = []
    for revision_id, fid, rstate, began, content in values:
        visible = fid not in blocked
        items.append({"fact_id": fid, "state": rstate, "began": began,
                      "current": revision_id == head, "this_memory": fid == fact_id,
                      "visible": visible,
                      "content": (content or "")[:TEXT_CHARS] if visible else None})
    legacy = source == LEGACY
    return {"slot_id": slot_id, "legacy": legacy, "state": state,
            "subject": None if legacy else subject, "relation": None if legacy else relation,
            "qualifier": None if legacy else (qualifier or None), "key_source": source,
            "values": items}


def for_fact(conn: sqlite3.Connection, profile_id: str, fact_id: str, *,
             hidden: Callable[[list[str]], set[str]] = lambda ids: set()) -> dict:
    """The fact's slots, open disputes and source-file state."""
    try:
        rows = conn.execute(_SLOTS, (fact_id, profile_id)).fetchall()
    except sqlite3.OperationalError:  # a store before M069
        rows = []
    slots = [_slot_view(conn, r, fact_id, hidden) for r in rows]
    open_disputes = disputes.for_facts(conn, profile_id, [fact_id], hidden=hidden).get(fact_id, [])
    source = leases.check_for_facts(conn, [fact_id]).get(fact_id)
    return {"fact_id": fact_id, "slots": slots, "disputes": open_disputes, "source": source,
            "origin": _origin(conn, fact_id)}


def _origin(conn: sqlite3.Connection, fact_id: str) -> dict | None:
    """Where the memory came from, in ids and words (WS-5): the stored class, else computed
    from the raw signals now; never written here. ``None`` when it cannot be read."""
    try:
        found = fact_trust.origin_of(conn, fact_id)
    except sqlite3.Error:
        return None
    return {k: v for k, v in found.items() if k != "basis"} if found else None


__all__ = ["for_fact"]
