# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The slot a caller names on a save: ``remember(slot=..., slot_value=...)`` (D-36).

A claim belongs to the memory, not to each fact extracted from it. It travels
in the memory's metadata under a reserved key, set by the door only after the
caller's own metadata has had every reserved key stripped
(``core.metadata_guard``), so a caller can set it only through the checked
``slot`` field. It is journaled with the save, so a save the writer accepts
late keeps its claim.

The slot pass binds the claim to the memory's first fact only (the verbatim
copy the daemon writes before enrichment): binding every fact of a memory that
split into several would make the memory dispute with itself. The memory's
other facts are keyed by the Tier 0 patterns as before.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from superlocalmemory.storage.json_columns import load_json_column
from superlocalmemory.truth.slot_keys import parse_spec

#: Reserved (``_slm_*``): a caller's metadata can never set it directly.
METADATA_KEY = "_slm_slot"

_CLAIM = """
SELECT m.metadata_json
  FROM atomic_facts f
  JOIN memories m ON m.memory_id = f.memory_id
 WHERE f.fact_id = ?
   AND instr(m.metadata_json, ?) > 0
   AND f.rowid = (SELECT MIN(g.rowid) FROM atomic_facts g WHERE g.memory_id = f.memory_id)
"""


def encode(spec: str | None, value: str | None) -> dict[str, str] | None:
    """The metadata entry for a checked claim, or None when no slot is named.

    Raises ``slot_keys.SlotSpecError`` for a malformed one: the door refuses
    the save before anything is stored.
    """
    parsed = parse_spec(spec, value)
    if parsed is None:
        return None
    entry = {"spec": parsed[0]}
    if parsed[1]:
        entry["value"] = parsed[1]
    return entry


def claim_for_fact(conn: sqlite3.Connection, fact_id: str) -> tuple[str, str | None] | None:
    """``(spec, value)`` when ``fact_id`` is the first fact of a memory with a claim."""
    try:
        row = conn.execute(_CLAIM, (fact_id, METADATA_KEY)).fetchone()
    except sqlite3.OperationalError:  # a store without memories: no claims there
        return None
    if row is None:
        return None
    return _decode(row[0])


def _decode(metadata_json: Any) -> tuple[str, str | None] | None:
    try:
        entry = load_json_column(metadata_json, {}, column="memories.metadata_json").get(METADATA_KEY)
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("spec"), str):
        return None
    value = entry.get("value")
    return entry["spec"], value if isinstance(value, str) and value.strip() else None


__all__ = ["METADATA_KEY", "claim_for_fact", "encode"]
