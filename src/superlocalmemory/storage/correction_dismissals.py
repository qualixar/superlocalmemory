# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Remembering which machine proposals the user dismissed (issue #309).

A case's id and idempotency key are deterministic (operation, facts, reason), so
processing a save again, for example a requeued save, would file a dismissed
case again. ``slm corrections dismiss`` (core/dismissed_cases.py) leaves its
audit row in ``correction_cases_overtaken`` with ``user_action = 'dismissed'``;
that row is the tombstone. Rows left by a person's delete, replace or edit are
not tombstones, exactly as before.
"""

from __future__ import annotations

import sqlite3

#: ``user_action`` of a case that was moved by the bulk dismissal.
DISMISSED_USER_ACTION = "dismissed"


def dismissed_in_bulk(conn: sqlite3.Connection, profile_id: str, case_id: str) -> bool:
    """True when ``case_id`` was dismissed in bulk and has not been put back."""
    try:
        row = conn.execute(
            "SELECT 1 FROM correction_cases_overtaken WHERE case_id = ? AND profile_id = ? "
            "AND user_action = ? AND restored_at IS NULL LIMIT 1",
            (case_id, profile_id, DISMISSED_USER_ACTION)).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return False  # nothing was ever overtaken on this store
        raise
    return row is not None


__all__ = ["DISMISSED_USER_ACTION", "dismissed_in_bulk"]
