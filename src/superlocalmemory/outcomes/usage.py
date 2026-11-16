# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Usage join: tie each sign that a memory was used to the delivery it came from (W3-2).

A usage signal is only evidence about a delivery, so it is stored against one:
``injection_usage`` holds (delivery, fact, signal). The estimates of WS-3 then
read usage beside the receipt's propensity, and a memory that was never
delivered can never look used.

Two joins, each exact rather than "the latest anywhere":

* **by session** (marker echo): the agent quoted a memory's marker in a later
  tool call of the same conversation; the signal joins to the latest delivery
  of that fact in that session at or before the moment it was seen;
* **by question** (explicit outcome): a report naming facts and the question
  they answered joins to the latest delivery of each fact whose keyed query
  digest equals the report's. A report without its question joins nothing.

A signal that matches no delivery is not stored (it is counted by the caller):
the fact reached the agent some way that wrote no receipt. Rows are idempotent
on (delivery, fact, signal), so a replayed event changes nothing. These
functions never commit and never raise for a store without the tables.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

#: The agent quoted the marker in an Edit/Write/NotebookEdit call: it acted on it.
MARKER_EDIT = "marker_edit"
#: The agent quoted the marker in any other tool call.
MARKER_TOOL = "marker_tool"
#: Prefix of an explicit outcome (``explicit_success``, ``explicit_failure``, ...).
EXPLICIT = "explicit_"

_LATEST_IN_SESSION = """
SELECT r.delivery_id FROM injection_receipts r
  JOIN injection_deliveries d ON d.delivery_id = r.delivery_id
 WHERE r.fact_id = ? AND r.delivered = 1 AND d.profile_id = ? AND d.session_id = ?
   AND julianday(d.delivered_at) <= julianday(?)
 ORDER BY julianday(d.delivered_at) DESC, d.delivery_id DESC LIMIT 1
"""

_LATEST_FOR_QUESTION = """
SELECT r.delivery_id FROM injection_receipts r
  JOIN injection_deliveries d ON d.delivery_id = r.delivery_id
 WHERE r.fact_id = ? AND r.delivered = 1 AND d.profile_id = ? AND d.query_digest = ?
   AND julianday(d.delivered_at) <= julianday(?)
 ORDER BY julianday(d.delivered_at) DESC, d.delivery_id DESC LIMIT 1
"""

_INSERT = ("INSERT OR IGNORE INTO injection_usage (delivery_id, fact_id, profile_id, signal,"
           " value, observed_at) VALUES (?, ?, ?, ?, ?, ?)")


def stamp(at: datetime) -> str:
    """One timestamp format for deliveries and usage (UTC, microseconds)."""
    return at.astimezone(UTC).isoformat(timespec="microseconds")


def from_ms(observed_ms: int) -> str:
    return stamp(datetime.fromtimestamp(observed_ms / 1000, UTC))


def _join(conn: sqlite3.Connection, sql: str, anchor: str, profile_id: str,
          fact_ids: Iterable[str], signal: str, observed_at: str,
          value: float | None) -> int:
    joined = 0
    try:
        for fid in dict.fromkeys(f for f in fact_ids if f):
            row = conn.execute(sql, (fid, profile_id, anchor, observed_at)).fetchone()
            if row is None:
                continue
            cur = conn.execute(_INSERT, (row[0], fid, profile_id, signal, value, observed_at))
            joined += cur.rowcount
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return 0  # a store before M075: nothing to join to
        raise
    return joined


def join_session(conn: sqlite3.Connection, *, profile_id: str, session_id: str,
                 fact_ids: Iterable[str], signal: str, observed_at: str,
                 value: float | None = None) -> int:
    """Rows added for a signal seen in ``session_id``; 0 without a real session."""
    from superlocalmemory.core.session_identity import is_conversation

    if not is_conversation(session_id, profile_id):
        return 0
    return _join(conn, _LATEST_IN_SESSION, session_id, profile_id, fact_ids, signal,
                 observed_at, value)


def join_question(conn: sqlite3.Connection, *, profile_id: str, query_digest: str | None,
                  fact_ids: Iterable[str], signal: str, observed_at: str,
                  value: float | None = None) -> int:
    """Rows added for a signal about the answer to one question (by its keyed digest)."""
    if not query_digest:
        return 0
    return _join(conn, _LATEST_FOR_QUESTION, query_digest, profile_id, fact_ids, signal,
                 observed_at, value)


def query_digest(slm_home, query: str) -> str | None:
    """The delivery-side digest of ``query``, or None when no key exists yet (so
    no delivery can carry one) or the query is empty. Never creates the key."""
    from pathlib import Path

    from superlocalmemory.core.receipt_key import KEY_FILE, digest

    if not query:
        return None
    try:
        key = (Path(slm_home) / "keys" / KEY_FILE).read_bytes()
    except OSError:
        return None
    if len(key) < 32:
        return None
    return digest(key, b"query:" + query.encode())


__all__ = ["EXPLICIT", "MARKER_EDIT", "MARKER_TOOL", "from_ms", "join_question",
           "join_session", "query_digest", "stamp"]
