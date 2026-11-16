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

**Content overlap** (W3-2b, opt-in by ``outcomes.usage``): at the end of a turn
the Stop hook scores, in its own process, how much of each memory delivered in
the session reappears in the agent's reply, and sends only (fact id, score).
The reply never leaves the hook. Every score is kept (the highest per
delivery), with no threshold: whether a score means "used" is calibrated later
against marker and explicit labels, not chosen here.

A signal that matches no delivery is not stored (it is counted by the caller):
the fact reached the agent some way that wrote no receipt. Rows are idempotent
on (delivery, fact, signal), so a replayed event changes nothing. These
functions never commit and never raise for a store without the tables.
"""

from __future__ import annotations

import logging
import re
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
#: Share of a memory's distinctive words and numbers found in the agent's reply.
OVERLAP = "overlap"

#: A memory with fewer distinctive tokens than this is too short to score
#: (routing only: "the" and one name would match any reply).
MIN_CONTENT_TOKENS = 3
#: Memories compared per turn, the session's most recently delivered first
#: (routing only: bounds the Stop hook's work).
MAX_COMPARED = 30

_TOKEN = re.compile(r"\d+(?:[.,:]\d+)*|[^\W\d_]+", re.UNICODE)
_STOP = frozenset("""about above after again also because been before being below between both
could does doing down during each from further have having here into itself just more most
other over same should some such than that their them then there these they this those through
under until very what when where which while will with would your yours""".split())

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
#: Overlap keeps the highest score a delivery reached over the session's turns.
_KEEP_MAX = ("INSERT INTO injection_usage (delivery_id, fact_id, profile_id, signal, value,"
             " observed_at) VALUES (?, ?, ?, ?, ?, ?)"
             " ON CONFLICT (delivery_id, fact_id, signal) DO UPDATE SET"
             " observed_at = CASE WHEN excluded.value > COALESCE(injection_usage.value, -1)"
             " THEN excluded.observed_at ELSE injection_usage.observed_at END,"
             " value = MAX(COALESCE(injection_usage.value, -1), excluded.value)")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall((text or "").casefold())
            if t[0].isdigit() or (len(t) >= 4 and t not in _STOP)}


def content_overlap(memory: str, reply: str) -> float | None:
    """Share of the memory's distinctive tokens (numbers, words of 4+ letters
    that are not function words) that appear in ``reply``; None when the
    memory has too few to say."""
    wanted = _tokens(memory)
    if len(wanted) < MIN_CONTENT_TOKENS:
        return None
    return len(wanted & _tokens(reply)) / len(wanted)


_DELIVERED_IN_SESSION = """
SELECT r.fact_id, f.content FROM injection_receipts r
  JOIN injection_deliveries d ON d.delivery_id = r.delivery_id
  JOIN atomic_facts f ON f.fact_id = r.fact_id
 WHERE d.profile_id = ? AND d.session_id = ? AND r.delivered = 1
 GROUP BY r.fact_id
 ORDER BY MAX(julianday(d.delivered_at)) DESC
 LIMIT ?
"""


def delivered_in_session(conn: sqlite3.Connection, profile_id: str,
                         session_id: str) -> list[tuple[str, str]]:
    """(fact id, text) of memories delivered in this conversation, latest first."""
    from superlocalmemory.core.session_identity import is_conversation

    if not is_conversation(session_id, profile_id):
        return []
    try:
        return [(r[0], r[1] or "") for r in conn.execute(
            _DELIVERED_IN_SESSION, (profile_id, session_id, MAX_COMPARED))]
    except sqlite3.OperationalError:
        return []


def overlap_scores(conn: sqlite3.Connection, profile_id: str, session_id: str,
                   reply: str) -> list[tuple[str, float]]:
    """(fact id, score) for this turn's reply; nothing but ids and numbers."""
    if not reply:
        return []
    out = []
    for fid, text in delivered_in_session(conn, profile_id, session_id):
        score = content_overlap(text, reply)
        if score is not None:
            out.append((fid, round(score, 4)))
    return out


def stamp(at: datetime) -> str:
    """One timestamp format for deliveries and usage (UTC, microseconds)."""
    return at.astimezone(UTC).isoformat(timespec="microseconds")


def from_ms(observed_ms: int) -> str:
    return stamp(datetime.fromtimestamp(observed_ms / 1000, UTC))


def _join(conn: sqlite3.Connection, sql: str, anchor: str, profile_id: str,
          fact_ids: Iterable[str], signal: str, observed_at: str,
          value: float | None, values: dict[str, float] | None = None) -> int:
    joined = 0
    insert = _KEEP_MAX if values is not None else _INSERT
    try:
        for fid in dict.fromkeys(f for f in fact_ids if f):
            row = conn.execute(sql, (fid, profile_id, anchor, observed_at)).fetchone()
            if row is None:
                continue
            v = values[fid] if values is not None else value
            cur = conn.execute(insert, (row[0], fid, profile_id, signal, v, observed_at))
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


def join_overlap(conn: sqlite3.Connection, *, profile_id: str, session_id: str,
                 scores: Iterable[tuple[str, float]], observed_at: str) -> int:
    """Store each delivered memory's overlap with a reply, keeping its highest."""
    from superlocalmemory.core.session_identity import is_conversation

    values = {fid: float(score) for fid, score in scores
              if fid and isinstance(score, (int, float)) and 0.0 <= score <= 1.0}
    if not values or not is_conversation(session_id, profile_id):
        return 0
    return _join(conn, _LATEST_IN_SESSION, session_id, profile_id, list(values), OVERLAP,
                 observed_at, None, values)


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


__all__ = ["EXPLICIT", "MARKER_EDIT", "MARKER_TOOL", "MAX_COMPARED", "MIN_CONTENT_TOKENS",
           "OVERLAP", "content_overlap", "delivered_in_session", "from_ms", "join_overlap",
           "join_question", "join_session", "overlap_scores", "query_digest", "stamp"]
