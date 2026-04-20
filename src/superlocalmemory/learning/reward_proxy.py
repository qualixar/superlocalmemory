# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3.5 + §5.6

"""Proxy settlement for bandit plays (v3.4.22 only).

LLD reference: ``.backup/active-brain/lld/LLD-03-contextual-bandit-and-ensemble.md``
Section 3.5 and 5.6.

Replaced in v3.4.22 by ``reward_from_outcomes.py`` — DO NOT extend this
module beyond the proxy window contract.

Policy (§3.5):
  For each ``bandit_plays`` row where ``settled_at IS NULL`` and
  ``age(played_at) > 60 s``:
    1. Read top-3 ``learning_signals.fact_id`` for the same query_id.
    2. Search ``tool_events`` within +30 s of played_at, LIKE '%fact_id%'.
    3. If hit → reward=1.0, kind='proxy_position'.
    4. Else if ``_requery_detected`` (NFC topic sig match within 30 s) →
       reward=0.0, kind='proxy_requery'.
    5. Else if ``age(played_at) > 120 s`` → reward=0.5, kind='default'.
    6. Else: skip (not yet settleable).

Hard rules:
  - P1: settlement window 60–120 s.
  - P2: requery uses NFC-normalised topic signature from LLD-01 §4.2.
  - B6: never writes raw query text anywhere.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from superlocalmemory.core.topic_signature import compute_topic_signature
from superlocalmemory.learning.bandit import ContextualBandit

logger = logging.getLogger(__name__)

_MIN_AGE_SEC = 60
_MAX_AGE_SEC = 120
_EVIDENCE_WINDOW_SEC = 30
_REQUERY_WINDOW_SEC = 30


def _parse_iso(ts: str) -> datetime | None:
    """Parse a best-effort ISO timestamp → tz-aware datetime (UTC if naive)."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _open(db_path: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        logger.debug("reward_proxy: open %s failed: %s", db_path, exc)
        return None


def _fetch_unsettled(
    learning_conn: sqlite3.Connection,
    profile_id: str,
    now: datetime,
) -> list[sqlite3.Row]:
    try:
        return learning_conn.execute(
            "SELECT play_id, query_id, played_at, stratum "
            "FROM bandit_plays "
            "WHERE profile_id = ? AND settled_at IS NULL "
            "ORDER BY played_at ASC LIMIT 500",
            (profile_id,),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("reward_proxy: fetch_unsettled: %s", exc)
        return []


def _top3_fact_ids(
    learning_conn: sqlite3.Connection,
    query_id: str,
) -> list[str]:
    """Return up to 3 fact_ids for this query_id ordered by position ASC.

    Returns [] if learning_signals is missing or has no rows for this qid.
    """
    try:
        rows = learning_conn.execute(
            "SELECT fact_id FROM learning_signals "
            "WHERE query_id = ? AND position < 3 "
            "ORDER BY position ASC LIMIT 3",
            (query_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows if r and r[0]]


def _tool_event_hit(
    memory_conn: sqlite3.Connection,
    played_at: datetime,
    fact_ids: list[str],
) -> bool:
    """True iff any tool_events row references any fact_id within +30 s."""
    if not fact_ids:
        return False
    start = played_at.isoformat(timespec="seconds")
    end = (played_at + timedelta(seconds=_EVIDENCE_WINDOW_SEC)).isoformat(
        timespec="seconds",
    )
    # tool_events schema varies by install; we check for the table first.
    try:
        tbl = memory_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='tool_events'"
        ).fetchone()
        if tbl is None:
            return False
    except sqlite3.Error:
        return False

    # E.2 (v3.4.22 perf): previously ran N separate LIKE '%fid%' scans,
    # each doing a full table scan of ``tool_events`` inside the 30-second
    # window. Instead, fetch the small window once and scan its payloads
    # in Python — typically 0-few rows in a 30 s window, far cheaper than
    # N LIKE passes against a growing table. Still O(rows_in_window * len(fact_ids))
    # worst case but the constant factor is tiny (a substring check).
    try:
        candidate_rows = memory_conn.execute(
            "SELECT payload_json FROM tool_events "
            "WHERE occurred_at BETWEEN ? AND ? "
            "  AND payload_json IS NOT NULL",
            (start, end),
        ).fetchall()
    except sqlite3.Error:
        return False
    if not candidate_rows:
        return False
    fid_list = [fid for fid in fact_ids if fid]
    if not fid_list:
        return False
    for row in candidate_rows:
        payload = row[0] or ""
        for fid in fid_list:
            if fid in payload:
                return True
    return False


def _requery_detected(
    memory_conn: sqlite3.Connection,
    played_at: datetime,
    query_id: str,
) -> bool:
    """True iff a follow-up query within 30 s has matching NFC topic sig.

    Uses the same topic-signature algorithm as LLD-01 §4.2. The original
    query text is NOT stored (B6) — we need the hash that was computed at
    recall time. Two schemas cover this:

      1. learning_signals.query_text_hash — populated by signal_worker.
      2. Any requery logged as a tool_event with a topic-signature field.

    We do (1). The window is the next 30 s after played_at.
    """
    start = played_at.isoformat(timespec="seconds")
    end = (played_at + timedelta(seconds=_REQUERY_WINDOW_SEC)).isoformat(
        timespec="seconds",
    )
    # Look up the play's own query hash on memory_conn's learning-mirror IF
    # it exists — otherwise skip quietly.
    try:
        tbl = memory_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='tool_events'"
        ).fetchone()
    except sqlite3.Error:
        return False
    if tbl is None:
        return False

    # Read *query* text from tool_events payload for within-window events;
    # compute topic sig on each, compare against the original query's sig.
    try:
        rows = memory_conn.execute(
            "SELECT payload_json FROM tool_events "
            "WHERE occurred_at > ? AND occurred_at <= ? "
            "  AND tool_name = 'recall' LIMIT 20",
            (start, end),
        ).fetchall()
    except sqlite3.Error:
        return False
    if not rows:
        return False

    # We don't have the original query text on the bandit row (B6).
    # The original is recoverable via tool_events at played_at (recall event).
    try:
        seed_row = memory_conn.execute(
            "SELECT payload_json FROM tool_events "
            "WHERE occurred_at <= ? AND tool_name = 'recall' "
            "ORDER BY occurred_at DESC LIMIT 1",
            (played_at.isoformat(timespec="seconds"),),
        ).fetchone()
    except sqlite3.Error:
        seed_row = None
    seed_query = _extract_query(seed_row[0] if seed_row else None)
    if not seed_query:
        return False
    # P2: NFC-normalised topic signature.
    seed_sig = compute_topic_signature(seed_query)
    for r in rows:
        q = _extract_query(r[0])
        if not q:
            continue
        if compute_topic_signature(q) == seed_sig:
            return True
    return False


def _extract_query(payload_json: str | None) -> str:
    """Pull a ``query`` / ``text`` field out of a tool_events payload."""
    if not payload_json:
        return ""
    try:
        import json as _json
        obj = _json.loads(payload_json)
    except (ValueError, TypeError):
        return ""
    if not isinstance(obj, dict):
        return ""
    for key in ("query", "text", "prompt"):
        v = obj.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def settle_stale_plays(
    profile_id: str,
    db_path: Path | str,
    tool_events_db: Path | str,
    *,
    now: datetime | None = None,
    bandit: ContextualBandit | None = None,
) -> int:
    """Settle every unsettled bandit play whose evidence window has passed.

    Returns the number of plays settled. Never raises.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    learning_conn = _open(Path(db_path))
    if learning_conn is None:
        return 0

    # tool_events lives in memory.db; if missing, evidence lookups simply fail.
    memory_conn = _open(Path(tool_events_db))
    owns_bandit = bandit is None
    if bandit is None:
        bandit = ContextualBandit(Path(db_path), profile_id=str(profile_id))

    settled = 0
    try:
        rows = _fetch_unsettled(learning_conn, str(profile_id), current)
        for row in rows:
            played = _parse_iso(row["played_at"])
            if played is None:
                continue
            age = (current - played).total_seconds()
            if age < _MIN_AGE_SEC:
                continue  # not yet settleable

            top3 = _top3_fact_ids(learning_conn, row["query_id"])
            reward: float | None = None
            kind = "default"
            if memory_conn is not None and _tool_event_hit(
                memory_conn, played, top3,
            ):
                reward = 1.0
                kind = "proxy_position"
            elif memory_conn is not None and _requery_detected(
                memory_conn, played, row["query_id"],
            ):
                reward = 0.0
                kind = "proxy_requery"
            elif age > _MAX_AGE_SEC:
                # P1: uncertain default after 120 s window closes.
                reward = 0.5
                kind = "default"
            else:
                # Between 60 and 120 s with no evidence yet — wait.
                continue

            if bandit.update(int(row["play_id"]), reward, kind=kind):
                settled += 1
    finally:
        try:
            learning_conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass
        if memory_conn is not None:
            try:
                memory_conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass
        # Don't close a caller-owned bandit instance.
        _ = owns_bandit

    return settled


__all__ = ("settle_stale_plays",)
