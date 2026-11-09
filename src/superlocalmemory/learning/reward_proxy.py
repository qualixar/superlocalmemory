# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §3.5 + §5.6

"""Proxy settlement for bandit plays (v3.4.22 only).


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

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from superlocalmemory.core.topic_signature import compute_topic_signature
from superlocalmemory.learning.bandit import ContextualBandit
from superlocalmemory.learning.engagement_features import (
    EngagementFeatures,
    extract_features,
)
from superlocalmemory.learning.reward_model import score

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
        # shown_fact_ids arrives with M044; the NULL alias keeps this working
        # on a store where it has not run yet.
        for columns in (
            "play_id, query_id, played_at, stratum, shown_fact_ids",
            "play_id, query_id, played_at, stratum, NULL AS shown_fact_ids",
        ):
            try:
                return learning_conn.execute(
                    f"SELECT {columns} FROM bandit_plays "
                    "WHERE profile_id = ? AND settled_at IS NULL "
                    "ORDER BY played_at ASC LIMIT 500",
                    (profile_id,),
                ).fetchall()
            except sqlite3.Error:
                continue
        return []
    except sqlite3.Error as exc:
        logger.debug("reward_proxy: fetch_unsettled: %s", exc)
        return []


def _shown_fact_ids(
    learning_conn: sqlite3.Connection,
    play_id: int,
) -> list[str]:
    """The fact_ids this play recorded at recall time (M044), or [].

    Preferred over the ``learning_signals`` lookup below because it does not
    depend on the exposure enqueue, which is off because it wrote a row per
    displayed memory per query and badly inflated the ranking-phase counter.
    With no signals rows there is nothing to look for, so plays fell through to
    the age-based default and arms stayed on their priors.

    Returns [] on a store where M044 has not run — "no such column" is an
    ``sqlite3.Error`` and is caught, so an unmigrated install falls back to the
    old lookup rather than losing settlement entirely.
    """
    try:
        row = learning_conn.execute(
            "SELECT shown_fact_ids FROM bandit_plays WHERE play_id = ?",
            (int(play_id),),
        ).fetchone()
    except sqlite3.Error:
        return []
    raw = (row[0] if row else None) or ""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(f) for f in parsed[:3] if f]


def _top3_fact_ids(
    learning_conn: sqlite3.Connection,
    query_id: str,
) -> list[str]:
    """Return up to 3 fact_ids for this query_id ordered by position ASC.

    Returns [] if learning_signals is missing or has no rows for this qid.
    Kept as the fallback for plays written before M044 and for installs that
    do run the exposure enqueue.
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
    profile_id: str | None = None,
) -> bool:
    """True iff any tool_events row references any fact_id within +30 s.

    Tenant-scoped: without a profile predicate, one tenant's reward signal
    would be reinforced by another tenant's tool events firing in the same
    30 s window — cross-contaminated behavioral learning.
    """
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
    # Add the profile predicate only when tool_events actually has the column
    # (schema varies by install).
    scope_sql, scope_params = "", ()
    if profile_id is not None:
        try:
            cols = {r[1] for r in memory_conn.execute(
                "PRAGMA table_info(tool_events)").fetchall()}
            if "profile_id" in cols:
                scope_sql = " AND profile_id = ?"
                scope_params = (profile_id,)
        except sqlite3.Error:
            pass
    try:
        candidate_rows = memory_conn.execute(
            "SELECT payload_json FROM tool_events "
            "WHERE occurred_at BETWEEN ? AND ? "
            "  AND payload_json IS NOT NULL" + scope_sql,
            (start, end, *scope_params),
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
    profile_id: str | None = None,
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

    # Tenant scope (guarded — tool_events schema varies): without it, another
    # profile's recall events in the same window count as this profile's
    # requeries, corrupting reward attribution.
    scope_sql, scope_params = "", ()
    if profile_id is not None:
        try:
            cols = {r[1] for r in memory_conn.execute(
                "PRAGMA table_info(tool_events)").fetchall()}
            if "profile_id" in cols:
                scope_sql = " AND profile_id = ?"
                scope_params = (profile_id,)
        except sqlite3.Error:
            pass

    # Read *query* text from tool_events payload for within-window events;
    # compute topic sig on each, compare against the original query's sig.
    try:
        rows = memory_conn.execute(
            "SELECT payload_json FROM tool_events "
            "WHERE occurred_at > ? AND occurred_at <= ? "
            "  AND tool_name = 'recall'" + scope_sql + " LIMIT 20",
            (start, end, *scope_params),
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
            "WHERE occurred_at <= ? AND tool_name = 'recall'" + scope_sql + " "
            "ORDER BY occurred_at DESC LIMIT 1",
            (played_at.isoformat(timespec="seconds"), *scope_params),
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


def _default_deadline(
    learning_conn: sqlite3.Connection, play: sqlite3.Row,
) -> float:
    """Age in seconds at which this play may be defaulted to 0.5.

    120 s for a play that recorded nothing — no evidence can ever arrive for
    it, so holding it open buys nothing but an unsettled row.

    Longer for a play that recorded which memories it showed (M044), because a
    reported outcome CAN settle it and two minutes is not long enough for
    anyone to decide whether an answer helped. Defaulting first would make the
    neutral fallback win a race it should not be in — which is how 1,405
    consecutive plays came to apply exactly 0.5.
    """
    try:
        from superlocalmemory.learning.reward_from_outcomes import GRACE_SEC
    except Exception:  # pragma: no cover — defensive
        return float(_MAX_AGE_SEC)
    try:
        raw = play["shown_fact_ids"]
    except (IndexError, KeyError):
        return float(_MAX_AGE_SEC)
    # PARSE, do not test the raw string. "[]" is truthy, so a play that recorded
    # no memories used to buy the full grace window it could never use: nothing
    # can overlap an empty set, so it stayed unsettled forever, and the
    # retention sweep only removes SETTLED rows. Reproduced: "[]" returned
    # 900 s where it should return 120 s.
    try:
        parsed = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        parsed = []
    has_evidence = bool(parsed) if isinstance(parsed, list) else False
    return (
        max(float(_MAX_AGE_SEC), GRACE_SEC) if has_evidence
        else float(_MAX_AGE_SEC)
    )


#: Once a play is older than this with nothing observed, it is closed without
#: touching the posterior. Leaving it open forever would make every settler
#: pass rescan it; settling it at a neutral value is what this module stopped
#: doing. Closed-and-unjudged is the third option that was missing.
_ABSTAIN_EXPIRY_SEC = 900


def _expire_play(learning_conn: sqlite3.Connection, play_id: int) -> bool:
    """Close a play without applying any reward.

    The posterior is deliberately untouched: nothing was observed, so there is
    nothing to learn, and a neutral update would shrink the arm's variance
    around its prior and make it harder to move once evidence does arrive.
    """
    try:
        learning_conn.execute(
            "UPDATE bandit_plays SET settled_at = ?, settlement_type = ? "
            "WHERE play_id = ? AND settled_at IS NULL",
            (datetime.now(timezone.utc).isoformat(), "unobserved", int(play_id)),
        )
        learning_conn.commit()
        return True
    except sqlite3.Error as exc:
        logger.debug("reward_proxy: expire play %s: %s", play_id, exc)
        return False


def _session_for_play(
    memory_conn: sqlite3.Connection | None,
    query_id: str,
    played_at: datetime,
    profile_id: str,
) -> str:
    """The conversation this play happened in, or "" when it cannot be named.

    The play and its outcome ticket are minted with different uuids on the same
    recall, so the query_id is tried first and a time-and-profile match is the
    fallback. An empty answer means no observation is possible, which the
    caller turns into an abstention rather than a guess.
    """
    if memory_conn is None:
        return ""
    try:
        row = memory_conn.execute(
            "SELECT session_id FROM pending_outcomes WHERE recall_query_id = ? "
            "LIMIT 1", (str(query_id),),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
        window_ms = 5000
        centre = int(played_at.timestamp() * 1000)
        row = memory_conn.execute(
            "SELECT session_id FROM pending_outcomes "
            "WHERE profile_id = ? AND created_at_ms BETWEEN ? AND ? "
            "ORDER BY ABS(created_at_ms - ?) LIMIT 1",
            (str(profile_id), centre - window_ms, centre + window_ms, centre),
        ).fetchone()
        return str(row[0]) if row and row[0] else ""
    except sqlite3.Error:
        return ""


def _ips_for_play(
    learning_conn: sqlite3.Connection, play_id: int, stratum: str, profile_id: str,
):
    """Inverse-propensity weight for this play against its stratum's arms."""
    from superlocalmemory.learning.propensity import ips_weight

    try:
        row = learning_conn.execute(
            "SELECT arm_id FROM bandit_plays WHERE play_id = ?", (int(play_id),),
        ).fetchone()
        if not row or not row[0]:
            return ips_weight(None, None)
        arm_id = str(row[0])
        rows = learning_conn.execute(
            "SELECT arm_id, alpha, beta FROM bandit_arms "
            "WHERE profile_id = ? AND stratum = ?",
            (str(profile_id), str(stratum or "")),
        ).fetchall()
    except sqlite3.Error:
        return ips_weight(None, None)

    mine, others = None, []
    for candidate, alpha, beta in rows:
        if str(candidate) == arm_id:
            mine = (float(alpha), float(beta))
        else:
            others.append((float(alpha), float(beta)))
    return ips_weight(mine, others)


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

            # The play's own record first; the signals table only as a
            # fallback for rows written before M044.
            top3 = (
                _shown_fact_ids(learning_conn, row["play_id"])
                or _top3_fact_ids(learning_conn, row["query_id"])
            )
            # The ladder this replaces asked one question — did a recalled
            # fact_id appear verbatim in a later tool event — and answered 0.5
            # whenever it could not tell. Both halves failed: nothing makes an
            # agent echo the marker, and 0.5 is not an absence of judgement but
            # a confident one, tightening the posterior around its prior.
            requeried = bool(
                memory_conn is not None and _requery_detected(
                    memory_conn, played, row["query_id"],
                    profile_id=str(profile_id),
                )
            )
            marker_hit = bool(
                memory_conn is not None and _tool_event_hit(
                    memory_conn, played, top3, profile_id=str(profile_id),
                )
            )
            session_id = _session_for_play(
                memory_conn, str(row["query_id"] or ""), played, str(profile_id),
            )
            if memory_conn is not None and session_id:
                features = extract_features(
                    memory_conn,
                    session_id=session_id,
                    profile_id=str(profile_id),
                    fact_ids=top3,
                    recalled_at=played,
                    requeried=requeried,
                    marker_hit=marker_hit,
                )
            else:
                features = EngagementFeatures(
                    requeried=requeried, marker_hit=marker_hit,
                )

            decision = score(features)
            if decision.reward is None:
                # Nothing observed. Wait while the window is still open, then
                # close the play unjudged rather than inventing a number.
                if age > _ABSTAIN_EXPIRY_SEC:
                    _expire_play(learning_conn, int(row["play_id"]))
                continue

            estimate = _ips_for_play(
                learning_conn, int(row["play_id"]),
                str(row["stratum"] or ""), str(profile_id),
            )
            if bandit.update(
                int(row["play_id"]), decision.reward,
                kind=decision.kind, weight=estimate.weight,
            ):
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
        # v3.4.33: close the threadlocal bandit connection so pool threads
        # from asyncio.to_thread don't leak file descriptors to learning.db.
        try:
            from superlocalmemory.learning.bandit import close_threadlocal_conn
            close_threadlocal_conn()
        except Exception:  # pragma: no cover — defensive
            pass

    return settled


__all__ = ("settle_stale_plays",)
