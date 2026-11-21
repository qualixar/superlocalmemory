# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Settle a bandit play from an authenticated outcome.

WHY THIS FILE EXISTS
--------------------
``reward_proxy.py`` has said since v3.4.22:

    Replaced in v3.4.22 by ``reward_from_outcomes.py`` — DO NOT extend this
    module beyond the proxy window contract.

That file was never written. So the replacement never arrived, and the module
that tells you not to extend it remained the only settlement path — one that
cannot work. Measured, not inferred:

* ``reward_proxy._tool_event_hit`` selects ``payload_json`` from ``tool_events``
  ``WHERE occurred_at BETWEEN ? AND ?``. **Neither column exists** on that
  table, on this install or in any DDL in the codebase. The query raises
  ``no such column: payload_json``, a bare ``except sqlite3.Error`` returns
  False, and the "cited" branch of the ladder is therefore unreachable.
* Even with the names corrected there would be nothing to find: **0 of 2,002**
  ``tool_events`` rows on a live store contain a 16-hex token, let alone a
  real ``fact_id``. That table records tool names and summaries, not the
  memories a recall returned.

So every settlement fell through to the 120-second default of 0.5, which is
visible in the arms as an exact tie::

    SUM(alpha) = SUM(beta) = 867.5 = 165 priors + 1,405 plays x 0.5

WHAT THIS USES INSTEAD
----------------------
``action_outcomes`` (memory.db, M006) — the table an *explicit* outcome report
already writes, with a reward the reporter chose:
``success=1.0 / failure=0.0 / partial=0.5``. That is an authenticated signal
about whether the memories helped, which is exactly what a play needs and what
an exposure is not.

It carries ``recall_query_id``, a join key straight to ``bandit_plays.query_id``.
Since 4.1.25 that id is the one recall returns as ``query_id`` (one identity
from engine to play), and every surface writes it through
``outcome_events``. When it is present the outcome settles exactly
that play; only an outcome that names no recall falls back to overlap between
its ``fact_ids_json`` and the ``shown_fact_ids`` the play recorded (M044).

An outcome is spent once: its id is written on the play it settled, in the
same transaction as the posterior change, under a unique index (M055).

THE GRACE WINDOW, AND WHY IT IS NOT 120 SECONDS
-----------------------------------------------
``reward_proxy`` defaults a play to 0.5 once it is 120 seconds old. A human or
an agent deciding whether a memory helped does not do so within two minutes, so
that deadline would claim every play before any real outcome arrived — the
default would win a race it should not be in.

A play that recorded ``shown_fact_ids`` therefore waits ``_GRACE_SEC`` (default
900) before it may be defaulted. A play with no recorded facts keeps the old
120-second behaviour, because nothing can ever settle it from evidence and
holding it open buys nothing.

NOTHING HERE FABRICATES A REWARD. If no outcome is reported, the play settles
neutral and is *labelled* ``default`` so it stays distinguishable from a
genuinely neutral outcome. Those two were indistinguishable before, which is why
this went unnoticed for months.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from superlocalmemory.learning.bandit import ContextualBandit
from superlocalmemory.learning.pcos import update_scores
from superlocalmemory.storage.json_columns import load_json_column

logger = logging.getLogger(__name__)

__all__ = ["settle_from_outcomes", "GRACE_SEC", "SETTLEMENT_KIND"]

#: How long a play that recorded its shown memories waits for a reported
#: outcome before it may be defaulted. Deliberately far longer than the proxy's
#: 120 s: see the module docstring.
GRACE_SEC: float = float(os.environ.get("SLM_OUTCOME_GRACE_SEC", "900"))

#: How far after a play an outcome may arrive and still be attributed to it.
#: Same order as the grace window — an outcome reported later than this is about
#: something else.
_MATCH_WINDOW_SEC: float = GRACE_SEC

#: Written to ``bandit_plays.settlement_type``. A settlement that came from a
#: real report must never be mistaken for the neutral default.
SETTLEMENT_KIND = "outcome_reported"

_MAX_PLAYS_PER_PASS = 500


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    raw = str(ts).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _open(path: Path) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        logger.debug("reward_from_outcomes: open %s failed: %s", path, exc)
        return None


def _shown(raw: object) -> set[str]:
    """``shown_fact_ids`` JSON -> set. Empty on anything unexpected."""
    if not raw:
        return set()
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {str(f) for f in parsed if f}


def _unsettled_plays(
    conn: sqlite3.Connection, profile_id: str,
) -> list[sqlite3.Row]:
    """Unsettled plays, oldest first. Degrades if M044 has not run."""
    for columns in (
        "play_id, query_id, played_at, shown_fact_ids",
        "play_id, query_id, played_at, NULL AS shown_fact_ids",
    ):
        try:
            return conn.execute(
                f"SELECT {columns} FROM bandit_plays "
                "WHERE profile_id = ? AND settled_at IS NULL "
                "ORDER BY played_at ASC LIMIT ?",
                (str(profile_id), _MAX_PLAYS_PER_PASS),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("reward_from_outcomes: play fetch (%s): %s", columns, exc)
    return []


def _candidate_outcomes(
    conn: sqlite3.Connection, profile_id: str, since: datetime,
) -> list[dict]:
    """Reported outcomes from ``since`` onward, newest first.

    Read once per pass rather than once per play: the window holds a handful of
    rows in practice, and a per-play query would scan this table N times on the
    settlement thread.
    """
    try:
        rows = conn.execute(
            "SELECT outcome_id, fact_ids_json, outcome, reward, timestamp, "
            "       recall_query_id "
            "FROM action_outcomes "
            "WHERE profile_id = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT 2000",
            (str(profile_id), since.isoformat()),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("reward_from_outcomes: outcome fetch: %s", exc)
        return []

    out: list[dict] = []
    for r in rows:
        at = _parse_iso(r["timestamp"])
        if at is None:
            continue
        try:
            facts = load_json_column(r["fact_ids_json"], [], column="action_outcomes.fact_ids_json")
        except (TypeError, ValueError):
            facts = []
        if not isinstance(facts, list):
            facts = []
        try:
            reward = float(r["reward"])
        except (TypeError, ValueError):
            continue
        out.append({
            "id": str(r["outcome_id"]),
            "at": at,
            "facts": {str(f) for f in facts if f},
            "reward": max(0.0, min(1.0, reward)),
            "query_id": str(r["recall_query_id"] or ""),
        })
    return out


def _spent_outcomes(conn: sqlite3.Connection, profile_id: str) -> set[str]:
    """Outcome ids already spent on a play — durable, across passes (M055)."""
    try:
        rows = conn.execute(
            "SELECT outcome_id FROM bandit_plays "
            "WHERE profile_id = ? AND outcome_id IS NOT NULL",
            (str(profile_id),),
        ).fetchall()
    except sqlite3.Error:
        return set()  # M055 not applied: only this pass can dedupe
    return {str(r[0]) for r in rows}


def _match(
    play_at: datetime,
    query_id: str,
    shown: set[str],
    outcomes: list[dict],
    claimed: set[str],
) -> dict | None:
    """The one outcome attributable to this play, or None.

    AN OUTCOME IS EVIDENCE ABOUT ONE RECALL, AND IS CONSUMED ONCE
    ------------------------------------------------------------
    ``claimed`` starts from every outcome id already written on a play
    (``bandit_plays.outcome_id``, M055) — not from an empty set per pass. The
    per-pass set let the next scheduler pass spend the same outcome on another
    play: one report moved three arms in three passes.

    AN OUTCOME THAT NAMES ITS RECALL IS ONLY ABOUT THAT RECALL
    ---------------------------------------------------------
    When an outcome carries a ``recall_query_id`` it may settle the play with
    that exact query id and no other. It used to fall through to the overlap
    match, so an older play that happened to show the same memory stole the
    credit for a report that explicitly named a different query.

    Overlap matching (memories in common, inside the window) remains for
    outcomes that name no recall — the strongest link available for them.
    Both paths require the outcome to fall inside the window.
    """
    horizon = play_at + timedelta(seconds=_MATCH_WINDOW_SEC)

    def _usable(o: dict) -> bool:
        return o["id"] not in claimed and play_at <= o["at"] <= horizon

    def _first(candidates: list[dict]) -> dict | None:
        return min(candidates, key=lambda o: o["at"]) if candidates else None

    if query_id:
        exact = _first([o for o in outcomes
                        if o["query_id"] == query_id and _usable(o)])
        if exact is not None:
            return exact
    if not shown:
        return None
    return _first([o for o in outcomes
                   if not o["query_id"] and _usable(o) and (o["facts"] & shown)])


def _credit_facts(memory_conn: sqlite3.Connection, profile_id: str,
                  outcome: dict, play_id: int, shown: set[str]) -> bool:
    """Fold one spent outcome into PCOS exactly once. Returns True if credited.

    The marker and the score change commit together in memory.db, so a crash
    between the bandit commit (learning.db) and this one is repaired by
    :func:`_recover_credits` without ever crediting twice.
    """
    judged = sorted(outcome["facts"] & shown) if shown else []
    try:
        memory_conn.execute("BEGIN IMMEDIATE")
        cur = memory_conn.execute(
            "UPDATE action_outcomes SET context_json = "
            "json_set(CASE WHEN json_valid(context_json) THEN context_json "
            "ELSE '{}' END, '$._credited_play', ?) "
            "WHERE outcome_id = ? AND (NOT json_valid(context_json) OR "
            "json_extract(context_json, '$._credited_play') IS NULL)",
            (int(play_id), outcome["id"]),
        )
        if cur.rowcount == 1 and judged:
            update_scores(memory_conn, profile_id, judged, outcome["reward"])
        memory_conn.execute("COMMIT")
        return cur.rowcount == 1
    except sqlite3.Error as exc:  # pragma: no cover — defensive
        memory_conn.execute("ROLLBACK")
        logger.debug("pcos credit skipped: %s", exc)
        return False


def _feed_shadow(learning_db: Path, profile_id: str, query_id: str,
                 reward: float) -> None:
    """A settled explicit reward is also an observation for a shadow test."""
    if not query_id:
        return
    try:
        from superlocalmemory.core.recall_pipeline import feed_recall_settled

        feed_recall_settled(learning_db=str(learning_db),
                            profile_id=str(profile_id), query_id=query_id,
                            reward=float(reward))
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("shadow feed skipped: %s", exc)


def _recover_credits(learning_conn: sqlite3.Connection,
                     memory_conn: sqlite3.Connection, profile_id: str,
                     outcomes: dict[str, dict]) -> int:
    """Credit PCOS for plays whose outcome was spent but never credited."""
    try:
        rows = learning_conn.execute(
            "SELECT play_id, outcome_id, shown_fact_ids FROM bandit_plays "
            "WHERE profile_id = ? AND outcome_id IS NOT NULL "
            "ORDER BY settled_at DESC LIMIT ?",
            (str(profile_id), _MAX_PLAYS_PER_PASS),
        ).fetchall()
    except sqlite3.Error:
        return 0
    repaired = 0
    for row in rows:
        outcome = outcomes.get(str(row["outcome_id"]))
        if outcome is not None and _credit_facts(
                memory_conn, profile_id, outcome, int(row["play_id"]),
                _shown(row["shown_fact_ids"])):
            repaired += 1
    return repaired


def settle_from_outcomes(
    profile_id: str,
    learning_db: Path | str,
    memory_db: Path | str,
    *,
    now: datetime | None = None,
    bandit: ContextualBandit | None = None,
) -> int:
    """Settle plays from reported outcomes. Returns the count. Never raises.

    Run BEFORE ``reward_proxy.settle_stale_plays``: the proxy defaults plays,
    and a real outcome arriving later would find nothing left to settle.
    One settlement = one outcome -> one play (durable, M055) -> one posterior
    update -> one PCOS credit (memories named AND shown) -> one shadow-test
    observation for that query id.
    """
    learning_conn = _open(Path(learning_db))
    if learning_conn is None:
        return 0
    memory_conn = _open(Path(memory_db))
    if memory_conn is None:
        learning_conn.close()
        return 0
    memory_conn.isolation_level = None  # explicit transactions in _credit_facts
    owns_bandit = bandit is None
    if bandit is None:
        bandit = ContextualBandit(Path(learning_db), profile_id=str(profile_id))
    try:
        return _settle_pass(learning_conn, memory_conn, bandit, profile_id,
                            Path(learning_db), now or datetime.now(timezone.utc))
    except sqlite3.Error as exc:  # pragma: no cover — defensive
        logger.warning("reward_from_outcomes: %s", exc)
        return 0
    finally:
        for conn in (learning_conn, memory_conn):
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover
                pass
        if owns_bandit:
            try:
                from superlocalmemory.learning.bandit import (
                    close_threadlocal_conn,
                )

                close_threadlocal_conn()
            except Exception:  # pragma: no cover — defensive
                pass


def _settle_pass(learning_conn: sqlite3.Connection,
                 memory_conn: sqlite3.Connection, bandit: ContextualBandit,
                 profile_id: str, learning_db: Path, current: datetime) -> int:
    plays = _unsettled_plays(learning_conn, profile_id)
    oldest = min((p for p in (_parse_iso(r["played_at"]) for r in plays)
                  if p is not None), default=current)
    horizon_start = min(oldest, current - timedelta(days=1))
    outcomes = _candidate_outcomes(memory_conn, profile_id, horizon_start)
    by_id = {o["id"]: o for o in outcomes}
    _recover_credits(learning_conn, memory_conn, profile_id, by_id)
    if not plays or not outcomes:
        return 0
    claimed = _spent_outcomes(learning_conn, profile_id)
    settled = 0
    for row in plays:
        played = _parse_iso(row["played_at"])
        if played is None:
            continue
        shown = _shown(row["shown_fact_ids"])
        qid = str(row["query_id"] or "")
        outcome = _match(played, qid, shown, outcomes, claimed)
        if outcome is None:
            continue
        if not bandit.update(int(row["play_id"]), outcome["reward"],
                             kind=SETTLEMENT_KIND, outcome_id=outcome["id"]):
            claimed.add(outcome["id"])  # spent elsewhere meanwhile
            continue
        settled += 1
        claimed.add(outcome["id"])
        # Credit ONLY the memories the outcome named that this play showed.
        _credit_facts(memory_conn, profile_id, outcome, int(row["play_id"]),
                      shown)
        _feed_shadow(learning_db, profile_id, qid, outcome["reward"])
    if settled:
        logger.info(
            "reward_from_outcomes: settled %d play(s) from reported outcomes",
            settled,
        )
    return settled
