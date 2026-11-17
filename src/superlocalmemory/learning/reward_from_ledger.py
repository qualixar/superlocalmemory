# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Settle bandit plays from what agents used of what the play surfaced (W3-6e, D-45).

A bandit play is the choice of one channel weighting for one recall
(``bandit_plays.query_id`` = the recall's ``query_id``). Its reward used to come
from an explicit outcome report or, failing that, from an engagement proxy that
looked 30 seconds ahead and closed the play unjudged after 15 minutes. The
injection ledger now names the recall each pull delivered
(``injection_deliveries.recall_query_id``) and records every use of each
delivered memory for 24 hours (``injection_usage``). So a play that a delivery
names is settled here, once that delivery's outcome window has closed:

* reward 1.0 when the agent acted on any memory the play surfaced (its
  ``shown_fact_ids``, else the delivery's top five), within the window
  (``outcomes.effects.USED_SIGNALS``), else 0.0;
* weighted by the same inverse-propensity estimate as the proxy, so arms the
  bandit rarely plays are not under-counted;
* kind ``ledger_usage``.

Order (``server.bandit_loops``): explicit outcomes first (a judgement wins),
then this, then the proxy, which leaves alone any play a delivery names until
its window has closed (``awaiting_ledger``), so it no longer closes plays the
ledger is about to judge. Never raises.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.outcomes import effects, usage

logger = logging.getLogger(__name__)

SETTLEMENT = "ledger_usage"
#: The memories of a play's answer that count, when the play recorded none.
TOP = 5

_DELIVERIES = ("SELECT delivery_id, julianday(delivered_at) + ? / 24.0 > julianday(?)"
               " FROM injection_deliveries WHERE profile_id = ? AND recall_query_id = ?")


def _deliveries(memory: sqlite3.Connection, profile_id: str, query_id: str,
                now: datetime) -> list[tuple[str, bool]]:
    """(delivery id, window still open) for each delivery naming this recall."""
    if not query_id:
        return []
    try:
        return [(r[0], bool(r[1])) for r in memory.execute(
            _DELIVERIES, (effects.OUTCOME_WINDOW_H, usage.stamp(now), profile_id, query_id))]
    except sqlite3.OperationalError:
        return []  # a store from before the ledger


def awaiting_ledger(memory: sqlite3.Connection | None, profile_id: str, query_id: str,
                    now: datetime) -> bool:
    """True while a delivery names this recall and its outcome window is open."""
    if memory is None:
        return False
    return any(open_ for _d, open_ in _deliveries(memory, profile_id, query_id, now))


def _shown(raw) -> list[str]:
    try:
        value = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _used(memory: sqlite3.Connection, delivery_ids: list[str], facts: list[str]) -> bool:
    marks_d = ",".join("?" * len(delivery_ids))
    fact_filter, params = "", []
    if facts:
        fact_filter = f" AND u.fact_id IN ({','.join('?' * len(facts))})"
        params = facts
    else:  # the play recorded nothing: its answer's top five
        fact_filter = (" AND u.fact_id IN (SELECT r.fact_id FROM injection_receipts r"
                       " WHERE r.delivery_id = u.delivery_id AND r.delivered = 1"
                       f" AND r.position <= {TOP})")
    sql = ("SELECT 1 FROM injection_usage u JOIN injection_deliveries d"
           " ON d.delivery_id = u.delivery_id"
           f" WHERE u.delivery_id IN ({marks_d}){fact_filter}"
           f" AND u.signal IN ({','.join('?' * len(effects.USED_SIGNALS))})"
           " AND julianday(u.observed_at) <= julianday(d.delivered_at) + ? / 24.0 LIMIT 1")
    return memory.execute(sql, (*delivery_ids, *params, *effects.USED_SIGNALS,
                                effects.OUTCOME_WINDOW_H)).fetchone() is not None


def settle_from_ledger(profile_id: str, learning_db: Path | str, memory_db: Path | str, *,
                       now: datetime | None = None, bandit=None) -> int:
    """Settle every unsettled play whose delivery's window has closed. Returns
    how many were settled."""
    from superlocalmemory.learning.bandit import ContextualBandit
    from superlocalmemory.learning.reward_proxy import _fetch_unsettled, _ips_for_play, _open

    current = now or datetime.now(UTC)
    learning = _open(Path(learning_db))
    if learning is None or not Path(memory_db).exists():
        return 0
    settled = 0
    try:
        memory = sqlite3.connect(f"{Path(memory_db).resolve().as_uri()}?mode=ro", uri=True,
                                 timeout=5.0)
    except sqlite3.Error:
        learning.close()
        return 0
    bandit = bandit or ContextualBandit(Path(learning_db), profile_id=str(profile_id))
    try:
        for play in _fetch_unsettled(learning, str(profile_id), current):
            found = _deliveries(memory, str(profile_id), str(play["query_id"] or ""), current)
            if not found or any(open_ for _d, open_ in found):
                continue  # not named by a delivery (the proxy's), or still observing
            reward = 1.0 if _used(memory, [d for d, _ in found],
                                  _shown(play["shown_fact_ids"])) else 0.0
            weight = _ips_for_play(learning, int(play["play_id"]), str(play["stratum"] or ""),
                                   str(profile_id)).weight
            if bandit.update(int(play["play_id"]), reward, kind=SETTLEMENT, weight=weight):
                settled += 1
    except sqlite3.Error as exc:
        logger.warning("ledger settlement stopped: %s", exc)
    finally:
        memory.close()
        learning.close()
    return settled


__all__ = ["SETTLEMENT", "awaiting_ledger", "settle_from_ledger"]
