# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Labelled ranker-training rows: exposures joined to authentic outcomes.

WHY THIS EXISTS
------------------------------
The trainer's join read ``action_outcomes`` from learning.db. That table lives
in memory.db; learning.db has none, so the query raised, the fetch returned
zero rows, and the ranker never trained however much feedback arrived. The
integration test for that join only passed because it copied the outcome rows
into learning.db first.

THE CONTRACT
------------
* **Features** come only from exposures — ``learning_signals`` rows of an
  exposure kind with a non-synthetic ``learning_features`` row — i.e. what the
  recall path recorded (asynchronously, via the signal worker) about each
  shown candidate, keyed by the recall's ``query_id``. An exposure is never
  counted as feedback (``signal_kinds``).
* **Labels** come only from outcomes that name the recall
  (``action_outcomes.recall_query_id``) and the memory (``fact_ids_json``).
  The earliest outcome wins, matching settlement. Displayed position is never
  used as a label.
* **Groups** are queries with at least one labelled memory. Other memories
  exposed in a judged query get the neutral reward 0.5 — they were shown and
  not singled out — and are marked ``label_source = 'unjudged_exposure'``.
  Queries nobody judged are left out entirely.

* **Ledger labels** (W3-6c, D-45): a pull delivery names the recall it
  answered (``injection_deliveries.recall_query_id``), so every memory it
  delivered is labelled once its outcome window has closed: 1.0 when the agent
  acted on it (``outcomes.effects.USED_SIGNALS`` within the window), else 0.0
  (``label_source`` ``ledger_used`` / ``ledger_unused``), with the receipt's
  logged inclusion propensity. A memory recall did not deliver (cut by the
  budget) gets no label. An explicit outcome for the same row wins.

memory.db is ATTACHed to the learning connection for this read only (SELECTs
only), so the join runs in SQLite and nothing is copied.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["NEUTRAL_REWARD", "fetch_labelled_rows", "ledger_source", "outcome_source"]

#: Reward given to an exposed but unjudged memory inside a judged query.
NEUTRAL_REWARD = 0.5

_EXPOSURE_TYPES = "('candidate', 'shown', 'legacy_feedback')"


def outcome_source(conn: sqlite3.Connection, memory_db: Path | None) -> str | None:
    """Qualified name of the outcome table visible on ``conn``, or None.

    A learning.db that carries its own ``action_outcomes`` (older test
    fixtures) is used as is. Otherwise ``memory_db`` is attached as ``mem``.
    """
    local = {r[1] for r in conn.execute("PRAGMA table_info(action_outcomes)")}
    if {"reward", "recall_query_id"} <= local:
        return "action_outcomes"
    if memory_db is None or not Path(memory_db).exists():
        return None
    try:
        # A plain path, not a file: URI — a connection opened without
        # uri=True would create a file literally named "file:...".
        conn.execute("ATTACH DATABASE ? AS mem", (str(Path(memory_db)),))
    except sqlite3.Error as exc:
        logger.warning("training rows: cannot attach memory.db: %s", exc)
        return None
    cols = {r[1] for r in conn.execute("PRAGMA mem.table_info(action_outcomes)")}
    if not {"reward", "recall_query_id"} <= cols:
        return None  # no outcome table, or M006 has not run on it
    return "mem.action_outcomes"


def _attached(conn: sqlite3.Connection, name: str) -> bool:
    return any(r[1] == name for r in conn.execute("PRAGMA database_list"))


def ledger_source(conn: sqlite3.Connection, memory_db: Path | None) -> str | None:
    """``"mem."`` when memory.db (attached as ``mem``) holds the injection ledger
    with ``recall_query_id``; ``""`` when this connection holds it itself; else None."""
    def has(prefix: str) -> bool:
        cols = {r[1] for r in conn.execute(f"PRAGMA {prefix}table_info(injection_deliveries)")}
        return "recall_query_id" in cols
    if has(""):
        return ""
    if not _attached(conn, "mem"):
        if memory_db is None or not Path(memory_db).exists():
            return None
        try:
            conn.execute("ATTACH DATABASE ? AS mem", (str(Path(memory_db)),))
        except sqlite3.Error as exc:
            logger.warning("training rows: cannot attach memory.db: %s", exc)
            return None
    return "mem." if has("mem.") else None


def _ledger_sql(prefix: str, synthetic_filter: str, n_signals: int) -> str:
    marks = ",".join("?" for _ in range(n_signals))
    return (
        "SELECT s.id AS signal_id, s.query_id, s.fact_id, s.position, "
        "       s.created_at, f.features_json, MIN(r.propensity) AS propensity, "
        "       MIN(r.position) AS shown_position, MIN(d.session_id) AS session_id, "
        "       MAX(EXISTS (SELECT 1 FROM " + prefix + "injection_usage u "
        "            WHERE u.delivery_id = r.delivery_id AND u.fact_id = r.fact_id "
        f"             AND u.signal IN ({marks}) "
        "              AND julianday(u.observed_at) <= julianday(d.delivered_at) + ? / 24.0"
        "       )) AS used "
        "FROM learning_signals s "
        "JOIN learning_features f "
        "  ON f.signal_id = s.id AND f.profile_id = s.profile_id "
        f"JOIN {prefix}injection_deliveries d "
        "  ON d.profile_id = s.profile_id AND d.recall_query_id = s.query_id "
        f"JOIN {prefix}injection_receipts r "
        "  ON r.delivery_id = d.delivery_id AND r.fact_id = s.fact_id AND r.delivered = 1 "
        "WHERE s.profile_id = ? AND s.query_id != '' "
        f"  AND s.signal_type IN {_EXPOSURE_TYPES}{synthetic_filter} "
        "  AND julianday(d.delivered_at) + ? / 24.0 <= julianday('now') "
        "GROUP BY s.id ORDER BY s.created_at DESC LIMIT ?"
    )


def _ledger_rows(conn: sqlite3.Connection, prefix: str, synthetic: str, *,
                 profile_id: str, limit: int) -> dict[int, dict]:
    from superlocalmemory.outcomes.effects import OUTCOME_WINDOW_H, USED_SIGNALS

    out: dict[int, dict] = {}
    try:
        rows = conn.execute(
            _ledger_sql(prefix, synthetic, len(USED_SIGNALS)),
            (*USED_SIGNALS, OUTCOME_WINDOW_H, profile_id, OUTCOME_WINDOW_H,
             int(limit))).fetchall()
    except sqlite3.OperationalError as exc:
        logger.debug("ledger labels unavailable: %s", exc)
        return out
    for row in rows:
        d = dict(row)
        used = bool(d.pop("used"))
        d["features"] = _features(d.pop("features_json"))
        d.update(outcome_reward=1.0 if used else 0.0, outcome_id=None,
                 label_source="ledger_used" if used else "ledger_unused")
        out[int(d["signal_id"])] = d
    return out


def _judged_sql(table: str, synthetic_filter: str) -> str:
    return (
        "SELECT s.id AS signal_id, s.query_id, s.fact_id, s.position, "
        "       s.created_at, f.features_json, o.reward AS outcome_reward, "
        "       o.outcome_id, o.timestamp AS outcome_at "
        "FROM learning_signals s "
        "JOIN learning_features f "
        "  ON f.signal_id = s.id AND f.profile_id = s.profile_id "
        f"JOIN {table} o "
        "  ON o.recall_query_id = s.query_id AND o.profile_id = s.profile_id "
        " AND o.reward IS NOT NULL "
        " AND EXISTS (SELECT 1 FROM json_each(o.fact_ids_json) j "
        "             WHERE j.value = s.fact_id) "
        "WHERE s.profile_id = ? AND s.query_id != '' "
        f"  AND s.signal_type IN {_EXPOSURE_TYPES}{synthetic_filter} "
        "  AND (julianday('now') - julianday(o.timestamp)) * 86400.0 >= ? "
        "ORDER BY s.created_at DESC LIMIT ?"
    )


def _exposures_sql(n_queries: int, synthetic_filter: str) -> str:
    marks = ",".join("?" for _ in range(n_queries))
    return (
        "SELECT s.id AS signal_id, s.query_id, s.fact_id, s.position, "
        "       s.created_at, f.features_json "
        "FROM learning_signals s "
        "JOIN learning_features f "
        "  ON f.signal_id = s.id AND f.profile_id = s.profile_id "
        f"WHERE s.profile_id = ? AND s.query_id IN ({marks}) "
        f"  AND s.signal_type IN {_EXPOSURE_TYPES}{synthetic_filter}"
    )


def _features(raw: object) -> dict:
    try:
        value = json.loads(str(raw or "{}"))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _earliest_per_signal(rows: list[sqlite3.Row]) -> dict[int, dict]:
    best: dict[int, dict] = {}
    for row in rows:
        d = dict(row)
        sid = int(d["signal_id"])
        if sid not in best or str(d["outcome_at"]) < str(best[sid]["outcome_at"]):
            best[sid] = d
    return best


def fetch_labelled_rows(
    conn: sqlite3.Connection, table: str | None, *, profile_id: str, limit: int,
    min_outcome_age_sec: int, include_synthetic: bool = False,
    ledger: str | None = None,
) -> list[dict]:
    """Rows for judged queries only, each with a real or neutral label.

    ``ledger`` is the table prefix from :func:`ledger_source` (None: no ledger)."""
    synthetic = "" if include_synthetic else " AND f.is_synthetic = 0"
    judged = {} if table is None else _earliest_per_signal(conn.execute(
        _judged_sql(table, synthetic),
        (profile_id, int(min_outcome_age_sec), int(limit)),
    ).fetchall())
    labelled = ({} if ledger is None else
                _ledger_rows(conn, ledger, synthetic, profile_id=profile_id, limit=limit))
    qids = sorted({str(d["query_id"]) for d in judged.values()})
    out: list[dict] = []
    for d in judged.values():
        d["features"] = _features(d.pop("features_json"))
        d["label_source"] = "outcome"
        d.pop("outcome_at", None)
        labelled.pop(int(d["signal_id"]), None)  # an explicit outcome wins
        out.append(d)
    out.extend(labelled.values())
    taken = {int(r["signal_id"]) for r in out}
    for start in range(0, len(qids), 500):
        chunk = qids[start:start + 500]
        for row in conn.execute(_exposures_sql(len(chunk), synthetic),
                                (profile_id, *chunk)).fetchall():
            if int(row["signal_id"]) in taken:
                continue
            d = dict(row)
            d["features"] = _features(d.pop("features_json"))
            d.update(outcome_reward=NEUTRAL_REWARD, outcome_id=None,
                     label_source="unjudged_exposure")
            out.append(d)
    out.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return out[: max(int(limit), 0)]
