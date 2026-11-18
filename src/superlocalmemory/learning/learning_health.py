# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Is SLM still learning? Every learner's state, measured (W3-6b, D-45).

SLM's learners stalled for months without anyone being told (F-48): the ranker
trained once, in April; the bandit stopped in July; feature logging fell from
22k rows a month to 2. Every status line still looked fine, because nothing
compared what a learner had learned with what it had been given. This module
does, for each learner, with one rule and no tuned threshold:

* ``learning``: its newest output is no older than its newest input by more
  than its own schedule (``cadence_h``, the interval its code runs on);
* ``stalled``: it has had new input for longer than its schedule allows;
* ``off``: a setting turns it off (the setting is named);
* ``waiting``: it learns only when an agent calls a tool, and none has yet;
* ``not_wired``: nothing in production feeds it (named so it is wired or
  removed, D-45 item 5);
* ``no_input``: nothing has arrived for it to learn from;
* ``retired``: deliberately not wired, with the reason (D-45 item 5).

Read-only, counts and times only, never memory text. Both stores are opened by
the caller with ``mode=ro``; a missing table reads as "not created".
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

LEARNING = "learning"
STALLED = "stalled"
OFF = "off"
WAITING = "waiting"
NOT_WIRED = "not_wired"
NO_INPUT = "no_input"
RETIRED = "retired"

#: Routing only: a learner that runs inside consolidation (every 6 h at most,
#: and only once the daemon has been idle) should have run within a day.
CONSOLIDATION_CADENCE_H = 24.0
#: Routing only: the ranker retrains once 24 h have passed with new labels
#: (``ranker_retrain_online``), so a day without a retrain is behind.
RANKER_CADENCE_H = 24.0
#: Routing only: co-retrieval is written by each recall; an hour of slack.
PER_RECALL_CADENCE_H = 1.0


@dataclass(frozen=True)
class Learner:
    name: str
    learns: str            # what it learns, in plain words
    rows: int | None       # None when its table was never created
    last: str | None       # its newest output (ISO), if any
    state: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _jd_to_iso(jd: float | None) -> str | None:
    if jd is None:
        return None
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return (epoch + timedelta(days=jd - 2440587.5)).isoformat(timespec="seconds")


#: The tables and time columns read here: fixed names, never caller input.
_TIME_COLUMNS = {
    "action_outcomes": "timestamp", "injection_usage": "observed_at",
    "fact_access_log": "accessed_at", "atomic_facts": "created_at",
    "learning_model_state": "trained_at", "co_retrieval_edges": "updated_at",
    "_store_patterns": "updated_at", "learning_signals": "created_at",
    "bandit_plays": "played_at", "agent_experiences": "created_at",
    "execution_learning_events": "created_at", "workflow_patterns": "created_at",
    "transferable_patterns": "NULL",
}


def _stat(conn: sqlite3.Connection | None, table: str,
          profile_id: str) -> tuple[int | None, float | None]:
    """(rows, newest time as a Julian day) of one profile in a table;
    (None, None) when the table does not exist."""
    if conn is None:
        return None, None
    column = _TIME_COLUMNS[table]
    try:
        row = conn.execute(f"SELECT COUNT(*), MAX(julianday({column})) FROM {table} "
                           "WHERE profile_id = ?", (profile_id,)).fetchone()
    except sqlite3.OperationalError:
        return None, None
    return int(row[0]), row[1]


def _flow(name: str, learns: str, out: tuple, inp: float | None, cadence_h: float,
          source: str) -> Learner:
    """The state of a learner fed by ``source`` whose input is newest at ``inp``."""
    rows, last = out
    if rows is None:
        return Learner(name, learns, None, None, NO_INPUT if inp is None else STALLED,
                       "not created yet" if inp is None else f"{source} is waiting and its "
                       "table was never created")
    if inp is None:
        return Learner(name, learns, rows, _jd_to_iso(last), NO_INPUT if not rows else LEARNING,
                       f"no {source} yet" if not rows else "up to date")
    behind_h = (inp - (last or 0.0)) * 24.0 if last else None
    if behind_h is None or behind_h > cadence_h:
        days = None if behind_h is None else round(behind_h / 24.0)
        since = ("never" if days is None else "under a day" if days < 1
                 else "1 day" if days == 1 else f"{days} days")
        return Learner(name, learns, rows, _jd_to_iso(last), STALLED,
                       f"{source} has arrived that it has not learned from ({since} behind)")
    return Learner(name, learns, rows, _jd_to_iso(last), LEARNING, "up to date")


def _gate(learning: sqlite3.Connection | None, profile_id: str) -> Learner:
    """The ranker candidate's offline gate verdict (D-45)."""
    import json

    learns = "checks a new ranker against the order you get, before anyone sees it"
    try:
        row = None if learning is None else learning.execute(
            "SELECT shadow_results_json, trained_at FROM learning_model_state"
            " WHERE profile_id = ? AND is_candidate = 1 LIMIT 1", (profile_id,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    if row is None:
        return Learner("ranker_gate", learns, 0, None, NO_INPUT, "no candidate trained yet")
    try:
        results = json.loads(row[0] or "{}")
    except (TypeError, ValueError):
        results = {}
    verdict = results.get("offline_gate") or {}
    if results.get("offline_certificate") is True:
        return Learner("ranker_gate", learns, 1, row[1], LEARNING,
                       "certified: the candidate is in the live test")
    return Learner("ranker_gate", learns, 1, row[1], WAITING,
                   str(verdict.get("reason") or "not checked yet"))


def _newest(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def _ranking_learners(learning: sqlite3.Connection | None, p: str, mode: str) -> list[Learner]:
    """The ranking, feature-logging and bandit learners (they depend on ``SLM_RANKING``)."""
    ranking_off = mode == "off"
    out: list[Learner] = []
    for name, learns, table in (
            ("ranking", "reorders recall with the learned model and bandit", None),
            ("feature_logging", "records what each recall offered, to train on",
             "learning_signals"),
            ("bandit", "tries channel weightings and keeps what works", "bandit_plays")):
        rows, last = _stat(learning, table, p) if table else (None, None)
        active = mode == "v2-ensemble" or (name == "ranking" and not ranking_off)
        out.append(Learner(name, learns, rows, _jd_to_iso(last),
                           LEARNING if active else OFF,
                           f"on ({mode})" if active else
                           ("SLM_RANKING is not set: recall keeps its fusion order"
                            if ranking_off else f"runs only with SLM_RANKING=v2-ensemble ({mode})")))
    return out


def _agent_learners(learning: sqlite3.Connection | None, p: str) -> list[Learner]:
    """The learners that learn only when an agent reports something."""
    out: list[Learner] = []
    for name, learns, table, tool in (
            ("agent_experiences", "what agents report worked or failed", "agent_experiences",
             "record_agent_experience"),
            ("execution_learning", "lessons from bounded-loop runs",
             "execution_learning_events", "observe_bounded_loop_execution_learning")):
        rows, last = _stat(learning, table, p)
        out.append(Learner(name, learns, rows, _jd_to_iso(last),
                           LEARNING if rows else WAITING,
                           "fed by agents" if rows else f"learns when an agent calls {tool}; "
                                                        "none has yet"))
    return out


def measure(learning: sqlite3.Connection | None, memory: sqlite3.Connection | None,
            profile_id: str, *, environ: Mapping[str, str] | None = None) -> list[Learner]:
    """Every learner's state for one profile."""
    from superlocalmemory.core.recall_pipeline import _resolve_ranking_mode

    env = os.environ if environ is None else environ
    mode = _resolve_ranking_mode(env)
    p = profile_id
    _, outcomes = _stat(memory, "action_outcomes", p)
    _, used = _stat(memory, "injection_usage", p)
    _, recalls = _stat(memory, "fact_access_log", p)
    _, saved = _stat(memory, "atomic_facts", p)
    out = [
        _flow("ranker_training", "which memories to put first, from what agents used",
              _stat(learning, "learning_model_state", p), _newest(outcomes, used),
              RANKER_CADENCE_H, "feedback"),
        _gate(learning, p),
        _flow("co_retrieval", "which memories belong together",
              _stat(learning, "co_retrieval_edges", p), recalls,
              PER_RECALL_CADENCE_H, "recall"),
        _flow("pattern_mining", "your interests, tools and habits",
              _stat(learning, "_store_patterns", p), saved,
              CONSOLIDATION_CADENCE_H, "new memory"),
    ]
    out += _ranking_learners(learning, p, mode)
    out += _agent_learners(learning, p)
    out.append(_flow("workflow_mining", "your recurring sequences of work and time-of-day habits "
                     "(kept in shadow: nothing reaches soft prompts yet)",
                     _stat(learning, "workflow_patterns", p), saved,
                     CONSOLIDATION_CADENCE_H, "new memory"))
    rows, _last = _stat(learning, "transferable_patterns", p)
    out.append(Learner("cross_project", "preferences carried from one workspace to another",
                       rows, None, RETIRED,
                       "not wired by design (D-45): it would carry one workspace's preferences "
                       "into another, which profile isolation forbids, and its input table is "
                       "an empty leftover"))
    return out


def summary(learners: list[Learner]) -> dict:
    counts: dict[str, int] = {}
    for item in learners:
        counts[item.state] = counts.get(item.state, 0) + 1
    return counts


__all__ = ["CONSOLIDATION_CADENCE_H", "LEARNING", "NOT_WIRED", "NO_INPUT", "OFF", "RETIRED",
           "PER_RECALL_CADENCE_H", "RANKER_CADENCE_H", "STALLED", "WAITING", "Learner",
           "measure", "summary"]
