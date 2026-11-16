# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The one writer for an explicit outcome report, shared by every surface.

WHY ONE WRITER
--------------
An outcome report ("these memories helped / did not / partly") is the label the
learning loop runs on. MCP, REST, the CLI and the Python tracker each used to
build their own ``action_outcomes`` INSERT, and they disagreed on what a report
*is*. REST wrote ``reward``, ``settled`` and ``settled_at``; MCP and the Python
``OutcomeTracker`` wrote none of them. The settler
(``reward_from_outcomes._candidate_outcomes``) skips a row whose reward is NULL,
so every outcome reported over MCP returned ``success: true`` and trained
nothing. Both answers looked identical to the caller.

This module is that contract, once:

* **Reward mapping.** ``success`` 1.0, ``partial`` 0.5, ``failure`` 0.0 —
  explicit, and anything else is refused rather than reinterpreted.
* **Attribution.** ``recall_query_id`` — the ``query_id`` recall returned —
  is stored when given, which ties the report to exactly one recall. Without
  it the settler may only match on overlapping memories, inside a window.
* **Provenance.** ``surface`` (mcp / rest / cli / python) and ``agent_id``
  are recorded in ``context_json`` so a label can always be traced to who
  reported it and how. No schema change on memory.db is needed for that.
* **Idempotence.** The same report twice is one event. With a
  ``recall_query_id`` (or a caller-supplied ``idempotency_key``) the outcome
  id is a digest of the report, so a retry hits the primary key and is
  ignored. Without either, a report identical in profile, memories, verdict,
  query text, surface and agent within ``DUPLICATE_WINDOW_SEC`` is treated as the same
  submission (a double click or a client retry), because there is no stable
  identity to key on.

A changed verdict about the same recall ("success", then "failure") is a new
event — it is new information. The settler still spends at most one outcome
per play, the earliest, so it cannot settle a recall twice.

Side effects run only when the event is new: the explicit-feedback bridge into
``learning_signals`` and the source-quality feed. Re-sending a report therefore
cannot double any downstream count either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DUPLICATE_WINDOW_SEC",
    "OutcomeReport",
    "OutcomeWrite",
    "PROVENANCE_KEY",
    "REWARD_BY_OUTCOME",
    "UnknownMemories",
    "VALID_SURFACES",
    "apply_side_effects",
    "record_feedback_outcome",
    "record_outcome_event",
    "reward_for",
    "write_outcome",
]

#: The explicit verdict-to-reward mapping. One table, read by every surface.
REWARD_BY_OUTCOME: dict[str, float] = {
    "success": 1.0,
    "partial": 0.5,
    "failure": 0.0,
}

#: Where a report came from. Stored as provenance, never trusted for auth.
VALID_SURFACES: frozenset[str] = frozenset(
    {
        "mcp",
        "rest",
        "cli",
        "dashboard",
        "python",
        "mcp_feedback",
    }
)

#: Key in ``context_json`` under which provenance is stored.
PROVENANCE_KEY = "_provenance"

#: How long an identical report without any identity counts as a resubmission.
DUPLICATE_WINDOW_SEC: float = 60.0

#: Learning-signal type and value per verdict, for the explicit-feedback bridge.
_SIGNAL_FOR_OUTCOME: dict[str, tuple[str, float]] = {
    "success": ("user_positive", 1.0),
    "failure": ("user_negative", 0.0),
    "partial": ("user_correction", 0.5),
}


class UnknownMemories(ValueError):
    """A report named memories that are not in the profile."""


@dataclass(frozen=True)
class OutcomeReport:
    """One explicit outcome report, before it is stored."""

    profile_id: str
    fact_ids: tuple[str, ...]
    outcome: str
    surface: str
    recall_query_id: str = ""
    agent_id: str = ""
    note: str = ""
    idempotency_key: str = ""
    query: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeWrite:
    """What the writer did. ``created`` is False for a duplicate."""

    outcome_id: str
    reward: float
    created: bool
    attributed: bool


def reward_for(outcome: str) -> float:
    """The reward for a verdict. Raises ``ValueError`` for anything else."""
    if outcome not in REWARD_BY_OUTCOME:
        raise ValueError(f"outcome must be one of {sorted(REWARD_BY_OUTCOME)}, not {outcome!r}")
    return REWARD_BY_OUTCOME[outcome]


def _normalised_ids(fact_ids: Any) -> tuple[str, ...]:
    out: list[str] = []
    for raw in fact_ids or ():
        fid = str(raw).strip()
        if fid and fid not in out:
            out.append(fid)
    return tuple(out)


def _fingerprint(report: OutcomeReport) -> str:
    """Digest of what the report says, independent of when it was sent."""
    parts = [
        report.profile_id,
        report.recall_query_id,
        report.outcome,
        ",".join(sorted(report.fact_ids)),
    ]
    if not report.recall_query_id:
        parts += [report.surface, report.agent_id, report.query]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _stable_outcome_id(report: OutcomeReport) -> str | None:
    """A deterministic id when the report has an identity, else None."""
    if report.idempotency_key:
        seed = f"{report.profile_id}\x1fkey\x1f{report.idempotency_key}"
    elif report.recall_query_id:
        seed = "fp\x1f" + _fingerprint(report)
    else:
        return None
    return "oc_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _columns(conn: sqlite3.Connection) -> set[str]:
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(action_outcomes)")}
    except sqlite3.Error:  # pragma: no cover — defensive
        return set()


def _check_memories(conn: sqlite3.Connection, report: OutcomeReport) -> None:
    marks = ",".join("?" for _ in report.fact_ids)
    rows = conn.execute(
        f"SELECT fact_id FROM atomic_facts WHERE profile_id = ? AND fact_id IN ({marks})",
        (report.profile_id, *report.fact_ids),
    ).fetchall()
    found = {str(r[0]) for r in rows}
    missing = [f for f in report.fact_ids if f not in found]
    if missing:
        raise UnknownMemories(
            f"{len(missing)} memory id(s) are not in profile {report.profile_id!r}"
        )


def _recent_duplicate(
    conn: sqlite3.Connection,
    report: OutcomeReport,
    fingerprint: str,
    now: datetime,
) -> str | None:
    since = (now - timedelta(seconds=DUPLICATE_WINDOW_SEC)).isoformat()
    try:
        row = conn.execute(
            "SELECT outcome_id FROM action_outcomes "
            "WHERE profile_id = ? AND timestamp >= ? "
            "AND json_extract(context_json, '$._provenance.fingerprint') = ? "
            "LIMIT 1",
            (report.profile_id, since, fingerprint),
        ).fetchone()
    except sqlite3.Error:  # pragma: no cover — JSON1 absent
        return None
    return str(row[0]) if row else None


def _context(report: OutcomeReport, fingerprint: str) -> str:
    ctx: dict[str, Any] = dict(report.extra)
    if report.note:
        ctx["note"] = report.note
    # Nested so a caller's own context keys are never overwritten.
    ctx[PROVENANCE_KEY] = {
        "surface": report.surface,
        "agent_id": report.agent_id or "",
        "fingerprint": fingerprint,
    }
    return json.dumps(ctx, separators=(",", ":"), sort_keys=True)


def _validate(report: OutcomeReport, *, need_ids: bool = True) -> OutcomeReport:
    reward_for(report.outcome)
    if report.surface not in VALID_SURFACES:
        raise ValueError(f"unknown outcome surface {report.surface!r}")
    ids = _normalised_ids(report.fact_ids)
    if need_ids and not ids:
        raise ValueError("an outcome must name at least one memory")
    qid = str(report.recall_query_id or "").strip()
    if len(qid) > 64:
        raise ValueError("recall_query_id must be at most 64 characters")
    return OutcomeReport(
        profile_id=str(report.profile_id),
        fact_ids=ids,
        outcome=report.outcome,
        surface=report.surface,
        recall_query_id=qid,
        agent_id=str(report.agent_id or "")[:120],
        note=str(report.note or "")[:1000],
        idempotency_key=str(report.idempotency_key or "").strip()[:128],
        query=str(report.query or "")[:200],
        extra=dict(report.extra),
    )


def record_outcome_event(
    conn: sqlite3.Connection,
    report: OutcomeReport,
    *,
    now: datetime | None = None,
    check_memories: bool = True,
) -> OutcomeWrite:
    """Store one report on ``conn`` (memory.db). The caller owns the commit.

    Degrades per column on a store where the deferred M006 has not run: the
    report is still recorded, it just cannot be settled — the honest
    reduction in capability.
    """
    report = _validate(report, need_ids=check_memories)
    current = now or datetime.now(timezone.utc)
    reward = reward_for(report.outcome)
    if check_memories:
        _check_memories(conn, report)
    fingerprint = _fingerprint(report)
    outcome_id = _stable_outcome_id(report)
    if outcome_id is None:
        dup = _recent_duplicate(conn, report, fingerprint, current)
        if dup is not None:
            return OutcomeWrite(dup, reward, False, False)
        outcome_id = "oc_" + uuid.uuid4().hex
    stamp = current.isoformat()
    cur, present = _insert_report(conn, report, outcome_id, reward, stamp, fingerprint)
    if cur.rowcount == 1 and report.surface != "mcp_feedback":
        from superlocalmemory.storage.retention_access import record_use_clock

        record_use_clock(conn, report.profile_id, report.fact_ids, outcome_id, stamp)
    if cur.rowcount == 1:
        _join_usage(conn, report, reward, current)
    attributed = bool(report.recall_query_id) and "recall_query_id" in present
    return OutcomeWrite(outcome_id, reward, cur.rowcount == 1, attributed)


def _join_usage(conn, report: OutcomeReport, reward: float, current: datetime) -> int:
    """W3-2: tie a new report to the deliveries that answered its question (by
    the keyed query digest), so explicit outcomes are read against what was
    delivered. Never raises: the report itself is what this write is for."""
    try:
        from superlocalmemory.outcomes import usage

        row = conn.execute("PRAGMA database_list").fetchone()
        if not row or not row[2]:
            return 0  # an in-memory store has no key beside it
        home = Path(row[2]).parent
        return usage.join_question(
            conn, profile_id=report.profile_id,
            query_digest=usage.query_digest(home, report.query),
            fact_ids=report.fact_ids, signal=usage.EXPLICIT + report.outcome,
            observed_at=usage.stamp(current), value=reward)
    except Exception as exc:
        logger.info("outcome usage join skipped: %s", exc)
        return 0


def _insert_report(conn, report, outcome_id, reward, stamp, fingerprint):
    cols = [
        "outcome_id",
        "profile_id",
        "query",
        "fact_ids_json",
        "outcome",
        "context_json",
        "timestamp",
    ]
    vals: list[Any] = [
        outcome_id,
        report.profile_id,
        report.query,
        json.dumps(list(report.fact_ids)),
        report.outcome,
        _context(report, fingerprint),
        stamp,
    ]
    present = _columns(conn)
    optional = (
        ("reward", reward),
        ("settled", 1),
        ("settled_at", stamp),
        ("recall_query_id", report.recall_query_id or None),
    )
    for name, value in optional:
        if name in present:
            cols.append(name)
            vals.append(value)
    cur = conn.execute(
        f"INSERT OR IGNORE INTO action_outcomes ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})",
        tuple(vals),
    )
    return cur, present


def apply_side_effects(
    written: OutcomeWrite,
    report: OutcomeReport,
    *,
    memory_db_path: Path,
    learning_db_path: Path,
    bridge_signals: bool = True,
) -> None:
    """Feed a NEW event to explicit feedback and source quality. Never raises."""
    if not written.created:
        return
    if bridge_signals:
        _bridge_signals(report, learning_db_path)
    try:
        from superlocalmemory.learning.source_quality import (
            update_source_quality_for_reward,
        )

        update_source_quality_for_reward(
            memory_db_path=Path(memory_db_path),
            learning_db_path=Path(learning_db_path),
            profile_id=report.profile_id,
            outcome_id=written.outcome_id,
            fact_ids=list(report.fact_ids),
            reward=written.reward,
        )
    except Exception as exc:  # noqa: BLE001 — the event is already durable
        logger.debug("source-quality outcome feed skipped: %s", exc)


def _bridge_signals(report: OutcomeReport, learning_db_path: Path) -> None:
    try:
        from superlocalmemory.learning.feedback import FeedbackCollector

        collector = FeedbackCollector(Path(learning_db_path))
        sig_type, value = _SIGNAL_FOR_OUTCOME[report.outcome]
        for fid in report.fact_ids:
            collector.record_explicit(
                profile_id=report.profile_id,
                fact_id=fid,
                signal_type=sig_type,
                value=value,
                channel=f"outcome:{report.surface}",
            )
    except Exception as exc:  # noqa: BLE001 — the event is already durable
        logger.debug("outcome -> feedback bridge skipped: %s", exc)


def write_outcome(
    report: OutcomeReport,
    *,
    memory_db_path: Path,
    learning_db_path: Path | None = None,
    bridge_signals: bool = True,
    check_memories: bool = True,
) -> OutcomeWrite:
    """Store a report under the process write lock, then run side effects.

    The entry point every surface calls. ``learning_db_path`` defaults to the
    ``learning.db`` beside ``memory_db_path`` — the layout the daemon uses.
    """
    from superlocalmemory.storage.memory_write import memory_write

    memory_db_path = Path(memory_db_path)
    learning = Path(learning_db_path or memory_db_path.parent / "learning.db")
    with memory_write(memory_db_path) as conn:
        written = record_outcome_event(
            conn,
            report,
            check_memories=check_memories,
        )
    apply_side_effects(
        written,
        _validate(report, need_ids=check_memories),
        memory_db_path=memory_db_path,
        learning_db_path=learning,
        bridge_signals=bridge_signals,
    )
    return written


#: report_feedback's vocabulary, mapped onto the outcome verdicts.
_FEEDBACK_TO_OUTCOME: dict[str, str] = {
    "relevant": "success",
    "partial": "partial",
    "irrelevant": "failure",
}


def record_feedback_outcome(
    engine: Any, profile_id: str, fact_id: str, feedback: str, recall_query_id: str
) -> bool:
    """Store a per-memory judgement of an identified recall as an outcome.

    Used by MCP ``report_feedback`` when the caller quotes the recall's
    ``query_id``. The explicit-feedback signal was already written by that
    tool, so the bridge is off here. Never raises; returns whether a new
    event was stored.
    """
    verdict = _FEEDBACK_TO_OUTCOME.get(feedback)
    if verdict is None or not recall_query_id:
        return False
    try:
        written = write_outcome(
            OutcomeReport(
                profile_id=profile_id,
                fact_ids=(fact_id,),
                outcome=verdict,
                surface="mcp_feedback",
                recall_query_id=recall_query_id,
            ),
            memory_db_path=Path(engine._db.db_path),  # noqa: SLF001
            learning_db_path=Path(engine._config.base_dir) / "learning.db",  # noqa: SLF001
            bridge_signals=False,
        )
    except Exception as exc:  # noqa: BLE001 — feedback already recorded
        logger.debug("feedback outcome not stored: %s", exc)
        return False
    return written.created
