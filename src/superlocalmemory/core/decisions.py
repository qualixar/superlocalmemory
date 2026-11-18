# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Who decided what: the one record every decision leaves (4.2.30, D-49, I-10).

SuperLocalMemory is a memory system for agents, so an agent through MCP, the command line or
HTTP, a script or a person on the dashboard may each release a memory the instruction firewall
holds, settle a dispute, give a blind audit label, and apply or roll back a set of proposed
changes. Nothing here asks whether the caller is "really a person": no such check can be made
by software that does not control the caller, and a check that cannot be enforced is not
made. Safety comes from provenance rules that need nobody (``trust/firewall``) and from this
record: each decision names **who** (an agent id, a client name or ``dashboard``), the
**channel** it came through (``mcp``, ``cli``, ``http`` or ``dashboard``), **when**, and can be
undone through the same doors that made it. The dashboard lists them ("Recent decisions by your
agents") so a person can see or undo any of them without being needed for anything to work.

``M084_decision_log`` holds one row per decision with ids and plain words, never memory text.
Writing is best effort for the *log* only: a store without the table (not yet migrated)
records nothing and the decision itself still happens; the domain tables
(``trust_approvals``, ``proposal_sets``, ``slot_disputes``, ``judge_labels``) carry their own
actor and channel, and ``slm doctor --truth`` (I-10) checks those.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

#: The four channels a decision can arrive through (never a claim about who is behind it).
CHANNELS: tuple[str, ...] = ("mcp", "cli", "http", "dashboard")
PROMOTION, PROPOSAL, DISPUTE, LABEL = "promotion", "proposal", "dispute", "label"
#: Signing memories again under the current key (``slm signing resign``). It changes no memory
#: text, origin or delivery, only integrity marks, and the marks it replaces could no longer be
#: checked; so it is recorded (who, channel, counts) but has no undo.
SIGNING = "signing"
KINDS: tuple[str, ...] = (PROMOTION, PROPOSAL, DISPUTE, LABEL, SIGNING)
#: Actions a caller can take back, by kind (a rejection or a rollback is not undone again).
UNDOABLE: dict[str, frozenset[str]] = {
    PROMOTION: frozenset({"promote"}), PROPOSAL: frozenset({"apply"}),
    DISPUTE: frozenset({"settle"}), LABEL: frozenset({"label"}),
}
DEFAULT_ACTOR = "client:local"
MAX_ACTOR = 80
MAX_SUMMARY = 200
MAX_LIST = 200
_ACTOR_BAD = re.compile(r"[^A-Za-z0-9_.:@-]+")


@dataclass(frozen=True)
class Decider:
    """Who decided and through which channel (both recorded, neither a proof of a person)."""

    actor: str
    channel: str

    @property
    def ref(self) -> str:
        """``actor@channel``: the compact form stored on a receipt's ``action_ref``."""
        return f"{self.actor}@{self.channel}"


def clean_actor(raw: Any, default: str = DEFAULT_ACTOR) -> str:
    """A name safe to store and to show: letters, digits and ``_ . : @ -``, at most 80 chars."""
    text = _ACTOR_BAD.sub("-", " ".join(str(raw or "").split())).strip("-")
    return text[:MAX_ACTOR] or default


def clean_channel(raw: Any, default: str = "http") -> str:
    value = str(raw or "").strip().lower()
    return value if value in CHANNELS else default


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _has_log(conn: sqlite3.Connection) -> bool:
    try:
        return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                 "AND name='decision_log'").fetchone())
    except sqlite3.Error:
        return False


def log(conn: sqlite3.Connection, *, profile_id: str, kind: str, subject_id: str, action: str,
        decider: Decider, summary: str = "", detail: dict | None = None) -> str:
    """Record one decision in the caller's transaction. Returns its id (``""`` if not logged).

    ``detail`` holds only what undoing it needs (class names, lists of wishes): never text.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown decision kind: {kind}")
    if not _has_log(conn):
        return ""
    decision_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO decision_log (decision_id, profile_id, kind, subject_id, action, actor, "
        "channel, summary, detail_json, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (decision_id, profile_id, kind, str(subject_id), action, clean_actor(decider.actor),
         clean_channel(decider.channel), str(summary)[:MAX_SUMMARY],
         json.dumps(detail or {}, sort_keys=True), _now()))
    return decision_id


def _view(row: Any) -> dict:
    undone = row[10] is not None
    try:
        detail = json.loads(row[8] or "{}")
    except ValueError:
        detail = {}
    return {"decision_id": row[0], "profile_id": row[1], "kind": row[2], "subject_id": row[3],
            "action": row[4], "actor": row[5], "channel": row[6], "summary": row[7],
            "detail": detail, "decided_at": row[9], "undone": undone,
            "undone_at": row[10], "undone_by": row[11], "undone_channel": row[12],
            "can_undo": not undone and row[4] in UNDOABLE.get(row[2], frozenset())}


_COLS = ("decision_id, profile_id, kind, subject_id, action, actor, channel, summary, "
         "detail_json, decided_at, reverted_at, reverted_by, reverted_channel")


def recent(conn: sqlite3.Connection, profile_id: str, limit: int = 50) -> list[dict]:
    """The profile's decisions, newest first. A store before M084 has none."""
    if not _has_log(conn):
        return []
    rows = conn.execute(
        f"SELECT {_COLS} FROM decision_log WHERE profile_id = ? "
        "ORDER BY decided_at DESC, rowid DESC LIMIT ?",
        (profile_id, max(1, min(int(limit), MAX_LIST)))).fetchall()
    return [_view(tuple(r)) for r in rows]


def get(conn: sqlite3.Connection, profile_id: str, decision_id: str) -> dict | None:
    if not _has_log(conn):
        return None
    row = conn.execute(f"SELECT {_COLS} FROM decision_log WHERE decision_id = ? "
                       "AND profile_id = ?", (decision_id, profile_id)).fetchone()
    return _view(tuple(row)) if row else None


def open_decision(conn: sqlite3.Connection, profile_id: str, kind: str,
                  subject_id: str) -> dict | None:
    """The newest undoable decision of ``kind`` about ``subject_id`` that has not been undone
    (a withdrawal, a rollback, a reopen or a retraction is the undo itself, never listed here)."""
    actions = sorted(UNDOABLE.get(kind, ()))
    if not actions or not _has_log(conn):
        return None
    marks = ",".join("?" * len(actions))
    row = conn.execute(
        f"SELECT {_COLS} FROM decision_log WHERE profile_id = ? AND kind = ? AND subject_id = ? "
        f"AND action IN ({marks}) AND reverted_at IS NULL "
        "ORDER BY decided_at DESC, rowid DESC LIMIT 1",
        (profile_id, kind, str(subject_id), *actions)).fetchone()
    return _view(tuple(row)) if row else None


def mark_undone(conn: sqlite3.Connection, decision_id: str, decider: Decider) -> bool:
    """Note that ``decider`` took the decision back. False if it was already undone."""
    if not decision_id or not _has_log(conn):
        return False
    cursor = conn.execute(
        "UPDATE decision_log SET reverted_at = ?, reverted_by = ?, reverted_channel = ? "
        "WHERE decision_id = ? AND reverted_at IS NULL",
        (_now(), clean_actor(decider.actor), clean_channel(decider.channel), decision_id))
    return cursor.rowcount == 1


def counts(conn: sqlite3.Connection, profile_id: str) -> dict[str, int]:
    """``total``, ``undone`` and ``open`` decisions of a profile (Memory Health's one line)."""
    if not _has_log(conn):
        return {"total": 0, "undone": 0, "open": 0}
    total, undone = conn.execute(
        "SELECT COUNT(*), COUNT(reverted_at) FROM decision_log WHERE profile_id = ?",
        (profile_id,)).fetchone()
    return {"total": int(total), "undone": int(undone), "open": int(total) - int(undone)}


__all__ = ["CHANNELS", "DEFAULT_ACTOR", "DISPUTE", "Decider", "KINDS", "LABEL", "PROMOTION",
           "PROPOSAL", "SIGNING", "UNDOABLE", "clean_actor", "clean_channel", "counts", "get", "log",
           "mark_undone", "open_decision", "recent"]
