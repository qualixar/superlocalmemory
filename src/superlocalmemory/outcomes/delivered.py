# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What SuperLocalMemory recently showed your agents, and why (W3, D-41, §8.10.3).

The dashboard's "Shown to your agents" card reads this. Every time memories are put in front of
an agent there is a delivery row (``injection_deliveries``) and one receipt per memory
(``injection_receipts``): at the start of a conversation, with a prompt, on a cue, or when an
agent asked. This module groups the recent ones by conversation and gives each memory a reason
in plain words: how it was chosen (its role and rank), whether it is still current, disputed or
from a file that changed, how old it is and, while the origin work is on, where it came from.

Read-only: it opens nothing for writing and the caller passes a ``mode=ro`` connection. It never
shows a memory its caller cannot see: a memory outside the profile is dropped, and ``hidden``
(the remote-view filter the disputes card uses) drops the rest. It is fail-open per part: a
store missing a table or a column gives an empty part, never an error.

What a receipt does not keep is the answer check's verdict for each memory, so "checked or not"
is not on the card: saying so would be a guess. The reasons are derived from what is stored and
from the memory's state now; the "why" an agent saw at the time travelled with the recall result.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.trust import origin_class as oc

logger = logging.getLogger(__name__)

#: Deliveries listed by default and at most (a page size, routing-only).
DEFAULT_DELIVERIES = 20
MAX_DELIVERIES = 100
#: Memories listed per delivery (a page size, routing-only; the count of the rest is kept).
MEMORIES_PER_DELIVERY = 8
#: Memories the harm gate would withhold that are listed (a page size; the count is kept).
LISTED = 5
#: How much of a memory's text the card shows.
PREVIEW_CHARS = 200

#: The kind of delivery and how a person would say it. ``before a web search`` runs through the
#: same hook queue as a prompt (``hooks/before_web_hook``), so the two are one kind.
KINDS: dict[str, tuple[str, str]] = {
    "session_init": ("session", "At the start of a conversation"),
    "session_context": ("session", "At the start of a conversation"),
    "hook_recall": ("prompt", "With a prompt or before a web search (automatic)"),
    "cue": ("cue", "On a cue: a file, command, error, repository or date"),
    "mcp_recall": ("asked", "When the agent asked memory"),
    "cli_recall": ("asked", "When asked at the command line"),
    "http_recall": ("asked", "When asked over HTTP"),
    "view_recall": ("asked", "When you searched in the dashboard"),
}
UNKNOWN_KIND = ("other", "By another route")

_ROLES = {"pinned": "A pinned memory", "standing": "A standing rule", "core": "Part of your core memory",
          "recall": "Found by search", "recent": "A recent memory",
          "session_summary": "A summary of an earlier conversation"}
_RANKED = frozenset({"recall", "recent", "session_summary"})

_STANDING = {"current": "Still current", "disputed": "Disputed: another memory says something different",
             "source changed": "Its source file has changed since", "source missing": "Its source file is gone"}

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def _clean(text: str) -> str:
    """One line, no control or direction-changing characters (a memory is untrusted text)."""
    return " ".join(_CONTROL.sub(" ", str(text or "")).split())[:PREVIEW_CHARS]


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    """Rows, or none for a store that predates the table or column."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) or "no such column" in str(exc):
            return []
        raise


def ago(then: str | None, now: datetime) -> str:
    """``3 days ago`` for an ISO time; empty when it cannot be read."""
    try:
        when = datetime.fromisoformat(str(then).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    seconds = int((now - when).total_seconds())
    if seconds < 0:
        return "just now"
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = seconds // size
            if unit == "day" and n >= 60:
                return f"{n // 30} months ago"
            return f"{n} {unit}{'' if n == 1 else 's'} ago"
    return "just now"


def _standing(conn: sqlite3.Connection, profile_id: str, ids: list[str],
              hidden: Callable[[list[str]], set[str]]) -> dict[str, str]:
    """``{fact_id: standing}`` for the facts that are not simply current (disputed, source changed)."""
    from superlocalmemory.truth import disputes, leases

    out: dict[str, str] = {}
    try:
        for fact_id, view in leases.check_for_facts(conn, ids).items():
            if view.get("state") in (leases.CHANGED, leases.MISSING):
                out[fact_id] = f"source {view['state']}"
    except Exception:  # noqa: BLE001 - a part of the card, never the whole
        logger.debug("lease state unreadable", exc_info=True)
    try:
        for fact_id in disputes.for_facts(conn, profile_id, ids, hidden=hidden):
            out[fact_id] = "disputed"  # a dispute outranks a changed file in the words
    except Exception:  # noqa: BLE001
        logger.debug("dispute state unreadable", exc_info=True)
    return out


def _origins(conn: sqlite3.Connection, ids: list[str], slm_home: Path | None) -> dict[str, dict]:
    """Where each memory came from, only while the origin work is on or in shadow."""
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.trust import fact_trust

    try:
        if not FeatureFlags(slm_home).observing("trust.taint"):
            return {}
        return fact_trust.for_results(conn, ids)
    except Exception:  # noqa: BLE001
        logger.debug("origin unreadable", exc_info=True)
        return {}


def _facts(conn: sqlite3.Connection, profile_id: str, ids: list[str],
           hidden: Callable[[list[str]], set[str]]) -> dict[str, dict]:
    """The memories of this profile the caller may see: preview and when they were saved."""
    from superlocalmemory.truth import disputes

    marks = ",".join("?" * len(ids))
    own = {r[0] for r in _rows(conn, f"SELECT fact_id FROM atomic_facts WHERE profile_id = ? "
                                     f"AND fact_id IN ({marks})", (profile_id, *ids))}
    blocked = hidden(sorted(own)) if own else set()
    details = disputes._details(conn, sorted(own - blocked))
    return {i: d for i, d in details.items() if i in own and i not in blocked}


def _memory(receipt: tuple, fact: dict, notes: dict, now: datetime) -> dict[str, Any]:
    fact_id, role, position, _eligible, _included = receipt
    standing = notes["standing"].get(fact_id, "current")
    origin = notes["origins"].get(fact_id) or {}
    how = _ROLES.get(role, f"As {role}" if role else "A result")
    if role in _RANKED or role not in _ROLES:
        how += f" (rank {int(position) + 1})"
    said = ago(fact.get("saved_at"), now)
    taint = origin.get("taint") or origin.get("class")
    why = {"how": how, "standing": standing, "standing_words": _STANDING[standing],
           "said": said, "origin": oc.label(taint) if taint else None}
    parts = [how + ".", _STANDING[standing] + "."]
    if said:
        parts.append(f"Saved {said}.")
    if why["origin"]:
        parts.append(f"Origin: {why['origin']}.")
    return {"fact_id": fact_id, "preview": _clean(fact.get("content", "")), "role": role,
            "why": {**why, "words": " ".join(parts)}}


def _deliveries(conn: sqlite3.Connection, profile_id: str, limit: int) -> list[tuple]:
    return _rows(
        conn, "SELECT delivery_id, session_id, client, path, delivered_at FROM injection_deliveries "
        "WHERE profile_id = ? ORDER BY delivered_at DESC LIMIT ?", (profile_id, int(limit)))


def _receipts(conn: sqlite3.Connection, profile_id: str, ids: list[str]) -> dict[str, list[tuple]]:
    marks = ",".join("?" * len(ids))
    out: dict[str, list[tuple]] = {}
    for did, fact_id, role, pos, eligible, included, delivered in _rows(
            conn, "SELECT delivery_id, fact_id, role, position, eligible, included, delivered "
            f"FROM injection_receipts WHERE profile_id = ? AND delivery_id IN ({marks}) "
            "ORDER BY delivery_id, position", (profile_id, *ids)):
        out.setdefault(did, []).append((fact_id, role, pos, eligible, included, delivered))
    return out


def recent(conn: sqlite3.Connection, profile_id: str, *, limit: int = DEFAULT_DELIVERIES,
           hidden: Callable[[list[str]], set[str]] = lambda ids: set(),
           slm_home: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Recent deliveries grouped by conversation, newest first (JSON-ready)."""
    now = now or datetime.now(UTC)
    limit = max(1, min(int(limit), MAX_DELIVERIES))
    deliveries = _deliveries(conn, profile_id, limit)
    if not deliveries:
        return {"groups": [], "deliveries": 0, "memories": 0}
    receipts = _receipts(conn, profile_id, [d[0] for d in deliveries])
    fact_ids = sorted({r[0] for rs in receipts.values() for r in rs if r[5]})
    facts = _facts(conn, profile_id, fact_ids, hidden) if fact_ids else {}
    seen = sorted(facts)
    notes = {"standing": _standing(conn, profile_id, seen, hidden) if seen else {},
             "origins": _origins(conn, seen, slm_home) if seen else {}}
    groups: dict[str, dict[str, Any]] = {}
    shown = 0
    for did, session, client, path, at in deliveries:
        kind, words = KINDS.get(path, UNKNOWN_KIND)
        rs = receipts.get(did, [])
        memories = [_memory(r[:5], facts[r[0]], notes, now) for r in rs if r[5] and r[0] in facts]
        left_out = sum(1 for r in rs if not r[4])
        shown += len(memories)
        entry = {"delivery_id": did, "at": at, "ago": ago(at, now), "path": path, "kind": kind,
                 "kind_words": words, "left_out": left_out,
                 "more": max(0, len(memories) - MEMORIES_PER_DELIVERY),
                 "memories": memories[:MEMORIES_PER_DELIVERY]}
        key = session or f"{client or 'unknown'}:{did}"
        group = groups.setdefault(key, {"session_id": session, "client": client, "last_at": at,
                                        "first_at": at, "deliveries": []})
        group["first_at"] = min(group["first_at"], at)
        group["deliveries"].append(entry)
    ordered = sorted(groups.values(), key=lambda g: g["last_at"], reverse=True)
    for g in ordered:
        g["ago"] = ago(g["last_at"], now)
    return {"groups": ordered, "deliveries": len(deliveries), "memories": shown}


def gates(conn: sqlite3.Connection, profile_id: str, *, slm_home: Path | None = None,
          hidden: Callable[[list[str]], set[str]] = lambda ids: set(),
          now: datetime | None = None) -> dict[str, Any]:
    """The two numbers next to the list: prompts the need gate skipped (with its measured bound)
    and the memories the harm gate would withhold (shadow: it never acts in 4.2.30).

    The harm gate's memories are listed only if this caller may see them.
    """
    from superlocalmemory.core import need_gate
    from superlocalmemory.outcomes import memory_report

    when = now or datetime.now(UTC)
    need = memory_report._need_gate(conn, profile_id, when)  # noqa: SLF001 - same section, fail-open
    harm = memory_report._harm_gate(conn, profile_id, when, slm_home)  # noqa: SLF001
    ids = [str(i) for i in harm.get("would_withhold") or []]
    visible = _facts(conn, profile_id, ids, hidden) if ids else {}
    return {"need_gate": {"turns": need.get("turns"), "skipped": need.get("skipped"),
                          "rules_bound": need.get("rules_bound"), "alpha": need.get("alpha"),
                          "days": need_gate.WINDOW_DAYS},
            "harm_gate": {"state": harm.get("state"), "acting": False, "tests": harm.get("tests"),
                          "facts_both_arms": harm.get("facts_both_arms"),
                          "facts_ever_left_out": harm.get("facts_ever_left_out"),
                          "would_withhold_count": sum(1 for i in ids if i in visible),
                          "would_withhold": [{"fact_id": i, "preview": _clean(visible[i].get("content", ""))}
                                             for i in ids if i in visible][:LISTED]}}


__all__ = ["DEFAULT_DELIVERIES", "KINDS", "LISTED", "MAX_DELIVERIES", "MEMORIES_PER_DELIVERY", "ago",
           "gates", "recent"]
