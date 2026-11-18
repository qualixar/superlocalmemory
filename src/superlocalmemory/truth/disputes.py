# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open disputes, for the agents and people who settle them (D-37, D-39).

Most callers are agents, so a dispute travels inside the recall result itself:
both sides, newest first, each with when and where it was said, and the next
action (``ask_user``: ask which is true now, then settle). People settle the
same disputes on the dashboard. Every door (MCP ``settle_dispute``,
``slm disputes settle``, ``POST /api/disputes/{id}/settle``, the dashboard
buttons) calls :func:`settle`, so there is one route for a person's answer.

A person's answer is a lawful basis for retiring the losing memory (I-10), as
``remember(replaces=…)`` is: the loser is retired through the correction
ledger and the dispute records who answered and through which channel. A
person's answer is never a blind label, so it never feeds a certificate.

Nothing here reads memory text beyond the facts the dispute names, and the
other side's text passes the same visibility filter as any recall result.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from superlocalmemory.trust.supersession import ORIGIN_BELOW_HEAD
from superlocalmemory.truth import slot_store
from superlocalmemory.truth.slot_repair import ROLLED_BACK
from superlocalmemory.truth.slot_resolver import APPLIED, retire_superseded

#: What a caller should do on meeting an open dispute.
ASK_USER = "ask_user"
KEEP_BOTH = "both"
UNRELATED = "unrelated"
#: Channels a person's answer may arrive through (recorded on the dispute).
CHANNELS = ("agent", "cli", "dashboard", "http")
#: Characters of the other side's text a recall result carries.
SIDE_TEXT_CHARS = 600
MAX_LISTING = 200

SETTLED, STALE = "settled", "stale"
STALE_MESSAGE = ("The value changed since this question was asked, so it was closed; "
                 "nothing else was changed.")

QUESTION = "Which is true now?"
ROLLED_BACK_QUESTION = ("SuperLocalMemory undid an earlier change between these two. "
                        "Which is true now?")
#: The newer memory comes from a source SuperLocalMemory trusts less than the one it
#: disagrees with (or the source of one is unknown): it will not choose between them.
ORIGIN_QUESTION = ("One of these came from a source SuperLocalMemory trusts less, so it "
                   "will not choose between them. Which is true now?")
HOW_TO_SETTLE = ("settle_dispute(dispute_id, keep=<fact_id of the side that is true now> | "
                 "'both' | 'unrelated')")

_DISPUTES = """
SELECT d.dispute_id, d.reason_code, d.opened_at, d.revision_a, d.revision_b,
       ra.fact_id, ra.valid_from, ra.recorded_at,
       rb.fact_id, rb.valid_from, rb.recorded_at, s.head_revision_id
  FROM slot_disputes d
  JOIN slot_revisions ra ON ra.revision_id = d.revision_a
  JOIN slot_revisions rb ON rb.revision_id = d.revision_b
  JOIN memory_slots s ON s.slot_id = d.slot_id
 WHERE d.state = 'open' AND d.profile_id = ?"""


class SettleRefused(ValueError):
    """The answer cannot be applied; nothing was changed."""


@dataclass(frozen=True)
class _Row:
    dispute_id: str
    reason: str
    opened_at: str
    sides: tuple[tuple[str, str | None, bool], ...]  # (fact_id, said_at, current)


def _rows(conn: sqlite3.Connection, profile_id: str, where: str = "",
          params: Sequence[object] = (), limit: int | None = None) -> list[_Row]:
    sql = _DISPUTES + where + " ORDER BY d.opened_at DESC, d.dispute_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    try:
        raw = conn.execute(sql, (profile_id, *params)).fetchall()
    except sqlite3.OperationalError:  # a store before M069: nothing can be disputed
        return []
    out = []
    for (did, reason, opened, rev_a, rev_b, fa, va, ra, fb, vb, rb, head) in raw:
        sides = ((fa, va or ra, head == rev_a), (fb, vb or rb, head == rev_b))
        out.append(_Row(did, reason, opened, sides))
    return out


def _details(conn: sqlite3.Connection, fact_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(dict.fromkeys(fact_ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT f.fact_id, f.memory_id, f.content, m.created_at, m.metadata_json "
        f"FROM atomic_facts f LEFT JOIN memories m ON m.memory_id = f.memory_id "
        f"WHERE f.fact_id IN ({marks})", ids).fetchall()
    out = {}
    for fact_id, memory_id, content, created_at, meta_json in rows:
        try:
            meta = json.loads(meta_json or "{}") or {}
        except (TypeError, ValueError):
            meta = {}
        source = meta.get("_slm_source") if isinstance(meta.get("_slm_source"), dict) else {}
        agent = meta.get("agent_id")
        out[fact_id] = {"memory_id": memory_id, "content": content or "",
                        "saved_at": created_at,
                        "agent": agent if isinstance(agent, str) and agent else None,
                        "file": source.get("relpath") or source.get("path") or None}
    return out


def _view(row: _Row, details: dict[str, dict], hidden: set[str]) -> dict:
    sides = []
    for fact_id, said_at, current in row.sides:
        d = details.get(fact_id, {})
        side = {"fact_id": fact_id, "memory_id": d.get("memory_id"),
                "said_at": said_at or d.get("saved_at"), "current": current,
                "agent": d.get("agent"), "file": d.get("file")}
        if fact_id in hidden:
            side["content"] = None
            side["visible"] = False
        else:
            side["content"] = (d.get("content") or "")[:SIDE_TEXT_CHARS]
            side["visible"] = True
        sides.append(side)
    sides.sort(key=lambda s: s["said_at"] or "", reverse=True)  # newest first (D-39)
    return {"dispute_id": row.dispute_id, "state": "disputed", "reason": row.reason,
            "opened_at": row.opened_at,
            "question": (ROLLED_BACK_QUESTION if row.reason == ROLLED_BACK
                         else ORIGIN_QUESTION if row.reason == ORIGIN_BELOW_HEAD else QUESTION),
            "next": ASK_USER, "how_to_settle": HOW_TO_SETTLE, "sides": sides}


def for_facts(conn: sqlite3.Connection, profile_id: str, fact_ids: Sequence[str], *,
              hidden: Callable[[list[str]], set[str]] = lambda ids: set()) -> dict[str, list[dict]]:
    """``{fact_id: [open dispute views]}`` for the facts a recall is returning.

    ``hidden`` names, among the other sides, those this caller may not see:
    their text is withheld, never the fact that a dispute exists.
    """
    ids = [f for f in dict.fromkeys(fact_ids) if f]
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = _rows(conn, profile_id, f" AND (ra.fact_id IN ({marks}) OR rb.fact_id IN ({marks}))",
                 (*ids, *ids))
    if not rows:
        return {}
    involved = {f for r in rows for f, _w, _c in r.sides}
    others = sorted(involved - set(ids))
    blocked = hidden(others) if others else set()
    details = _details(conn, involved)
    out: dict[str, list[dict]] = {}
    for row in rows:
        view = _view(row, details, blocked)
        for fact_id, _w, _c in row.sides:
            if fact_id in ids:
                out.setdefault(fact_id, []).append(view)
    return out


def listing(conn: sqlite3.Connection, profile_id: str, limit: int = 50, *,
            hidden: Callable[[list[str]], set[str]] = lambda ids: set()) -> list[dict]:
    """The profile's open disputes, newest first (CLI, MCP, dashboard)."""
    rows = _rows(conn, profile_id, limit=max(1, min(int(limit), MAX_LISTING)))
    involved = sorted({f for r in rows for f, _w, _c in r.sides})
    blocked = hidden(involved) if involved else set()
    details = _details(conn, involved)
    return [_view(r, details, blocked) for r in rows]


def open_count(conn: sqlite3.Connection, profile_id: str) -> int:
    try:
        return int(conn.execute("SELECT COUNT(*) FROM slot_disputes WHERE profile_id = ? "
                                "AND state = 'open'", (profile_id,)).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def _resolution(row: tuple, keep: str) -> str:
    _did, fact_a, fact_b = row
    if keep == KEEP_BOTH:
        return "both"
    if keep == UNRELATED:
        return "unrelated"
    if keep == fact_a:
        return "a"
    if keep == fact_b:
        return "b"
    raise SettleRefused("keep must be the fact_id of one side, 'both' or 'unrelated'")


def settle(conn: sqlite3.Connection, *, profile_id: str, dispute_id: str, keep: str,
           channel: str, actor: str = "") -> dict:
    """A person's answer to one open dispute, applied in the caller's transaction.

    ``keep`` is the fact id of the side that is true now (the other side is
    retired through the correction ledger), ``both`` (both stay current) or
    ``unrelated`` (the two were never about the same thing; the newer one leaves
    the slot). Raises :class:`SettleRefused` when nothing can be applied, and
    :class:`LookupError` for a dispute that is not open. Returns ``status``
    ``stale`` (and closes the dispute) when the slot moved on since it opened.
    """
    if channel not in CHANNELS:
        raise SettleRefused(f"channel must be one of {CHANNELS}")
    keep = (keep or "").strip()
    row = conn.execute(
        "SELECT d.dispute_id, ra.fact_id, rb.fact_id FROM slot_disputes d "
        "JOIN slot_revisions ra ON ra.revision_id = d.revision_a "
        "JOIN slot_revisions rb ON rb.revision_id = d.revision_b "
        "WHERE d.dispute_id = ? AND d.profile_id = ? AND d.state = 'open'",
        (dispute_id, profile_id)).fetchone()
    if row is None:
        raise LookupError(f"no open dispute {dispute_id}")
    resolution = _resolution(row, keep)
    scope = conn.execute("SELECT s.scope FROM slot_disputes d JOIN memory_slots s "
                         "ON s.slot_id = d.slot_id WHERE d.dispute_id = ?",
                         (dispute_id,)).fetchone()[0]
    by = f"person:{channel}" + (f":{actor[:80]}" if actor else "")
    conn.execute("SAVEPOINT settle")
    try:
        retired = slot_store.resolve_dispute(conn, dispute_id, resolution, resolved_by=by)
        for loser in retired:
            successor = slot_store.successor_fact(conn, dispute_id, loser)
            ledger, _case = retire_superseded(conn, profile_id=profile_id, scope=scope,
                                              predecessor=loser, successor=successor)
            if ledger != APPLIED:
                raise SettleRefused(f"the correction ledger did not apply it ({ledger})")
    except slot_store.StaleDisputeError:
        # Returned, not raised: the caller's transaction must commit the dismissal.
        conn.execute("ROLLBACK TO settle")
        conn.execute("RELEASE settle")
        slot_store.dismiss_dispute(conn, dispute_id, by=f"{by}:stale")
        return {"dispute_id": dispute_id, "status": STALE, "resolution": None, "retired": [],
                "settled_by": by, "message": STALE_MESSAGE}
    except Exception:
        conn.execute("ROLLBACK TO settle")
        conn.execute("RELEASE settle")
        raise
    conn.execute("RELEASE settle")
    return {"dispute_id": dispute_id, "status": SETTLED, "resolution": resolution,
            "retired": list(retired), "settled_by": by}


__all__ = ["ASK_USER", "CHANNELS", "KEEP_BOTH", "SETTLED", "STALE", "SettleRefused", "UNRELATED", "for_facts",
           "listing", "open_count", "settle"]
