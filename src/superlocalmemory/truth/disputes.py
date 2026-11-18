# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Open disputes, for the agents and people who settle them (D-37, D-39).

Most callers are agents, so a dispute travels inside the recall result itself:
both sides, newest first, each with when and where it was said, and the next
action (``ask_user``: ask which is true now, then settle). Any caller settles
the same disputes (D-49): an agent, a script or a person on the dashboard. Every
door (MCP ``settle_dispute``, ``slm disputes settle``,
``POST /api/disputes/{id}/settle``, the dashboard buttons) calls :func:`settle`, so
there is one route for an answer.

A caller's recorded answer is a lawful basis for retiring the losing memory (I-10),
as ``remember(replaces=…)`` is: the loser is retired through the correction ledger
and the dispute records who answered (an agent id, a client name or ``dashboard``),
through which channel (``mcp``, ``cli``, ``http``, ``dashboard``) and when, in
``resolved_by`` as ``actor@channel`` and in the decision log. :func:`reopen` takes
the answer back: the retired memory returns and the question is open again. An
answer is never a blind label, so it never feeds a certificate.

Nothing here reads memory text beyond the facts the dispute names, and the
other side's text passes the same visibility filter as any recall result.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from superlocalmemory.core import decisions
from superlocalmemory.trust.supersession import ORIGIN_BELOW_HEAD
from superlocalmemory.truth import slot_store
from superlocalmemory.truth.slot_repair import ROLLED_BACK
from superlocalmemory.truth.slot_resolver import APPLIED, retire_superseded

#: What a caller should do on meeting an open dispute.
ASK_USER = "ask_user"
KEEP_BOTH = "both"
UNRELATED = "unrelated"
#: Channels an answer may arrive through (recorded on the dispute; never a claim about who).
CHANNELS = decisions.CHANNELS
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


class ReopenRefused(ValueError):
    """The answer cannot be taken back (the value moved on since); nothing was changed."""


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
    """A caller's answer to one open dispute, applied in the caller's transaction.

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
    by = decisions.Decider(decisions.clean_actor(actor), channel).ref
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
    decisions.log(conn, profile_id=profile_id, kind=decisions.DISPUTE, subject_id=dispute_id,
                  action="settle", decider=decisions.Decider(actor, channel),
                  summary=_SUMMARY[resolution], detail={"resolution": resolution})
    return {"dispute_id": dispute_id, "status": SETTLED, "resolution": resolution,
            "retired": list(retired), "settled_by": by}


_SUMMARY = {"a": "Settled a dispute: kept the earlier value",
            "b": "Settled a dispute: kept the newer value",
            "both": "Settled a dispute: both values are true",
            "unrelated": "Settled a dispute: the two are unrelated"}


def reopen(conn: sqlite3.Connection, *, profile_id: str, dispute_id: str, channel: str,
           actor: str = "") -> dict:
    """Take a settled dispute's answer back: the memory it retired returns, the slot is as it
    was before the answer, and the question is open again. Recorded like any decision.

    Raises :class:`LookupError` for a dispute that is not settled in this profile and
    :class:`ReopenRefused` when the slot moved on since (nothing is changed then).
    """
    if channel not in CHANNELS:
        raise ReopenRefused(f"channel must be one of {CHANNELS}")
    row = conn.execute(
        "SELECT d.slot_id, d.revision_a, d.revision_b, d.reason_code, d.resolution, "
        "s.head_revision_id, s.version FROM slot_disputes d JOIN memory_slots s "
        "ON s.slot_id = d.slot_id WHERE d.dispute_id = ? AND d.profile_id = ? "
        "AND d.state = 'resolved'", (dispute_id, profile_id)).fetchone()
    if row is None:
        raise LookupError(f"no settled dispute {dispute_id}")
    slot_id, rev_a, rev_b, reason, resolution, head, version = row
    if head != (rev_b if resolution == "b" else rev_a):
        raise ReopenRefused("The value changed since this was settled, so it cannot be "
                            "reopened.")
    if conn.execute("SELECT 1 FROM slot_disputes WHERE slot_id = ? AND revision_a = ? "
                    "AND revision_b = ? AND state = 'open'",
                    (slot_id, rev_a, rev_b)).fetchone():
        raise ReopenRefused("The same question is already open.")
    who = decisions.Decider(decisions.clean_actor(actor), channel)
    conn.execute("SAVEPOINT reopen")
    try:
        restored = _restore_retired(conn, profile_id, resolution, rev_a, rev_b)
        _restore_slot(conn, slot_id, int(version), rev_a, rev_b, reason)
        conn.execute("UPDATE slot_disputes SET state = 'open', resolution = NULL, "
                     "resolved_by = NULL, resolved_at = NULL WHERE dispute_id = ?",
                     (dispute_id,))
    except Exception:
        conn.execute("ROLLBACK TO reopen")
        conn.execute("RELEASE reopen")
        raise
    conn.execute("RELEASE reopen")
    earlier = decisions.open_decision(conn, profile_id, decisions.DISPUTE, dispute_id)
    if earlier is not None:
        decisions.mark_undone(conn, earlier["decision_id"], who)
    decisions.log(conn, profile_id=profile_id, kind=decisions.DISPUTE, subject_id=dispute_id,
                  action="reopen", decider=who, summary="Reopened a dispute",
                  detail={"was": resolution})
    return {"dispute_id": dispute_id, "status": "open", "restored": restored,
            "reopened_by": who.ref}


def _restore_retired(conn: sqlite3.Connection, profile_id: str, resolution: str, rev_a: str,
                     rev_b: str) -> list[str]:
    """Roll back the correction case that retired the losing memory (a or b answers only)."""
    if resolution not in ("a", "b"):
        return []
    from superlocalmemory.storage.correction_cases import transition_on_connection
    from superlocalmemory.truth.slot_resolver import ACTOR, REASON

    loser = conn.execute("SELECT fact_id FROM slot_revisions WHERE revision_id = ?",
                         (rev_a if resolution == "b" else rev_b,)).fetchone()[0]
    case = conn.execute(
        "SELECT case_id, version FROM correction_cases WHERE profile_id = ? AND "
        "predecessor_fact_id = ? AND status = 'applied' AND reason_code = ? LIMIT 1",
        (profile_id, loser, REASON)).fetchone()
    if case is None:
        raise ReopenRefused("The retired memory was already restored another way.")
    transition_on_connection(
        conn, case_id=case[0], expected_version=int(case[1]), actor=ACTOR,
        operation_id=f"dispute-reopen:{case[0]}", from_status="applied",
        to_status="rolled_back", mutate_temporal=True,
        is_profile_active=lambda candidate: candidate == profile_id,
        is_actor_trusted=lambda actor: actor == ACTOR)
    return [loser]


def _restore_slot(conn: sqlite3.Connection, slot_id: str, version: int, rev_a: str,
                  rev_b: str, reason: str) -> None:
    """The slot as it stood when the dispute opened: head ``a``, challenger ``b`` disputed."""
    conn.execute("UPDATE slot_revisions SET state = 'head', superseded_by = NULL "
                 "WHERE revision_id = ?", (rev_a,))
    conn.execute("UPDATE slot_revisions SET state = 'disputed', superseded_by = NULL, "
                 "reason_code = ? WHERE revision_id = ?", (reason, rev_b))
    slot_store._move_head(conn, slot_id, version, rev_a, "disputed",
                          datetime.now(UTC).isoformat())


__all__ = ["ASK_USER", "CHANNELS", "KEEP_BOTH", "ReopenRefused", "SETTLED", "STALE",
           "SettleRefused", "UNRELATED", "for_facts", "listing", "open_count", "reopen", "settle"]
