# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""An edit made in a Markdown file is a proposal, never a write (D-17): the ``wiki_edit`` kind.

Reading an edited block back (``compiler/wiki.sync``) admits its text the way every
other edit of a memory is admitted: as the successor of a correction case waiting for
review (``core/mutations.update_fact_authorized``: the secret screen runs on the
text, the successor is stored unsearchable, origin channel ``external_edit``). This
module is the ProposalSet around that case:

* **apply** (a person's decision: no certificate covers an edit of a person's
  words) is a compare-and-swap. The block was written against a slot version; if the
  slot is still at that version (or the fact fills no slot and is still current) the
  correction case is applied and the new memory is marked ``user_confirmed``, because a
  person approved this exact text. If the slot moved on since, the edit is not allowed
  to replace anything: the new text is kept as a memory *next to* the slot's current
  value (applied without retiring the old one) and opens a **dispute** (reason
  ``external_edit_conflict``), so the person is asked which is true now, in the same
  place as every other dispute. A slot with no value left to dispute declines the edit.
* **rollback** undoes what apply did and nothing else: the ledger restores the
  replaced memory, the earlier class of the new memory returns, a conflict's dispute
  is dismissed (refused once a person has settled it).

``edit_text`` says what new text an edited block means for its fact: a span replaces
that sentence inside the memory, a story line replaces its first sentence, a
current-value line replaces the whole value; the other template lines are computed
from slot values and cannot be edited here (edit the memory they cite).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from superlocalmemory.compiler.page_text import one_line
from superlocalmemory.compiler.proposal_kinds import (
    ApplyContext,
    Handler,
    ProposalError,
    _glance,
    _rollback_cases,
    _transition,
    register_handler,
)
from superlocalmemory.compiler.versions import select_rows as _select

KIND = "wiki_edit"
CONFLICT = "external_edit_conflict"
TEXT_LIMIT = 20_000


def edit_text(sentence: Mapping[str, Any], edited: str, content: str) -> str | None:
    """The memory text that results from editing one page sentence, or ``None`` if this
    kind of sentence cannot be edited here (or the edit does not fit it)."""
    old, new = str(sentence["text"]), " ".join(edited.split())
    if not new or new == one_line(old):
        return None
    if sentence["type"] == "span":
        return content.replace(old, new, 1) if old in content else None
    template = sentence.get("template")
    if template == "slot_value":
        flat = one_line(content)
        prefix = old[: len(old) - len(flat)] if old.endswith(flat) else None
        return new[len(prefix):].strip() or None if prefix and new.startswith(prefix) else None
    if template == "story_step":
        prefix = old.split(": ", 1)[0] + ": "
        first = old[len(prefix):]
        if not new.startswith(prefix) or first not in content:
            return None
        return content.replace(first, new[len(prefix):].strip(), 1)
    return None


def edit_key(profile_id: str, fact_id: str, base_hash: str, new_text: str) -> str:
    """One edited block is proposed once: the same base and the same new words, the same key."""
    digest = hashlib.sha256(f"{profile_id}\0{fact_id}\0{base_hash}\0{new_text}".encode()).hexdigest()
    return digest[:32]


# -- the handler ---------------------------------------------------------------------------------

def _case(conn: Any, profile_id: str, case_id: str) -> dict:
    rows = _select(conn, "SELECT status, predecessor_fact_id, successor_fact_id FROM "
                   "correction_cases WHERE case_id = ? AND profile_id = ?", (case_id, profile_id))
    if not rows:
        raise ProposalError("The correction behind this edit is no longer there.")
    return rows[0]


def _content(conn: Any, profile_id: str, fact_id: str) -> str:
    rows = _select(conn, "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                   (fact_id, profile_id))
    if not rows:
        raise ProposalError("A memory named by the edit is not in this profile.")
    return str(rows[0]["content"])


def _validate(conn: Any, profile_id: str, payload: dict) -> list[str]:
    case = _case(conn, profile_id, str(payload.get("case_id") or ""))
    fact, successor = str(payload.get("fact_id") or ""), str(payload.get("successor_fact_id") or "")
    if (case["predecessor_fact_id"], case["successor_fact_id"]) != (fact, successor):
        raise ProposalError("The correction does not belong to this edit.")
    if case["status"] != "proposed":
        raise ProposalError("Only an edit still waiting for review can be proposed.")
    _content(conn, profile_id, successor)
    return [fact, successor]


def _describe(conn: Any, profile_id: str, payload: dict) -> str:
    old = _glance(_content(conn, profile_id, payload["fact_id"]))
    new = _glance(_content(conn, profile_id, payload["successor_fact_id"]))
    return f"Replace “{old}” with “{new}” (edited in the Markdown wiki)"


def _slot_moved(conn: Any, profile_id: str, base: Mapping[str, Any]) -> tuple[bool, dict | None]:
    """Compare-and-swap: has the slot the block was written against changed since?"""
    slot_id = base.get("slot_id")
    if not slot_id:
        return False, None
    rows = _select(conn, "SELECT slot_id, subject_key, relation_key, qualifier, version, "
                   "head_revision_id FROM memory_slots WHERE slot_id = ? AND profile_id = ?",
                   (slot_id, profile_id))
    if not rows:
        return True, None
    return int(rows[0]["version"]) != base.get("slot_version"), rows[0]


def _fact_is_current(conn: Any, profile_id: str, fact_id: str) -> bool:
    from superlocalmemory.compiler.page_templates import live_facts

    return fact_id in live_facts(conn, profile_id, [fact_id])


def _apply(conn: Any, ctx: ApplyContext, payload: dict) -> dict:
    base = payload.get("base") or {}
    case_id, fact, successor = payload["case_id"], payload["fact_id"], payload["successor_fact_id"]
    moved, slot = _slot_moved(conn, ctx.profile_id, base)
    if not moved and not base.get("slot_id") and not _fact_is_current(conn, ctx.profile_id, fact):
        moved = True  # no slot to compare: the memory itself changed since the file was written
    if not moved:
        _transition(conn, ctx, case_id, "proposed", "applied", temporal=True)
        before = _select(conn, "SELECT * FROM fact_trust WHERE fact_id = ?", (successor,))
        from superlocalmemory.trust import fact_trust

        fact_trust.mark_confirmed(conn, successor, by=ctx.actor_id, channel="proposal")
        return {"outcome": "applied", "cases": [case_id], "successor": successor,
                "trust_before": before[0] if before else None}
    if slot and slot["head_revision_id"]:
        # The edit is kept as a memory next to the slot's current value, and the person is
        # asked which is true now. Applied without touching the replaced memory (no
        # retirement): a retirement here would be the swap this compare-and-swap refused.
        _transition(conn, ctx, case_id, "proposed", "applied", temporal=False)
        dispute = _open_conflict(conn, ctx.profile_id, slot, successor)
        return {"outcome": "conflict", "cases": [case_id], "successor": successor, **dispute}
    _transition(conn, ctx, case_id, "proposed", "rejected", temporal=False)
    return {"outcome": "declined", "cases": [], "successor": successor}


def _open_conflict(conn: Any, profile_id: str, slot: dict, successor: str) -> dict:
    """The edit's text against the slot's current value, as a dispute."""
    from superlocalmemory.truth import slot_store
    from superlocalmemory.truth.slot_keys import SlotKey

    text = _content(conn, profile_id, successor)
    key = SlotKey(subject_key=slot["subject_key"], relation_key=slot["relation_key"],
                  qualifier=slot["qualifier"], cardinality="single", key_source="external_edit",
                  confidence=0.0, value_hash="edit:" + hashlib.sha256(text.encode()).hexdigest()[:24])
    scope = _select(conn, "SELECT scope FROM atomic_facts WHERE fact_id = ?", (successor,))[0]["scope"]
    got = slot_store.bind(conn, profile_id=profile_id, scope=scope or "personal", fact_id=successor,
                          key=key, decided_by="external_edit")
    if not got.dispute_id:  # a slot that has a head and a different value always disputes
        raise ProposalError("The slot changed while the edit was being applied; try again.")
    conn.execute("UPDATE slot_disputes SET reason_code = ? WHERE dispute_id = ?",
                 (CONFLICT, got.dispute_id))
    conn.execute("UPDATE slot_revisions SET reason_code = ? WHERE revision_id = ?",
                 (CONFLICT, got.revision_id))
    return {"dispute_id": got.dispute_id, "revision_id": got.revision_id, "slot_id": got.slot_id}


def _rollback(conn: Any, ctx: ApplyContext, payload: dict, inverse: dict) -> None:
    if inverse.get("outcome") == "applied":
        _rollback_cases(conn, ctx, inverse)
        before = inverse.get("trust_before")
        if before is None:
            conn.execute("DELETE FROM fact_trust WHERE fact_id = ?", (inverse["successor"],))
        else:
            columns = list(before)
            conn.execute(
                "INSERT OR REPLACE INTO fact_trust (" + ", ".join(columns) + ") VALUES ("
                + ", ".join("?" for _ in columns) + ")", tuple(before[c] for c in columns))
        return
    if inverse.get("outcome") != "conflict":
        return
    from superlocalmemory.truth import slot_store

    open_now = _select(conn, "SELECT 1 AS p FROM slot_disputes WHERE dispute_id = ? AND state = 'open'",
                       (inverse["dispute_id"],))
    if not open_now:
        raise ProposalError("The dispute this edit opened was settled since, so it was not rolled "
                            "back. Undo the settlement first.")
    slot_store.dismiss_dispute(conn, inverse["dispute_id"], by="rollback")
    conn.execute("UPDATE slot_revisions SET state = 'unbound' WHERE revision_id = ? "
                 "AND state = 'disputed'", (inverse["revision_id"],))
    conn.execute(
        "UPDATE memory_slots SET version = version + 1, state = CASE WHEN EXISTS (SELECT 1 FROM "
        "slot_disputes d WHERE d.slot_id = memory_slots.slot_id AND d.state = 'open') "
        "THEN 'disputed' ELSE 'current' END WHERE slot_id = ?", (inverse["slot_id"],))
    _transition(conn, ctx, inverse["cases"][0], "applied", "rolled_back", temporal=False)


def _reject(conn: Any, ctx: ApplyContext, payload: dict) -> None:
    case = _case(conn, ctx.profile_id, payload["case_id"])
    if case["status"] == "proposed":  # a declined edit leaves no pending correction behind
        _transition(conn, ctx, payload["case_id"], "proposed", "rejected", temporal=False)


register_handler(KIND, Handler(_validate, _describe, _apply, _rollback, _reject))

__all__ = ["CONFLICT", "KIND", "edit_key", "edit_text"]
