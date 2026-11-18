# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Undo any recorded decision through the same door that made it (4.2.30, D-49).

``core/decisions`` lists what callers decided. Each kind has its own reversal, and every
reversal is also a door of its own (``slm trust withdraw``, ``slm proposals rollback``,
``POST /api/disputes/{id}/reopen``, ``POST /api/audit/{id}/retract``). This module is the one
place that picks the right one for a row of the log, so the dashboard's Undo button and an
agent's ``POST /api/decisions/{id}/undo`` do exactly what the specific door does, and the undo is
recorded with who took the decision back and through which channel.
"""

from __future__ import annotations

import sqlite3

from superlocalmemory.core import decisions


class UndoRefused(ValueError):
    """The decision cannot be undone (already undone, or the value moved on); nothing changed."""


def undo(conn: sqlite3.Connection, *, profile_id: str, decision_id: str,
         decider: decisions.Decider, engine: object | None = None,
         runtime: object | None = None) -> dict:
    """Reverse one logged decision. ``LookupError`` if there is no such decision in the profile,
    :class:`UndoRefused` if it cannot be undone now. ``engine`` and ``runtime`` are the daemon's,
    needed only by an import, whose rollback is a background job that removes memories."""
    found = decisions.get(conn, profile_id, decision_id)
    if found is None:
        raise LookupError(f"no decision {decision_id}")
    if found["undone"]:
        raise UndoRefused("That decision was already undone.")
    if not found["can_undo"]:
        raise UndoRefused("That kind of decision is not undone; it is itself an undo or a "
                          "rejection.")
    kind, subject = found["kind"], found["subject_id"]
    try:
        if kind == decisions.PROMOTION:
            from superlocalmemory.trust import firewall

            done = firewall.withdraw(conn, subject, by=decider.actor, channel=decider.channel)
        elif kind == decisions.PROPOSAL:
            from superlocalmemory.compiler import proposals

            done = proposals.rollback_set(
                conn, profile_id=profile_id, set_id=subject,
                basis=proposals.Basis.decision(decider.actor, decider.channel))
        elif kind == decisions.DISPUTE:
            from superlocalmemory.truth import disputes

            done = disputes.reopen(conn, profile_id=profile_id, dispute_id=subject,
                                   channel=decider.channel, actor=decider.actor)
        elif kind == decisions.IMPORT:
            done = _undo_import(engine, runtime, profile_id, subject, decider)
        elif kind == decisions.LABEL:
            from superlocalmemory.core import audit_review

            done = audit_review.retract(conn, profile_id=profile_id, item_id=subject,
                                        channel=decider.channel, actor=decider.actor)
        else:  # pragma: no cover - the kinds are closed in core/decisions
            raise UndoRefused(f"unknown decision kind {kind}")
    except LookupError as exc:
        raise UndoRefused(f"There is nothing left to undo: {exc}") from None
    except ValueError as exc:  # StateRefused / ReopenRefused / ProposalError subclass it
        raise UndoRefused(str(exc)) from None
    return {"decision_id": decision_id, "kind": kind, "undone": True, **_plain(done)}


def _undo_import(engine: object | None, runtime: object | None, profile_id: str, run_id: str,
                 decider: decisions.Decider) -> dict:
    """Roll the whole import run back in the background (it can take a while for a big run)."""
    from superlocalmemory.portability import service

    if engine is None:
        raise UndoRefused("An import can only be undone through the running daemon.")
    try:
        run = service.start_undo(engine, runtime, profile_id=profile_id, run_id=run_id,
                                 decider=decider)
    except service.ImportRefused as exc:
        raise UndoRefused(str(exc)) from None
    return {"state": run["state"], "run_id": run_id}


def _plain(done: object) -> dict:
    """The reversal's own answer, only when it is a flat mapping (a set's view is not)."""
    if not isinstance(done, dict):
        return {}
    return {k: v for k, v in done.items() if isinstance(v, (str, int, float, bool, list))
            and k not in ("proposals",)}


__all__ = ["UndoRefused", "undo"]
