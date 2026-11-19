# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saving or confirming a memory as a rule or decision, under the minimal firewall (19 Nov).

Since the owner's 19 Nov decision the instruction firewall's floor is ``agent_inferred``: an
agent's own words are trusted for instruction slots directly, so a rule or decision an agent
saves (or whose kind it confirms) needs no promotion. This module keeps its doors and records
nothing for those facts. What it still does:

* a fact that is already trusted (``agent_inferred`` or above, or ``legacy``) is skipped: no
  promotion, no decision-log entry;
* a fact below the floor (a web page, a tool's output, another agent's message, an imported file,
  model text, unknown) is **not** promoted. It stays held and the ask is put on the waiting list
  (``firewall.request``), as the pin door does;
* **a correction keeps a rule working** (:func:`inherit`): when a correction is applied to a fact
  that was already trusted and the successor carries the rule or decision kind, the successor is
  checked the same way: trusted passes, below the floor is held and asked for. A predecessor that
  was not trusted passes nothing on. Rolling the correction back takes a promotion this module
  recorded back (:func:`take_back`);
* nothing happens while the firewall is not acting (``trust.taint`` off or shadow, or the kill
  switch on): nothing is held then.

A guess is not a decision: the background classifier (``storage/memory_kind_writes.apply_batch``)
never calls this module. It runs inside the canonical writer's own transaction, beside the kind
write or the save it follows, so the kind and its ask are one commit.

Ids, class names and counts only in logs.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from typing import Any

from superlocalmemory.core import decisions
from superlocalmemory.storage.memory_kinds import is_confirmed
from superlocalmemory.trust import fact_trust, firewall

logger = logging.getLogger(__name__)

#: The declared kinds that count as a promotion (the values of ``MemoryKind``).
KINDS = frozenset({firewall.RULE, firewall.DECISION})
#: How the kind reached the memory: the lead words of the decision summary.
SAVED, CONFIRMED, CORRECTED = "saved", "confirmed", "corrected"
VIA = "declared_kind"
#: Recorded on a promotion a correction passed on, so its rollback can take exactly that back.
CORRECTION_VIA = "correction"
_SAVEPOINT = "declared_promotion"


def wanted(kind: object) -> bool:
    """True when a ``kind`` declared now should be promoted: a rule or decision, firewall acting."""
    return str(kind or "").strip().lower() in KINDS and firewall.active()


def _promote_one(conn: sqlite3.Connection, fact_id: str, kind: str, *, by: str, channel: str,
                 how: str, via: str = VIA) -> str:
    """``skipped`` (already trusted, or no such fact) or ``held`` (below the floor: asked for).

    ``how`` and ``via`` are kept for the callers' sake; since the floor is ``agent_inferred`` a
    rule an agent saves needs no promotion, so nothing is promoted here and nothing is recorded
    for a trusted fact.
    """
    origin = fact_trust.origin_of(conn, fact_id)
    if origin is None or firewall.is_trusted_taint(origin["taint"]):
        return "skipped"
    firewall.request(conn, fact_id, kind, by=by, channel=channel)
    return "held"


def promote(conn: sqlite3.Connection, profile_id: str, facts: Iterable[tuple[str, str]], *,
            by: str, channel: str, how: str, via: str = VIA) -> dict[str, Any]:
    """Check each ``(fact_id, kind)`` whose kind is a rule or decision, as ``by`` through
    ``channel``: a fact below the floor is held and asked for. Only facts of ``profile_id`` are
    touched. Returns ``{"promoted": [], "held": [...]}`` (fact ids; ``promoted`` stays empty
    under the minimal firewall); empty while the firewall is not acting."""
    out: dict[str, list[str]] = {"promoted": [], "held": []}
    if not firewall.active() or channel not in firewall.CHANNELS:
        return out
    for fact_id, raw_kind in facts:
        kind = str(raw_kind or "").strip().lower()
        if kind not in KINDS:
            continue
        owned = conn.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
                             (fact_id, profile_id)).fetchone()
        if owned is None:
            continue
        verdict = _promote_one(conn, fact_id, kind, by=by, channel=channel, how=how, via=via)
        if verdict in out:
            out[verdict].append(fact_id)
    return out


def inherit(conn: sqlite3.Connection, profile_id: str, predecessor_id: str, successor_id: str,
            *, by: str, channel: str) -> dict[str, Any]:
    """On an applied correction: ask for the successor when what it retires was trusted.

    Runs inside the apply transaction, after the kind was carried over. The predecessor must be
    trusted (a promotion, a grandfathered memory or a class at or above the floor) and the
    successor must carry a confirmed rule or decision kind; a successor at or above the floor
    needs nothing, and a lower class is held and asked for. Empty while the firewall is not
    acting.
    """
    none: dict[str, list[str]] = {"promoted": [], "held": []}
    if not firewall.active():
        return none
    row = conn.execute("SELECT memory_kind, memory_kind_source FROM atomic_facts "
                       "WHERE fact_id = ? AND profile_id = ?", (successor_id, profile_id)).fetchone()
    if row is None or not is_confirmed(row[1]):
        return none
    before = fact_trust.origin_of(conn, predecessor_id)
    if before is None or not firewall.is_trusted_taint(before["taint"]):
        return none
    return promote(conn, profile_id, [(successor_id, row[0])], by=by, channel=channel,
                   how=CORRECTED, via=CORRECTION_VIA)


def take_back(conn: sqlite3.Connection, profile_id: str, successor_id: str, *, by: str,
              channel: str) -> bool:
    """On a rolled-back correction: withdraw the promotion :func:`inherit` recorded, if any.

    The successor is retired by the rollback anyway; this keeps the decision log honest (the
    promotion is marked undone, not left open) and the memory back under its own class.
    """
    found = decisions.open_decision(conn, profile_id, decisions.PROMOTION, successor_id)
    if found is None or found.get("detail", {}).get("via") != CORRECTION_VIA:
        return False
    conn.execute(f"SAVEPOINT {_SAVEPOINT}")
    try:
        firewall.withdraw(conn, successor_id, by=by, channel=channel)
    except (sqlite3.Error, LookupError, ValueError) as exc:
        conn.execute(f"ROLLBACK TO {_SAVEPOINT}")
        logger.warning("correction rollback: promotion of %s not withdrawn (%s)", successor_id,
                       type(exc).__name__)
        return False
    finally:
        conn.execute(f"RELEASE {_SAVEPOINT}")
    return True


__all__ = ["CONFIRMED", "CORRECTED", "CORRECTION_VIA", "KINDS", "SAVED", "VIA", "inherit",
           "promote", "take_back", "wanted"]
