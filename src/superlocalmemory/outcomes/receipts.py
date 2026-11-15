# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Injection receipts: plan a delivery, apply the randomiser, record it (D-22).

A delivery surface (session start, later every path) calls:

1. ``plan(...)``: the candidates and their randomised choices, or ``None``
   when receipts are off (flag or kill switch) or the key cannot be read;
2. ``deliver(...)``: renders what remains after withholding, records the
   receipts, and returns the rendering.

Two rules make withholding safe:

* **Protection comes from the store, not from the surface.** Before choosing,
  ``plan`` looks every candidate up: a pinned fact or a confirmed standing rule
  is never eligible, whatever role the surface gave it and whatever the config
  says (W0-01). If the lookup fails, nothing on that delivery is eligible.
* **Nothing is withheld unless its receipt is stored.** ``deliver`` writes the
  receipts before the rendering leaves; if the write fails (store busy, tables
  missing, any error), it renders again with everything included and records
  nothing.

The write waits at most ``WRITE_WAIT_S`` in all (the process lock and SQLite's
busy wait share it); a failed write may add one more render. It never raises.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from superlocalmemory.outcomes import randomiser
from superlocalmemory.outcomes.randomiser import Candidate, Choice

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Longest a delivery waits for the writer (routing only: a latency budget).
WRITE_WAIT_S = 0.25

HOLDOUT_ON = "on"
HOLDOUT_OFF = "off"

_PROVENANCE_ID = re.compile(r"fact_id=([A-Za-z0-9_\-]+)")
#: Kinds whose confirmed facts are standing rules (core/standing_rules.py).
_STANDING_KINDS = ("rule", "decision")
_CONFIRMED_SOURCES = ("user", "caller")


@dataclass(frozen=True)
class DeliveryPlan:
    delivery_id: str
    profile_id: str
    session_id: str | None
    client: str | None
    path: str
    holdout_state: str
    epsilon: float
    key_id: str
    query_digest: str | None
    assignment: randomiser.Assignment

    @property
    def choices(self) -> tuple[Choice, ...]:
        return self.assignment.choices

    @property
    def withheld(self) -> frozenset[str]:
        return frozenset(c.fact_id for c in self.choices if not c.included)


def _flags(slm_home: Path):
    from superlocalmemory.core.feature_flags import FeatureFlags

    return FeatureFlags(slm_home)


def protected_roles(db_path: Path | None, fact_ids: Sequence[str]) -> dict[str, str] | None:
    """Fact id -> ``pinned`` / ``standing`` for the candidates the store protects.

    None when the store cannot be read: the caller then withholds nothing.
    """
    ids = sorted({f for f in fact_ids if f})
    if not ids:
        return {}
    if db_path is None:
        return None
    marks = ",".join("?" * len(ids))
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=WRITE_WAIT_S)
        try:
            pinned, standing = _protected(conn, ids, marks)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.info("injection receipt: protection lookup failed (%s); nothing withheld", exc)
        return None
    return {**{f: "standing" for f in standing}, **{f: "pinned" for f in pinned}}


def _protected(conn: sqlite3.Connection, ids: list[str], marks: str) -> tuple[set, set]:
    """Pinned and standing ids, from whichever of the store's sources exist (a store
    without a ``pinned_facts`` table simply has no pins there)."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    if "fact_id" not in cols:
        raise sqlite3.OperationalError("no atomic_facts table")
    pinned: set[str] = set()
    if "pinned" in cols:
        pinned |= {r[0] for r in conn.execute(
            f"SELECT fact_id FROM atomic_facts WHERE fact_id IN ({marks}) "  # noqa: S608
            "AND COALESCE(pinned, 0) = 1", ids)}
    if "pinned_facts" in tables:
        pinned |= {r[0] for r in conn.execute(
            f"SELECT fact_id FROM pinned_facts WHERE fact_id IN ({marks})", ids)}  # noqa: S608
    standing: set[str] = set()
    if {"memory_kind", "memory_kind_source"} <= cols:
        kinds = ",".join("?" * len(_STANDING_KINDS))
        sources = ",".join("?" * len(_CONFIRMED_SOURCES))
        standing = {r[0] for r in conn.execute(
            f"SELECT fact_id FROM atomic_facts WHERE fact_id IN ({marks}) "  # noqa: S608
            f"AND memory_kind IN ({kinds}) AND memory_kind_source IN ({sources})",
            (*ids, *_STANDING_KINDS, *_CONFIRMED_SOURCES))}
    return pinned, standing


def _protect(candidates: list[Candidate], db_path: Path | None) -> tuple[list[Candidate], bool]:
    """The candidates with the store's protection applied, and whether withholding may run."""
    roles = protected_roles(db_path, [c.fact_id for c in candidates])
    if roles is None:
        return candidates, False
    return [Candidate(c.fact_id, roles.get(c.fact_id, c.role), c.score) for c in candidates], True


def plan(*, slm_home: Path, db_path: Path | None, profile_id: str, session_id: str | None,
         path: str, query: str, candidates: Iterable[Candidate], client: str | None = None,
         flags=None) -> DeliveryPlan | None:
    """The delivery's choices, or ``None`` when no receipt will be written."""
    from superlocalmemory.core.receipt_key import digest, load_or_create

    try:
        flags = flags if flags is not None else _flags(slm_home)
        if not flags.observing("outcomes.receipts"):
            return None
        protected, readable = _protect(list(candidates), db_path)
        holdout_on = flags.enabled("outcomes.holdout") and readable
        key = load_or_create(slm_home)
        delivery_id = f"dlv_{uuid.uuid4().hex}"
        epsilon = randomiser.EPSILON if holdout_on else 0.0
        assignment = randomiser.choose(key, delivery_id, path, protected,
                                       holdout_on=holdout_on, epsilon=epsilon)
    except Exception as exc:  # a receipt is never worth a failed delivery
        logger.warning("injection receipt plan skipped: %s", exc)
        return None
    return DeliveryPlan(
        delivery_id=delivery_id, profile_id=profile_id, session_id=session_id or None,
        client=client or None, path=path,
        holdout_state=HOLDOUT_ON if holdout_on else HOLDOUT_OFF, epsilon=epsilon,
        key_id=hashlib.sha256(key).hexdigest()[:16],
        query_digest=digest(key, b"query:" + query.encode()) if query else None,
        assignment=assignment)


def shown_in_text(text: str) -> frozenset[str]:
    """Fact ids the shared renderer stamped into ``text`` (its provenance tags)."""
    return frozenset(_PROVENANCE_ID.findall(text or ""))


def deliver(plan_: DeliveryPlan | None, db_path: Path | None, items: Sequence[T], *,
            fact_id: Callable[[T], str], render: Callable[[list[T]], R],
            shown: Callable[[R], Iterable[str]]) -> R:
    """Render ``items`` less the withheld ones, record, and return the rendering.

    ``items`` are the randomisable candidates; ``render`` adds whatever else
    the surface shows. If the receipts cannot be stored, everything is shown.
    """
    if plan_ is None or db_path is None:
        return render(list(items))
    withheld = plan_.withheld
    out = render([i for i in items if fact_id(i) not in withheld])
    try:
        delivered = frozenset(shown(out)) - withheld
    except Exception as exc:
        logger.warning("injection receipt: delivered set unreadable: %s", exc)
        delivered = frozenset()
    if record(db_path, plan_, delivered):
        return out
    return render(list(items)) if withheld else out


def record(db_path: Path, plan_: DeliveryPlan, delivered: frozenset[str],
           *, now: datetime | None = None) -> bool:
    """Store one delivery and its receipts in one short transaction; never raises."""
    from superlocalmemory.storage.write_lock import get_write_lock

    started = time.monotonic()
    lock = get_write_lock(db_path)
    if not lock.acquire(timeout=WRITE_WAIT_S):
        logger.info("injection receipt skipped: writer busy")
        return False
    try:
        left = max(0.0, WRITE_WAIT_S - (time.monotonic() - started))
        conn = sqlite3.connect(str(db_path), timeout=left)
        try:
            conn.execute(f"PRAGMA busy_timeout={int(left * 1000)}")
            with conn:
                _insert(conn, plan_, delivered, now or datetime.now(UTC))
            return True
        finally:
            conn.close()
    except Exception as exc:
        logger.info("injection receipt skipped: %s", exc)
        return False
    finally:
        lock.release()


def _insert(conn: sqlite3.Connection, p: DeliveryPlan, delivered: frozenset[str],
            at: datetime) -> None:
    a = p.assignment
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " query_digest, candidates, delivered_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (p.delivery_id, p.profile_id, p.session_id, p.client, p.path, randomiser.DESIGN,
         p.holdout_state, p.epsilon, a.eligible_count, a.withhold_draw, a.pick_draw, p.key_id,
         p.query_digest, len(a.choices), at.isoformat()))
    conn.executemany(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, included, delivered)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(p.delivery_id, c.fact_id, p.profile_id, c.role, c.position, c.score,
          int(c.eligible), c.exclusion, c.propensity, int(c.included),
          int(c.included and c.fact_id in delivered)) for c in a.choices])


__all__ = ["HOLDOUT_OFF", "HOLDOUT_ON", "WRITE_WAIT_S", "DeliveryPlan", "deliver", "plan",
           "protected_roles", "record", "shown_in_text"]
