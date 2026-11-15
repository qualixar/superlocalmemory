# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Injection receipts: plan a delivery, apply the randomiser, record it (D-22).

A delivery surface (session start, later every path) calls:

1. ``plan(...)``: the candidates and their randomised choices, or ``None``
   when receipts are off (flag or kill switch) or the key cannot be read;
2. ``deliver(...)``: renders what remains after withholding, records the
   receipts, and returns the rendering.

The rule that makes withholding safe to measure: **nothing is withheld unless
its receipt is stored.** ``deliver`` writes the receipts before the rendering
leaves; if the write fails (store busy, tables missing, any error), it renders
again with everything included and records nothing, so the agent gets what it
would have got without 4.2.30 and no unrecorded withholding exists.

Writes are short and bounded (``WRITE_WAIT_S``); a delivery is never slowed by
more than that, and a failed write never raises to the surface.
"""

from __future__ import annotations

import logging
import re
import sqlite3
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


@dataclass(frozen=True)
class DeliveryPlan:
    delivery_id: str
    profile_id: str
    session_id: str
    path: str
    holdout_state: str
    epsilon: float
    query_digest: str | None
    choices: tuple[Choice, ...]

    @property
    def withheld(self) -> frozenset[str]:
        return frozenset(c.fact_id for c in self.choices if not c.included)


def _flags(slm_home: Path):
    from superlocalmemory.core.feature_flags import FeatureFlags

    return FeatureFlags(slm_home)


def plan(*, slm_home: Path, profile_id: str, session_id: str, path: str, query: str,
         candidates: Iterable[Candidate], flags=None) -> DeliveryPlan | None:
    """The delivery's choices, or ``None`` when no receipt will be written."""
    from superlocalmemory.core.receipt_key import digest, load_or_create

    try:
        flags = flags if flags is not None else _flags(slm_home)
        if not flags.observing("outcomes.receipts"):
            return None
        holdout_on = flags.enabled("outcomes.holdout")
        key = load_or_create(slm_home)
        delivery_id = f"dlv_{uuid.uuid4().hex}"
        epsilon = randomiser.EPSILON if holdout_on else 0.0
        choices = randomiser.choose(key, delivery_id, path, candidates,
                                    holdout_on=holdout_on, epsilon=epsilon)
    except Exception as exc:  # a receipt is never worth a failed delivery
        logger.warning("injection receipt plan skipped: %s", exc)
        return None
    return DeliveryPlan(
        delivery_id=delivery_id, profile_id=profile_id, session_id=session_id or "",
        path=path, holdout_state=HOLDOUT_ON if holdout_on else HOLDOUT_OFF,
        epsilon=epsilon, query_digest=digest(key, b"query:" + query.encode()) if query else None,
        choices=choices)


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

    lock = get_write_lock(db_path)
    if not lock.acquire(timeout=WRITE_WAIT_S):
        logger.info("injection receipt skipped: writer busy")
        return False
    try:
        conn = sqlite3.connect(str(db_path), timeout=WRITE_WAIT_S)
        try:
            conn.execute(f"PRAGMA busy_timeout={int(WRITE_WAIT_S * 1000)}")
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
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, path, design,"
        " holdout_state, epsilon, query_digest, candidates, delivered_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (p.delivery_id, p.profile_id, p.session_id, p.path, randomiser.DESIGN,
         p.holdout_state, p.epsilon, p.query_digest, len(p.choices), at.isoformat()))
    conn.executemany(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, draw, included, delivered)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(p.delivery_id, c.fact_id, p.profile_id, c.role, c.position, c.score,
          int(c.eligible), c.exclusion, c.propensity, c.draw, int(c.included),
          int(c.included and c.fact_id in delivered)) for c in p.choices])


__all__ = ["HOLDOUT_OFF", "HOLDOUT_ON", "WRITE_WAIT_S", "DeliveryPlan", "deliver", "plan",
           "record", "shown_in_text"]
