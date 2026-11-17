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
import queue
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from superlocalmemory.outcomes import randomiser, usage
from superlocalmemory.outcomes.randomiser import Candidate, Choice

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: Longest a delivery waits for the writer (routing only: a latency budget).
WRITE_WAIT_S = 0.25
#: Pull receipts waiting for the background writer; past this they are dropped
#: and counted (routing only: a memory bound, never a reason to slow a recall).
PULL_QUEUE_MAX = 256
#: The writer thread ends after this long with nothing to write and starts again
#: on the next receipt (routing only: no thread lingers in an idle process).
PULL_IDLE_S = 0.5

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
    #: The recall engine's id for the answer delivered (pulls; W3-6c, D-45).
    recall_query_id: str | None = None

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


@dataclass(frozen=True)
class Begun:
    """A delivery whose treatment was drawn before its fetch (W3-3, D-44): the
    channel arm changes the fetch itself, so it must be known first."""

    delivery_id: str
    treatment: randomiser.Treatment


def begin(*, slm_home: Path, path: str, flags=None) -> Begun | None:
    """Draw a delivery's treatment before it fetches, or None when receipts are off."""
    from superlocalmemory.core.receipt_key import load_or_create

    try:
        flags = flags if flags is not None else _flags(slm_home)
        if not flags.observing("outcomes.receipts"):
            return None
        key = load_or_create(slm_home)
        delivery_id = f"dlv_{uuid.uuid4().hex}"
        return Begun(delivery_id, randomiser.treat(
            key, delivery_id, path, holdout_on=flags.enabled("outcomes.holdout")))
    except Exception as exc:  # a receipt is never worth a failed delivery
        logger.warning("injection receipt draw skipped: %s", exc)
        return None


def plan(*, slm_home: Path, db_path: Path | None, profile_id: str, session_id: str | None,
         path: str, query: str, candidates: Iterable[Candidate], client: str | None = None,
         flags=None, begun: Begun | None = None,
         recall_query_id: str | None = None) -> DeliveryPlan | None:
    """The delivery's choices, or ``None`` when no receipt will be written.

    ``begun`` carries a treatment drawn before the fetch; it is kept as drawn
    (what the fetch did is what is recorded), and withholding still needs a
    readable store."""
    from superlocalmemory.core.receipt_key import digest, load_or_create

    try:
        flags = flags if flags is not None else _flags(slm_home)
        if not flags.observing("outcomes.receipts"):
            return None
        protected, readable = _protect(list(candidates), db_path)
        key = load_or_create(slm_home)
        if begun is not None:
            delivery_id, treatment = begun.delivery_id, begun.treatment
            holdout_on = treatment.randomised and readable
            randomised = treatment.randomised
        else:  # drawn after the fetch: only the memory arm is possible now
            delivery_id = f"dlv_{uuid.uuid4().hex}"
            holdout_on = randomised = flags.enabled("outcomes.holdout") and readable
            treatment = randomiser.treat(key, delivery_id, path, holdout_on=holdout_on,
                                         epsilon=randomiser.EPSILON if holdout_on else 0.0,
                                         channel_arm=False)
        epsilon = randomiser.EPSILON if randomised else 0.0
        assignment = randomiser.choose(key, delivery_id, path, protected,
                                       holdout_on=holdout_on, epsilon=epsilon,
                                       treatment=treatment)
    except Exception as exc:  # a receipt is never worth a failed delivery
        logger.warning("injection receipt plan skipped: %s", exc)
        return None
    return DeliveryPlan(
        delivery_id=delivery_id, profile_id=profile_id, session_id=session_id or None,
        client=client or None, path=path,
        holdout_state=HOLDOUT_ON if randomised else HOLDOUT_OFF, epsilon=epsilon,
        key_id=hashlib.sha256(key).hexdigest()[:16],
        query_digest=digest(key, b"query:" + query.encode()) if query else None,
        assignment=assignment, recall_query_id=recall_query_id or None)


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
    return deliver_recorded(plan_, db_path, items, fact_id=fact_id, render=render,
                            shown=shown)[0]


def deliver_recorded(plan_: DeliveryPlan | None, db_path: Path | None, items: Sequence[T], *,
                     fact_id: Callable[[T], str], render: Callable[[list[T]], R],
                     shown: Callable[[R], Iterable[str]]) -> tuple[R, bool]:
    """``deliver``, and whether the receipt was stored. A surface whose fetch was
    changed by the channel arm must undo that change when it was not (no
    treatment without its receipt)."""
    if plan_ is None or db_path is None:
        return render(list(items)), False
    withheld = plan_.withheld
    out = render([i for i in items if fact_id(i) not in withheld])
    try:
        delivered = frozenset(shown(out)) - withheld
    except Exception as exc:
        logger.warning("injection receipt: delivered set unreadable: %s", exc)
        delivered = frozenset()
    if record(db_path, plan_, delivered):
        return out, True
    return (render(list(items)) if withheld else out), False


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
    t = a.treatment
    conn.execute(
        "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path,"
        " design, holdout_state, epsilon, eligible_count, withhold_draw, pick_draw, key_id,"
        " query_digest, candidates, delivered_at, arm, arm_draw, dropped_channel, memory_share,"
        " recall_query_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (p.delivery_id, p.profile_id, p.session_id, p.client, p.path, randomiser.DESIGN,
         p.holdout_state, p.epsilon, a.eligible_count, a.withhold_draw, a.pick_draw, p.key_id,
         p.query_digest, len(a.choices), usage.stamp(at), t.arm, t.arm_draw, t.channel,
         t.memory_share, p.recall_query_id))
    conn.executemany(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position,"
        " score, eligible, exclusion, propensity, included, delivered)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(p.delivery_id, c.fact_id, p.profile_id, c.role, c.position, c.score,
          int(c.eligible), c.exclusion, c.propensity, int(c.included),
          int(c.included and c.fact_id in delivered)) for c in a.choices])


@dataclass(frozen=True)
class Pull:
    """One answered recall: what an agent asked for and was shown (W3-1)."""

    slm_home: Path
    db_path: Path
    profile_id: str
    session_id: str | None
    path: str
    query: str
    #: (fact id, score) per result, in the order shown; score may be None.
    results: tuple[tuple[str, float | None], ...]
    client: str | None = None
    #: When the answer left, not when the writer got to it: a usage signal is
    #: joined to deliveries at or before it, so a late write must not move this.
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: The recall engine's id for this answer (envelope ``query_id``; W3-6c).
    recall_query_id: str | None = None


def record_pull(pull: Pull, *, flags=None) -> bool:
    """Store one pull delivery now: every result delivered, nothing withheld.

    A pull path is never randomised (``randomiser.RANDOMISED_PATHS``), so each
    result is logged at propensity 1. Never raises; False when nothing was stored.
    """
    try:
        candidates = [Candidate(fid, "recall", None if score is None else float(score))
                      for fid, score in pull.results if fid]
        plan_ = plan(slm_home=pull.slm_home, db_path=pull.db_path, profile_id=pull.profile_id,
                     session_id=pull.session_id, path=pull.path, query=pull.query,
                     candidates=candidates, client=pull.client, flags=flags,
                     recall_query_id=pull.recall_query_id)
        if plan_ is None:
            return False
        if plan_.withheld:  # a pull never withholds; refuse rather than misrecord
            logger.warning("pull receipt refused: %s withheld on %s", len(plan_.withheld),
                           pull.path)
            return False
        return record(pull.db_path, plan_, frozenset(c.fact_id for c in candidates),
                      now=pull.at)
    except Exception as exc:
        logger.info("pull receipt skipped: %s", exc)
        return False


class _PullWriter:
    """One background thread that stores pull receipts, so a recall never waits."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Pull] = queue.Queue(maxsize=PULL_QUEUE_MAX)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.stored = 0

    def submit(self, pull: Pull) -> bool:
        try:
            self._queue.put_nowait(pull)
        except queue.Full:
            self.dropped += 1
            return False
        self._start()
        return True

    def _start(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="slm-pull-receipts",
                                                daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                pull = self._queue.get(timeout=PULL_IDLE_S)
            except queue.Empty:
                with self._lock:  # a receipt queued now finds no writer and starts one
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            try:
                if record_pull(pull):
                    self.stored += 1
            finally:
                self._queue.task_done()

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait until every submitted receipt is handled (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.005)
        return True


PULL_WRITER = _PullWriter()


def submit_pull(pull: Pull) -> bool:
    """Queue one pull receipt for the background writer; never blocks, never raises."""
    try:
        return PULL_WRITER.submit(pull)
    except Exception as exc:
        logger.info("pull receipt not queued: %s", exc)
        return False


def engine_paths(engine) -> tuple[Path | None, Path | None]:
    """(SLM home, memory.db) of a real engine, or (None, None).

    ``MemoryEngine`` keeps its config private (``_config``); reading only a
    public ``config`` found nothing on a real engine, so session-start receipts
    were never written there (found in W3-1).
    """
    config = getattr(engine, "config", None) or getattr(engine, "_config", None)
    home = getattr(config, "base_dir", None)
    db_path = getattr(getattr(engine, "db", None), "db_path", None)
    if isinstance(home, Path) and isinstance(db_path, Path):
        return home, db_path
    return None, None


def pull_from_results(engine, *, path: str, profile_id: str, session_id: str | None,
                      query: str, results, client: str | None = None,
                      recall_query_id: str | None = None) -> Pull | None:
    """The receipt for a serialized recall answer, or None without a real store."""
    from superlocalmemory.core.session_identity import is_conversation

    home, db_path = engine_paths(engine)
    if home is None or db_path is None or path not in randomiser.PULL_PATHS:
        return None
    rows = tuple((str(r.get("fact_id") or ""), r.get("score")) for r in (results or [])
                 if isinstance(r, dict))
    sid = session_id if is_conversation(session_id, profile_id) else None
    return Pull(home, db_path, profile_id, sid, path, query or "",
                tuple((f, s if isinstance(s, (int, float)) else None) for f, s in rows if f),
                client or None, recall_query_id=str(recall_query_id or "") or None)


__all__ = ["HOLDOUT_OFF", "HOLDOUT_ON", "PULL_IDLE_S", "PULL_QUEUE_MAX", "PULL_WRITER", "WRITE_WAIT_S",
           "Begun", "DeliveryPlan", "Pull", "begin", "deliver", "deliver_recorded",
           "engine_paths", "plan", "protected_roles",
           "pull_from_results", "record", "record_pull", "shown_in_text", "submit_pull"]
