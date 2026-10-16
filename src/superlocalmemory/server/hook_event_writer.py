# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The daemon's hook event writer: the only writer of hook events.

Hooks (and the ``/internal/hook-events`` route on their behalf) append events
to the spool (``hooks/event_spool.py``). One thread here moves them into
memory.db through the canonical writer path:

* ``tool_event`` rows go in through ``memory_write`` (process write lock +
  the shared busy timeout) in batches. Each batch commits together with the
  drain cursor for its file (a ``config`` row), so a crash part way through a
  file resumes after the last committed batch: no event twice, none lost.
* reward events (``fact_signal``, ``outcome_signal``, ``session_finalize``)
  run through ``EngagementRewardModel`` in strict mode under the same write
  lock. They are idempotent (a signal is a key set on a pending outcome, a
  settled outcome is not settled again), so a replay after a crash is safe.
  Each one is judged at the moment the hook observed it, so a signal that
  waited in the spool is not rejected for arriving late.
* Event ids already written are skipped: a hook that gave up on a slow daemon
  spools the same events the daemon may already have accepted.

A failing database raises out of ``drain_once``; nothing is consumed, the
background loop logs it and retries with backoff. Unreadable lines are
counted in the spool stats (``slm doctor`` shows them), never ignored.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    TypeAdapter,
    ValidationError,
)

from superlocalmemory.hooks import event_spool

logger = logging.getLogger(__name__)

CURSOR_PREFIX = "hook_spool_cursor:"
DEFAULT_BATCH = 200
DEFAULT_INTERVAL_S = 2.0
_MAX_BACKOFF_S = 30.0
_SEEN_CAP = 50_000
_SEED_ROWS = 20_000

_CONFIG_DDL = (
    "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, "
    "value TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
)


# ---------------------------------------------------------------------------
# Wire schema (validated at the route and again on every drained line)
# ---------------------------------------------------------------------------

_Id = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
_Text = Annotated[str, Field(min_length=1, max_length=256)]
_SignalName = Literal["dwell_ms", "requery", "edit", "cite"]
_SignalValue = Union[StrictBool, StrictInt, StrictFloat]


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_id: _Id
    profile_id: Annotated[str, Field(min_length=1, max_length=128)]


class ToolEventIn(_Event):
    kind: Literal["tool_event"]
    session_id: _Text
    tool_name: Annotated[str, Field(min_length=1, max_length=200)]
    event_type: Annotated[str, Field(max_length=64)] = "complete"
    project_path: Annotated[str, Field(max_length=500)] = ""
    input_summary: Annotated[str, Field(max_length=2000)] = ""
    output_summary: Annotated[str, Field(max_length=2000)] = ""
    created_at: Annotated[str, Field(min_length=1, max_length=64)]


class FactSignalIn(_Event):
    kind: Literal["fact_signal"]
    session_id: _Text
    fact_ids: Annotated[list[Annotated[str, Field(min_length=1, max_length=128)]],
                        Field(min_length=1, max_length=100)]
    signal_name: _SignalName
    signal_value: _SignalValue
    observed_ms: Annotated[int, Field(ge=0)]


class OutcomeSignalIn(_Event):
    kind: Literal["outcome_signal"]
    outcome_id: Annotated[str, Field(min_length=1, max_length=128)]
    signal_name: _SignalName
    signal_value: _SignalValue
    observed_ms: Annotated[int, Field(ge=0)]


class SessionFinalizeIn(_Event):
    kind: Literal["session_finalize"]
    session_id: _Text
    observed_ms: Annotated[int, Field(ge=0)]


HookEvent = Annotated[
    Union[ToolEventIn, FactSignalIn, OutcomeSignalIn, SessionFinalizeIn],
    Field(discriminator="kind"),
]


class HookEventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: Annotated[list[HookEvent], Field(min_length=1, max_length=32)]


_EVENT_ADAPTER: TypeAdapter = TypeAdapter(HookEvent)


def parse_line(raw: bytes):
    """One spooled line -> a validated event, or None when unreadable."""
    try:
        return _EVENT_ADAPTER.validate_json(raw)
    except (ValidationError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _insert_tool_events(conn: sqlite3.Connection, batch: list[ToolEventIn]) -> None:
    if not batch:
        return
    conn.executemany(
        "INSERT INTO tool_events (session_id, profile_id, project_path, tool_name,"
        " event_type, input_summary, output_summary, duration_ms, metadata, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        [(e.session_id, e.profile_id, e.project_path, e.tool_name, e.event_type,
          e.input_summary, e.output_summary,
          json.dumps({"event_id": e.event_id}, separators=(",", ":")), e.created_at)
         for e in batch],
    )


def _set_cursor(conn: sqlite3.Connection, key: str, value: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value, updated_at) "
        "VALUES (?, ?, datetime('now'))", (key, str(int(value))),
    )


def _apply_reward(db: Path, event) -> None:
    """Apply one reward event; raises on any database error (strict model)."""
    from superlocalmemory.learning.reward import EngagementRewardModel
    from superlocalmemory.storage.memory_write import _busy_timeout_ms

    observed = int(event.observed_ms)
    model = EngagementRewardModel(db, clock_ms=lambda: observed,
                                  busy_timeout_ms=_busy_timeout_ms(),
                                  raise_on_db_error=True)
    try:
        if isinstance(event, FactSignalIn):
            for oid in model.match_pending_for_fact_ids(
                    session_id=event.session_id, fact_ids=list(event.fact_ids)):
                model.register_signal(outcome_id=oid, signal_name=event.signal_name,
                                      signal_value=event.signal_value)
        elif isinstance(event, OutcomeSignalIn):
            model.register_signal(outcome_id=event.outcome_id,
                                  signal_name=event.signal_name,
                                  signal_value=event.signal_value)
        elif isinstance(event, SessionFinalizeIn):
            with model._lock:
                rows = model._get_conn().execute(
                    "SELECT outcome_id FROM pending_outcomes "
                    "WHERE session_id = ? AND status = 'pending'",
                    (event.session_id,),
                ).fetchall()
            for row in rows:
                model.finalize_outcome(outcome_id=row[0])
    finally:
        model.close()


class _SeenIds:
    """Bounded insertion-ordered set of event ids already written."""

    def __init__(self, cap: int = _SEEN_CAP) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._cap = cap

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._ids

    def add_all(self, ids: Iterable[str]) -> None:
        for event_id in ids:
            self._ids[event_id] = None
            self._ids.move_to_end(event_id)
        while len(self._ids) > self._cap:
            self._ids.popitem(last=False)


@dataclass(frozen=True)
class DrainReport:
    applied: int = 0
    duplicates: int = 0
    invalid: int = 0
    files: int = 0

    def __add__(self, other: "DrainReport") -> "DrainReport":
        return DrainReport(self.applied + other.applied, self.duplicates + other.duplicates,
                           self.invalid + other.invalid, self.files + other.files)


class HookEventWriter:
    """Drains the hook event spool into ``memory_db``. See module docstring."""

    def __init__(self, memory_db: str | Path, *, root: str | Path | None = None,
                 batch_size: int = DEFAULT_BATCH,
                 interval_s: float = DEFAULT_INTERVAL_S) -> None:
        self._db = Path(memory_db)
        self._root = event_spool.spool_root(root)
        self._batch = max(1, int(batch_size))
        self._interval_s = max(0.05, float(interval_s))
        self._seen = _SeenIds()
        self._ready = False
        self._drain_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- one pass -------------------------------------------------------
    def drain_once(self) -> DrainReport:
        """Drain everything waiting. Raises when the database refuses."""
        with self._drain_lock:
            self._prepare()
            total = DrainReport()
            for path in event_spool.claim_ready(self._root):
                total = total + self._drain_file(path)
            return total

    def _prepare(self) -> None:
        if self._ready:
            return
        from superlocalmemory.storage.memory_write import memory_write

        with memory_write(self._db) as conn:
            conn.execute(_CONFIG_DDL)
            has_tool_events = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tool_events'"
            ).fetchone()
            # No table: nothing to seed. A tool event would fail to insert
            # (and stay spooled); reward events do not need the table.
            rows = conn.execute(
                "SELECT metadata FROM tool_events WHERE metadata LIKE '{\"event_id\":%' "
                "ORDER BY id DESC LIMIT ?", (_SEED_ROWS,),
            ).fetchall() if has_tool_events else []
        ids = []
        for (raw,) in reversed(rows):
            try:
                ids.append(json.loads(raw)["event_id"])
            except (ValueError, KeyError, TypeError):
                continue
        self._seen.add_all(ids)
        self._ready = True

    def _read_cursor(self, key: str) -> int:
        from superlocalmemory.storage.memory_write import memory_write

        with memory_write(self._db) as conn:
            row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        try:
            return max(0, int(row[0])) if row else 0
        except (TypeError, ValueError):
            return 0

    def _drain_file(self, path: Path) -> DrainReport:
        from superlocalmemory.storage.memory_write import memory_write

        key = CURSOR_PREFIX + path.name
        try:
            lines = path.read_bytes().splitlines()
        except FileNotFoundError:  # erased (GDPR) between claim and read
            return DrainReport()
        profile = path.name.split(".", 1)[0]
        report = DrainReport(files=1)
        for begin in range(self._read_cursor(key), len(lines), self._batch):
            chunk = lines[begin:begin + self._batch]
            parsed = [parse_line(raw) for raw in chunk if raw.strip()]
            events = [e for e in parsed if e is not None]
            invalid = len(parsed) - len(events)
            report = report + self._apply_batch(events, key, begin + len(chunk))
            if invalid:
                event_spool.record_counts(self._root, profile, invalid=invalid)
                logger.warning("hook spool: %d unreadable line(s) in %s", invalid, path.name)
                report = report + DrainReport(invalid=invalid)
        path.unlink(missing_ok=True)
        with memory_write(self._db) as conn:
            conn.execute("DELETE FROM config WHERE key = ?", (key,))
        return report

    def _apply_batch(self, events: list, key: str, cursor: int) -> DrainReport:
        from superlocalmemory.storage.memory_write import memory_write
        from superlocalmemory.storage.write_lock import get_write_lock

        fresh, batch_ids = [], set()
        for event in events:
            if event.event_id in self._seen or event.event_id in batch_ids:
                continue
            batch_ids.add(event.event_id)
            fresh.append(event)
        rewards = [e for e in fresh if not isinstance(e, ToolEventIn)]
        if rewards:
            with get_write_lock(self._db):
                for event in rewards:
                    _apply_reward(self._db, event)
        with memory_write(self._db) as conn:
            _insert_tool_events(conn, [e for e in fresh if isinstance(e, ToolEventIn)])
            _set_cursor(conn, key, cursor)
        self._seen.add_all(e.event_id for e in fresh)
        return DrainReport(applied=len(fresh), duplicates=len(events) - len(fresh))

    # -- background thread ---------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="slm-hook-event-writer",
                                        daemon=True)
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Stop after one last drain. True when the thread has exited."""
        thread = self._thread
        if thread is None:
            return True
        self._stop.set()
        self._wake.set()
        thread.join(timeout_s)
        return not thread.is_alive()

    def _run(self) -> None:
        backoff = self._interval_s
        while True:
            stopping = self._stop.is_set()
            try:
                self.drain_once()
                backoff = self._interval_s
            except Exception as exc:  # noqa: BLE001 - logged, spool kept, retried
                backoff = min(_MAX_BACKOFF_S, max(backoff * 2, 1.0))
                logger.warning("hook event writer: drain failed (%s: %s); events stay "
                               "spooled, retry in %.0f s", type(exc).__name__, exc, backoff)
            if stopping:
                return
            self._wake.wait(backoff)
            self._wake.clear()


# ---------------------------------------------------------------------------
# Intake (the route and the hook socket both land here)
# ---------------------------------------------------------------------------

def accept_batch(batch: HookEventBatch, *, root: str | Path | None = None) -> tuple[int, int]:
    """Spool a validated batch (grouped per profile) and wake the writer.

    Returns ``(written, dropped)``. Dropped events (spool full) are counted
    in the spool stats by ``append_events``. Raises when the spool itself is
    unusable, so the caller does not ack and the hook keeps the events.
    """
    by_profile: dict[str, list[dict]] = {}
    for event in batch.events:
        by_profile.setdefault(event.profile_id, []).append(event.model_dump())
    written = dropped = 0
    for profile, events in by_profile.items():
        result = event_spool.append_events(events, profile=profile, root=root)
        written += result.written
        dropped += result.dropped
    if dropped:
        logger.warning("hook spool full: %d event(s) dropped (see `slm doctor`)", dropped)
    wake()
    return written, dropped


# ---------------------------------------------------------------------------
# Daemon singleton
# ---------------------------------------------------------------------------

_writer: HookEventWriter | None = None
_writer_guard = threading.Lock()


def start_writer(memory_db: str | Path) -> HookEventWriter:
    global _writer
    with _writer_guard:
        if _writer is None:
            _writer = HookEventWriter(memory_db)
        _writer.start()
        return _writer


def stop_writer(timeout_s: float = 5.0) -> bool:
    global _writer
    with _writer_guard:
        writer, _writer = _writer, None
    return True if writer is None else writer.stop(timeout_s)


def wake() -> None:
    writer = _writer
    if writer is not None:
        writer.wake()


def erase_profile_spool(profile_id: str, *, db_path: str | Path | None = None,
                        root: str | Path | None = None) -> int:
    """GDPR erase: drop ``profile_id``'s spooled events and drain cursors.

    Runs under this process's writer drain lock, so a batch in flight lands
    before (and is then removed by the profile sweep) and nothing of the
    profile is drained after. Returns the number of events removed.
    """
    writer = _writer
    guard = writer._drain_lock if writer is not None else threading.Lock()
    with guard:
        removed = event_spool.erase_profile(root, profile_id)
        if db_path is not None and Path(db_path).exists():
            from superlocalmemory.storage.memory_write import memory_write

            stem = event_spool.profile_stem(profile_id)
            with memory_write(Path(db_path)) as conn:
                has_config = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='config'"
                ).fetchone()
                if has_config:
                    conn.execute("DELETE FROM config WHERE key >= ? AND key < ?",
                                 (f"{CURSOR_PREFIX}{stem}.", f"{CURSOR_PREFIX}{stem}/"))
    return removed


__all__ = [
    "CURSOR_PREFIX", "DrainReport", "HookEvent", "HookEventBatch", "HookEventWriter",
    "accept_batch", "erase_profile_spool", "parse_line", "start_writer", "stop_writer",
    "wake",
]
