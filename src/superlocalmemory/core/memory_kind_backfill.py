# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

""""Classify my memories": a user-started, resumable, undoable run (LLD §6).

One daemon thread (``slm-kind-backfill``) works one batch at a time:

1. Pick the oldest run that needs work (queued, running or reverting).
2. Yield while a recall is in flight; yield (for a bounded time) while new
   memories wait to be enriched.
3. Read a batch of the run's own profile's facts — never a row merely shared
   into it, never a quarantined or erased one — with no write lock.
4. Ask the classifier with **no lock held**, inside ``background_work`` and a
   per-batch deadline: a model call can never stall the thread forever under
   steady recall load; a batch the model did not answer is retried later,
   then (bounded) typed by the rules.
5. Write the batch, its history rows and the run cursor in ONE short
   transaction, each row guarded by the values it was read with. A run paused
   or cancelled meanwhile rolls the whole batch back.

Never changes ``fact_type`` (I2), never replaces a kind a person or caller
confirmed, never sends memory text off the device unless the run itself was
confirmed for that. Undo walks the history newest-first and restores exactly
what was there (also guarded: a later edit by hand is never clobbered).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core import memory_kind_backfill_plan as plan
from superlocalmemory.core import memory_kind_runs as runs
from superlocalmemory.core import memory_kind_wiring as wiring
from superlocalmemory.core import recall_gate
from superlocalmemory.core.memory_kind_config import MemoryKindConfig
from superlocalmemory.core.memory_kind_runs import ACTIVE, REVERTIBLE
from superlocalmemory.encoding.memory_kind_rules import cue_kind, suggest_by_rules
from superlocalmemory.storage.memory_kind_store import MemoryKindStore

logger = logging.getLogger(__name__)

THREAD_NAME = "slm-kind-backfill"
SCHEMA_REASON = ("Memory kinds need this update's database step, which has not run yet "
                 "(usually because there was not enough free disk space for the safety "
                 "copy). Everything else keeps working.")
STARTING = "The memory engine is starting; try again in a moment."
ONLINE_PAUSE = ("Settings changed so that typing would now send memories online. "
                "Cancel this run and start a new one to confirm that.")
_POLL_S, _YIELD_S, _REVERT_PACE_S, _MAX_BACKLOG_YIELD_S = 1.0, 0.25, 0.05, 10.0
#: Pacing from the measured lock hold: after holding the write lock H, a batch
#: waits at least H x this before the next (the backfill's share of the lock
#: stays under 1 / (1 + this)), and x ``_BACKLOG_SHARE`` while new memories
#: wait to be enriched. It used to wait a flat 10 s per batch under any
#: backlog: 21,500 rows took 81 minutes on a large store.
_LOCK_SHARE, _BACKLOG_SHARE = 4.0, 19.0


class BackfillRefused(Exception):
    """A run request the user must change (code + plain message + payload)."""

    def __init__(self, code: str, message: str, payload: dict | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.payload = code, message, dict(payload or {})


@dataclass(frozen=True, slots=True)
class StepResult:
    action: str          # idle|yield|batch|revert|deferred|paused|finished|stopped|error
    delay: float
    run_id: str | None = None


def _lock_ms(result: Any, started: float) -> float:
    """The write-lock hold the store measured, else the call's wall time."""
    held = getattr(result, "write_ms", 0.0)
    return float(held) if held else (time.perf_counter() - started) * 1000.0


class BackfillRunner:
    def __init__(self, *, engine_supplier: Callable[[], Any],
                 store: MemoryKindStore | None = None,
                 classifier_supplier: Callable[[], Any] | None = None,
                 config: MemoryKindConfig | Callable[[], MemoryKindConfig] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 lease: Callable[[], Any] | None = None,
                 preempt: Callable[[], bool] | None = None,
                 revert_batch_size: int = 50, batch_budget_s: float = 10.0,
                 max_model_misses: int = 3, max_errors: int = 5) -> None:
        self._engine_supplier, self._store, self._config = engine_supplier, store, config
        self._classifier_supplier, self._clock = classifier_supplier, clock
        self._lease = lease or (lambda: contextlib.nullcontext(True))
        self._preempt = preempt or (lambda: False)
        self._revert_batch_size, self._batch_budget_s = revert_batch_size, batch_budget_s
        self._max_model_misses, self._max_errors = max_model_misses, max_errors
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._misses: dict[tuple[str, int], int] = {}
        self._errors = 0
        self._backlog_since: float | None = None
        self._deadline: float | None = None
        #: Batch size per backend (and 'revert'), adapted to the measured write time.
        self._sizes: dict[str, int] = {}
        self._recent: dict[str, tuple[float, ...]] = {}
        #: New memories waiting for enrichment go first (replaceable in tests).
        self._materializer_due: Callable[[Any], bool] = runs.materializer_due
        #: Write-lock hold of each batch (BEGIN IMMEDIATE to COMMIT), in ms.
        self.batch_write_ms: collections.deque[float] = collections.deque(maxlen=4096)
        #: Time each batch waited for the write lock behind other writers, in ms.
        self.batch_wait_ms: collections.deque[float] = collections.deque(maxlen=4096)
        self._last_hold_s = 0.0
        #: What the one automatic classification check found, per profile.
        self.upgrade: dict[str, dict] = {}

    # -- thread ------------------------------------------------------------

    def start(self) -> None:
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=THREAD_NAME, daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> bool:
        self._stop.set()
        with self._thread_lock:
            thread = self._thread
            if thread is not None:
                thread.join(timeout_s)
                if thread.is_alive():
                    return False
                self._thread = None
        return True

    def _loop(self) -> None:
        self._ensure_typed_once()
        while not self._stop.is_set():
            try:
                step = self.run_once()
            except Exception as exc:  # noqa: BLE001 — the thread must outlive any batch
                logger.warning("Memory kind run step failed (%s)", type(exc).__name__)
                step = StepResult("error", 5.0)
            if step.delay > 0:
                self._stop.wait(step.delay)

    # -- what the routes call ------------------------------------------------

    def create_run(self, profile_id: str, *, mode: str, requested_by: str,
                   confirm_data_leaves_device: bool = False) -> dict[str, Any]:
        if mode not in ("untyped", "refresh"):
            raise BackfillRefused("bad_request", "Mode must be 'untyped' or 'refresh'.")
        engine, db, store = self._parts(required=True)
        cfg = self._config_for(engine)
        choice = self._choice(engine, cfg)
        if choice.active == "off":
            raise BackfillRefused("disabled", plan.REASONS["off"])
        if runs.active_run(db, profile_id) is not None:
            raise BackfillRefused("run_active", "A classification run is already in "
                                  "progress for this profile.")
        facts = plan.count_candidates(db, profile_id, mode)
        if choice.leaves_device and confirm_data_leaves_device is not True:
            requests = math.ceil(facts / plan.MAX_MODEL_FACTS)
            message = (f"This run would send the text of {facts} memories to the Jev "
                       f"service ({requests} requests). Confirm to continue.")
            raise BackfillRefused("needs_confirmation", message, {
                "needs_confirmation": True, "facts": facts, "requests": requests,
                "leaves_device": True, "backend": choice.active, "detail": message})
        try:
            run = store.create_run(profile_id, backend=choice.active,
                                   recipe_id=plan.recipe_for(choice.active), mode=mode,
                                   requested_by=requested_by, total_estimate=facts)
        except sqlite3.IntegrityError as exc:
            raise BackfillRefused("run_active", "A classification run is already in "
                                  "progress for this profile.") from exc
        return run

    def pause(self, run_id: str, *, requested_by: str, profile_id: str | None = None) -> dict:
        return self._move(run_id, profile_id, "paused", ("queued", "running"),
                          "Only a queued or running run can be paused.")

    def resume(self, run_id: str, *, requested_by: str, profile_id: str | None = None) -> dict:
        return self._move(run_id, profile_id, "queued", ("paused",),
                          "Only a paused run can be resumed.", clear_error=True)

    def cancel(self, run_id: str, *, requested_by: str, profile_id: str | None = None) -> dict:
        return self._move(run_id, profile_id, "cancelled", ("queued", "running", "paused"),
                          "This run has already finished.", finished=True)

    def revert(self, run_id: str, *, requested_by: str, profile_id: str | None = None) -> dict:
        try:
            return self._move(run_id, profile_id, "reverting", REVERTIBLE,
                              "Pause, cancel or let the run finish before undoing it.")
        except sqlite3.IntegrityError as exc:
            raise BackfillRefused("run_active", "Finish or cancel the run in progress "
                                  "for this profile before undoing another.") from exc

    def status(self, profile_id: str) -> dict[str, Any]:
        """What the status card and ``slm kinds status`` show (LLD §6.6)."""
        engine, db, store = self._parts(required=False)
        cfg = self._config_for(engine)
        ready = bool(db is not None and db.has_memory_kind_columns())
        try:
            timings = tuple(self.batch_write_ms)
        except RuntimeError:          # appended to by the runner thread mid-copy
            timings = ()
        reason = SCHEMA_REASON if db is not None else STARTING
        out = runs.status_view(db if ready else None, store, cfg, self._choice(engine, cfg),
                               profile_id, timings, not_ready_reason=reason)
        # The start-up check of confirmed kinds runs in the background
        # (core/kind_reconcile_task.py): say whether it has finished.
        task = getattr(engine, "_kind_reconcile", None)
        out["confirmed_kind_check"] = task.snapshot() if task is not None else {
            "state": "not_started"}
        out["automatic_classification"] = self.upgrade.get(profile_id, {"state": "not_checked"})
        try:
            waits = tuple(self.batch_wait_ms)
        except RuntimeError:
            waits = ()
        out["batch_write_ms"] = {**out.get("batch_write_ms", {}),
                                 "lock_wait_max": max(waits, default=None)}
        return runs.explain_counts(out)

    # -- one step --------------------------------------------------------------

    def run_once(self) -> StepResult:
        engine, db, store = self._parts(required=False)
        if db is None or not db.has_memory_kind_columns():
            return StepResult("idle", _POLL_S)
        run = store.next_runnable()
        if run is None:
            return StepResult("idle", _POLL_S)
        if recall_gate.in_flight() > 0:
            return StepResult("yield", _YIELD_S, run["run_id"])
        try:
            with self._lease() as admitted:
                if admitted is None:
                    return StepResult("yield", _YIELD_S, run["run_id"])
                with recall_gate.background_work(preempt_requested=self._preempted):
                    if run["status"] == "reverting":
                        step = self._revert_step(store, run)
                    else:
                        step = self._forward_step(engine, db, store, run)
            self._errors = 0
            return step
        except Exception as exc:  # noqa: BLE001 — recorded on the run, then retried
            return self._failed_step(db, run["run_id"], exc)

    def _forward_step(self, engine: Any, db: Any, store: MemoryKindStore,
                      run: dict) -> StepResult:
        run_id, cfg = run["run_id"], self._config_for(engine)
        choice = self._choice(engine, cfg)
        if choice.active == "off" or (choice.leaves_device and run["backend"] != "jev"):
            reason = plan.REASONS["off"] if choice.active == "off" else ONLINE_PAUSE
            runs.transition(db, run_id, "paused", ("queued", "running"), last_error=reason)
            return StepResult("paused", 0.0, run_id)
        if run["status"] == "queued" and not runs.transition(
                db, run_id, "running", ("queued",), started=True):
            return StepResult("yield", 0.0, run_id)
        if self._materializer_due(db):
            if self._backlog_since is None:
                self._backlog_since = self._clock()
            gap = min(_MAX_BACKLOG_YIELD_S, max(_YIELD_S, self._last_hold_s * _BACKLOG_SHARE))
            if self._clock() - self._backlog_since < gap:
                return StepResult("yield", _YIELD_S, run_id)
        self._backlog_since = None
        backend, cursor = choice.active, int(run["cursor_rowid"])
        configured = int(cfg.batch_size.get(backend, 8))
        limit = min(configured, self._sizes.get(backend, configured))
        batch = store.select_batch(run["profile_id"], after_rowid=cursor,
                                   limit=limit, mode=run["mode"])
        if not batch:
            runs.transition(db, run_id, "completed", ("running",), finished=True)
            return StepResult("finished", 0.0, run_id)
        started = self._clock()
        assignments, note = self._classify(engine, batch, backend, (run_id, cursor))
        if assignments is None:
            return StepResult("deferred", note, run_id)
        old = plan.old_confidences(db, run["profile_id"], batch) if run["mode"] == "refresh" \
            else {}
        changes, _unchanged = plan.plan_changes(batch, assignments, old)
        t0 = time.perf_counter()
        result = (store.apply_batch(run_id, run["profile_id"], changes,
                                    new_cursor=batch[-1].rowid, actor="backfill",
                                    examined=len(batch))
                  if changes else runs.advance(db, run_id, batch[-1].rowid, len(batch)))
        self._adapt(backend, _lock_ms(result, t0), configured,
                    float(getattr(result, "wait_ms", 0.0) or 0.0))
        if not result.run_still_active:
            return StepResult("stopped", 0.0, run_id)
        if note:
            runs.note(db, run_id, note)
        pace = len(batch) / float(cfg.rate_per_second.get(backend, 8.0))
        pace = max(pace - (self._clock() - started), self._last_hold_s * _LOCK_SHARE)
        return StepResult("batch", max(0.0, pace), run_id)

    def _classify(self, engine: Any, batch: list, backend: str,
                  key: tuple[str, int]) -> tuple[list | None, Any]:
        """(assignments, note) or (None, retry delay) when a model must be retried."""
        rules = [suggest_by_rules(c.content, c.fact_type) for c in batch]
        model = self._ask(engine, batch, backend)
        if model is None:
            misses = self._misses.get(key, 0) + 1
            self._misses = {key: misses}
            if misses <= self._max_model_misses:
                return None, float(min(30, 2 ** misses))
            return rules, "The model did not answer for some memories; they were typed by rules."
        self._misses = {}
        if any(a is None for a in model):
            return None, _YIELD_S          # turned off mid-batch: the next step pauses
        decided = [plan.decide(m, r, cue_kind(c.content))
                   for m, r, c in zip(model, rules, batch)]
        return decided, ""

    def _ask(self, engine: Any, batch: list, backend: str) -> list | None:
        """Classifier output per fact; None when a model was due and did not answer.
        Runs with a per-batch deadline: idle waits end there, and the deadline
        also preempts any remaining model chunk (see ``_preempted``)."""
        self._deadline = time.monotonic() + self._batch_budget_s
        try:
            with recall_gate.idle_wait_deadline(self._deadline):
                return plan.ask_in_chunks(self._classifier_for(engine), batch, backend)
        finally:
            self._deadline = None

    def _revert_step(self, store: MemoryKindStore, run: dict) -> StepResult:
        limit = min(self._revert_batch_size, self._sizes.get("revert", self._revert_batch_size))
        t0 = time.perf_counter()
        result = store.revert_batch(run["run_id"], run["profile_id"], limit=limit)
        self._adapt("revert", _lock_ms(result, t0), self._revert_batch_size,
                    float(getattr(result, "wait_ms", 0.0) or 0.0))
        return StepResult("revert", max(_REVERT_PACE_S, self._last_hold_s * _LOCK_SHARE),
                          run["run_id"])

    def _adapt(self, key: str, write_ms: float, configured: int, wait_ms: float = 0.0) -> None:
        """Keep each batch write near the lock target (median of recent writes)."""
        self.batch_write_ms.append(write_ms)
        self.batch_wait_ms.append(wait_ms)
        self._last_hold_s = write_ms / 1000.0
        recent = (self._recent.get(key, ()) + (write_ms,))[-plan.WINDOW:]
        decided = plan.next_batch_size(self._sizes.get(key, configured), recent, configured)
        if decided is None:
            self._recent[key] = recent
        else:
            self._sizes[key] = decided
            self._recent[key] = ()

    def _failed_step(self, db: Any, run_id: str, exc: Exception) -> StepResult:
        self._errors += 1
        name = type(exc).__name__
        logger.warning("Memory kind run %s: batch failed (%s)", run_id, name)
        message = f"A batch failed ({name}); it will be retried."
        try:
            if self._errors >= self._max_errors:
                runs.transition(db, run_id, "failed", ("queued", "running", "reverting"),
                                 finished=True, last_error=f"Stopped after repeated errors "
                                 f"({name}). Resume or undo it from the dashboard.")
                self._errors = 0
            else:
                runs.record_error(db, run_id, message)
        except Exception:  # noqa: BLE001 — the store itself may be the problem
            pass
        return StepResult("error", float(min(60, 2 ** self._errors)), run_id)

    # -- helpers ---------------------------------------------------------------

    def _ensure_typed_once(self) -> None:
        """Queue each profile's one automatic classification, if it needs one."""
        try:
            from superlocalmemory.core.kind_upgrade import ensure_store_typed

            engine, db, store = self._parts(required=False)
            if db is None:
                return
            cfg = self._config_for(engine)
            self.upgrade = ensure_store_typed(db, store, self._choice(engine, cfg),
                                              bool(cfg.enabled))
        except Exception as exc:  # noqa: BLE001 - the runner must keep running
            logger.warning("Automatic classification check skipped (%s)", type(exc).__name__)

    def _preempted(self) -> bool:
        if self._stop.is_set() or self._preempt():
            return True
        deadline = self._deadline
        return deadline is not None and time.monotonic() >= deadline

    def _engine(self) -> Any | None:
        try:
            return self._engine_supplier()
        except Exception:  # noqa: BLE001
            return None

    def _parts(self, *, required: bool) -> tuple[Any, Any, MemoryKindStore | None]:
        engine = self._engine()
        db = getattr(engine, "db", None) or getattr(engine, "_db", None) \
            if engine is not None else None
        if db is None and self._store is not None:
            db = self._store.db
        if required and db is None:
            raise BackfillRefused("not_ready", STARTING)
        if required and not db.has_memory_kind_columns():
            raise BackfillRefused("schema_not_ready", SCHEMA_REASON)
        store = self._store or (MemoryKindStore(db) if db is not None else None)
        return engine, db, store

    def _store_for(self, engine: Any) -> MemoryKindStore:
        return self._parts(required=False)[2]

    def _config_for(self, engine: Any) -> MemoryKindConfig:
        cfg = self._config() if callable(self._config) else self._config
        return cfg if isinstance(cfg, MemoryKindConfig) else wiring.current_config(engine)

    def _classifier_for(self, engine: Any) -> Any | None:
        if self._classifier_supplier is not None:
            return self._classifier_supplier()
        found = getattr(engine, "_kind_classifier", None)
        return found if found is not None else wiring.build_kind_classifier(engine)

    def _choice(self, engine: Any, cfg: MemoryKindConfig) -> plan.BackendChoice:
        return plan.resolve_backfill_backend(cfg, wiring.engine_mode(engine),
                                             wiring.live_judge(engine),
                                             wiring.llm_available(engine))

    def _move(self, run_id: str, profile_id: str | None, to: str, expected: tuple,
              refusal: str, *, finished: bool = False, clear_error: bool = False) -> dict:
        _engine, db, store = self._parts(required=True)
        run = store.get_run(run_id)
        if run is None or (profile_id is not None and run["profile_id"] != profile_id):
            raise BackfillRefused("not_found", "No such classification run.")
        if not runs.transition(db, run_id, to, expected, finished=finished,
                                last_error="" if clear_error else None):
            raise BackfillRefused("bad_state", refusal)
        return store.get_run(run_id) or {}


__all__ = ["ACTIVE", "REVERTIBLE", "SCHEMA_REASON", "THREAD_NAME", "BackfillRefused",
           "BackfillRunner", "StepResult"]
