# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The pending materializer: drains durable ingestion work and legacy pending rows.

Everything it needs from the server process (engine, profile runtime, the pending
queue, event emission, actor identity, recall activity) is injected, so this
module imports nothing from the server, CLI or MCP layers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PendingProfileMismatchError(RuntimeError):
    """A legacy pending row no longer matches the admitted profile lease."""


def _no_reap(db) -> list[str]:
    return []


def _no_reconcile(*args, **kwargs) -> Any:
    return None


def _no_recalls() -> int:
    return 0


@dataclass(frozen=True)
class PassHooks:
    """Collaborators of one ingestion pass that live in the server process."""

    reap: Callable[[Any], list[str]] = _no_reap
    reconcile_manifest: Callable[..., None] = _no_reconcile
    reconcile_pending: Callable[[Any], Any] = _no_reconcile
    recalls_needing_embedder: Callable[[], int] = _no_recalls


def should_idle(pending: object, durable_complete: int, durable_failed: int) -> bool:
    """Whether the materializer pass earned a sleep before the next one.

    ``durable_failed`` used to suppress the sleep. The pass runs one operation
    at a time, so a single operation that cannot succeed kept this loop at full
    speed indefinitely, re-reaping and re-listing on every iteration. A failure
    is activity, not progress. GitHub #137.
    """
    return not pending and not durable_complete


def run_operation(
    runtime,
    engine_supplier,
    operation,
    *,
    expected_profile_id: str | None = None,
):
    """Run one bounded background unit against an admitted engine snapshot.

    Cooperative preemption: if a profile transition is already in progress,
    skip this materialization cycle entirely and return None.  The caller's
    loop retries on the next iteration, by which time the switch has committed
    and a clean admission is available.  This prevents the materializer from
    holding the operation lease during the transition drain window.
    """
    # Writer-priority: don't acquire a new lease when a transition is draining.
    if runtime is not None and (runtime.transitioning or runtime.background_paused):
        if expected_profile_id is not None:
            raise PendingProfileMismatchError(
                "pending materialization deferred during profile transition"
            )
        return None
    with runtime.operation() as snapshot:
        if (
            expected_profile_id is not None
            and snapshot.profile_id != expected_profile_id
        ):
            raise PendingProfileMismatchError(
                "pending profile changed before materializer admission"
            )
        # Resolve the engine only after admission. A concurrent mode/provider
        # reconfiguration may have replaced the module-level engine while this
        # worker was waiting at the transition barrier.
        engine = engine_supplier()
        if engine is None:
            return None
        engine_profile_id = getattr(engine, "_profile_id", None)
        if (
            expected_profile_id is not None
            and engine_profile_id != expected_profile_id
        ):
            raise PendingProfileMismatchError(
                "resident engine does not match pending profile"
            )
        from superlocalmemory.core.recall_gate import background_work
        with background_work(
            preempt_requested=lambda: bool(runtime is not None and runtime.transitioning),
        ):
            return operation(engine)


def ingestion_pass(
    engine,
    *,
    hooks: PassHooks,
    emit_event: Callable[..., None],
    limit: int = 50,
    min_queryable_age_seconds: float = 1.0,
) -> tuple[int, int]:
    """Materialize durable M018 work once; return ``(complete, failed)``."""
    # Recovery first, unconditionally: terminalizing exhausted leases is
    # pure SQL and must never wait on recall quiescence or embedder
    # warmth — a cold embedder blocked the reap forever on one operator
    # box, wedging the write pipeline until manual DB surgery (#131).
    db = getattr(engine, "_db", None)
    reaped = hooks.reap(db) if db is not None else []
    if reaped:
        logger.warning(
            "Materializer terminalized %d exhausted ingestion operation(s)",
            len(reaped),
        )
    # Yield while a recall may still need the embedder (question not embedded yet);
    # later steps yield on their own: embeds per text, the local judge per recall.
    if hooks.recalls_needing_embedder() > 0:
        return 0, 0

    # A local sentence-transformers cold start can take minutes.  Remember's
    # queryable projection is already durable, so defer enrichment until the
    # daemon warmup/health monitor has proved the worker ready.  This preserves
    # every enrichment layer while preventing a background cold load from
    # monopolizing the same worker needed by foreground recall.
    embedder = getattr(engine, "_embedder", None)
    if embedder is not None and hasattr(embedder, "is_warm"):
        try:
            if not bool(embedder.is_warm):
                return 0, 0
        except Exception:
            return 0, 0

    from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
    from superlocalmemory.core.ingestion_command import IngestionState

    command = build_engine_ingestion_command(engine)
    completed = failed = 0
    for operation in command.repository.list_materializable(
        limit=limit,
        min_queryable_age_seconds=min_queryable_age_seconds,
    ):
        try:
            result = command.materialize(operation.operation_id)
        except Exception as exc:
            failed += 1
            logger.warning(
                "Ingestion operation %s could not be materialized: %s",
                operation.operation_id,
                exc,
            )
            continue
        if result.state is IngestionState.COMPLETE:
            completed += 1
            hooks.reconcile_manifest(
                engine,
                result.operation_id,
                getattr(operation, "profile_id", ""),
                result.fact_ids,
            )
            emit_event(
                "memory.stored",
                payload={
                    "operation_id": result.operation_id,
                    "fact_ids": list(result.fact_ids),
                    "path": "canonical_materializer",
                    "content_preview": result.raw_content[:120],
                },
                source_agent="materializer",
            )
        else:
            failed += 1
            logger.warning(
                "Ingestion operation %s failed: %s",
                result.operation_id,
                result.last_error,
            )
    hooks.reconcile_pending(engine)
    return completed, failed


def legacy_item(engine, item: dict, *, actor_id: str) -> str:
    """Backfill one pre-M018 pending.db row through canonical ingestion."""
    from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
    from superlocalmemory.core.ingestion_command import (
        IngestionRequest,
        IngestionState,
    )

    expected_profile_id = str(item.get("profile_id") or "default")
    if getattr(engine, "_profile_id", None) != expected_profile_id:
        raise PendingProfileMismatchError(
            "legacy pending item does not match resident engine profile"
        )

    metadata_value = item.get("metadata") or "{}"
    try:
        metadata = (
            json.loads(metadata_value)
            if isinstance(metadata_value, str)
            else dict(metadata_value)
        )
    except (TypeError, ValueError):
        metadata = {}
    if item.get("tags"):
        metadata.setdefault("tags", item["tags"])
    scope = metadata.pop("scope", None) or "personal"
    shared_with = tuple(metadata.pop("shared_with", None) or ())
    source_type = str(metadata.pop("_slm_source_type", "legacy-pending"))
    idempotency_key = str(
        metadata.pop("_slm_idempotency_key", f"pending:{item['id']}")
    )
    command = build_engine_ingestion_command(engine)
    receipt = command.submit(IngestionRequest(
        content=item["content"],
        profile_id=expected_profile_id,
        source_type=source_type,
        idempotency_key=idempotency_key,
        metadata=metadata,
        scope=scope,
        shared_with=shared_with,
        trusted_actor_id=actor_id,
        session_id=str(metadata.get("session_id") or ""),
    ))
    result = command.materialize(receipt.operation_id)
    if result.state is not IngestionState.COMPLETE:
        raise RuntimeError(result.last_error or "legacy pending materialization failed")
    return result.operation_id


@dataclass(kw_only=True)
class PendingMaterializer:
    """Background service that drains ingestion operations and the legacy queue."""

    engine_supplier: Callable[[], Any]
    runtime_supplier: Callable[[], Any]
    pending_store: Any  # object with get_pending / mark_done / mark_failed
    emit_event: Callable[..., None]
    actor_id_supplier: Callable[[], str]
    recalls_in_flight: Callable[[], int]
    hooks: PassHooks = field(default_factory=PassHooks)
    # The collaborators below default to the module functions; the server
    # process passes call-time lookups so its own names stay patchable.
    idle_predicate: Callable[[object, int, int], bool] = should_idle
    ingestion_step: Callable[[Any, int], tuple[int, int]] | None = None
    legacy_step: Callable[[Any, dict], str] | None = None
    name: str = "pending_materializer"

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.ingestion_step is None:
            self.ingestion_step = lambda engine, limit: ingestion_pass(
                engine, hooks=self.hooks, emit_event=self.emit_event, limit=limit,
            )
        if self.legacy_step is None:
            self.legacy_step = lambda engine, item: legacy_item(
                engine, item, actor_id=self.actor_id_supplier(),
            )

    def start(self) -> None:
        """Drain M018 operations and backfill the legacy pending.db queue."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="pending-materializer",
        )
        self._thread.start()
        logger.info("Pending materializer started (recall-priority)")

    def stop(self, timeout_s: float = 5.0) -> bool:
        """Stop and join the background writer before closing its engine."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                logger.warning(
                    "Pending materializer did not stop within %.1fs", timeout_s,
                )
                return False
        self._thread = None
        return True

    def health(self) -> dict:
        thread = self._thread
        running = thread is not None and thread.is_alive()
        return {"state": "running" if running else "stopped", "detail": ""}

    def _loop(self) -> None:
        # Log first engine acquisition so we know the materializer is alive.
        engine_logged = False
        waiting_logged = False
        while not self._stop.is_set():
            try:
                # Read the suppliers on every iteration so the engine published
                # by the lifespan is picked up, never a stale reference.
                engine = self.engine_supplier()
                runtime = self.runtime_supplier()
                if engine is None or runtime is None:
                    if not waiting_logged:
                        logger.info(
                            "Materializer: waiting for engine/runtime to init..."
                        )
                        waiting_logged = True
                    time.sleep(0.5)
                    continue
                if not engine_logged:
                    logger.info("Materializer: engine acquired, starting drain loop")
                    engine_logged = True
                self._cycle(runtime)
            except Exception as exc:
                logger.warning("materializer loop error: %s", exc)
                time.sleep(5.0)

    def _cycle(self, runtime) -> None:
        cycle_result = run_operation(
            runtime,
            self.engine_supplier,
            # One operation per lease bounds profile-switch wait time without
            # allowing engine components to rebind halfway through an
            # enrichment pipeline.
            lambda admitted_engine: self.ingestion_step(admitted_engine, 1),
        )
        durable_complete, durable_failed = cycle_result or (0, 0)
        if runtime.background_paused:  # a model switch is swapping: no spin
            time.sleep(0.25)
            return
        # Only backfill legacy pending items enqueued under the active
        # profile — never materialize another profile's queued memory
        # under whichever profile happens to be active now.
        pending = self.pending_store.get_pending(
            limit=50, profile_id=runtime.snapshot.profile_id,
        )
        if self.idle_predicate(pending, durable_complete, durable_failed):
            time.sleep(1.0)
            return
        if pending:
            logger.info(
                "Materializer: backfilling %d legacy pending memories", len(pending),
            )
        for item in pending:
            if self._stop.is_set():
                break
            self._backfill_one(runtime, item)

    def _backfill_one(self, runtime, item: dict) -> None:
        waits = 0
        while self.recalls_in_flight() > 0 and waits < 60:
            time.sleep(0.5)
            waits += 1
        try:
            operation_id = run_operation(
                runtime,
                self.engine_supplier,
                lambda admitted_engine: self.legacy_step(admitted_engine, item),
                expected_profile_id=str(item.get("profile_id") or "default"),
            )
            if operation_id is None:
                raise RuntimeError("resident engine became unavailable")
            self.pending_store.mark_done(item["id"])
            self.emit_event(
                "memory.stored",
                payload={
                    "pending_id": item["id"],
                    "operation_id": operation_id,
                    "path": "legacy_pending_backfill",
                    "content_preview": item["content"][:120],
                },
                source_agent="materializer",
            )
        except PendingProfileMismatchError:
            # A profile transition committed after this row was fetched but
            # before it obtained an operation lease. Leave it pending, without
            # consuming a retry, so the owning profile can safely drain it later.
            logger.debug("Pending %d deferred after profile switch", item["id"])
        except Exception as exc:
            logger.warning("Pending %d failed: %s", item["id"], exc)
            self.pending_store.mark_failed(item["id"], str(exc))
