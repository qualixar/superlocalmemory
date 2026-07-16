# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Unified Daemon — single FastAPI process for ALL routes.

Replaces the dual-process architecture (stdlib daemon + FastAPI dashboard).
One MemoryEngine singleton shared by CLI, MCP, Dashboard, and Mesh routes.

Architecture:
  slm serve       → starts unified daemon (uvicorn on port 8765)
  slm remember X  → HTTP POST to daemon → instant
  slm recall X    → HTTP GET from daemon → instant
  slm dashboard   → opens browser to http://localhost:8765
  slm serve stop  → POST /stop → graceful uvicorn shutdown

Port 8765: primary (dashboard + API + daemon routes)
Port 8767: TCP redirect for backward compat (deprecated)

24/7 by default. Opt-in auto-kill: --idle-timeout=1800

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# v3.6.7: Tell mcp/server.py it is being imported inside the daemon process.
# This suppresses the three side-effect threads (mcp-warmup, parent-watchdog,
# stdin-eof-monitor) that are harmful when the MCP server runs embedded.
# Must be set BEFORE any import of superlocalmemory.mcp.server.
os.environ.setdefault("SLM_MCP_EMBEDDED", "1")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT
from superlocalmemory.infra.daemon_identity import (
    DaemonDescriptor,
    build_descriptor,
    clear_descriptor,
    descriptor_path,
    write_descriptor,
)
from superlocalmemory.infra.data_root import (
    assert_no_durable_root_conflict,
    canonical_data_root,
    state_path,
)

logger = logging.getLogger("superlocalmemory.unified_daemon")

_DEFAULT_PORT = 8765
_LEGACY_PORT = 8767
_ACTIVE_DAEMON_DESCRIPTOR: DaemonDescriptor | None = None


def _configured_daemon_port() -> int:
    """Return the configured bind port, falling back safely to the default."""
    try:
        return int(os.environ.get("SLM_DAEMON_PORT", "") or _DEFAULT_PORT)
    except ValueError:
        return _DEFAULT_PORT


def _process_descriptor(port: int, version: str, state: str) -> DaemonDescriptor:
    """Return this process's stable namespace/instance identity."""
    global _ACTIVE_DAEMON_DESCRIPTOR
    if _ACTIVE_DAEMON_DESCRIPTOR is None:
        descriptor = build_descriptor(
            port=port,
            version=version,
            pid=os.getpid(),
            instance_id=os.environ.get("SLM_DAEMON_INSTANCE_ID") or None,
            capability=os.environ.get("SLM_DAEMON_CAPABILITY") or None,
            state=state,
        )
        os.environ["SLM_DAEMON_INSTANCE_ID"] = descriptor.instance_id
        os.environ["SLM_DAEMON_CAPABILITY"] = descriptor.capability
        _ACTIVE_DAEMON_DESCRIPTOR = descriptor
    elif _ACTIVE_DAEMON_DESCRIPTOR.state != state:
        _ACTIVE_DAEMON_DESCRIPTOR = replace(
            _ACTIVE_DAEMON_DESCRIPTOR,
            state=state,
            port=port,
            version=version,
        )
    return _ACTIVE_DAEMON_DESCRIPTOR


def _publish_process_descriptor(
    port: int, version: str, state: str,
) -> DaemonDescriptor:
    """Atomically publish identity plus one-release PID/port mirrors."""
    descriptor = _process_descriptor(port, version, state)
    write_descriptor(descriptor)
    pid_file = descriptor_path().with_name("daemon.pid")
    port_file = descriptor_path().with_name("daemon.port")
    pid_file.write_text(str(descriptor.pid))
    port_file.write_text(str(descriptor.port))
    return descriptor


def _cleanup_process_descriptor(descriptor: DaemonDescriptor | None) -> None:
    """Remove lifecycle state only when this process still owns the instance."""
    if descriptor is None or not clear_descriptor(descriptor.instance_id):
        return
    for path, expected in (
        (descriptor_path().with_name("daemon.pid"), str(descriptor.pid)),
        (descriptor_path().with_name("daemon.port"), str(descriptor.port)),
    ):
        try:
            if path.read_text().strip() == expected:
                path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    content: str
    tags: str = ""
    metadata: dict | None = None  # v3.4.26: pass-through from MCP pool_store
    idempotency_key: str | None = None
    session_id: str = ""
    # v3.6.15 multi-scope: visibility of the new memory. ``None`` scope means
    # "use the configured default_scope" (personal). shared_with is the list of
    # profile_ids for scope='shared'.
    scope: str | None = None
    shared_with: list[str] | None = None


class SessionOpenRequest(BaseModel):
    # #49: local session-open warm (no model roundtrip needed)
    project_path: str = ""
    query: str = ""
    max_results: int = 10


class SessionCloseRequest(BaseModel):
    # #49: local session-close (e.g. a Claude /quit hook). Empty session_id
    # closes the most recent real session.
    session_id: str = ""


class ObserveRequest(BaseModel):
    content: str


# ---------------------------------------------------------------------------
# V3.4.37: Engine recall adapter — routes QueueConsumer through the daemon's
# in-process MemoryEngine instead of spawning a recall_worker subprocess.
# Saves ~800 MB by eliminating the duplicate engine.
# ---------------------------------------------------------------------------

class EngineRecallAdapter:
    """Adapts MemoryEngine.recall() to RecallPoolProtocol for QueueConsumer.

    The daemon already has a full MemoryEngine in-process. The QueueConsumer
    previously routed through WorkerPool → recall_worker subprocess, which
    loaded a SECOND MemoryEngine. This adapter eliminates that duplication.
    """

    def __init__(self, engine) -> None:
        self._engine = engine

    def recall(self, query: str, limit: int = 10, session_id: str = "") -> dict:
        response = self._engine.recall(
            query, limit=limit, session_id=session_id or None,
        )
        memory_ids = list({
            r.fact.memory_id for r in response.results[:limit]
            if r.fact.memory_id
        })
        memory_map = (
            self._engine._db.get_memory_content_batch(memory_ids)
            if memory_ids else {}
        )
        # v3.6.6: same shared chokepoint as the HTTP route — identical output.
        from superlocalmemory.server.recall_serializer import (
            recall_response_metadata,
            serialize_recall_response,
        )
        _rc = getattr(self._engine._config, "retrieval", None)
        results, no_confident_match = serialize_recall_response(
            response,
            limit=limit,
            memory_map={k: _sanitize_json_text(v) for k, v in memory_map.items()},
            per_fact_max=getattr(_rc, "recall_per_fact_max_chars", 2400),
            total_max=getattr(_rc, "recall_total_max_chars", 12000),
        )
        for _r in results:
            _r["content"] = _sanitize_json_text(_r.get("content", ""))
        return {
            "ok": True,
            "query": query,
            "query_type": response.query_type,
            "result_count": len(results),
            "retrieval_time_ms": round(response.retrieval_time_ms, 1),
            "channel_weights": {
                k: round(v, 3)
                for k, v in (response.channel_weights or {}).items()
            },
            "total_candidates": getattr(response, "total_candidates", 0),
            "results": results,
            "no_confident_match": no_confident_match,
            **recall_response_metadata(response),
        }


# ---------------------------------------------------------------------------
# v3.4.32: Recall-priority gate for the pending materializer.
# All /remember writes go to pending.db and return fast; a background
# thread drains pending while yielding to any in-flight /search.
# See ``superlocalmemory.core.recall_gate``.
# ---------------------------------------------------------------------------

from superlocalmemory.core.recall_gate import (
    begin_recall as _begin_recall,
    end_recall as _end_recall,
    in_flight as _recalls_in_flight,
)

# v3.4.38: Module-level engine reference for the pending materializer.
# Set by the FastAPI lifespan after engine.initialize(). Was missing before,
# causing "name '_engine' is not defined" errors that blocked materialization
# of pending memories — they accumulated forever, only being processed at
# daemon startup via engine._process_pending_memories().
_engine = None


def _emit_event(
    event_type: str,
    payload: dict | None = None,
    *,
    source_agent: str = "http_client",
) -> None:
    """Emit a best-effort EventBus event from an HTTP write path.

    Mirrors mcp.shared.emit_event but tags source_protocol="http" so the
    dashboard can distinguish HTTP traffic from MCP tool calls. Never raises
    — a bus failure must not affect the caller's response.
    """
    try:
        from superlocalmemory.infra.event_bus import EventBus
        from superlocalmemory.server.routes.helpers import DB_PATH
        bus = EventBus.get_instance(DB_PATH)
        bus.emit(
            event_type,
            payload=payload,
            source_agent=source_agent,
            source_protocol="http",
        )
    except Exception as exc:
        logger.debug("EventBus emit failed (%s): %s", event_type, exc)


# v3.4.53: Limit concurrent full (non-fast) recalls. Without this, N parallel
# /recall calls spawn N × 6-channel threads → Ollama serialises, reranker
# lock queues, and total wall time is N × single-recall-time. 3 concurrent
# full recalls gives parallelism benefit without resource oversaturation.
import asyncio as _asyncio
_recall_semaphore = _asyncio.Semaphore(3)

# v3.4.52: Embedding model warm state. Set to True by the async pre-warm
# thread once Ollama has loaded the embedding model. /health reports this
# so MCP clients can wait for warm state before issuing recall calls.
_embedding_warm: bool = False


def _sanitize_json_text(text: str) -> str:
    """Strip control characters that break JSON serialization.

    Facts ingested from agent conversations can contain raw \\n, \\r, \\t,
    null bytes, and other ASCII control chars (0x00-0x1F) that survive
    database round-trips but cause ``json.JSONDecodeError: Invalid control
    character`` when FastAPI serialises the /recall response payload.

    We replace them with spaces rather than dropping them so the byte
    length is preserved and :300 truncation semantics stay predictable.
    Python's ``str.isprintable()`` is too aggressive (it also drops
    Unicode line separators), so we target only the ASCII control range.
    """
    if not text:
        return text
    # Fast path: most facts are clean JSON text. Check in C before allocating.
    if all(c >= " " or c in "\n\r\t" for c in text):
        return text
    return "".join(c if c >= " " or c in "\n\r\t" else " " for c in text)


# ---------------------------------------------------------------------------
# Observation debounce buffer (migrated from daemon.py)
# ---------------------------------------------------------------------------

class ObserveBuffer:
    """Durable observation admission with a short duplicate window.

    An accepted observation is submitted to M018 before ``enqueue`` returns.
    The timer clears only the in-memory duplicate set; it never owns evidence
    or delays persistence.
    """

    def __init__(self, debounce_sec: float = 3.0):
        self._debounce_sec = debounce_sec
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._engine = None

    def set_engine(self, engine) -> None:
        self._engine = engine

    def enqueue(self, content: str, *, trusted_actor_id: str = "") -> dict:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._lock:
            if content_hash in self._seen:
                return {"captured": False, "reason": "duplicate within debounce window"}
            self._seen.add(content_hash)
            window_size = len(self._seen)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._clear_seen)
            self._timer.daemon = True
            self._timer.start()
        _emit_event(
            "memory.observed",
            payload={
                "content_hash": content_hash,
                "content_preview": content[:120],
                "buffer_size": window_size,
            },
        )
        if self._engine is None:
            with self._lock:
                self._seen.discard(content_hash)
            return {
                "captured": False,
                "durable": False,
                "reason": "memory engine unavailable",
            }

        try:
            from superlocalmemory.hooks.auto_capture import AutoCapture
            from superlocalmemory.core.engine_ingestion import (
                build_engine_ingestion_command,
            )
            from superlocalmemory.core.ingestion_command import IngestionRequest

            decision = AutoCapture().evaluate(content)
            if not decision.capture:
                _emit_event(
                    "memory.dropped",
                    payload={
                        "reason": decision.reason,
                        "content_preview": content[:120],
                    },
                )
                return {
                    "captured": False,
                    "durable": False,
                    "reason": decision.reason,
                    "category": decision.category,
                    "confidence": round(decision.confidence, 3),
                }

            scope_config = getattr(self._engine._config, "scope", None)
            scope = getattr(scope_config, "default_scope", "personal")
            command = build_engine_ingestion_command(self._engine)
            receipt = command.submit(IngestionRequest(
                content=content,
                profile_id=self._engine._profile_id,
                source_type="http-observe",
                idempotency_key=f"observe:v1:{content_hash}",
                metadata={
                    "source": "auto-capture",
                    "category": decision.category,
                    "confidence": decision.confidence,
                },
                scope=scope,
                trusted_actor_id=trusted_actor_id or _materializer_actor_id(),
            ))
            _emit_event(
                "memory.captured",
                payload={
                    "operation_id": receipt.operation_id,
                    "category": decision.category,
                    "confidence": decision.confidence,
                    "content_preview": content[:120],
                },
            )
            return {
                "captured": True,
                "durable": True,
                "queued": receipt.state.value != "complete",
                "operation_id": receipt.operation_id,
                "fact_ids": list(receipt.fact_ids),
                "materialization_state": receipt.state.value,
                "category": decision.category,
                "confidence": round(decision.confidence, 3),
            }
        except Exception as exc:
            with self._lock:
                self._seen.discard(content_hash)
            logger.warning(
                "ObserveBuffer: durable admission failed for content %.40r: %s",
                content,
                exc,
            )
            _emit_event(
                "memory.dropped",
                payload={
                    "reason": "durable admission failed",
                    "content_preview": content[:120],
                },
            )
            return {
                "captured": False,
                "durable": False,
                "reason": "durable admission failed",
                "error": str(exc),
            }

    def _clear_seen(self) -> None:
        with self._lock:
            self._seen.clear()
            self._timer = None

    def _flush(self) -> None:
        """Compatibility alias: no evidence is buffered in V3.7."""
        self._clear_seen()

    def flush_sync(self) -> None:
        """Clear duplicate-window state for shutdown."""
        if self._timer is not None:
            self._timer.cancel()
        self._clear_seen()


_observe_buffer = ObserveBuffer(
    debounce_sec=float(os.environ.get("SLM_OBSERVE_DEBOUNCE_SEC", "3.0"))
)


# ---------------------------------------------------------------------------
# Idle watchdog (opt-in)
# ---------------------------------------------------------------------------

_last_activity = time.monotonic()


def _start_idle_watchdog(timeout_sec: int) -> None:
    """Auto-shutdown after idle. Only if timeout > 0."""
    if timeout_sec <= 0:
        return

    def _watch():
        while True:
            time.sleep(30)
            idle = time.monotonic() - _last_activity
            if idle > timeout_sec:
                logger.info("Daemon idle for %ds, shutting down", int(idle))
                os.kill(os.getpid(), signal.SIGTERM)
                break

    t = threading.Thread(target=_watch, daemon=True, name="idle-watchdog")
    t.start()


# ---------------------------------------------------------------------------
# Legacy port TCP redirect (backward compat for port 8767)
# ---------------------------------------------------------------------------

async def _start_legacy_redirect(primary_port: int, legacy_port: int) -> None:
    """Start TCP redirect from legacy_port → primary_port.

    Simple byte-level proxy. No shared event loop with uvicorn — runs
    in its own asyncio task within the same loop.
    """
    _deprecation_warned = False

    async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        nonlocal _deprecation_warned
        if not _deprecation_warned:
            logger.warning(
                "Request on deprecated port %d. Update config to use port %d.",
                legacy_port, primary_port,
            )
            _deprecation_warned = True

        try:
            upstream_r, upstream_w = await asyncio.open_connection("127.0.0.1", primary_port)
            await asyncio.gather(
                _pipe(reader, upstream_w),
                _pipe(upstream_r, writer),
            )
        except Exception:
            pass
        finally:
            writer.close()

    async def _pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
        try:
            while True:
                data = await src.read(8192)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception:
            pass
        finally:
            try:
                dst.close()
            except Exception:
                pass

    try:
        server = await asyncio.start_server(_handle_client, "127.0.0.1", legacy_port)
        logger.info("Legacy redirect: port %d → %d (deprecated)", legacy_port, primary_port)
        await server.serve_forever()
    except OSError:
        logger.info("Port %d in use (old daemon?), skipping legacy redirect", legacy_port)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize engine, workers, and optional services on startup."""
    global _last_activity

    engine = None
    config = None

    # Register the SSE bridge inside the application lifespan.  FastAPI's
    # legacy ``on_event`` hook is deprecated and, more importantly, made a
    # second startup mechanism compete with the daemon's existing lifespan.
    from superlocalmemory.server.routes.events import register_event_listener
    register_event_listener()

    # H-21 (Stage 8) — first-boot-after-upgrade notice. Compare the cached
    # version marker against the current package version; if they differ
    # (fresh install or upgrade), log a one-time banner with a link to the
    # CHANGELOG. Non-fatal; any filesystem error is swallowed.
    try:
        try:
            from importlib.metadata import version as _pkg_version
            _slm_version = _pkg_version("superlocalmemory")
        except Exception:
            _slm_version = "unknown"
        _version_marker = state_path(".last_version")
        _prev = None
        if _version_marker.exists():
            try:
                _prev = _version_marker.read_text(encoding="utf-8").strip()
            except OSError:
                _prev = None
        # S9-SKEP-15: the version marker is written AFTER the migration
        # block succeeds (see below). A failed migration must NOT cause
        # the next successful start to skip the upgrade banner — the
        # banner is the operator's cue that a new version just landed.
        _want_write_marker = _prev != _slm_version
        if _want_write_marker:
            if _prev is None:
                logger.info(
                    "[slm] first boot on v%s — run `slm status` to see your "
                    "memory overview. Changelog: "
                    "https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md",
                    _slm_version,
                )
            else:
                logger.info(
                    "[slm] upgraded %s → %s. Data migrations run in a moment; "
                    "your 18k+ atomic facts are preserved. Changelog: "
                    "https://github.com/qualixar/superlocalmemory/blob/main/CHANGELOG.md",
                    _prev, _slm_version,
                )
    except Exception as _exc:  # pragma: no cover — never block startup
        logger.debug("version-banner skipped: %s", _exc)
        _want_write_marker = False
        _version_marker = None
        _slm_version = None

    # LLD-06 §7.3 / LLD-07 §4.1 — run additive schema migrations BEFORE
    # engine init so later queries see the expected columns/tables.
    # Non-fatal: any failure here is logged and the daemon still starts.
    try:
        from superlocalmemory.storage.migration_runner import apply_all
        _home = canonical_data_root()
        _learning_db = _home / "learning.db"
        _memory_db = _home / "memory.db"
        _result = apply_all(_learning_db, _memory_db)
        _applied = _result.get("applied", [])
        _failed = _result.get("failed", [])
        if _applied:
            logger.info("migrations applied: %s", _applied)
        if _failed:
            logger.warning("migrations failed (non-fatal): %s", _failed)
        application.state.migration_result = _result
        # S9-SKEP-15: only commit the new `.last_version` AFTER migrations
        # complete with zero failures. A partial upgrade (schema didn't
        # land) must retain the old marker so the next successful start
        # still fires the upgrade banner — otherwise the operator loses
        # the one signal that tells them a version just changed.
        if (
            _want_write_marker
            and _version_marker is not None
            and _slm_version is not None
            and not _failed
        ):
            try:
                _version_marker.parent.mkdir(parents=True, exist_ok=True)
                _version_marker.write_text(_slm_version, encoding="utf-8")
            except OSError:
                pass  # non-fatal
    except Exception as _exc:
        logger.warning("migration runner crashed (non-fatal): %s", _exc)
        application.state.migration_result = {
            "applied": [], "skipped": [], "failed": ["_runner_crash"],
            "details": {"_crash": str(_exc)},
        }

    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine

        # v3.4.54: one-time migration config.json → 3-mode system
        SLMConfig.migrate_to_3mode()

        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()

        # Enforce WAL mode for concurrent reads
        db = getattr(engine, '_db', None) or getattr(engine, '_storage', None)
        if db and hasattr(db, 'execute'):
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                pass

        application.state.engine = engine
        application.state.config = config
        # v3.4.38: Wire module-level _engine for the pending materializer.
        global _engine
        _engine = engine
        logger.info("Unified daemon: MemoryEngine initialized (mode=%s)", config.mode.value)

        # v3.5.0: Backend Orchestrator — CozoDB (graph) + LanceDB (vector) backends.
        # Initialise AFTER engine so the retrieval channels exist to receive backends.
        # Migrates edges/embeddings automatically; fail-soft (non-blocking).
        _cozo_backend = None
        _lancedb_backend = None
        try:
            from superlocalmemory.core.backend_orchestrator import (
                BackendOrchestrator, set_orchestrator,
            )
            orch = BackendOrchestrator(config=config, db=engine._db)
            orch.on_daemon_start()
            set_orchestrator(orch)
            _cozo_backend = orch.get_graph_backend()
            _lancedb_backend = orch.get_vector_backend()
            # Inject CozoDB into entity_graph channel (already has the param).
            re = getattr(engine, '_retrieval_engine', None)
            if re is not None:
                eg = getattr(re, '_entity', None)
                if eg is not None and _cozo_backend is not None:
                    try:
                        eg._cozo = _cozo_backend
                        logger.info("CozoDB backend wired into entity_graph channel")
                    except Exception as exc:
                        logger.warning("CozoDB channel injection failed: %s", exc)
            logger.info("BackendOrchestrator: ready (cozo=%s, lancedb=%s)",
                         "active" if _cozo_backend else "off",
                         "active" if _lancedb_backend else "off")
        except Exception as exc:
            logger.warning("BackendOrchestrator init failed (non-fatal): %s", exc)

        # LLD-07 §4 — deferred migrations (e.g. M006 reward column) need to
        # run AFTER MemoryEngine.initialize() has bootstrapped runtime tables
        # like action_outcomes. Non-fatal by contract.
        try:
            from superlocalmemory.storage.migration_runner import apply_deferred
            _deferred = apply_deferred(_learning_db, _memory_db)
            _d_applied = _deferred.get("applied", [])
            _d_failed = _deferred.get("failed", [])
            if _d_applied:
                logger.info("deferred migrations applied: %s", _d_applied)
            if _d_failed:
                logger.warning(
                    "deferred migrations failed (non-fatal, trainer falls "
                    "back to position proxy): %s", _d_failed,
                )
            # Merge into the migration result already on app state so the
            # dashboard sees one consolidated picture.
            _mr = getattr(application.state, "migration_result", None) or {
                "applied": [], "skipped": [], "failed": [], "details": {},
            }
            _mr.setdefault("applied", []).extend(_d_applied)
            _mr.setdefault("skipped", []).extend(_deferred.get("skipped", []))
            _mr.setdefault("failed", []).extend(_d_failed)
            _mr.setdefault("details", {}).update(_deferred.get("details", {}))
            application.state.migration_result = _mr
        except Exception as _dexc:  # pragma: no cover — defensive
            logger.warning(
                "deferred migration runner crashed (non-fatal): %s", _dexc,
            )

        # S9-DASH-02: start the outcome-queue worker so recall →
        # pending_outcomes is actually produced. Before v3.4.22 this
        # producer had zero callers and the closed-loop pipeline was
        # dark. Worker drains at 250 ms cadence; one SQLite INSERT per
        # event via EngagementRewardModel.record_recall.
        try:
            from superlocalmemory.learning.outcome_queue import start_worker
            start_worker(_memory_db)
        except Exception as _oqexc:  # pragma: no cover — defensive
            logger.debug("outcome_queue start failed (non-fatal): %s", _oqexc)

        # Set up observe buffer
        _observe_buffer.set_engine(engine)

        # v3.4.52: Ensure covering indexes for SpreadingActivation queries.
        # SQLite 3.45+ streaming merge (UNION ALL + ORDER BY + LIMIT) uses
        # these to seek directly to top-K rows per subquery, avoiding a
        # full sort.  Without them full 6-channel recall takes 7-10s on
        # >1M edges (the SpreadingActivation 4-UNION query disk-sorts every
        # node's neighbor list on each call).  With them: sub-second.
        try:
            import sqlite3 as _sqlite3
            _idx_conn = _sqlite3.connect(str(_memory_db))
            _idx_conn.execute("PRAGMA journal_mode=WAL")
            _idx_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source_weight "
                "ON graph_edges(profile_id, source_id, weight DESC)"
            )
            _idx_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target_weight "
                "ON graph_edges(profile_id, target_id, weight DESC)"
            )
            _idx_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assoc_source_weight "
                "ON association_edges(profile_id, source_fact_id, weight DESC)"
            )
            _idx_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assoc_target_weight "
                "ON association_edges(profile_id, target_fact_id, weight DESC)"
            )
            _idx_conn.close()
        except Exception as _idx_exc:
            logger.debug("SpreadingActivation covering indexes skipped: %s", _idx_exc)

        # V3.4.37: Removed WorkerPool.warmup() — the recall_worker subprocess
        # duplicated the daemon's MemoryEngine (800+ MB). QueueConsumer now
        # uses the daemon's engine directly via EngineRecallAdapter.
        # WorkerPool is still available as fallback for dashboard/chat routes.

        # The reranker constructor has already started its background warmup.
        # Never block daemon publication here: a first-time model download or
        # ONNX compilation previously held every CLI/MCP request for 120s.
        # Until it is ready, retrieval uses its deterministic fallback scorer;
        # the worker upgrades subsequent recalls without changing their API.
        retrieval_eng = getattr(engine, '_retrieval_engine', None)

        # V3.4.11: Pre-warm embedding worker (load ONNX model on startup)
        # Without this, first recall takes 60-90s for model load.
        # Same pattern as reranker warmup above.
        # v3.4.52: Sets module-level _embedding_warm flag so /health can
        # report readiness. Combined with keep_alive=-1 in ollama_embedder.py
        # this keeps the embedding model resident forever after first warm-up.
        import threading
        global _embedding_warm
        _embedding_warm = False
        def _warmup_embedder():
            global _embedding_warm
            try:
                embedder = getattr(retrieval_eng, '_embedder', None) if retrieval_eng else None
                if embedder and hasattr(embedder, 'embed'):
                    embedder.embed("warmup")
                    _embedding_warm = True
                    logger.info("Embedding worker pre-warmed (model resident, keep_alive=-1)")
            except Exception as exc:
                logger.warning("Embedding warmup failed: %s", exc)

        def _warmup_recall():
            """v3.4.62: Fire a full 6-channel recall after embedding warms up.

            Loads the graph_edges table (347K rows, ~100 MB) into the SQLite
            page cache. Without this, the first user query takes 15-24s because
            it reads graph_edges from disk. After this warmup completes, all
            subsequent queries hit the warm page cache at <2s.

            Runs after embedding warm (embed first so recall can use it).
            Named 'recall-warmup' so it appears clearly in thread dumps.
            """
            import time as _t
            for _ in range(60):
                if _embedding_warm:
                    break
                _t.sleep(0.5)
            try:
                t0 = _t.monotonic()
                # Fire 2 warmup queries: one to load the graph page cache,
                # second to warm the reranker subprocess + all 6 channels.
                # Without this, dashboard POST /api/search hits 11s cold.
                for wq in ("memory recall performance", "context injection retrieval"):
                    engine.recall(wq, limit=5)
                elapsed = round((_t.monotonic() - t0) * 1000)
                logger.info(
                    "Recall engine pre-warmed in %dms", elapsed,
                )
            except Exception as exc:
                logger.warning("Recall warmup failed (non-fatal): %s", exc)

        def _backfill_vector_store():
            """v3.5.0: index facts whose embeddings exist in atomic_facts but
            are missing from the sqlite-vec store. Facts stored before dual-write
            was complete were never indexed, so semantic + hopfield only saw a
            fraction of the corpus (observed: 5.8k indexed of 17.2k embedded).
            Idempotent (skips when the store is already complete), non-blocking,
            fail-soft. Runs once after a 3.5.0 upgrade, then no-ops every restart.
            """
            import time as _t
            from pathlib import Path as _P
            for _ in range(120):
                if _embedding_warm:
                    break
                _t.sleep(0.5)
            try:
                from superlocalmemory.retrieval.vector_store import (
                    VectorStore, VectorStoreConfig,
                )
                db = engine._db
                db_path = getattr(db, "db_path", None) or getattr(db, "_db_path", None)
                if db_path is None:
                    return
                dim = getattr(getattr(config, "embedding", None), "dimension", 768) or 768
                vs = VectorStore(_P(db_path), VectorStoreConfig(dimension=dim))
                if not vs.available:
                    return
                try:
                    profiles = list(db.list_profiles()) or ["default"]
                except Exception:
                    profiles = ["default"]
                for pid in profiles:
                    facts = db.get_all_facts(pid)
                    with_emb = [
                        (f.fact_id, getattr(f, "profile_id", pid) or pid, f.embedding)
                        for f in facts
                        if getattr(f, "embedding", None) and len(f.embedding) == dim
                    ]
                    if not with_emb:
                        continue
                    if vs.count(pid) >= int(len(with_emb) * 0.98):
                        continue  # already complete — no-op
                    n = vs.rebuild_from_facts(with_emb)
                    logger.info(
                        "VS backfill[%s]: indexed %d of %d embedded facts",
                        pid, n, len(with_emb),
                    )
            except Exception as exc:
                logger.warning("Vector store backfill failed (non-fatal): %s", exc)

        threading.Thread(target=_warmup_embedder, daemon=True, name="embed-warmup").start()
        threading.Thread(target=_warmup_recall, daemon=True, name="recall-warmup").start()
        threading.Thread(target=_backfill_vector_store, daemon=True, name="vs-backfill").start()

        # v3.6.8: Runtime recall-health monitor. The three warmups above run
        # ONCE at boot; on a long-running daemon the graph page cache gets
        # evicted and the embedder can start returning None while _embedding_warm
        # still claims True — both silently degrade recall to keyword-only BM25
        # (observed: session_init "pool.recall timed out → DEGRADED MODE", 7×/2d).
        # This monitor re-warms + actively probes the semantic channel + self-heals
        # a dead embedder for the daemon's whole life. See server/recall_health.py.
        try:
            from superlocalmemory.server.recall_health import (
                start_recall_health_monitor,
            )
            _rh_thread, _rh_stop, _ = start_recall_health_monitor(engine)
            application.state.recall_health_stop = _rh_stop
        except Exception as _rh_exc:
            logger.warning(
                "recall-health monitor start failed (non-fatal): %s", _rh_exc,
            )
            application.state.recall_health_stop = None

        # v3.4.37: QueueConsumer uses daemon's engine directly via adapter.
        # Previously routed through WorkerPool → recall_worker subprocess,
        # which loaded a duplicate MemoryEngine (~800 MB waste).
        try:
            from superlocalmemory.core.queue_consumer import QueueConsumer
            from superlocalmemory.core.recall_queue import RecallQueue
            _queue_db = state_path("recall_queue.db")
            _recall_queue = RecallQueue(_queue_db)
            _queue_consumer = QueueConsumer(
                queue=_recall_queue,
                pool=EngineRecallAdapter(engine),
            )
            _queue_consumer.start()
            application.state.queue_consumer = _queue_consumer
            application.state.recall_queue = _recall_queue
            logger.info("QueueConsumer started (recall_queue.db)")

            # v3.4.36: Start persistent hook daemon (Unix socket server).
            # Eliminates Python subprocess startup for each recall hook call.
            try:
                from superlocalmemory.hooks.hook_daemon import HookDaemon
                _hook_daemon = HookDaemon(queue_db_path=_queue_db)
                _hook_daemon.start()
                application.state.hook_daemon = _hook_daemon
            except Exception as _hd_exc:
                logger.warning("HookDaemon start failed (non-fatal): %s", _hd_exc)
                application.state.hook_daemon = None
        except Exception as _qc_exc:
            logger.warning("QueueConsumer start failed (non-fatal): %s", _qc_exc)
            application.state.queue_consumer = None
            application.state.recall_queue = None

    except Exception:
        logger.exception("Engine init failed")  # auto-includes traceback
        application.state.engine = None
        application.state.config = None

    application.state.observe_buffer = _observe_buffer

    # Phase B: Start health monitor
    try:
        from superlocalmemory.core.health_monitor import HealthMonitor
        health_config = getattr(config, 'health', None)
        # v3.6.9 BUG-A: env override + RAM-scaled default (HealthMonitor computes
        # 40% of physical RAM when budget=0). SLM_RSS_BUDGET_MB takes priority.
        _env_budget = int(os.environ.get("SLM_RSS_BUDGET_MB", "0") or 0)
        _cfg_budget = getattr(health_config, 'global_rss_budget_mb', 0) if health_config else 0
        monitor = HealthMonitor(
            global_rss_budget_mb=_env_budget or _cfg_budget or 0,
            heartbeat_timeout_sec=getattr(health_config, 'heartbeat_timeout_sec', 60) if health_config else 60,
            check_interval_sec=getattr(health_config, 'health_check_interval_sec', 15) if health_config else 15,
            enable_structured_logging=getattr(health_config, 'enable_structured_logging', True) if health_config else True,
        )
        monitor.start()
        application.state.health_monitor = monitor
    except Exception as exc:
        logger.debug("Health monitor init: %s", exc)
        application.state.health_monitor = None

    # Phase C: Start mesh broker
    try:
        mesh_enabled = getattr(config, 'mesh_enabled', True) if config else True
        if mesh_enabled:
            from superlocalmemory.mesh.broker import MeshBroker
            db_path = config.db_path if config else state_path("memory.db")
            mesh_broker = MeshBroker(str(db_path))
            mesh_broker.start_cleanup()
            application.state.mesh_broker = mesh_broker
            logger.info("Mesh broker started")
        else:
            application.state.mesh_broker = None
    except Exception as exc:
        logger.debug("Mesh broker init: %s", exc)
        application.state.mesh_broker = None

    # Start idle watchdog if configured
    idle_timeout = int(os.environ.get("SLM_DAEMON_IDLE_TIMEOUT", "0"))
    if config and hasattr(config, 'daemon_idle_timeout'):
        idle_timeout = idle_timeout or config.daemon_idle_timeout
    _start_idle_watchdog(idle_timeout)

    # Start legacy port redirect
    enable_legacy = os.environ.get("SLM_DISABLE_LEGACY_PORT", "").lower() not in ("1", "true")
    if enable_legacy:
        identity = application.state.daemon_descriptor
        asyncio.create_task(_start_legacy_redirect(identity.port, _LEGACY_PORT))

    # V3.4.22 LLD-02: signal-worker background drainer (S8-SK-01 fix).
    # Without this, ``signals.enqueue`` fills a bounded queue and drops
    # silently after ~250 recalls — learning_signals never populates,
    # Phase 3 never activates, the whole Living Brain stays cold.
    if os.environ.get("SLM_SIGNALS_ENABLED", "1") != "0":
        try:
            from superlocalmemory.learning import signal_worker as _sw
            _learning_db = state_path("learning.db")
            _sw.start(_learning_db)
            application.state.signal_worker_started = True
            logger.info("signal_worker started on %s", _learning_db)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("signal_worker failed to start: %s", exc)
            application.state.signal_worker_started = False

    # V3.4.22 LLD-05: cross-platform adapter sync loop
    if os.environ.get("SLM_CROSS_PLATFORM_SYNC_DISABLED", "").lower() not in ("1", "true"):
        try:
            from superlocalmemory.cli.context_commands import build_default_adapters
            from superlocalmemory.hooks.sync_loop import schedule as _schedule_sync
            _schedule_sync(build_default_adapters())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("cross-platform sync loop failed to start: %s", exc)

    # V3.4.22 LLD-03: bandit reward proxy settler + retention sweep loops
    if os.environ.get("SLM_BANDIT_DISABLED", "0") != "1":
        try:
            from superlocalmemory.server.bandit_loops import (
                schedule_bandit_loops,
            )
            schedule_bandit_loops(application, config)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("bandit loops failed to start: %s", exc)

    global _start_time
    _start_time = time.monotonic()
    _last_activity = time.monotonic()
    # v3.4.23: pre-format the ready message. Previous code passed a ternary as
    # the log format string with a fixed 2-arg tuple; when idle_timeout<=0 the
    # chosen branch had only one %d, triggering a TypeError on every startup.
    # Python's logging module then wrote the full stack to stderr. Because the
    # call runs inside FastAPI's stacked merged_lifespan, each dump was ~30 KB
    # and the error log grew to tens of MB within a day.
    _display_port = _configured_daemon_port()
    if idle_timeout <= 0:
        _ready_msg = f"Unified daemon ready on port {_display_port} (24/7 mode)"
    else:
        _ready_msg = (
            f"Unified daemon ready on port {_display_port} "
            f"(idle timeout: {idle_timeout}s)"
        )
    logger.info(_ready_msg)

    # Start optimize proxy httpx client if mounted
    try:
        _opt_proxy = getattr(application.state, "optimize_proxy", None)
        if _opt_proxy is not None:
            await _opt_proxy.startup()
    except Exception as _exc:  # pragma: no cover — defensive
        logger.warning("optimize_proxy startup failed (non-fatal): %s", _exc)

    # V3.6: Mount optimize API routes + restore persisted metrics + start flush loop
    try:
        from superlocalmemory.server.routes.optimize import router as optimize_router
        from superlocalmemory.optimize.metrics.counters import MetricsCollector
        from superlocalmemory.optimize.metrics.persistence import MetricsPersistence
        from superlocalmemory.optimize.storage.db import CacheDB
        application.include_router(optimize_router)

        # Restore persisted metrics counters on startup.
        # #48 fix: build ONE CacheDB + MetricsPersistence and reuse them for the
        # life of the daemon. The old code constructed a new CacheDB() on every
        # 60s flush, which re-ran schema init ("Schema initialized" log spam),
        # the corruption check, and AES-key derivation on every tick.
        _metrics_db = CacheDB()
        _metrics_persistence = MetricsPersistence()
        _metrics_persistence.load(MetricsCollector.get_instance(), _metrics_db)

        # Periodic flush — every 60s (OPT-005: guard + task ref for shutdown)
        _metrics_flush_task = getattr(application.state, "_optimize_flush_task", None)
        if _metrics_flush_task is None or _metrics_flush_task.done():
            async def _metrics_flush_loop():
                while True:
                    await asyncio.sleep(60)
                    try:
                        _metrics_persistence.flush(
                            MetricsCollector.get_instance(), _metrics_db
                        )
                    except Exception as e:
                        logger.warning("metrics flush error: %s", e)
            application.state._optimize_flush_task = asyncio.create_task(
                _metrics_flush_loop()
            )
    except Exception as e:
        logger.warning("optimize module not available: %s", e)

    # v3.6.7: Start MCP Streamable-HTTP session manager (GOTCHA #1).
    # streamable_http_app() carries its own Starlette lifespan that initialises
    # an anyio task group inside the session manager. Without entering that
    # lifespan every POST /mcp 500s with "Task group is not initialized."
    # AsyncExitStack enters the context only when _mcp_app was mounted; if the
    # mount failed (non-fatal) the daemon starts normally without HTTP MCP.
    # v3.6.9 (#34): wrap the MCP lifespan in shield so an unhandled exception
    # or tool-level cancellation inside a session manager task group cannot
    # propagate out and trigger uvicorn's graceful-shutdown handler.
    async with AsyncExitStack() as _mcp_stack:
        if _mcp_app is not None:
            try:
                await _mcp_stack.enter_async_context(
                    _mcp_app.router.lifespan_context(_mcp_app)
                )
                logger.info("MCP HTTP session manager started (Streamable-HTTP on /mcp)")
            except Exception as _mcp_lifespan_exc:
                logger.warning(
                    "MCP HTTP session manager failed to start (non-fatal, stdio still works): %s",
                    _mcp_lifespan_exc,
                )

        yield

    # Cancel optimize metrics flush loop + run final flush before shutdown
    try:
        _flush_task = getattr(application.state, "_optimize_flush_task", None)
        if _flush_task is not None and not _flush_task.done():
            _flush_task.cancel()
            try:
                await _flush_task
            except asyncio.CancelledError:
                pass
        # Final flush to persist the last window (H-04: use singleton)
        try:
            from superlocalmemory.optimize.metrics.persistence import MetricsPersistence
            from superlocalmemory.optimize.metrics.counters import MetricsCollector
            from superlocalmemory.optimize.storage.db import CacheDB as _FinalCacheDB
            MetricsPersistence().flush(
                MetricsCollector.get_instance(),
                _FinalCacheDB.get_default(),
            )
        except Exception:
            pass
    except Exception:  # pragma: no cover — defensive
        pass

    # Shutdown optimize proxy httpx client (symmetric with startup)
    try:
        _opt_proxy = getattr(application.state, "optimize_proxy", None)
        if _opt_proxy is not None:
            await _opt_proxy.shutdown()
    except Exception as _exc:  # pragma: no cover — defensive
        logger.warning("optimize_proxy shutdown failed (non-fatal): %s", _exc)

    # S9-W4 C2: symmetric shutdown. Prior version only flushed the
    # observe-buffer + signal_worker + engine. The following long-lived
    # subsystems lived on ``application.state`` but were never
    # explicitly cancelled / joined, so uvicorn's
    # ``timeout_graceful_shutdown=10`` silently killed live threads
    # mid-commit: HealthMonitor probes, MeshBroker cleanup thread,
    # bandit settler asyncio tasks, and the process-wide cost-log
    # connection cache. A WAL commit interrupted mid-flight could
    # leave ``evolution_llm_cost_log`` with torn rows.
    #
    # New policy: every subsystem that stored a handle on
    # ``application.state`` MUST be stopped here, in reverse start
    # order. Each stop is wrapped in try/except so one failure does
    # not skip the rest.
    _observe_buffer.flush_sync()

    # S9-DASH-02: stop outcome-queue worker (final drain on graceful
    # shutdown). Any events left unpersisted are logged but not
    # replayed — signal capture is not load-bearing on correctness.
    try:
        from superlocalmemory.learning.outcome_queue import stop_worker
        _oq_remaining = stop_worker(timeout_s=2.0)
        if _oq_remaining:
            logger.info(
                "outcome_queue shutdown: %d events dropped on flush",
                _oq_remaining,
            )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("outcome_queue stop failed: %s", exc)

    # Cancel bandit asyncio tasks (LLD-03). ``bandit_loops`` stashes
    # them at ``application.state.bandit_tasks``; if the attr is
    # missing we skip.
    _bandit_tasks = getattr(application.state, "bandit_tasks", None)
    if _bandit_tasks:
        try:
            for _t in _bandit_tasks:
                try:
                    _t.cancel()
                except Exception:  # pragma: no cover
                    pass
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("bandit_tasks cancel failed: %s", exc)

    # v3.4.36: Stop HookDaemon (Unix socket server).
    _hd = getattr(application.state, "hook_daemon", None)
    if _hd is not None:
        try:
            _hd.stop()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("hook_daemon stop failed: %s", exc)

    # v3.4.26: Stop QueueConsumer (recall_queue.db drainer).
    _qc = getattr(application.state, "queue_consumer", None)
    if _qc is not None:
        try:
            _qc.stop()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("queue_consumer stop failed: %s", exc)
    _rq = getattr(application.state, "recall_queue", None)
    if _rq is not None:
        try:
            _rq.close()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("recall_queue close failed: %s", exc)

    # v3.6.8: Stop recall-health monitor (owns a daemon thread).
    _rh_stop = getattr(application.state, "recall_health_stop", None)
    if _rh_stop is not None:
        try:
            _rh_stop.set()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("recall_health monitor stop failed: %s", exc)

    # Stop HealthMonitor (health_monitor.py owns a daemon thread).
    _health = getattr(application.state, "health_monitor", None)
    if _health is not None:
        try:
            stop_fn = getattr(_health, "stop", None)
            if callable(stop_fn):
                stop_fn()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("health_monitor stop failed: %s", exc)

    # Stop MeshBroker cleanup thread.
    _mesh = getattr(application.state, "mesh_broker", None)
    if _mesh is not None:
        try:
            stop_fn = getattr(_mesh, "stop_cleanup", None)
            if callable(stop_fn):
                stop_fn()
            else:  # pragma: no cover — older broker versions
                stop_fn = getattr(_mesh, "stop", None)
                if callable(stop_fn):
                    stop_fn()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("mesh_broker stop failed: %s", exc)

    # LLD-02 SW3: flush pending signals to DB before closing. Bounded 3 s
    # to keep daemon shutdown snappy; drops + counts anything unwritten.
    if getattr(application.state, "signal_worker_started", False):
        try:
            from superlocalmemory.learning import signal_worker as _sw
            _sw.stop(timeout=3.0)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("signal_worker shutdown flush failed: %s", exc)

    # Close the process-wide evolution cost-log connection cache
    # BEFORE engine.close so fsyncs land under our own control, not
    # under uvicorn's SIGTERM timeout. ``_close_cost_conns`` is
    # idempotent — the atexit hook is still registered but won't
    # re-close since the cache is cleared.
    try:
        from superlocalmemory.evolution import llm_dispatch as _ld
        _ld._close_cost_conns()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("evolution cost-conn cache close failed: %s", exc)

    # Drop the trigram cache conn symmetrically.
    try:
        from superlocalmemory.learning import trigram_index as _ti
        _ti._reset_cache_conn()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("trigram cache conn close failed: %s", exc)

    # Flush the perf-log fd explicitly (the atexit hook still fires
    # but explicit close here is cheap insurance against uvicorn
    # killing the process before atexit runs).
    try:
        from superlocalmemory.hooks._outcome_common import _perf_log_flush
        _perf_log_flush()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("perf_log flush failed: %s", exc)

    if engine is not None:
        try:
            engine.close()
        except Exception:
            pass
    _cleanup_process_descriptor(
        getattr(application.state, "daemon_descriptor", None),
    )
    logger.info("Unified daemon shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _configure_mcp_transport_settings(fastmcp) -> bool:
    """Apply the current transport mode without leaking singleton state.

    ``superlocalmemory.mcp.server.server`` is process-global.  App factories
    are invoked more than once by tests and embedded hosts, so both flags must
    be assigned on every call; an earlier stateless app must not silently turn
    a later default app stateless.  Keeping this small policy separate also
    lets tests exercise the wiring without reloading FastMCP and rebuilding
    hundreds of Pydantic models in a native-heavy Python process.
    """
    from superlocalmemory.core.remote_mode import mcp_stateless

    stateless = bool(mcp_stateless())
    fastmcp.settings.stateless_http = stateless
    fastmcp.settings.json_response = stateless
    return stateless


def create_app() -> FastAPI:
    """Create the unified FastAPI application."""
    from superlocalmemory.server.routes.helpers import SLM_VERSION

    application = FastAPI(
        title="SuperLocalMemory V3 — Unified Daemon",
        description="Memory + Dashboard + Mesh — one process, one engine.",
        version=SLM_VERSION,
        lifespan=lifespan,
    )
    identity_port = _configured_daemon_port()
    application.state.daemon_descriptor = _process_descriptor(
        identity_port, SLM_VERSION, "ready",
    )

    # -- Middleware --
    from superlocalmemory.server.security_middleware import SecurityHeadersMiddleware
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(GZipMiddleware, minimum_size=1000)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8765", "http://127.0.0.1:8765",
            "http://localhost:8767", "http://127.0.0.1:8767",  # legacy compat
            "http://localhost:8417", "http://127.0.0.1:8417",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Content-Type", "Authorization", "X-SLM-API-Key",
            "X-SLM-Daemon-Capability", "X-SLM-Target-Instance",
        ],
    )

    # -- Register all dashboard routes (from existing api.py) --
    _register_dashboard_routes(application)

    # -- Mesh routes (Phase C) --
    try:
        from superlocalmemory.server.routes.mesh import router as mesh_router
        application.include_router(mesh_router)
    except ImportError:
        pass

    # -- Entity routes (Phase D) --
    try:
        from superlocalmemory.server.routes.entity import router as entity_router
        application.include_router(entity_router)
    except ImportError:
        pass

    # -- Ingestion route (Phase E) --
    try:
        from superlocalmemory.server.routes.ingest import router as ingest_router
        application.include_router(ingest_router)
    except ImportError:
        pass

    # -- Brain route (LLD-04 v2: /api/v3/brain + deprecated shims) --
    try:
        from superlocalmemory.server.routes.brain import (
            router as brain_router,
        )
        from superlocalmemory.server.middleware.security_headers import (
            SecurityHeadersMiddleware as StrictSecurityHeadersMiddleware,
        )
        application.include_router(brain_router)
        # Strict CSP / XFO / XCTO / Referrer-Policy — applies to every
        # response including the Brain route. Added as the outermost
        # middleware so it overrides the legacy security_middleware's
        # looser CSP on requests that pass through this strict wall.
        application.add_middleware(StrictSecurityHeadersMiddleware)
    except ImportError as exc:  # pragma: no cover — defensive wiring
        logger.warning("brain router not wired: %s", exc)

    # -- Prewarm route (LLD-01 §4.4 — S8-SK-02 fix) --
    # POST /internal/prewarm populates active_brain_cache after every
    # tool_use. Without this handler, the async hook POSTs to a 404 and
    # the cache never gets populated, which made every UserPromptSubmit
    # a structural miss. All 4 auth gates applied inside the route.
    try:
        from superlocalmemory.server.routes.prewarm import (
            router as prewarm_router,
        )
        application.include_router(prewarm_router)
    except ImportError as exc:  # pragma: no cover — defensive wiring
        logger.warning("prewarm router not wired: %s", exc)

    # -- Token route — auto-inject install token into the local dashboard --
    # GET /internal/token returns the install token to loopback+origin-
    # scoped browser callers so brain.js (and any future token-gated
    # dashboard fetch) can include X-Install-Token without ever asking
    # the non-technical user to paste it. Non-browser clients (MCP, CLI,
    # IDE adapters) keep reading ~/.superlocalmemory/.install_token
    # directly and sending the header themselves.
    try:
        from superlocalmemory.server.routes.token import (
            router as token_router,
        )
        application.include_router(token_router)
    except ImportError as exc:  # pragma: no cover — defensive wiring
        logger.warning("token router not wired: %s", exc)

    # ── Optimize proxy (optional, fail-open) ──────────────────────────────
    # Mounts on the existing daemon port 8765. Proxy routes carry provider
    # API keys (x-api-key, Authorization), NOT the SLM API key. Auth-exempt
    # path prefixes are configured below in the auth_middleware block.
    try:
        from superlocalmemory.optimize.config import _set_config_store, get_shared_store
        from superlocalmemory.optimize.proxy.server import ProxyApp, build_proxy_router

        # ONE shared ConfigStore for daemon + routes + watchdog + proxy reload
        # (fixes W-05 fresh-store-per-request; powers runtime hot-reload).
        _opt_store = get_shared_store()
        _set_config_store(_opt_store)
        _opt_cfg = _opt_store.get()
        # W-03 fix: the proxy path is gated by proxy_enabled ALONE. The master
        # `enabled` gates only the SDK adapter, never the proxy mount.
        if _opt_cfg.proxy_enabled:
            _proxy = ProxyApp(config=_opt_cfg)
            application.state.optimize_proxy = _proxy
            _proxy_router = build_proxy_router(_proxy)
            # prefix="" — proxy claims /v1/*, /v1beta/* directly.
            application.include_router(_proxy_router, prefix="")
            # v3.6.10: runtime hot-reload — rebuild the proxy HookChain whenever
            # optimize.json changes so cache_enabled / compress_enabled can be
            # toggled INDEPENDENTLY from the UI with no restart. UI save fires the
            # callback immediately; external edits are caught by the 2s watchdog.
            _opt_store.register_change_callback(_proxy.reload_from_config)
            _opt_store.start_watchdog()
            logger.info(
                "optimize.proxy mounted on /v1/*, /v1beta/*  port=8765 "
                "(runtime cache/compress hot-reload enabled)"
            )
        else:
            application.state.optimize_proxy = None
    except ImportError:
        application.state.optimize_proxy = None
        logger.debug("optimize.proxy not installed — skipping")
    except Exception as _exc:  # pragma: no cover — defensive
        application.state.optimize_proxy = None
        logger.warning("optimize.proxy mount failed (non-fatal): %s", _exc)

    # -- Daemon-specific routes --
    _register_daemon_routes(application)

    # -- v3.6.7: MCP Streamable-HTTP transport at /mcp --
    # Mount the FastMCP server as a Starlette ASGI sub-app so ALL clients
    # (Claude Code sessions, subagents, desktop, hermes) share ONE daemon
    # process instead of spawning an `slm mcp` subprocess per connection.
    # The session manager lifespan is started in lifespan() via AsyncExitStack.
    # Fail-open: if import or mount fails, stdio transport keeps working.
    #
    # streamable_http_path is set to "/" so that when mounted at "/mcp" the
    # effective user-facing endpoint is exactly http://127.0.0.1:8765/mcp.
    # (FastAPI strips the mount prefix before passing the request to the
    # sub-app, so the sub-app's internal route must be "/".)
    try:
        from superlocalmemory.mcp.server import server as _mcp_fastmcp
        _mcp_fastmcp.settings.streamable_http_path = "/"
        _mcp_fastmcp._session_manager = None  # Defensive reset for idempotency
        # v3.6.9 (#36): configure DNS-rebinding protection from env.
        # Default: localhost-only (safe). Set SLM_MCP_ALLOWED_HOSTS=192.168.x.y:*
        # (comma-separated, e.g. "192.168.50.144:*,slm.lan:*") to open to a LAN.
        # Use "*" to disable protection entirely (trusted private network only).
        # TransportSecuritySettings imported lazily here so that MCP mount
        # works on older SDK versions when SLM_MCP_ALLOWED_HOSTS is not set.
        _mcp_allowed = os.environ.get("SLM_MCP_ALLOWED_HOSTS", "").strip()
        if _mcp_allowed:
            from mcp.server.transport_security import TransportSecuritySettings
            if _mcp_allowed == "*":
                _mcp_fastmcp.settings.transport_security = TransportSecuritySettings(
                    enable_dns_rebinding_protection=False,
                )
            else:
                _hosts = [h.strip() for h in _mcp_allowed.split(",") if h.strip()]
                _mcp_fastmcp.settings.transport_security = TransportSecuritySettings(
                    enable_dns_rebinding_protection=True,
                    allowed_hosts=_hosts,
                    allowed_origins=[f"http://{h}" for h in _hosts],
                )
            logger.info("MCP transport security: allowed_hosts=%r", _mcp_allowed)
        # v3.6.12 (issue #39): stateless MCP transport for distributed/gateway
        # deployments. SLM's Streamable-HTTP is stateful by default — every call
        # must replay the Mcp-Session-Id from the initialize handshake. A gateway
        # (MCP Hub, LAN forwarder) that doesn't replay it gets "-32600 Session
        # not found" (the mesh-tools symptom in #39). Stateless mode treats each
        # request independently so any forwarder works. Default OFF (loopback
        # clients keep full stateful sessions); enabled by SLM_REMOTE=1 or
        # SLM_MCP_STATELESS=1. Per-agent /mcp/{agent_id} routing is unaffected
        # (path-based, not session-based).
        from superlocalmemory.core.remote_mode import is_remote_mode
        if _configure_mcp_transport_settings(_mcp_fastmcp):
            if is_remote_mode():
                logger.warning(
                    "MCP transport: STATELESS mode ON (SLM_REMOTE) — LAN "
                    "gateways/hubs may forward tool calls without a session id. "
                    "Per-session isolation is relaxed; intended for trusted networks."
                )
            else:
                logger.warning(
                    "MCP transport: STATELESS mode ON (SLM_MCP_STATELESS alone) "
                    "— session isolation relaxed for LOOPBACK clients. Intended "
                    "for a local gateway/hub (e.g. MCP Hub) on 127.0.0.1 only; "
                    "the token endpoint stays loopback-only without SLM_REMOTE."
                )
        global _mcp_app
        _mcp_app = _mcp_fastmcp.streamable_http_app()

        # v3.6.10: per-agent-ID routing — /mcp/{agent_id} extracts the agent
        # identity from the URL path and places it in a ContextVar so all MCP
        # tools (remember, recall, etc.) automatically use the correct namespace.
        # AgentIDExtractorASGI lives in mcp/agent_context so it is unit-testable
        # (tests/test_mcp/test_agent_context.py) rather than buried inline here.
        from superlocalmemory.mcp.agent_context import AgentIDExtractorASGI

        application.mount("/mcp", AgentIDExtractorASGI(_mcp_app))
        logger.info(
            "MCP HTTP transport mounted at /mcp (Streamable HTTP, port %d; "
            "per-agent routing enabled)",
            _configured_daemon_port(),
        )
    except Exception as _mcp_exc:  # pragma: no cover — defensive
        logger.warning("MCP HTTP mount failed (non-fatal, stdio still works): %s", _mcp_exc)

    return application


def _register_dashboard_routes(application: FastAPI) -> None:
    """Mount all existing dashboard routes from server/routes/*.

    Extracted from api.py's create_app() to avoid duplicate MemoryEngine.
    """
    from superlocalmemory.server.api import UI_DIR

    # Rate limiting (graceful)
    try:
        from superlocalmemory.infra.rate_limiter import RateLimiter
        from superlocalmemory.core.remote_mode import (
            rate_limit_config,
            is_rate_limit_exempt,
        )
        # v3.6.12 (issue #40): thresholds are env-tunable (SLM_RATE_LIMIT_WRITE/
        # READ/WINDOW) so distributed/LAN operators can raise them. Defaults
        # unchanged (30 writes / 120 reads per 60s) for the local case.
        _rl_write, _rl_read, _rl_window = rate_limit_config()
        _write_limiter = RateLimiter(max_requests=_rl_write, window_seconds=_rl_window)
        _read_limiter = RateLimiter(max_requests=_rl_read, window_seconds=_rl_window)

        # S9-DASH-09: loopback (127.0.0.1 / ::1) is always the dashboard
        # itself — it legitimately makes many rapid reads (Brain + tabs +
        # polling). Rate-limiting our own UI produces 429s that cascade
        # into blank panels. CORS already restricts origins to localhost,
        # so we don't lose the anti-abuse posture for external callers.
        # v3.6.12 (issue #40): in SLM_REMOTE mode an allowlisted LAN browser is
        # the user's own dashboard doing the same rapid polling, so it is exempt
        # too (is_rate_limit_exempt) — otherwise normal polling trips 429.

        @application.middleware("http")
        async def rate_limit_middleware(request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if is_rate_limit_exempt(client_ip):
                return await call_next(request)
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            limiter = _write_limiter if is_write else _read_limiter
            allowed, remaining = limiter.is_allowed(client_ip)
            if not allowed:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests."},
                    headers={"Retry-After": str(getattr(limiter, 'window', 60))},
                )
            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
    except Exception as _rl_exc:
        # v3.6.12 (failopen-4): don't silently swallow — a missing rate limiter
        # is anti-abuse degradation worth a log line (unlike auth, this may
        # fail-open: rate limiting is not a security boundary).
        logger.warning("Rate-limit middleware not installed (%s)", _rl_exc)

    # Auth middleware (graceful)
    try:
        from superlocalmemory.infra.auth_middleware import (
            authorize_http_mcp_request,
            check_api_key,
        )
        from superlocalmemory.server.write_identity import (
            require_http_mutation_actor,
        )

        # Auth-exempt path prefixes — proxy routes carry provider API keys
        # (x-api-key for Anthropic, Authorization: Bearer for OpenAI, x-goog-api-key
        # for Gemini), never X-SLM-API-Key. Verified: auth_middleware.py:50-82
        # returns False for POST when api_key file exists and X-SLM-API-Key
        # is absent.
        _AUTH_EXEMPT_PREFIXES = ("/v1/", "/v1beta/")

        @application.middleware("http")
        async def auth_middleware(request, call_next):
            # Exempt proxy paths — LLM clients carry provider keys, not SLM keys.
            if request.url.path.startswith(_AUTH_EXEMPT_PREFIXES):
                return await call_next(request)
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            records_recall_telemetry = request.url.path.startswith("/recall")
            requires_mutation_actor = is_write or records_recall_telemetry
            headers = dict(request.headers)
            client_host = request.client.host if request.client else ""
            if request.url.path.startswith("/mcp") and not authorize_http_mcp_request(
                headers,
                client_host=client_host,
            ):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Remote HTTP MCP requires a configured SLM API key."
                    },
                )
            # v3.6.12 (csrf-1): defense-in-depth CSRF/DNS-rebinding guard on
            # state-changing requests. A cross-origin browser Origin is rejected;
            # loopback origins (the local dashboard) always pass, and LAN origins
            # pass only when explicitly allowlisted in SLM_REMOTE mode. Non-browser
            # clients (CLI/MCP/curl) send no Origin and are unaffected.
            if requires_mutation_actor:
                _origin = headers.get("origin", "") or headers.get("Origin", "")
                if _origin:
                    _ok_origin = any(_origin.startswith(p) for p in (
                        "http://127.0.0.1", "https://127.0.0.1",
                        "http://localhost", "https://localhost",
                        "http://[::1]", "https://[::1]",
                    ))
                    if not _ok_origin:
                        from superlocalmemory.core.remote_mode import is_remote_origin_allowed
                        _ok_origin = is_remote_origin_allowed(_origin)
                    if not _ok_origin:
                        from fastapi.responses import JSONResponse
                        return JSONResponse(
                            status_code=403,
                            content={"error": "cross-origin request rejected"},
                        )
                _mesh_secret = None
                if request.url.path.startswith("/mesh"):
                    _mesh_broker = getattr(application.state, "mesh_broker", None)
                    _mesh_secret = getattr(_mesh_broker, "_shared_secret", None)
                try:
                    request.state.authenticated_actor = require_http_mutation_actor(
                        request,
                        getattr(application.state, "daemon_descriptor", None),
                        actor_kind="http-route",
                        mesh_secret=_mesh_secret,
                    )
                except Exception as _identity_exc:
                    from fastapi import HTTPException as _HTTPException
                    from fastapi.responses import JSONResponse
                    if isinstance(_identity_exc, _HTTPException):
                        return JSONResponse(
                            status_code=_identity_exc.status_code,
                            content={"error": str(_identity_exc.detail)},
                        )
                    raise
            if not check_api_key(headers, is_write=is_write):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key."},
                )
            return await call_next(request)
    except Exception as _auth_exc:
        # v3.6.12 (failopen-1): security middleware must NEVER fail open silently.
        # The old `except (ImportError, Exception): pass` meant any failure to
        # install the auth gate left ALL write endpoints unauthenticated. Instead
        # log critically and install a fail-CLOSED fallback: writes from
        # non-loopback clients are rejected (loopback dashboard keeps working).
        logger.critical(
            "Auth middleware failed to install (%s) — installing fail-CLOSED "
            "fallback; non-loopback writes will be rejected.", _auth_exc,
        )
        try:
            from superlocalmemory.hooks.prewarm_auth import is_loopback as _is_lb
        except Exception:
            def _is_lb(h: str) -> bool:
                return h in ("127.0.0.1", "::1", "localhost")

        @application.middleware("http")
        async def _failclosed_auth(request, call_next):
            if request.url.path.startswith(("/v1/", "/v1beta/")):
                return await call_next(request)
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            client_host = request.client.host if request.client else ""
            if is_write and not _is_lb(client_host):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=503,
                    content={"error": "Auth subsystem unavailable; writes disabled."},
                )
            return await call_next(request)

    # Static files
    from fastapi.staticfiles import StaticFiles
    UI_DIR.mkdir(exist_ok=True)
    application.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    # Route modules
    from superlocalmemory.server.routes.memories import router as memories_router
    from superlocalmemory.server.routes.stats import router as stats_router
    from superlocalmemory.server.routes.profiles import router as profiles_router
    from superlocalmemory.server.routes.backup import router as backup_router
    from superlocalmemory.server.routes.data_io import router as data_io_router
    from superlocalmemory.server.routes.events import router as events_router
    from superlocalmemory.server.routes.agents import router as agents_router
    from superlocalmemory.server.routes.ws import router as ws_router, manager as ws_manager
    from superlocalmemory.server.routes.v3_api import router as v3_router
    from superlocalmemory.server.routes.adapters import router as adapters_router

    application.include_router(memories_router)
    application.include_router(stats_router)
    application.include_router(profiles_router)
    application.include_router(backup_router)
    application.include_router(data_io_router)

    # Optional routers — ImportError-safe so missing modules don't crash startup
    try:
        from superlocalmemory.server.routes.tiers import router as tiers_router
        application.include_router(tiers_router)
    except ImportError:
        logger.debug("tiers_router not available")

    try:
        from superlocalmemory.server.routes.evolution import router as evolution_router
        application.include_router(evolution_router)
    except ImportError:
        logger.debug("evolution_router not available")
    application.include_router(events_router)
    application.include_router(agents_router)
    application.include_router(ws_router)
    application.include_router(v3_router)
    application.include_router(adapters_router)

    # v3.4.1 chat SSE
    for _mod_name in ("chat",):
        try:
            _mod = __import__(
                f"superlocalmemory.server.routes.{_mod_name}", fromlist=["router"],
            )
            application.include_router(_mod.router)
        except (ImportError, Exception):
            pass

    # Optional routers
    for _mod_name in ("learning", "lifecycle", "behavioral", "compliance", "insights", "timeline"):
        try:
            _mod = __import__(
                f"superlocalmemory.server.routes.{_mod_name}", fromlist=["router"],
            )
            application.include_router(_mod.router)
        except (ImportError, Exception):
            pass

    # Wire WebSocket manager
    import superlocalmemory.server.routes.profiles as _profiles_mod
    import superlocalmemory.server.routes.data_io as _data_io_mod
    _profiles_mod.ws_manager = ws_manager
    _data_io_mod.ws_manager = ws_manager

    # Root page
    from fastapi.responses import HTMLResponse, JSONResponse

    # v3.4.23: /api/version — dashboard polls this to detect daemon upgrades
    # and auto-reload stale tabs (see ui/js/core.js::checkVersionFingerprint).
    try:
        from superlocalmemory import __version__ as _SLM_VERSION
    except Exception:  # pragma: no cover — defensive
        _SLM_VERSION = "unknown"

    @application.get("/api/version")
    async def api_version():
        return JSONResponse({"version": _SLM_VERSION})

    # v3.4.55: Mode switching & config API for the dashboard UI.
    # The auto-settings.js expects /api/v3/* endpoints. These routes
    # bridge the 3-mode config system to the existing settings page.

    @application.get("/api/v3/auto")
    async def v3_auto_detect():
        """Auto-detect available providers from environment."""
        import os as _os
        providers = []
        if _os.environ.get("OPENROUTER_API_KEY"):
            providers.append({"id": "openrouter", "name": "OpenRouter", "has_key": True})
        if _os.environ.get("OPENAI_API_KEY"):
            providers.append({"id": "openai", "name": "OpenAI", "has_key": True})
        if _os.environ.get("ANTHROPIC_API_KEY"):
            providers.append({"id": "anthropic", "name": "Anthropic", "has_key": True})
        # Ollama is always available as a local option if the server is reachable
        try:
            import httpx as _hx
            _r = _hx.get("http://localhost:11434/api/tags", timeout=2.0)
            ollama_models = []
            if _r.status_code == 200:
                ollama_models = [m["name"] for m in _r.json().get("models", [])]
            providers.append({
                "id": "ollama", "name": "Ollama (local)",
                "has_key": False, "running": True,
                "models": ollama_models,
            })
        except Exception:
            providers.append({
                "id": "ollama", "name": "Ollama (local)",
                "has_key": False, "running": False, "models": [],
            })
        return {"providers": providers}

    @application.get("/", response_class=HTMLResponse)
    async def root():
        index_path = UI_DIR / "index.html"
        if not index_path.exists():
            return (
                "<html><head><title>SuperLocalMemory V3</title></head>"
                "<body style='font-family:Arial;padding:40px'>"
                "<h1>SuperLocalMemory V3 — Unified Daemon</h1>"
                "<p><a href='/docs'>API Documentation</a></p>"
                "</body></html>"
            )
        # v3.4.23: substitute version placeholder so the dashboard can detect
        # upgrades and auto-reload. Read fresh each request (daemon uptime is
        # days, but we want zero caching surprises during development).
        html = index_path.read_text()
        return html.replace("__SLM_VERSION__", _SLM_VERSION)

def _register_daemon_routes(application: FastAPI) -> None:
    """Add daemon-specific routes for CLI integration."""
    global _last_activity

    from superlocalmemory.server.routes.helpers import get_engine_lazy

    def _get_engine_or_503():
        """Lazy-init engine; raise 503 if init fails.

        Shared by every daemon route so a mode switch that nulled
        ``application.state.engine`` never leaves the daemon stuck in
        503 until restart.
        """
        engine = get_engine_lazy(application.state)
        if engine is None:
            raise HTTPException(503, detail="Engine not initialized")
        return engine

    def _require_daemon_actor(request: Request) -> str:
        """Authenticate the private capability for this exact process."""
        from superlocalmemory.server.write_identity import require_daemon_actor

        return require_daemon_actor(
            request,
            getattr(application.state, "daemon_descriptor", None),
        )

    def _require_write_actor(request: Request) -> str:
        """Authenticate a local write and return its trusted actor.

        Caller-provided agent labels are audit metadata only.  A mutating
        daemon client may borrow the process actor only after proving the
        private capability for this exact instance.  Same-origin dashboard
        writers instead present the install token, which is never the daemon
        process capability.
        """
        from superlocalmemory.server.write_identity import require_write_actor

        return require_write_actor(
            request,
            getattr(application.state, "daemon_descriptor", None),
            actor_kind="dashboard",
        )

    @application.get("/health")
    async def health():
        _update_activity()
        # Non-blocking peek: report status without forcing a re-init.
        engine = getattr(application.state, "engine", None)
        migration_result = getattr(application.state, "migration_result", None)
        migration_failures = list(
            (migration_result or {}).get("failed", []) or []
        )
        migration_details = (migration_result or {}).get("details", {}) or {}
        migrations_ready = bool(migration_result) and not migration_failures
        if migration_details.get("_crash"):
            migrations_ready = False
        readiness = {
            "engine": engine is not None,
            "migrations": migrations_ready,
            "retrieval": bool(_embedding_warm),
            "migration_failures": migration_failures,
        }
        base_ready = all((readiness["engine"], readiness["migrations"]))
        fully_ready = base_ready and readiness["retrieval"]
        runtime_state = (
            "ready" if fully_ready else "warming" if base_ready else "not_ready"
        )
        # v3.6.8: surface the recall-health verdict so a silently-degraded
        # recall path (warm-but-broken embedder) is VISIBLE, never silent.
        try:
            from superlocalmemory.server.recall_health import get_recall_health
            _recall_health = get_recall_health()
        except Exception:
            _recall_health = {"recall_healthy": None}
        identity = getattr(application.state, "daemon_descriptor", None)
        return {
            "status": "ok",
            "ready": fully_ready,
            "readiness": readiness,
            "pid": os.getpid(),
            "engine": "initialized" if engine else "unavailable",
            "version": getattr(application, 'version', 'unknown'),
            # v3.4.52: clients can poll this to wait for embedding model
            # readiness before issuing recall calls.
            "embedding_warm": _embedding_warm,
            # v3.6.8: True iff the semantic channel actually fired on the last
            # health probe; includes self-heal counters.
            "recall_health": _recall_health,
            **(identity.public_health_fields() if identity is not None else {}),
            # Runtime readiness is more precise than descriptor lifecycle.
            # A process can be alive and identity-valid while retrieval warms.
            "state": runtime_state,
        }

    @application.get("/recall")
    async def recall(
        request: Request,
        q: str = "", query: str = "", limit: int = CANONICAL_RECALL_LIMIT,
        session_id: str = "",
        fast: bool = False,
        full: bool = False,
        include_source: bool = False,
        include_global: bool | None = None,
        include_shared: bool | None = None,
    ):
        _update_activity()
        search_query = q or query  # Accept both ?q= and ?query= for compatibility
        engine = _get_engine_or_503()
        if not search_query:
            return {"results": [], "count": 0, "query_type": "none", "retrieval_time_ms": 0}
        # S9-DASH-02: session_id for the outcome-queue producer.
        # Priority: ?session_id= > X-SLM-Session-Id header > synthetic
        # "http:<ts>". Without a session_id the recall still works
        # (outcome just can't be hook-matched).
        effective_sid = session_id
        if not effective_sid:
            effective_sid = request.headers.get("X-SLM-Session-Id", "")
        if not effective_sid:
            import time as _t
            effective_sid = f"http:{int(_t.time() * 1000)}"
        recall_actor = getattr(request.state, "authenticated_actor", "")
        if not recall_actor:
            from superlocalmemory.server.write_identity import (
                require_http_mutation_actor,
            )
            recall_actor = require_http_mutation_actor(
                request,
                getattr(application.state, "daemon_descriptor", None),
                actor_kind="http-recall",
            )
        # v3.4.32: mark recall in-flight so the pending materializer pauses
        # v3.4.52: run engine.recall() in a thread-pool executor so the
        # FastAPI event loop stays responsive for /health, /remember, and
        # concurrent /recall requests. Without this, a single slow full
        # recall (reranker timeout, cold embedder) blocks ALL endpoints.
        import asyncio
        _begin_recall()
        # v3.4.53: Full (non-fast) recalls are gated by a semaphore to
        # prevent resource oversaturation. Ollama serialises concurrent
        # embedding calls and the reranker subprocess has a single lock —
        # queuing more than ~3 concurrent full recalls just adds latency.
        # Fast recalls retain the bounded retrieval channels but skip remote
        # agentic verification, so they do not need the full-recall semaphore.
        if not fast:
            await _recall_semaphore.acquire()
        try:
            response = await asyncio.to_thread(
                engine.recall,
                search_query, limit=limit, session_id=effective_sid,
                agent_id=recall_actor,
                fast=fast,
                include_global=include_global,
                include_shared=include_shared,
            )
            # v3.4.26: return the same field shape as recall_worker so
            # MCP processes proxying through the daemon get recall_trace-
            # compatible data without a second round trip.
            memory_ids = list({
                r.fact.memory_id for r in response.results[:limit]
                if r.fact.memory_id
            })
            memory_map = (
                engine._db.get_memory_content_batch(memory_ids)
                if memory_ids else {}
            )
            # v3.6.6: single shared serialization chokepoint — budget + source
            # discipline + no_confident_match, identical across every surface.
            from superlocalmemory.server.recall_serializer import (
                recall_response_metadata,
                serialize_recall_response,
            )
            _rc = getattr(engine._config, "retrieval", None)
            results, no_confident_match = serialize_recall_response(
                response,
                limit=limit,
                memory_map={k: _sanitize_json_text(v) for k, v in memory_map.items()},
                per_fact_max=getattr(_rc, "recall_per_fact_max_chars", 2400),
                total_max=getattr(_rc, "recall_total_max_chars", 12000),
                full=full,
                include_source=include_source,
            )
            for _r in results:
                _r["content"] = _sanitize_json_text(_r.get("content", ""))
            return {
                "ok": True,
                "query": search_query,
                "query_type": response.query_type,
                "result_count": len(results),
                "retrieval_time_ms": round(response.retrieval_time_ms, 1),
                "channel_weights": {
                    k: round(v, 3)
                    for k, v in (response.channel_weights or {}).items()
                },
                "total_candidates": getattr(response, "total_candidates", 0),
                "results": results,
                "count": len(results),
                "no_confident_match": no_confident_match,
                **recall_response_metadata(response),
            }
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))
        finally:
            if not fast:
                _recall_semaphore.release()
            _end_recall()

    @application.post("/remember")
    async def remember(
        req: RememberRequest,
        request: Request,
        wait: bool = False,
    ):
        """Persist through the durable canonical ingestion state machine.

        The default path returns after the relational/FTS projection is
        queryable.  ``wait=true`` materializes the same operation inline; the
        background worker handles all other queryable operations.
        """
        trusted_actor_id = _require_write_actor(request)
        _update_activity()
        engine = _get_engine_or_503()

        # v3.6.15 multi-scope: resolve the write scope. ``None`` (not specified
        # by the caller) → the configured default_scope (personal). Shared
        # memory is opt-in, so the default keeps every write private.
        _scope_cfg = getattr(engine._config, "scope", None)
        scope = req.scope or getattr(_scope_cfg, "default_scope", "personal")
        shared_with = req.shared_with

        try:
            from superlocalmemory.core.engine_ingestion import (
                build_engine_ingestion_command,
            )
            from superlocalmemory.core.ingestion_command import (
                IngestionRequest,
                IngestionState,
            )

            meta = {}
            if req.tags:
                meta["tags"] = req.tags
            extra = getattr(req, "metadata", None)
            if isinstance(extra, dict):
                meta.update(extra)
            command = build_engine_ingestion_command(engine)
            receipt = command.submit(IngestionRequest(
                content=req.content,
                profile_id=engine._profile_id,
                source_type="http",
                idempotency_key=req.idempotency_key or uuid.uuid4().hex,
                metadata=meta,
                scope=scope,
                shared_with=tuple(shared_with or ()),
                trusted_actor_id=trusted_actor_id,
                session_id=req.session_id,
            ))

            result = command.materialize(receipt.operation_id) if wait else receipt
            if result.state is IngestionState.FAILED:
                raise RuntimeError(result.last_error or "materialization failed")

            fact_ids = list(result.fact_ids)
            _emit_event(
                "memory.stored" if wait else "memory.queued",
                payload={
                    "operation_id": result.operation_id,
                    "fact_ids": fact_ids,
                    "tags": req.tags or "",
                    "content_preview": req.content[:120],
                    "path": "remember_sync" if wait else "remember_queryable",
                },
            )
            return {
                "ok": True,
                "fact_ids": fact_ids,
                "count": len(fact_ids),
                "operation_id": result.operation_id,
                # One-release compatibility alias. The durable operation ID is
                # opaque and replaces the integer pending.db row identifier.
                "pending_id": result.operation_id,
                "status": "stored" if wait else "queryable",
                "materialization_state": result.state.value,
                "note": (
                    "canonical ingestion complete"
                    if wait
                    else "queryable now; canonical enrichment pending"
                ),
            }
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    @application.post("/observe")
    async def observe(req: ObserveRequest, request: Request):
        _update_activity()
        from superlocalmemory.server.write_identity import (
            authenticated_request_actor,
        )
        actor_id = authenticated_request_actor(
            request,
            getattr(application.state, "daemon_descriptor", None),
            actor_kind="http-observe",
        )
        result = _observe_buffer.enqueue(
            req.content,
            trusted_actor_id=actor_id,
        )
        return result

    # v3.4.26: CCQ consolidation via daemon so MCP clients don't need to
    # import CognitiveConsolidator (which pulls sentence-transformers).
    @application.post("/consolidate/cognitive")
    async def consolidate_cognitive_endpoint(body: dict, request: Request):
        _update_activity()
        engine = _get_engine_or_503()
        from superlocalmemory.server.route_mutations import (
            authorize_route_mutation,
        )
        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-cognitive-consolidation",
            profile_id=body.get("profile_id") or engine.profile_id,
        )
        try:
            pid = body.get("profile_id") or engine.profile_id
            from superlocalmemory.encoding.cognitive_consolidator import (
                CognitiveConsolidator,
            )
            consolidator = CognitiveConsolidator(db=engine._db)
            result = consolidator.run_pipeline(pid)
            authorization.complete()
            return {
                "ok": True,
                "profile_id": pid,
                "clusters_processed": result.clusters_processed,
                "blocks_created": result.blocks_created,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # v3.4.26: run_maintenance via daemon so MCP doesn't import
    # EbbinghausCurve, ForgettingScheduler, or ConsolidationWorker.
    @application.post("/maintenance/run")
    async def run_maintenance_endpoint(body: dict, request: Request):
        _update_activity()
        engine = _get_engine_or_503()
        from superlocalmemory.server.route_mutations import (
            authorize_route_mutation,
        )
        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-maintenance",
            profile_id=body.get("profile_id") or engine.profile_id,
        )
        try:
            pid = body.get("profile_id") or engine.profile_id
            results: dict = {}
            try:
                from superlocalmemory.core.maintenance import run_maintenance as _run_maint
                maint_result = _run_maint(engine._db, engine._config, pid)
                results["langevin"] = {"updated": maint_result.get("updated", 0)}
            except Exception as exc:
                results["langevin"] = {"error": str(exc)}
            try:
                from superlocalmemory.math.ebbinghaus import EbbinghausCurve
                from superlocalmemory.learning.forgetting_scheduler import (
                    ForgettingScheduler,
                )
                ebb = EbbinghausCurve(engine._config.forgetting)
                sched = ForgettingScheduler(
                    engine._db, ebb, engine._config.forgetting,
                )
                results["forgetting"] = sched.run_decay_cycle(pid, force=False)
            except Exception as exc:
                results["forgetting"] = {"error": str(exc)}
            try:
                from superlocalmemory.learning.consolidation_worker import (
                    ConsolidationWorker,
                )
                cw = ConsolidationWorker(
                    engine._db.db_path,
                    engine._db.db_path.parent / "learning.db",
                )
                count = cw._generate_patterns(pid, False)
                results["behavioral"] = {"patterns_mined": count}
            except Exception as exc:
                results["behavioral"] = {"error": str(exc)}
            authorization.complete()
            return {"ok": True, "profile": pid, **results}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    @application.get("/status")
    async def status():
        _update_activity()
        # Non-blocking peek — status must never force a re-init.
        engine = getattr(application.state, "engine", None)
        fact_count = engine.fact_count if engine else 0
        mode = engine._config.mode.value if engine and hasattr(engine, '_config') else "unknown"
        return {
            "status": "running",
            "pid": os.getpid(),
            "uptime_s": round(time.monotonic() - (_start_time or time.monotonic())),
            "mode": mode,
            "fact_count": fact_count,
            "idle_s": round(time.monotonic() - _last_activity),
            "port": application.state.daemon_descriptor.port,
            "legacy_port": _LEGACY_PORT,
        }

    @application.get("/list")
    async def list_facts(limit: int = 50):
        _update_activity()
        engine = _get_engine_or_503()
        try:
            facts = engine.list_facts(limit=limit)
            items = [
                {
                    "content": f.content[:100],
                    "fact_type": getattr(f.fact_type, 'value', str(f.fact_type)),
                    "created_at": (f.created_at or "")[:19],
                    "fact_id": f.fact_id,
                }
                for f in facts
            ]
            return {"results": items, "count": len(items)}
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    @application.post("/stop")
    async def stop(request: Request):
        """Gracefully stop only the capability-bound process instance."""
        _require_daemon_actor(request)
        logger.info("Stop requested via API")
        _observe_buffer.flush_sync()
        # Signal uvicorn to shut down gracefully
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "stopping"}

    @application.post("/session/open")
    async def session_open(req: SessionOpenRequest, request: Request):
        """#49: Open a session locally — warm recall context with no model
        roundtrip, so a shell/session-start hook can call it directly
        (`slm session open`) instead of going through the MCP tool.
        """
        _update_activity()
        engine = _get_engine_or_503()
        if req.query:
            query = req.query
        elif req.project_path:
            query = f"project context {req.project_path}"
        else:
            query = "recent important decisions"
        try:
            from superlocalmemory.server.write_identity import (
                authenticated_request_actor,
            )
            actor_id = authenticated_request_actor(
                request,
                getattr(application.state, "daemon_descriptor", None),
                actor_kind="http-session-open",
            )
            resp = engine.recall(
                query,
                limit=req.max_results,
                agent_id=actor_id,
            )
            results = (
                getattr(resp, "results", None)
                or getattr(resp, "memories", None)
                or []
            )
            return {"ok": True, "query": query, "warmed": len(results)}
        except HTTPException:
            raise
        except Exception as exc:
            # Warming is best-effort — never fail the session-open hook.
            return {"ok": True, "query": query, "warmed": 0, "warning": str(exc)}

    @application.post("/session/close")
    async def session_close(req: SessionCloseRequest, request: Request):
        """#49: Close a session locally (e.g. a Claude /quit hook calling
        `slm session close`). Creates per-entity temporal summary events.
        An empty session_id closes the most recent real session.
        """
        _update_activity()
        engine = _get_engine_or_503()
        from superlocalmemory.server.route_mutations import (
            authorize_route_mutation,
        )
        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-session-close",
            profile_id=engine.profile_id,
        )
        sid = req.session_id
        if not sid:
            # Fall back to the most recent session that has memories.
            try:
                db = getattr(engine, "_db", None) or getattr(engine, "db", None)
                if db is not None and hasattr(db, "execute"):
                    rows = db.execute(
                        "SELECT session_id FROM memories "
                        "WHERE session_id != '' ORDER BY created_at DESC LIMIT 1",
                        (),
                    )
                    if rows:
                        sid = str(rows[0][0])
            except Exception as exc:
                logger.debug("session_close fallback lookup failed: %s", exc)
        if not sid:
            return {"ok": True, "session_id": "", "summary_events_created": 0,
                    "message": "no session to close"}
        try:
            created = engine.close_session(sid)
            authorization.complete()
            return {"ok": True, "session_id": sid,
                    "summary_events_created": int(created)}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))


def _update_activity():
    global _last_activity
    _last_activity = time.monotonic()


_start_time: float | None = None

# v3.6.7: Starlette app returned by mcp_server.streamable_http_app().
# Set in create_app(); consumed by lifespan() to start the session manager.
_mcp_app = None


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def _start_memory_watchdog() -> None:
    """v3.4.7: Background watchdog that kills child workers exceeding memory limit.

    Prevents the orphan worker memory explosion that caused 16GB+ RAM usage.
    Checks every 60 seconds. Kills workers over 2GB RSS. Auto-restarts them
    on next request (workers are lazy-spawned).
    """
    import threading

    MAX_WORKER_MB = int(os.environ.get("SLM_MAX_WORKER_MB", "2500"))

    def watchdog_loop():
        while True:
            time.sleep(15)  # V3.4.37: 15s (was 60s) — catch spikes faster
            try:
                import psutil
                parent = psutil.Process(os.getpid())
                for child in parent.children(recursive=True):
                    try:
                        rss_mb = child.memory_info().rss / (1024 * 1024)
                        if rss_mb > MAX_WORKER_MB:
                            logger.warning(
                                "Memory watchdog: killing %s (PID %d, %.0f MB > %d MB limit)",
                                child.name(), child.pid, rss_mb, MAX_WORKER_MB,
                            )
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                pass  # psutil not available — watchdog disabled
            except Exception as exc:
                logger.debug("Memory watchdog error: %s", exc)

    t = threading.Thread(target=watchdog_loop, daemon=True, name="memory-watchdog")
    t.start()
    logger.info("Memory watchdog started (limit: %d MB per worker)", MAX_WORKER_MB)


_materializer_stop = threading.Event()
_materializer_thread: threading.Thread | None = None


def _materializer_actor_id() -> str:
    """Return the process-owned actor identity used by background writes."""
    descriptor = _ACTIVE_DAEMON_DESCRIPTOR
    if descriptor is None:
        from superlocalmemory.server.routes.helpers import SLM_VERSION

        descriptor = _process_descriptor(_DEFAULT_PORT, SLM_VERSION, "ready")
    return f"daemon-capability:{descriptor.capability_fingerprint}"


def _materialize_ingestion_one_pass(
    engine,
    *,
    limit: int = 50,
    min_queryable_age_seconds: float = 1.0,
) -> tuple[int, int]:
    """Materialize durable M018 work once; return ``(complete, failed)``."""
    # The durable queue shares the embedder/LLM with foreground recall just
    # like the legacy pending queue.  Yield before even constructing/claiming
    # work so an active user recall cannot suffer priority inversion.
    if _recalls_in_flight() > 0:
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
            _emit_event(
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
    return completed, failed


def _materialize_legacy_pending_item(engine, item: dict) -> str:
    """Backfill one pre-M018 pending.db row through canonical ingestion."""
    from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
    from superlocalmemory.core.ingestion_command import (
        IngestionRequest,
        IngestionState,
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
        profile_id=engine._profile_id,
        source_type=source_type,
        idempotency_key=idempotency_key,
        metadata=metadata,
        scope=scope,
        shared_with=shared_with,
        trusted_actor_id=_materializer_actor_id(),
        session_id=str(metadata.get("session_id") or ""),
    ))
    result = command.materialize(receipt.operation_id)
    if result.state is not IngestionState.COMPLETE:
        raise RuntimeError(result.last_error or "legacy pending materialization failed")
    return result.operation_id


def _start_pending_materializer() -> None:
    """Drain M018 operations and backfill the legacy pending.db queue."""
    global _materializer_thread

    def _loop():
        from superlocalmemory.cli.pending_store import (
            get_pending, mark_done, mark_failed,
        )
        # v3.4.38: log first engine acquisition so we know materializer is alive
        _engine_logged = False
        _waiting_logged = False
        while not _materializer_stop.is_set():
            try:
                # v3.4.38: Read fresh module global on every iteration so we
                # pick up the engine after lifespan sets it. Use the import
                # trick to ensure we're reading the live module attribute,
                # not a stale local reference.
                import superlocalmemory.server.unified_daemon as _ud
                engine = _ud._engine
                if engine is None:
                    if not _waiting_logged:
                        logger.info("Materializer: waiting for engine to init...")
                        _waiting_logged = True
                    time.sleep(0.5)
                    continue
                if not _engine_logged:
                    logger.info("Materializer: engine acquired, starting drain loop")
                    _engine_logged = True

                durable_complete, durable_failed = _materialize_ingestion_one_pass(
                    engine,
                    limit=50,
                )
                pending = get_pending(limit=50)
                if not pending and not durable_complete and not durable_failed:
                    time.sleep(1.0)
                    continue
                if pending:
                    logger.info(
                        "Materializer: backfilling %d legacy pending memories",
                        len(pending),
                    )
                for item in pending:
                    if _materializer_stop.is_set():
                        break
                    waits = 0
                    while _recalls_in_flight() > 0 and waits < 60:
                        time.sleep(0.5)
                        waits += 1
                    try:
                        operation_id = _materialize_legacy_pending_item(engine, item)
                        mark_done(item["id"])
                        _emit_event(
                            "memory.stored",
                            payload={
                                "pending_id": item["id"],
                                "operation_id": operation_id,
                                "path": "legacy_pending_backfill",
                                "content_preview": item["content"][:120],
                            },
                            source_agent="materializer",
                        )
                    except Exception as exc:
                        logger.warning(
                            "Pending %d failed: %s", item["id"], exc,
                        )
                        mark_failed(item["id"], str(exc))
            except Exception as exc:
                logger.warning("materializer loop error: %s", exc)
                time.sleep(5.0)

    _materializer_thread = threading.Thread(
        target=_loop, daemon=True, name="pending-materializer",
    )
    _materializer_thread.start()
    logger.info("Pending materializer started (recall-priority)")


def start_server(port: int = _DEFAULT_PORT) -> None:
    """Start the unified daemon. Blocks until stopped."""
    global _start_time
    assert_no_durable_root_conflict()
    import uvicorn

    # v3.4.23: rotate oversized logs before anything else so both the CLI
    # path (`slm serve`) and the LaunchAgent path (__main__) are covered.
    try:
        rotate_oversized_logs()
    except Exception:
        pass  # never block startup on log housekeeping

    from superlocalmemory.server.routes.helpers import SLM_VERSION

    _publish_process_descriptor(port, SLM_VERSION, "starting")
    _start_time = time.monotonic()

    try:
        from superlocalmemory.migrations.v3_4_25_to_v3_4_26 import (
            is_ready as _is_ready, migrate as _migrate,
        )
        _data = canonical_data_root()
        if not _is_ready(_data):
            _migrate(_data)
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "v3.4.26 migration on daemon start failed: %s", exc,
        )

    # v3.4.7: Start memory watchdog to prevent runaway workers
    _start_memory_watchdog()

    # v3.4.32: Continuous pending-queue materializer with recall priority.
    _start_pending_materializer()

    log_dir = state_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Bind address. `SLM_DAEMON_HOST` is the canonical name; `SLM_HOST` is
    # accepted as a shorter alias (issue #23). Set either to 0.0.0.0 to serve
    # a shared instance over a trusted private network (e.g. WireGuard mesh).
    bind_host = (
        os.environ.get("SLM_DAEMON_HOST")
        or os.environ.get("SLM_HOST")
        or "127.0.0.1"
    )

    config = uvicorn.Config(
        app="superlocalmemory.server.unified_daemon:create_app",
        factory=True,
        host=bind_host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=10,
    )
    server = uvicorn.Server(config)

    _publish_process_descriptor(port, SLM_VERSION, "ready")

    try:
        server.run()
    finally:
        _cleanup_process_descriptor(_ACTIVE_DAEMON_DESCRIPTOR)


# ---------------------------------------------------------------------------
# v3.4.23 — Startup log rotation
# ---------------------------------------------------------------------------
# The LaunchAgent plist redirects stdout/stderr to daemon.log and
# daemon-error.log. Those files are managed by launchd, not Python, so
# Python's RotatingFileHandler cannot prune them. If any bug ever writes
# large amounts of data to stderr (the v3.4.22 logger-format bug produced
# ~30 KB per startup and the file grew to 69 MB), end users end up with a
# disk-eating log they never knew existed.
#
# rotate_oversized_logs() is a belt-and-suspenders guard: every time the
# daemon starts, if either log exceeds MAX_LOG_BYTES we rename the current
# file to ".1" (keeping one rotated copy) and truncate the original so
# launchd's open file descriptor keeps working. This is cheap, stateless,
# and independent of whatever caused the overflow.
# ---------------------------------------------------------------------------

_MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB


def rotate_oversized_logs(log_dir: Optional[Path] = None,
                          max_bytes: int = _MAX_LOG_BYTES) -> None:
    """Rotate daemon.log and daemon-error.log at startup if oversized.

    Keeps one rotated copy (.1). Safe under concurrent start attempts:
    rename is atomic on POSIX, and truncation is idempotent.
    """
    log_dir = log_dir or state_path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    for name in ("daemon.log", "daemon-error.log", "daemon.json.log"):
        path = log_dir / name
        try:
            if not path.exists() or path.stat().st_size <= max_bytes:
                continue
            rotated = log_dir / f"{name}.1"
            try:
                if rotated.exists():
                    rotated.unlink()
            except Exception:
                pass
            try:
                path.rename(rotated)
            except Exception:
                # If rename fails (e.g., file is the open stderr fd under
                # launchd), fall back to truncation so we at least reclaim
                # disk without breaking the redirect.
                try:
                    with open(path, "w"):
                        pass
                except Exception:
                    pass
                continue
            # Re-create the original path as empty so launchd's redirect
            # keeps appending to a fresh file.
            try:
                path.touch()
            except Exception:
                pass
        except Exception:
            # Log rotation must never prevent daemon startup.
            continue


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Rotate first, then configure logging, so the first log line lands in a
    # freshly-sized file.
    rotate_oversized_logs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # v3.6.9 (#33): honour SLM_DAEMON_PORT env so operators can configure the
    # port without changing the launch command. --port= arg takes precedence.
    port = int(os.environ.get("SLM_DAEMON_PORT", "") or _DEFAULT_PORT)
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
    if "--start" in sys.argv:
        start_server(port=port)
    else:
        print("Usage: python -m superlocalmemory.server.unified_daemon --start [--port=8765]")
