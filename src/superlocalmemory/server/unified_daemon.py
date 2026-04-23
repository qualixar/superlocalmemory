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
License: Elastic-2.0
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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

logger = logging.getLogger("superlocalmemory.unified_daemon")

_DEFAULT_PORT = 8765
_LEGACY_PORT = 8767
_PID_FILE = Path.home() / ".superlocalmemory" / "daemon.pid"
_PORT_FILE = Path.home() / ".superlocalmemory" / "daemon.port"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    content: str
    tags: str = ""
    metadata: dict | None = None  # v3.4.26: pass-through from MCP pool_store


class ObserveRequest(BaseModel):
    content: str


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


# ---------------------------------------------------------------------------
# Observation debounce buffer (migrated from daemon.py)
# ---------------------------------------------------------------------------

class ObserveBuffer:
    """Thread-safe debounce buffer for observation processing.

    Buffers observations for a configurable window, deduplicates by content
    hash, then processes as a batch via the singleton MemoryEngine.
    """

    def __init__(self, debounce_sec: float = 3.0):
        self._debounce_sec = debounce_sec
        self._buffer: list[str] = []
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._engine = None

    def set_engine(self, engine) -> None:
        self._engine = engine

    def enqueue(self, content: str) -> dict:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        with self._lock:
            if content_hash in self._seen:
                return {"captured": False, "reason": "duplicate within debounce window"}
            self._seen.add(content_hash)
            self._buffer.append(content)
            buf_size = len(self._buffer)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._flush)
            self._timer.daemon = True
            self._timer.start()
        return {"captured": True, "queued": True, "buffer_size": buf_size}

    def _flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._seen.clear()
            self._timer = None

        if self._engine is None:
            return

        try:
            from superlocalmemory.hooks.auto_capture import AutoCapture
            auto = AutoCapture(engine=self._engine)
            for content in batch:
                try:
                    decision = auto.evaluate(content)
                    if decision.capture:
                        auto.capture(content, category=decision.category)
                except Exception:
                    pass
            logger.info("Observe debounce: processed %d observations", len(batch))
        except Exception:
            pass

    def flush_sync(self) -> None:
        """Force flush for shutdown."""
        if self._timer is not None:
            self._timer.cancel()
        self._flush()


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

    # H-21 (Stage 8) — first-boot-after-upgrade notice. Compare the cached
    # version marker against the current package version; if they differ
    # (fresh install or upgrade), log a one-time banner with a link to the
    # CHANGELOG. Non-fatal; any filesystem error is swallowed.
    try:
        from pathlib import Path as _VP
        try:
            from importlib.metadata import version as _pkg_version
            _slm_version = _pkg_version("superlocalmemory")
        except Exception:
            _slm_version = "unknown"
        _version_marker = _VP.home() / ".superlocalmemory" / ".last_version"
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
        from pathlib import Path as _P
        from superlocalmemory.storage.migration_runner import apply_all
        _home = _P.home() / ".superlocalmemory"
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
            "applied": [], "skipped": [], "failed": [],
            "details": {"_crash": str(_exc)},
        }

    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine

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
        logger.info("Unified daemon: MemoryEngine initialized (mode=%s)", config.mode.value)

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

        # Pre-warm workers (background)
        from superlocalmemory.core.worker_pool import WorkerPool
        WorkerPool.shared().warmup()

        # Force reranker warmup
        retrieval_eng = getattr(engine, '_retrieval_engine', None)
        if retrieval_eng:
            reranker = getattr(retrieval_eng, '_reranker', None)
            if reranker and hasattr(reranker, 'warmup_sync'):
                reranker.warmup_sync(timeout=120)

        # V3.4.11: Pre-warm embedding worker (load ONNX model on startup)
        # Without this, first recall takes 60-90s for model load.
        # Same pattern as reranker warmup above.
        import threading
        def _warmup_embedder():
            try:
                embedder = getattr(retrieval_eng, '_embedder', None) if retrieval_eng else None
                if embedder and hasattr(embedder, 'embed'):
                    embedder.embed("warmup")
                    logger.info("Embedding worker pre-warmed (ONNX model loaded)")
            except Exception as exc:
                logger.warning("Embedding warmup failed: %s", exc)
        threading.Thread(target=_warmup_embedder, daemon=True, name="embed-warmup").start()

    except Exception as exc:
        logger.warning("Engine init failed: %s", exc)
        application.state.engine = None
        application.state.config = None

    application.state.observe_buffer = _observe_buffer

    # Phase B: Start health monitor
    try:
        from superlocalmemory.core.health_monitor import HealthMonitor
        health_config = getattr(config, 'health', None)
        monitor = HealthMonitor(
            global_rss_budget_mb=getattr(health_config, 'global_rss_budget_mb', 4096) if health_config else 4096,
            heartbeat_timeout_sec=getattr(health_config, 'heartbeat_timeout_sec', 60) if health_config else 60,
            check_interval_sec=getattr(health_config, 'health_check_interval_sec', 30) if health_config else 30,
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
            db_path = config.db_path if config else Path.home() / ".superlocalmemory" / "memory.db"
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
        asyncio.create_task(_start_legacy_redirect(_DEFAULT_PORT, _LEGACY_PORT))

    # V3.4.22 LLD-02: signal-worker background drainer (S8-SK-01 fix).
    # Without this, ``signals.enqueue`` fills a bounded queue and drops
    # silently after ~250 recalls — learning_signals never populates,
    # Phase 3 never activates, the whole Living Brain stays cold.
    if os.environ.get("SLM_SIGNALS_ENABLED", "1") != "0":
        try:
            from superlocalmemory.learning import signal_worker as _sw
            from pathlib import Path as _P
            _learning_db = _P.home() / ".superlocalmemory" / "learning.db"
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
    if idle_timeout <= 0:
        _ready_msg = f"Unified daemon ready on port {_DEFAULT_PORT} (24/7 mode)"
    else:
        _ready_msg = (
            f"Unified daemon ready on port {_DEFAULT_PORT} "
            f"(idle timeout: {idle_timeout}s)"
        )
    logger.info(_ready_msg)

    yield

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
    _PID_FILE.unlink(missing_ok=True)
    _PORT_FILE.unlink(missing_ok=True)
    logger.info("Unified daemon shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create the unified FastAPI application."""
    from superlocalmemory.server.routes.helpers import SLM_VERSION

    application = FastAPI(
        title="SuperLocalMemory V3 — Unified Daemon",
        description="Memory + Dashboard + Mesh — one process, one engine.",
        version=SLM_VERSION,
        lifespan=lifespan,
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
        allow_headers=["Content-Type", "Authorization", "X-SLM-API-Key"],
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

    # -- Daemon-specific routes --
    _register_daemon_routes(application)

    return application


def _register_dashboard_routes(application: FastAPI) -> None:
    """Mount all existing dashboard routes from server/routes/*.

    Extracted from api.py's create_app() to avoid duplicate MemoryEngine.
    """
    from superlocalmemory.server.api import UI_DIR

    # Rate limiting (graceful)
    try:
        from superlocalmemory.infra.rate_limiter import RateLimiter
        _write_limiter = RateLimiter(max_requests=30, window_seconds=60)
        _read_limiter = RateLimiter(max_requests=120, window_seconds=60)

        # S9-DASH-09: loopback (127.0.0.1 / ::1) is always the dashboard
        # itself — it legitimately makes many rapid reads (Brain + tabs +
        # polling). Rate-limiting our own UI produces 429s that cascade
        # into blank panels. CORS already restricts origins to localhost,
        # so we don't lose the anti-abuse posture for external callers.
        _LOOPBACK_IPS = frozenset({"127.0.0.1", "::1", "localhost"})

        @application.middleware("http")
        async def rate_limit_middleware(request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if client_ip in _LOOPBACK_IPS:
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
    except (ImportError, Exception):
        pass

    # Auth middleware (graceful)
    try:
        from superlocalmemory.infra.auth_middleware import check_api_key

        @application.middleware("http")
        async def auth_middleware(request, call_next):
            is_write = request.method in ("POST", "PUT", "DELETE", "PATCH")
            headers = dict(request.headers)
            if not check_api_key(headers, is_write=is_write):
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key."},
                )
            return await call_next(request)
    except (ImportError, Exception):
        pass

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
    from superlocalmemory.server.routes.events import (
        router as events_router, register_event_listener,
    )
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

    # Startup event for event listener
    @application.on_event("startup")
    async def startup_event():
        register_event_listener()


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

    @application.get("/health")
    async def health():
        _update_activity()
        # Non-blocking peek: report status without forcing a re-init.
        engine = getattr(application.state, "engine", None)
        return {
            "status": "ok",
            "pid": os.getpid(),
            "engine": "initialized" if engine else "unavailable",
            "version": getattr(application, 'version', 'unknown'),
        }

    @application.get("/recall")
    async def recall(
        request: Request,
        q: str = "", query: str = "", limit: int = 20,
        session_id: str = "",
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
        # v3.4.32: mark recall in-flight so the pending materializer pauses
        _begin_recall()
        try:
            response = engine.recall(
                search_query, limit=limit, session_id=effective_sid,
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
            results = []
            for r in response.results[:limit]:
                fact_type = getattr(r.fact, "fact_type", None)
                lifecycle = getattr(r.fact, "lifecycle", None)
                results.append({
                    "fact_id": r.fact.fact_id,
                    "memory_id": r.fact.memory_id,
                    "content": r.fact.content,
                    "source_content": memory_map.get(r.fact.memory_id, ""),
                    "score": round(r.score, 4),
                    "confidence": round(r.confidence, 4),
                    "trust_score": round(r.trust_score, 4),
                    "channel_scores": {
                        k: round(v, 4)
                        for k, v in (r.channel_scores or {}).items()
                    },
                    "fact_type": fact_type.value
                        if fact_type and hasattr(fact_type, "value")
                        else getattr(r.fact, "fact_type", ""),
                    "lifecycle": lifecycle.value
                        if lifecycle and hasattr(lifecycle, "value") else "",
                    "access_count": getattr(r.fact, "access_count", 0),
                    "evidence_chain": list(
                        getattr(r, "evidence_chain", []) or []
                    ),
                })
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
            }
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))
        finally:
            _end_recall()

    @application.post("/remember")
    async def remember(req: RememberRequest, wait: bool = False):
        """v3.4.32: Async by default — writes to pending.db, returns pending_id
        in <100ms. Materializer thread drains at low priority, yielding to
        /search. Pass ``?wait=true`` for legacy synchronous behavior (blocks
        on the embedder until facts are written).
        """
        _update_activity()
        engine = _get_engine_or_503()

        if wait:
            try:
                metadata = {"tags": req.tags} if req.tags else {}
                extra = getattr(req, "metadata", None)
                if isinstance(extra, dict):
                    metadata.update(extra)
                fact_ids = engine.store(req.content, metadata=metadata)
                return {"ok": True, "fact_ids": fact_ids, "count": len(fact_ids)}
            except Exception as exc:
                raise HTTPException(500, detail=str(exc))

        try:
            from superlocalmemory.cli.pending_store import store_pending
            meta = {}
            if req.tags:
                meta["tags"] = req.tags
            extra = getattr(req, "metadata", None)
            if isinstance(extra, dict):
                meta.update(extra)
            pending_id = store_pending(
                req.content, tags=req.tags or "", metadata=meta,
            )
            return {
                "ok": True,
                "pending_id": pending_id,
                "status": "queued",
                "note": "materialized async; pass ?wait=true for legacy sync",
            }
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    @application.post("/observe")
    async def observe(req: ObserveRequest):
        _update_activity()
        result = _observe_buffer.enqueue(req.content)
        return result

    # v3.4.26: CCQ consolidation via daemon so MCP clients don't need to
    # import CognitiveConsolidator (which pulls sentence-transformers).
    @application.post("/consolidate/cognitive")
    async def consolidate_cognitive_endpoint(body: dict):
        _update_activity()
        engine = _get_engine_or_503()
        try:
            pid = body.get("profile_id") or engine.profile_id
            from superlocalmemory.encoding.cognitive_consolidator import (
                CognitiveConsolidator,
            )
            consolidator = CognitiveConsolidator(db=engine._db)
            result = consolidator.run_pipeline(pid)
            return {
                "ok": True,
                "profile_id": pid,
                "clusters_processed": result.clusters_processed,
                "blocks_created": result.blocks_created,
            }
        except Exception as exc:
            raise HTTPException(500, detail=str(exc))

    # v3.4.26: run_maintenance via daemon so MCP doesn't import
    # EbbinghausCurve, ForgettingScheduler, or ConsolidationWorker.
    @application.post("/maintenance/run")
    async def run_maintenance_endpoint(body: dict):
        _update_activity()
        engine = _get_engine_or_503()
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
            return {"ok": True, "profile": pid, **results}
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
            "port": _DEFAULT_PORT,
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
    async def stop():
        """Graceful shutdown via uvicorn's mechanism."""
        logger.info("Stop requested via API")
        _observe_buffer.flush_sync()
        # Signal uvicorn to shut down gracefully
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "stopping"}


def _update_activity():
    global _last_activity
    _last_activity = time.monotonic()


_start_time: float | None = None


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

    MAX_WORKER_MB = 4096  # 4GB per worker — ONNX full model is 1.6GB + overhead

    def watchdog_loop():
        while True:
            time.sleep(60)
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


def _start_pending_materializer() -> None:
    """Background thread: drains pending.db, yields to active /search calls.

    Poll loop:
    1. Fetch up to 5 pending rows.
    2. For each row: if any /search is in flight, sleep 500ms (yield priority).
    3. Call engine.store(), mark_done or mark_failed.
    4. Sleep 2s between polls when idle (empty queue).
    """
    global _materializer_thread

    def _loop():
        from superlocalmemory.cli.pending_store import (
            get_pending, mark_done, mark_failed,
        )
        while not _materializer_stop.is_set():
            try:
                engine = _engine  # may be None briefly at startup
                if engine is None:
                    time.sleep(2.0)
                    continue
                pending = get_pending(limit=5)
                if not pending:
                    time.sleep(2.0)
                    continue
                for item in pending:
                    if _materializer_stop.is_set():
                        break
                    # Yield to recalls: wait until none in flight
                    waits = 0
                    while _recalls_in_flight() > 0 and waits < 60:
                        time.sleep(0.5)
                        waits += 1
                    try:
                        import json as _json
                        md_str = item.get("metadata") or "{}"
                        try:
                            md = _json.loads(md_str)
                        except Exception:
                            md = {}
                        if item.get("tags"):
                            md.setdefault("tags", item["tags"])
                        engine.store(item["content"], metadata=md)
                        mark_done(item["id"])
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
    import uvicorn

    # v3.4.23: rotate oversized logs before anything else so both the CLI
    # path (`slm serve`) and the LaunchAgent path (__main__) are covered.
    try:
        rotate_oversized_logs()
    except Exception:
        pass  # never block startup on log housekeeping

    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))
    _PORT_FILE.write_text(str(port))
    _start_time = time.monotonic()

    try:
        from superlocalmemory.migrations.v3_4_25_to_v3_4_26 import (
            is_ready as _is_ready, migrate as _migrate,
        )
        _data = Path(os.environ.get("SLM_DATA_DIR")
                     or Path.home() / ".superlocalmemory")
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

    log_dir = Path.home() / ".superlocalmemory" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config = uvicorn.Config(
        app="superlocalmemory.server.unified_daemon:create_app",
        factory=True,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=10,
    )
    server = uvicorn.Server(config)

    try:
        server.run()
    finally:
        _PID_FILE.unlink(missing_ok=True)
        _PORT_FILE.unlink(missing_ok=True)


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
    log_dir = log_dir or (Path.home() / ".superlocalmemory" / "logs")
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
    port = _DEFAULT_PORT
    for arg in sys.argv:
        if arg.startswith("--port="):
            port = int(arg.split("=")[1])
    if "--start" in sys.argv:
        start_server(port=port)
    else:
        print("Usage: python -m superlocalmemory.server.unified_daemon --start [--port=8765]")
