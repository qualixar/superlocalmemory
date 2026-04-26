# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Enterprise-grade health monitoring for the SLM Unified Daemon.

Monitors:
  - Global RSS budget (kill heaviest worker if over limit)
  - Worker heartbeat (kill unresponsive workers after 60s)
  - Structured JSON logging (daemon.json.log alongside text logs)
  - Extensible health check registry (Phase C/D/E add checks)

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger("superlocalmemory.health_monitor")

# Try psutil — graceful fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.info("psutil not available — health monitoring limited")


# ---------------------------------------------------------------------------
# Health Check Registry (extensible by other phases)
# ---------------------------------------------------------------------------

_HEALTH_CHECKS: list[Callable[[], dict]] = []


def register_health_check(check_fn: Callable[[], dict]) -> None:
    """Register a health check function. Returns dict with name, status, detail."""
    _HEALTH_CHECKS.append(check_fn)


def run_all_health_checks() -> list[dict]:
    """Run all registered health checks. Returns list of results."""
    results = []
    for check_fn in _HEALTH_CHECKS:
        try:
            results.append(check_fn())
        except Exception as e:
            results.append({
                "name": getattr(check_fn, '__name__', 'unknown'),
                "status": "error",
                "detail": str(e),
            })
    return results


# ---------------------------------------------------------------------------
# Structured JSON Logger (additive — does NOT replace text logs)
# ---------------------------------------------------------------------------

_json_logger: logging.Logger | None = None


def setup_structured_logging(log_dir: Path | None = None) -> None:
    """Set up JSON structured logging alongside existing text logs.

    Creates a separate daemon.json.log file with RotatingFileHandler.
    Text logs continue working unchanged.
    """
    global _json_logger

    log_dir = log_dir or (Path.home() / ".superlocalmemory" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    json_log_path = log_dir / "daemon.json.log"

    _json_logger = logging.getLogger("superlocalmemory.structured")
    _json_logger.setLevel(logging.INFO)
    _json_logger.propagate = False  # Don't send to text handler

    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(
        str(json_log_path), maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _json_logger.addHandler(handler)


def log_structured(**fields) -> None:
    """Emit a structured JSON log entry.

    Always includes: timestamp, level. Caller provides the rest.
    Example fields: worker_pid, memory_rss_mb, operation, latency_ms, message.
    """
    if _json_logger is None:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": fields.pop("level", "info"),
        **fields,
    }
    try:
        _json_logger.info(json.dumps(entry, default=str))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Health Monitor Thread
# ---------------------------------------------------------------------------

class HealthMonitor:
    """Background thread monitoring worker health, RSS budget, heartbeats.

    Self-healing: if the monitor itself crashes, it logs and retries
    with exponential backoff (max 5 min sleep).
    """

    # SLM worker command-line identifiers for child process filtering
    _WORKER_IDENTIFIERS = (
        "superlocalmemory.core.embedding_worker",
        "superlocalmemory.core.reranker_worker",
        "superlocalmemory.core.recall_worker",
    )

    def __init__(
        self,
        global_rss_budget_mb: int = 2500,
        heartbeat_timeout_sec: int = 60,
        check_interval_sec: int = 15,
        enable_structured_logging: bool = True,
    ):
        self._budget_mb = global_rss_budget_mb
        self._heartbeat_timeout = heartbeat_timeout_sec
        self._interval = check_interval_sec
        self._enable_logging = enable_structured_logging
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._consecutive_failures = 0

    def start(self) -> None:
        """Start the health monitor in a daemon thread."""
        if not PSUTIL_AVAILABLE:
            logger.warning("Health monitor disabled: psutil not installed")
            return

        if self._enable_logging:
            setup_structured_logging()

        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="health-monitor",
        )
        self._thread.start()
        logger.info("Health monitor started (budget=%dMB, heartbeat=%ds)",
                     self._budget_mb, self._heartbeat_timeout)

        # Register built-in health checks
        register_health_check(self._check_daemon_health)
        register_health_check(self._check_worker_health)
        register_health_check(self._check_memory_budget)

    def stop(self) -> None:
        self._stop_event.set()

    def _run_loop(self) -> None:
        """Main monitoring loop with self-healing."""
        while not self._stop_event.is_set():
            try:
                self._check_once()
                self._consecutive_failures = 0
            except Exception as exc:
                self._consecutive_failures += 1
                backoff = min(300, 30 * self._consecutive_failures)
                logger.error("Health check failed (%d consecutive): %s. Backoff %ds.",
                             self._consecutive_failures, exc, backoff)
                log_structured(
                    level="error", operation="health_check",
                    message=f"Health check failed: {exc}",
                    consecutive_failures=self._consecutive_failures,
                )
                self._stop_event.wait(backoff)
                continue

            self._stop_event.wait(self._interval)

    def _check_once(self) -> None:
        """Single health check cycle."""
        proc = psutil.Process(os.getpid())
        daemon_rss_mb = proc.memory_info().rss / (1024 * 1024)

        # Find SLM worker children only (not adapters or other children)
        children = proc.children(recursive=True)
        slm_workers = []
        for child in children:
            try:
                cmdline = " ".join(child.cmdline()).lower()
                if any(ident in cmdline for ident in self._WORKER_IDENTIFIERS):
                    rss_mb = child.memory_info().rss / (1024 * 1024)
                    slm_workers.append({
                        "pid": child.pid,
                        "rss_mb": round(rss_mb, 1),
                        "cmdline": cmdline[:80],
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        total_rss_mb = daemon_rss_mb + sum(w["rss_mb"] for w in slm_workers)

        # Structured log entry
        log_structured(
            level="info",
            operation="health_check",
            memory_rss_mb=round(daemon_rss_mb, 1),
            total_rss_mb=round(total_rss_mb, 1),
            worker_count=len(slm_workers),
            workers=slm_workers,
            budget_mb=self._budget_mb,
        )

        # RSS budget enforcement
        if total_rss_mb > self._budget_mb and slm_workers:
            heaviest = max(slm_workers, key=lambda w: w["rss_mb"])
            logger.warning(
                "RSS budget exceeded (%.0fMB > %dMB). Killing heaviest worker PID %d (%.0fMB)",
                total_rss_mb, self._budget_mb, heaviest["pid"], heaviest["rss_mb"],
            )
            log_structured(
                level="warning",
                operation="rss_budget_kill",
                killed_pid=heaviest["pid"],
                killed_rss_mb=heaviest["rss_mb"],
                total_rss_mb=round(total_rss_mb, 1),
            )
            try:
                psutil.Process(heaviest["pid"]).terminate()
            except psutil.NoSuchProcess:
                pass

        # Heartbeat checks delegated to WorkerPool (Phase B wiring)
        # WorkerPool tracks last_heartbeat per worker. HealthMonitor
        # reads it here. Actual heartbeat protocol is in worker_pool.py.
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            last_hb = getattr(pool, '_last_heartbeat', {})
            now = time.monotonic()
            for wpid, last_time in list(last_hb.items()):
                if now - last_time > self._heartbeat_timeout:
                    logger.warning("Worker PID %d unresponsive (no heartbeat for %ds). Killing.",
                                   wpid, int(now - last_time))
                    log_structured(
                        level="warning",
                        operation="heartbeat_kill",
                        worker_pid=wpid,
                        seconds_since_heartbeat=round(now - last_time),
                    )
                    try:
                        psutil.Process(wpid).terminate()
                    except psutil.NoSuchProcess:
                        pass
                    del last_hb[wpid]
        except Exception:
            pass  # WorkerPool not initialized yet — fine

    # -- Built-in health checks for registry --

    def _check_daemon_health(self) -> dict:
        if not PSUTIL_AVAILABLE:
            return {"name": "daemon", "status": "unknown", "detail": "psutil unavailable"}
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        return {
            "name": "daemon",
            "status": "ok" if rss_mb < 500 else "warning",
            "detail": f"PID {os.getpid()}, RSS {rss_mb:.0f}MB",
        }

    def _check_worker_health(self) -> dict:
        try:
            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            wpid = pool.worker_pid
            if wpid:
                return {"name": "workers", "status": "ok", "detail": f"Worker PID {wpid}"}
            return {"name": "workers", "status": "warning", "detail": "No active worker"}
        except Exception as e:
            return {"name": "workers", "status": "error", "detail": str(e)}

    def _check_memory_budget(self) -> dict:
        if not PSUTIL_AVAILABLE:
            return {"name": "memory", "status": "unknown", "detail": "psutil unavailable"}
        proc = psutil.Process(os.getpid())
        total_rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        total_mb = total_rss / (1024 * 1024)
        status = "ok" if total_mb < self._budget_mb else "critical"
        return {
            "name": "memory",
            "status": status,
            "detail": f"{total_mb:.0f}MB / {self._budget_mb}MB budget",
        }
