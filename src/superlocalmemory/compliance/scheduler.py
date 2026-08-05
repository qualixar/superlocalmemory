# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Background scheduler that periodically enforces retention rules.

Runs on a configurable interval (default: 1 hour) using daemon threads
so the scheduler does not prevent process exit.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from .retention import RetentionEngine

logger = logging.getLogger(__name__)

# Default: run every hour
DEFAULT_INTERVAL_SECONDS = 3600


class RetentionScheduler:
    """Background scheduler that periodically enforces retention rules.

    Uses daemon threading — does not prevent process exit. The scheduler
    runs RetentionEngine.enforce() on all profiles at each interval.
    """

    def __init__(
        self,
        retention_engine: RetentionEngine | None = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        *,
        db_path: str | Path | None = None,
    ) -> None:
        if retention_engine is None and db_path is None:
            raise ValueError("retention_engine or db_path is required")
        self._engine = retention_engine
        if self._engine is not None:
            self._engine._autocommit = False
        self._db_path = Path(db_path) if db_path is not None else None
        self._interval = interval_seconds
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background scheduler.

        Does nothing if already running. Schedules the first enforcement
        cycle after interval_seconds.
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._schedule_next()
            logger.info(
                "Retention scheduler started (interval=%ds)",
                self._interval,
            )

    def stop(self) -> None:
        """Stop the background scheduler.

        Cancels the pending timer and waits briefly for an active cycle. Safe
        to call even if not running.
        """
        timer: threading.Timer | None
        with self._lock:
            self._running = False
            timer = self._timer
            self._timer = None
            if timer is not None:
                timer.cancel()
        if timer is not None and timer is not threading.current_thread():
            timer.join(timeout=10.0)
            if timer.is_alive():
                logger.warning("Retention scheduler did not stop within 10 seconds")
        logger.info("Retention scheduler stopped")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_once(self) -> dict[str, Any]:
        """Run retention enforcement once (for testing / manual trigger).

        Returns:
            Dict with enforcement results across all profiles.
        """
        return self._execute_cycle()

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next enforcement cycle."""
        self._timer = threading.Timer(self._interval, self._run_cycle)
        self._timer.daemon = True
        self._timer.start()

    def _run_cycle(self) -> None:
        """Run one enforcement cycle, then schedule the next."""
        try:
            self._execute_cycle()
        except Exception as exc:
            # Scheduler must not crash — log and continue
            logger.error("Retention scheduler cycle failed: %s", exc)
        finally:
            with self._lock:
                if self._running:
                    self._schedule_next()

    def _execute_cycle(self) -> dict[str, Any]:
        """Core retention enforcement logic.

        Discovers all profiles with retention rules and enforces each.
        """
        if self._db_path is not None:
            from superlocalmemory.storage.memory_write import memory_write

            with memory_write(self._db_path) as connection:
                return self._execute_with_engine(
                    RetentionEngine(connection, autocommit=False)
                )
        if self._engine is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("retention scheduler has no engine")
        return self._execute_with_engine(self._engine)

    @staticmethod
    def _execute_with_engine(engine: RetentionEngine) -> dict[str, Any]:
        """Enforce one cycle through the supplied short-lived engine."""
        results: list[dict[str, Any]] = []

        try:
            db = engine._db
            rows = db.execute(
                "SELECT DISTINCT profile_id FROM retention_rules"
            ).fetchall()
            profile_ids = [r[0] for r in rows]
        except sqlite3.OperationalError:
            profile_ids = []

        for profile_id in profile_ids:
            savepoint = "slm_retention_scheduler_profile"
            try:
                engine._db.execute(f"SAVEPOINT {savepoint}")
                result = engine.enforce(profile_id)
                engine._db.execute(f"RELEASE SAVEPOINT {savepoint}")
                results.append(result)
            except Exception as exc:
                try:
                    engine._db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    engine._db.execute(f"RELEASE SAVEPOINT {savepoint}")
                except sqlite3.Error:
                    logger.exception(
                        "Retention rollback failed for profile '%s'", profile_id
                    )
                logger.error(
                    "Retention enforcement failed for profile '%s': %s",
                    profile_id, exc,
                )
                results.append({
                    "profile_id": profile_id,
                    "error": str(exc),
                })

        return {
            "profiles_processed": len(profile_ids),
            "results": results,
        }
