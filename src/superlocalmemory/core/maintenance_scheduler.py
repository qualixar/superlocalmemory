# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Background Maintenance Scheduler.

V3.3.13: Periodically triggers Langevin/Ebbinghaus/Sheaf maintenance
so users don't need to call run_maintenance manually.

Configurable interval via ForgettingConfig.scheduler_interval_minutes.
Defaults to 30 min. Disabled during benchmarks (no config.forgetting.enabled).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """Background scheduler for periodic math maintenance.

    Runs Langevin/Sheaf/Fisher maintenance at configurable intervals.
    Thread-safe. Auto-stops on garbage collection or explicit stop().
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: SLMConfig,
        profile_id: str = "default",
    ) -> None:
        self._db = db
        self._config = config
        self._profile_id = profile_id
        self._timer: threading.Timer | None = None
        self._running = False
        self._interval = config.forgetting.scheduler_interval_minutes * 60.0

    def start(self) -> None:
        """Start the periodic scheduler. Idempotent."""
        if self._running:
            return
        self._running = True
        self._schedule_next()
        logger.info(
            "Maintenance scheduler started (interval=%dm)",
            self._config.forgetting.scheduler_interval_minutes,
        )

    def stop(self) -> None:
        """Stop the scheduler. Idempotent."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("Maintenance scheduler stopped")

    def _schedule_next(self) -> None:
        """Schedule the next maintenance run."""
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        """Execute maintenance + auto-backup check, then schedule next run."""
        if not self._running:
            return
        for profile_id in self._profile_ids():
            try:
                from superlocalmemory.core.maintenance import run_maintenance
                counts = run_maintenance(self._db, self._config, profile_id)
                logger.info("Scheduled maintenance complete for %s: %s", profile_id, counts)
            except Exception as exc:
                logger.warning("Scheduled maintenance failed for %s: %s", profile_id, exc)

            # V3.4.11: Graph pruning (remove orphan edges)
            try:
                from superlocalmemory.core.graph_pruner import prune_graph
                prune_stats = prune_graph(self._db.db_path, profile_id)
                removed = prune_stats["total_before"] - prune_stats["total_after"]
                if removed > 0:
                    logger.info("Graph pruning for %s: %d edges removed", profile_id, removed)
            except Exception as exc:
                logger.debug("Graph pruning skipped for %s: %s", profile_id, exc)

            # Lifecycle evaluation must cover every stored profile, not only
            # whichever profile was active when the engine started.
            try:
                from superlocalmemory.core.tier_manager import evaluate_tiers
                stats = evaluate_tiers(self._db, profile_id)
                demoted = stats["demoted_to_warm"] + stats["demoted_to_cold"] + stats["demoted_to_archive"]
                if demoted > 0:
                    logger.info("Tier evaluation for %s: %d facts demoted", profile_id, demoted)
            except Exception as exc:
                logger.debug("Tier evaluation skipped for %s: %s", profile_id, exc)

            # v3.6.6 F-5: Daily core-block recompile with hygiene.
            try:
                from superlocalmemory.core.block_hygiene import _recompile_core_blocks
                _recompile_core_blocks(self._db, self._config, profile_id)
            except Exception as exc:
                logger.debug("Core-block recompile skipped for %s: %s", profile_id, exc)

        # V3.4.10: Check if auto-backup is due
        try:
            from superlocalmemory.infra.backup import BackupManager
            manager = BackupManager(db_path=self._db.db_path)
            filename = manager.check_and_backup()
            if filename:
                logger.info("Auto-backup created: %s", filename)
                self._sync_cloud_destinations(manager)
        except Exception as exc:
            logger.debug("Auto-backup check skipped: %s", exc)

        try:
            from superlocalmemory.cli.pending_store import cleanup_stale
            stats = cleanup_stale()
            if stats["total"] > 0:
                logger.info("Pending cleanup: %s", stats)
        except Exception as exc:
            logger.debug("Pending cleanup skipped: %s", exc)

        self._schedule_next()

    def _profile_ids(self) -> tuple[str, ...]:
        """Return every persisted profile with a deterministic fallback."""
        try:
            rows = self._db.execute(
                "SELECT profile_id FROM profiles ORDER BY profile_id",
                (),
            )
            profiles = tuple(
                dict.fromkeys(str(row["profile_id"]) for row in rows if row["profile_id"])
            )
            if profiles:
                return profiles
        except Exception as exc:
            logger.debug("Profile enumeration failed: %s", exc)
        return (self._profile_id,)

    def _sync_cloud_destinations(self, manager: object) -> None:
        """Push latest backup to configured cloud destinations."""
        try:
            from superlocalmemory.infra.cloud_backup import sync_all_destinations
            sync_all_destinations(self._db.db_path)
        except ImportError:
            pass  # cloud_backup module not available yet
        except Exception as exc:
            logger.warning("Cloud sync failed (non-critical): %s", exc)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
