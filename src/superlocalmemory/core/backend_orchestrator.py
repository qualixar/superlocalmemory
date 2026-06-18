# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory v3.4.5 — Backend Orchestrator.

Central coordinator for multi-backend architecture.
Manages CozoDB, LanceDB, and TierManager lifecycle.
Handles auto-migration, fallback, and incremental sync.

This is the ONLY module that imports all three backends.
Other modules call BackendOrchestrator methods.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton (set by daemon, read by store_pipeline)
# ---------------------------------------------------------------------------

_orchestrator: BackendOrchestrator | None = None


def get_orchestrator() -> BackendOrchestrator | None:
    """Return the global BackendOrchestrator singleton."""
    return _orchestrator


def set_orchestrator(orch: BackendOrchestrator) -> None:
    """Set the global BackendOrchestrator singleton."""
    global _orchestrator
    _orchestrator = orch


# ---------------------------------------------------------------------------
# BackendOrchestrator
# ---------------------------------------------------------------------------

class BackendOrchestrator:
    """Central coordinator for multi-backend architecture.

    Lifecycle:
      on_daemon_start() → migrate backends → ready
      sync_new_fact() → called from store_pipeline after SQLite write
      health_check() → returns status of all backends
    """

    def __init__(self, config: SLMConfig, db: DatabaseManager) -> None:
        self._config = config
        self._db = db
        self._data_dir = Path(getattr(config, "data_dir", None) or config.base_dir)
        self._cozo: Any = None
        self._lancedb: Any = None
        self._tiers: Any = None
        self._backend_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Daemon Startup
    # ------------------------------------------------------------------

    def on_daemon_start(self) -> None:
        """Called once on daemon startup. Order matters (F-11: rebalance before migration)."""
        logger.info("BackendOrchestrator: daemon starting")

        # 1. Apply schema (if not already applied)
        self._apply_schema_v345()

        # 2. Initialize TierManager (always)
        try:
            from superlocalmemory.core.tier_manager import evaluate_tiers, set_backends
            self._tiers = evaluate_tiers
            set_backends(cozo=self._cozo, lancedb=self._lancedb)
            logger.info("BackendOrchestrator: TierManager initialized")
        except Exception as exc:
            logger.warning("TierManager init failed (non-fatal): %s", exc)

        # 3. Run initial tier rebalance FIRST (F-11: before migration)
        try:
            from superlocalmemory.core.tier_manager import evaluate_tiers as rebalance
            result = rebalance(self._db)
            logger.info("BackendOrchestrator: initial rebalance — %s",
                         result.get("total_evaluated", "?"))
        except Exception as exc:
            logger.warning("Initial rebalance failed (non-fatal): %s", exc)

        # 4. Initialize CozoDB if available
        cozo_available = self._detect_cozo()
        if cozo_available:
            self._init_cozo()

        # 5. Initialize LanceDB if available
        lancedb_available = self._detect_lancedb()
        if lancedb_available:
            self._init_lancedb()

        # 6. Auto-migrate
        if self._cozo:
            status = self._cozo_status()
            if status in ("not_initialized", "migrating"):
                if status == "migrating":
                    logger.warning("CozoDB migration interrupted — rebuilding")
                self._migrate_cozo()

        if self._lancedb:
            status = self._lancedb_status()
            if status in ("not_initialized", "migrating"):
                if status == "migrating":
                    logger.warning("LanceDB migration interrupted — rebuilding")
                self._migrate_lancedb()

        logger.info("BackendOrchestrator: daemon ready (cozo=%s, lancedb=%s)",
                     "active" if self._cozo and self._cozo_status() == "active" else "off",
                     "active" if self._lancedb and self._lancedb_status() == "active" else "off")

    # ------------------------------------------------------------------
    # Incremental Sync (F-04: called from store_pipeline)
    # ------------------------------------------------------------------

    def sync_new_fact(self, fact: Any) -> None:
        """Sync a newly stored fact to CozoDB and LanceDB.

        Called AFTER SQLite write in store_pipeline.
        Non-blocking, best-effort. Failures are logged, not raised.
        """
        try:
            tier = getattr(fact, "lifecycle", "active")
        except Exception:
            tier = "active"

        if tier in ("active", "warm"):
            if self._cozo and self._cozo_status() == "active":
                self._sync_fact_entities(fact)

            if self._lancedb and self._lancedb_status() == "active":
                self._sync_fact_embedding(fact)

    def _sync_fact_entities(self, fact: Any) -> None:
        """Sync fact's entities and edges to CozoDB."""
        try:
            entities = getattr(fact, "canonical_entities", []) or []
            for eid in entities:
                self._cozo.add_entity(eid, eid, "concept", {})
                # Add edges from this entity to existing ones
                for other in entities:
                    if other != eid:
                        self._cozo.add_edge(eid, other, "co_occurs", 1.0)
        except Exception as exc:
            logger.debug("CozoDB incremental sync skipped: %s", exc)

    def _sync_fact_embedding(self, fact: Any) -> None:
        """Sync fact's embedding to LanceDB."""
        try:
            embedding = getattr(fact, "embedding", None)
            if embedding:
                tier = getattr(fact, "lifecycle", "active")
                self._lancedb.add_vectors(
                    [fact.fact_id], [embedding], [tier],
                )
        except Exception as exc:
            logger.debug("LanceDB incremental sync skipped: %s", exc)

    # ------------------------------------------------------------------
    # Backend Access
    # ------------------------------------------------------------------

    def get_graph_backend(self) -> Any:
        """Return active graph backend or None (caller falls back to NetworkX)."""
        if self._cozo and self._cozo_status() == "active":
            return self._cozo
        return None

    def get_vector_backend(self) -> Any:
        """Return active vector backend or None."""
        if self._lancedb and self._lancedb_status() == "active":
            return self._lancedb
        return None

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Comprehensive health status for dashboard + CLI."""
        result: dict[str, Any] = {
            "sqlite": {"status": "active"},
            "cozo": {"status": "not_available"},
            "lancedb": {"status": "not_available"},
            "tiers": {},
            "warnings": [],
        }

        try:
            from superlocalmemory.core.tier_manager import get_tier_stats
            result["tiers"] = get_tier_stats(self._db)
        except Exception:
            pass

        if self._cozo:
            try:
                result["cozo"] = self._cozo.health_check()
            except Exception as exc:
                result["cozo"] = {"status": "error", "error": str(exc)}
        else:
            result["warnings"].append(
                "CozoDB not active. Install: pip install superlocalmemory[cozo]"
            )

        if self._lancedb:
            try:
                result["lancedb"] = self._lancedb.health_check()
            except Exception as exc:
                result["lancedb"] = {"status": "error", "error": str(exc)}
        else:
            result["warnings"].append(
                "LanceDB not active. Install: pip install superlocalmemory[lancedb]"
            )

        return result

    # ------------------------------------------------------------------
    # Internal: Detection
    # ------------------------------------------------------------------

    def _detect_cozo(self) -> bool:
        gb = getattr(self._config, "graph_backend", "auto") or "auto"
        if gb == "sqlite":
            return False
        if gb in ("auto", "cozo"):
            try:
                import pycozo  # noqa: F401
                return True
            except ImportError:
                return False
        return False

    def _detect_lancedb(self) -> bool:
        vb = getattr(self._config, "vector_backend", "auto") or "auto"
        if vb == "sqlite-vec":
            return False
        if vb in ("auto", "lancedb"):
            try:
                import lancedb  # noqa: F401
                return True
            except ImportError:
                return False
        return False

    # ------------------------------------------------------------------
    # Internal: Init
    # ------------------------------------------------------------------

    def _init_cozo(self) -> None:
        try:
            from superlocalmemory.graph.cozo_backend import CozoDBGraphBackend
            cozo_path = self._data_dir / "cozo"
            cozo_path.mkdir(parents=True, exist_ok=True)
            self._cozo = CozoDBGraphBackend(str(cozo_path / "graph"))
            self._update_status("cozo", "not_initialized")
            logger.info("CozoDB initialized at %s", cozo_path)
        except Exception as exc:
            logger.warning("CozoDB init failed: %s", exc)
            self._cozo = None

    def _init_lancedb(self) -> None:
        try:
            from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend
            lance_path = self._data_dir / "lance"
            self._lancedb = LanceDBVectorBackend(str(lance_path))
            self._update_status("lancedb", "not_initialized")
            logger.info("LanceDB initialized at %s", lance_path)
        except Exception as exc:
            logger.warning("LanceDB init failed: %s", exc)
            self._lancedb = None

    # ------------------------------------------------------------------
    # Internal: Migration
    # ------------------------------------------------------------------

    def _migrate_cozo(self) -> None:
        self._update_status("cozo", "migrating")

        def _run():
            conn = sqlite3.connect(str(self._data_dir / "memory.db"))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA query_only=ON")  # F-07: read-only in migration thread
            try:
                count = self._cozo.bulk_import_from_sqlite(conn)
                self._update_status("cozo", "active", count)
                logger.info("CozoDB migration complete: %d edges", count)
            except Exception as exc:
                logger.error("CozoDB migration failed: %s", exc)
                self._update_status("cozo", "failed", error=str(exc))
            finally:
                conn.close()

        threading.Thread(target=_run, daemon=True).start()

    def _migrate_lancedb(self) -> None:
        self._update_status("lancedb", "migrating")

        def _run():
            conn = sqlite3.connect(str(self._data_dir / "memory.db"))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA query_only=ON")
            try:
                count = self._lancedb.bulk_import_from_sqlite(conn)
                self._update_status("lancedb", "active", count)
                logger.info("LanceDB migration complete: %d vectors", count)
            except Exception as exc:
                logger.error("LanceDB migration failed: %s", exc)
                self._update_status("lancedb", "failed", error=str(exc))
            finally:
                conn.close()

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal: Status
    # ------------------------------------------------------------------

    def _cozo_status(self) -> str:
        return self._backend_cache.get("cozo", "not_initialized")

    def _lancedb_status(self) -> str:
        return self._backend_cache.get("lancedb", "not_initialized")

    def _update_status(self, name: str, status: str,
                        count: int = 0, error: str = "") -> None:
        self._backend_cache[name] = status
        try:
            # #47 fix: DatabaseManager has no `.conn`; execute() commits itself.
            self._db.execute(
                "INSERT OR REPLACE INTO backend_status "
                "(backend_name, status, record_count, error_message, last_sync_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                (name, status, count, error),
            )
        except Exception as exc:
            logger.debug("backend_status update failed for %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Internal: Schema
    # ------------------------------------------------------------------

    def _apply_schema_v345(self) -> None:
        try:
            from superlocalmemory.storage.schema_v345 import (
                apply_migration, schema_version_applied,
            )
            # #47 fix: use raw_connection() — DatabaseManager has no `.conn`,
            # so the old code raised AttributeError that was silently swallowed,
            # leaving the v3.4.5 migration (access_count_30d) permanently unapplied.
            with self._db.raw_connection() as conn:
                if not schema_version_applied(conn):
                    result = apply_migration(conn)
                    if result.get("errors"):
                        logger.warning("Schema v3.4.5 had errors: %s", result["errors"])
        except ImportError:
            logger.debug("schema_v345 not found — skipping")
        except Exception as exc:
            logger.warning("Schema v3.4.5 apply failed (non-fatal): %s", exc)
