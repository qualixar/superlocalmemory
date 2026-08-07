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

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def _module_spec_present(module: str) -> bool:
    """True if ``module`` can be imported, WITHOUT importing it.

    ``find_spec`` only resolves the loader; it never executes the module, so
    native packages (lancedb, pycozo) cannot spawn background runtimes during
    availability probes. Mirrors component_registry._module_present.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        # A broken/partial install can raise inside find_spec — treat as absent.
        return False

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
      on_daemon_start() → initialize bounded backend state → ready
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
        """Initialize bounded backend state without delaying daemon readiness."""
        logger.info("BackendOrchestrator: daemon starting")

        # 1. Apply schema (if not already applied)
        self._apply_schema_v345()

        # 2. Initialize TierManager (always). Backends are refreshed after
        # optional projections have been opened below.
        try:
            from superlocalmemory.core.tier_manager import evaluate_tiers
            self._tiers = evaluate_tiers
            logger.info("BackendOrchestrator: tier evaluator registered")
        except Exception as exc:
            logger.warning("TierManager init failed (non-fatal): %s", exc)

        # Full-database tier evaluation belongs to MaintenanceScheduler. Running
        # it here blocks FastAPI lifespan readiness on mature upgrade databases
        # and occurs before optional projection backends are fully registered.

        self._recover_interrupted_scale_promotion()

        # Backends may be installed with the product, but installing a wheel
        # is not authorization to mutate an existing data root.  Only a
        # verified, explicit promotion may initialize and migrate projections.
        if getattr(self._config, "scale_engine_state", "local_core") != "promoted":
            logger.info(
                "Scale Engine remains on Local Core (state=%s)",
                getattr(self._config, "scale_engine_state", "local_core"),
            )
            # v3.8.5: schedule a background check that auto-promotes to
            # Cozo+LanceDB only once the DB is large enough that they beat the
            # SQLite graph.  A no-op (and never even starts the build) for the
            # vast majority of installs, which sit far below the threshold.
            self._maybe_schedule_auto_promote()
            return

        # 3. Initialize CozoDB if available
        cozo_available = self._detect_cozo()
        if cozo_available:
            self._init_cozo()

        # 4. Initialize LanceDB if available
        lancedb_available = self._detect_lancedb()
        if lancedb_available:
            self._init_lancedb()

        # A promoted stage is already parity-verified.  Never rebuild it at
        # startup: automatic migration would bypass the staged lifecycle and
        # could make the active projection diverge from canonical SQLite.
        if self._cozo:
            self._update_status("cozo", "active", self._cozo.health_check().get("edges", 0))
        if self._lancedb:
            self._update_status("lancedb", "active", self._lancedb.health_check().get("vectors", 0))
        try:
            from superlocalmemory.core.tier_manager import set_backends
            set_backends(cozo=self._cozo, lancedb=self._lancedb)
        except Exception as exc:
            logger.warning("TierManager backend registration failed (non-fatal): %s", exc)

        logger.info("BackendOrchestrator: daemon ready (cozo=%s, lancedb=%s)",
                     "active" if self._cozo and self._cozo_status() == "active" else "off",
                     "active" if self._lancedb and self._lancedb_status() == "active" else "off")

    def _maybe_schedule_auto_promote(self) -> None:
        """Schedule a delayed, one-shot scale auto-promote check (v3.8.5).

        Fires well after boot warmup so it never competes for CPU / the write
        lock during the startup window — the daemon keeps serving canonical
        SQLite throughout.  A no-op (never even starts the build) unless
        auto-promotion is enabled AND the DB has grown past the threshold where
        a graph DB actually beats the well-indexed SQLite graph.
        """
        import os
        import threading

        cfg = self._config
        if not getattr(cfg, "scale_auto_promote_enabled", True):
            return
        if getattr(cfg, "scale_engine_state", "local_core") != "local_core":
            return
        try:
            delay = float(os.environ.get("SLM_AUTO_PROMOTE_DELAY_S", "300"))
        except (TypeError, ValueError):
            delay = 300.0
        timer = threading.Timer(delay, self._auto_promote_if_at_scale)
        timer.daemon = True
        timer.start()

    def _count_default_edges(self) -> int:
        """graph_edges count for the default profile (fail-soft → 0)."""
        try:
            rows = self._db.execute(
                "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = 'default'"
            )
            return int(rows[0]["c"]) if rows else 0
        except Exception:
            return 0

    def _auto_promote_if_at_scale(self) -> None:
        """Build + promote the Cozo/Lance projection iff the DB is at scale.

        Uses the SAME staged parity gate as the manual CLI path
        (prepare → verify → promote).  Any failure leaves canonical SQLite
        selected — the projection is derived data, never the source of truth.
        The promoted backends only serve after the next daemon restart, so this
        logs a clear, actionable message rather than swapping under a live
        process.
        """
        try:
            import os

            cfg = self._config
            if getattr(cfg, "scale_engine_state", "local_core") != "local_core":
                return
            threshold = int(
                os.environ.get("SLM_AUTO_PROMOTE_MIN_EDGES", "")
                or getattr(cfg, "scale_auto_promote_min_edges", 1_000_000)
            )
            edges = self._count_default_edges()
            if edges < threshold:
                logger.info(
                    "Scale auto-promote: %d edges < threshold %d — Local Core "
                    "(SQLite) stays optimal; no projection built.",
                    edges, threshold,
                )
                return
            logger.info(
                "Scale auto-promote: %d edges >= threshold %d — building "
                "Cozo+LanceDB projection in the background (SQLite keeps serving).",
                edges, threshold,
            )
            from superlocalmemory.core.scale_engine import ScaleEngineManager

            mgr = ScaleEngineManager(cfg, profile_id="default")
            prepared = mgr.prepare()
            stage_id = prepared.get("stage_id")
            mgr.verify(stage_id)
            mgr.promote(stage_id)
            logger.warning(
                "Scale Engine AUTO-PROMOTED to Cozo+LanceDB at %d edges. RESTART "
                "the daemon (`slm restart`) to activate the backends; until then "
                "it keeps serving canonical SQLite.",
                edges,
            )
        except Exception as exc:
            # Derived-data failure must never take down Local Core.
            logger.warning(
                "Scale auto-promote skipped — staying on Local Core / SQLite: %s",
                exc,
            )

    def _recover_interrupted_scale_promotion(self) -> None:
        """Repair an interrupted promotion; never auto-mutate a legacy root."""
        try:
            from superlocalmemory.core.scale_engine import ScaleEngineManager

            result = ScaleEngineManager(
                self._config,
                profile_id="default",
            ).recover_interrupted_promotion()
            if result:
                logger.warning("Scale Engine promotion recovery: %s", result)
        except Exception as exc:
            # A scale projection is derived data. Startup must keep serving
            # canonical SQLite even if optional recovery itself is unhealthy.
            logger.error(
                "Scale Engine recovery requires repair; Local Core remains active: %s",
                exc,
            )

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
        """Synchronize one fact's canonical entity bridge and fact edges."""
        try:
            # Retrying ingestion must not retain stale fact/entity links.
            self._cozo.remove_fact(fact.fact_id)
            entities = getattr(fact, "canonical_entities", []) or []
            profile_id = getattr(fact, "profile_id", "default") or "default"
            for eid in entities:
                rows = self._db.execute(
                    "SELECT canonical_name, entity_type, fact_count FROM canonical_entities "
                    "WHERE entity_id = ? AND profile_id = ?",
                    (eid, profile_id),
                )
                if rows:
                    entity = dict(rows[0])
                    self._cozo.add_entity(
                        eid,
                        entity.get("canonical_name") or eid,
                        entity.get("entity_type") or "concept",
                        {"fact_count": int(entity.get("fact_count") or 0)},
                        profile_id,
                    )
            self._cozo.add_fact_entities(fact.fact_id, entities, profile_id)
            for row in self._db.execute(
                "SELECT source_id, target_id, edge_type, weight FROM graph_edges "
                "WHERE profile_id = ? AND (source_id = ? OR target_id = ?)",
                (profile_id, fact.fact_id, fact.fact_id),
            ):
                edge = dict(row)
                self._cozo.add_edge(
                    edge["source_id"], edge["target_id"], edge.get("edge_type") or "related",
                    float(edge.get("weight") or 1.0), profile_id=profile_id,
                )
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
                    getattr(fact, "profile_id", "default") or "default",
                )
        except Exception as exc:
            logger.debug("LanceDB incremental sync skipped: %s", exc)

    def sync_deleted_fact(self, fact_id: str) -> None:
        """Remove a fact from derived projections after canonical deletion."""
        if self._cozo and self._cozo_status() == "active":
            try:
                self._cozo.remove_fact(fact_id)
            except Exception as exc:
                logger.warning("Cozo deletion sync failed for %s: %s", fact_id[:16], exc)
        if self._lancedb and self._lancedb_status() == "active":
            try:
                self._lancedb.remove_vector(fact_id)
            except Exception as exc:
                logger.warning("Lance deletion sync failed for %s: %s", fact_id[:16], exc)

    def sync_changed_fact(self, fact_id: str) -> None:
        """Refresh projections after an authorized canonical fact update."""
        fact = self._db.get_fact(fact_id)
        if fact is not None:
            self.sync_new_fact(fact)

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

    def graph_retrieval_ready(self) -> bool:
        """Whether Cozo can be injected into entity recall.

        Cozo carries both canonical entity mappings and fact graph edges.  The
        entity channel still shadows every projected result against SQLite and
        fails closed on any mismatch, so availability never weakens recall.
        """
        return bool(self._cozo and self._cozo_status() == "active")

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
            # find_spec only resolves the loader — never executes the module.
            # Native pycozo import can spawn background runtimes; do not probe
            # availability by importing (Python 3.14 GC race class).
            return _module_spec_present("pycozo")
        return False

    def _detect_lancedb(self) -> bool:
        vb = getattr(self._config, "vector_backend", "auto") or "auto"
        if vb == "sqlite-vec":
            return False
        if vb in ("auto", "lancedb"):
            # find_spec never executes the module. `import lancedb` starts
            # LanceDBBackgroundEventLoop and segfaults under Python 3.14 GC
            # when the full suite races cleanup — never import for a yes/no.
            return _module_spec_present("lancedb")
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
        except BaseException as exc:
            # PyO3 exposes Rust panics as PanicException(BaseException), not
            # Exception. An incompatible optional projection must never abort
            # daemon startup or hide canonical SQLite memory. Re-raise genuine
            # process-control exceptions; preserve the graph and degrade Cozo.
            if not isinstance(exc, Exception) and type(exc).__name__ != "PanicException":
                raise
            logger.warning("CozoDB init failed: %s", exc)
            self._cozo = None

    def _init_lancedb(self) -> None:
        try:
            from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend
            lance_path = self._data_dir / "lance"
            # v3.7.6 (#72): honor the configured embedding width instead of the
            # hardcoded 768d, so custom endpoints (e.g. 1024d Qwen3-Embedding) work.
            dimension = getattr(
                getattr(self._config, "embedding", None), "dimension", None
            )
            self._lancedb = LanceDBVectorBackend(str(lance_path), dimension=dimension)
            self._update_status("lancedb", "not_initialized")
            logger.info("LanceDB initialized at %s", lance_path)
        except Exception as exc:
            logger.warning("LanceDB init failed: %s", exc)
            self._lancedb = None

    # v3.7.9 (scale MEDIUM-2): _migrate_cozo/_migrate_lancedb were dead code —
    # never called anywhere — and bypassed the staged prepare→verify→promote
    # safety envelope (no fingerprint, no parity, no backup). Removed so a future
    # caller cannot re-import outside the lifecycle. Emergency re-imports must go
    # through the Scale Engine lifecycle.

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
                apply_migration,
                schema_version_applied,
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
