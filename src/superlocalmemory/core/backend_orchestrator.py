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

from superlocalmemory.core.projection_drain import ProjectionDrain
from superlocalmemory.storage import projection_outbox

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
    """Install the global BackendOrchestrator, retiring the previous one.

    A plain assignment here leaked a whole native thread pool per call.
    ``_hot_reconfigure_engine`` builds a fresh orchestrator on every
    reconfigure and profile switch, and a LanceDB connection owns a Rust tokio
    runtime sized to the host's cores -- so the old runtime, and the old drain
    worker still writing into the projection that was just swapped out, both
    stayed alive with no reference left to stop them. GitHub #137.

    Retiring the predecessor must never prevent the successor being installed,
    hence the guard: a half-installed orchestrator is worse than a leak.
    """
    global _orchestrator
    previous = _orchestrator
    _orchestrator = orch
    if previous is not None and previous is not orch:
        try:
            previous.stop()
        except Exception:
            logger.warning("retiring the previous orchestrator failed", exc_info=True)


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
        # Given accessors, not backends: a promotion or a rollback replaces
        # them underneath the worker, and a reference captured here would keep
        # writing into the projection that was just swapped out.
        self._drain = ProjectionDrain(
            db, self.get_graph_backend, self.get_vector_backend,
        )

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

        # Reconcile what the config CLAIMS against what is on disk, before
        # anything reads either -- and before the early return below, because
        # the stores that need reconciling are exactly the ones that take it.
        #
        # This used to sit after that return, so it ran only for a store already
        # in the promoted state. The real case it was written for is a store
        # whose settings name a graph and a vector backend, whose state is
        # `verified` rather than `promoted`, and where neither directory exists:
        # the settings kept the claim, the reconcile never ran, and every
        # restart preserved it. The test asserted the call appeared before
        # another call in the source text, which is true either way.
        self._reconcile_backend_selection()

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
            # Started even with no projection open. A pass with no backend
            # returns without touching a row, and starting it here means a
            # promotion that completes mid-session has a worker waiting for it
            # rather than a queue nobody is reading.
            self._drain.start()
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

        self._drain.start()

        logger.info(
            "BackendOrchestrator: daemon ready (cozo=%s, lancedb=%s, queued=%d)",
            "active" if self._cozo and self._cozo_status() == "active" else "off",
            "active" if self._lancedb and self._lancedb_status() == "active" else "off",
            projection_outbox.depth(self._db),
        )

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
        # Only a store that has finished promoting has nothing left to do.
        #
        # This used to skip every state except ``local_core``, which made
        # ``prepared`` and ``verified`` terminal: the daemon above returns
        # early for anything that is not ``promoted``, so the backends never
        # started, and this refused to finish the promotion that would have
        # started them. A store that got as far as building and checking its
        # projection then sat on SQLite forever while its own config named Cozo
        # and LanceDB as the backends — measured on a real store whose
        # ``backend_status`` read lancedb=not_initialized under
        # ``scale_engine_state=verified``.
        #
        # ``run_auto_promote`` already resumes a half-finished stage
        # (``_resumable_stage``) and applies the size threshold and the
        # repair-required check itself, so it is the right place for every
        # decision except "there is nothing left to do".
        if str(getattr(cfg, "scale_engine_state", "local_core")).lower() == "promoted":
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
            # Same rule as the scheduler that armed this timer: only a store
            # that has finished has nothing left to do. Fixing the scheduler
            # alone would have armed a timer whose callback still refused.
            if str(getattr(cfg, "scale_engine_state", "local_core")).lower() == "promoted":
                return
            threshold = int(
                os.environ.get("SLM_AUTO_PROMOTE_MIN_EDGES", "")
                or getattr(cfg, "scale_auto_promote_min_edges", 100_000)
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

    def _reconcile_backend_selection(self) -> None:
        """Stop the config claiming a backend the store does not have.

        THE STATE THIS REPAIRS

        On a real store: ``graph_backend='cozo'``, ``vector_backend='lancedb'``,
        ``scale_engine_state='verified'`` -- and neither the ``cozo/`` nor the
        ``lance/`` directory existed, with no promotion journal to explain it.
        Something wrote the selection a completed promotion writes, without a
        promotion having completed.

        Nothing corrected it. ``recover_interrupted_promotion`` acts only when a
        journal exists, so with no journal it returns immediately and the claim
        survives every restart. The dashboard then reports the configured backend
        while retrieval uses SQLite, which is the disagreement a person notices
        last and trusts first.

        WHAT THIS DOES NOT DO

        It does not disable anything. ``auto`` still detects and initialises both
        projections when their libraries are installed, so the only thing removed
        is the false claim. It leaves ``verified`` alone -- that is a legitimate
        waypoint meaning "parity checked, not yet promoted" -- and only resets
        ``promoted``, which asserts a swap that plainly did not happen. And it
        never touches a selection whose directory is present, nor one with a
        journal still open, because those belong to the promotion lifecycle.
        """
        try:
            from superlocalmemory.core.scale_engine import ScaleEngineManager

            manager = ScaleEngineManager(self._config, profile_id="default")
            if manager.promotion_journal_path.exists():
                return  # the recovery path owns this
            cozo_path, lance_path = manager.active_paths
        except Exception as exc:  # noqa: BLE001 -- reconciliation is best effort
            logger.debug("Backend reconciliation skipped: %s", exc)
            return

        corrections: list[str] = []
        graph = getattr(self._config, "graph_backend", "auto") or "auto"
        if graph not in ("auto", "sqlite") and not cozo_path.exists():
            corrections.append(f"graph_backend {graph!r} -> 'auto' (no {cozo_path.name}/)")
            self._config.graph_backend = "auto"
        vector = getattr(self._config, "vector_backend", "auto") or "auto"
        if vector not in ("auto", "sqlite-vec") and not lance_path.exists():
            corrections.append(
                f"vector_backend {vector!r} -> 'auto' (no {lance_path.name}/)"
            )
            self._config.vector_backend = "auto"
        state = getattr(self._config, "scale_engine_state", "") or ""
        if state == "promoted" and not (cozo_path.exists() or lance_path.exists()):
            corrections.append("scale_engine_state 'promoted' -> 'local_core'")
            self._config.scale_engine_state = "local_core"

        if not corrections:
            return
        try:
            self._config.save()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Backend selection is inconsistent with the data directory and "
                "could not be corrected (%s): %s", exc, "; ".join(corrections),
            )
            return
        logger.warning(
            "Backend selection did not match the data directory; corrected: %s",
            "; ".join(corrections),
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
        """Signal that a stored fact needs projecting.

        The projection itself is not written here. It used to be — inline, on
        the caller's thread, with every failure swallowed into a debug line —
        and that is the defect the outbox replaced. The intent to project was
        already committed to ``projection_outbox`` in the same SQLite
        transaction as the fact, so all that is left to do is wake the worker
        that owns the projections.

        Kept as a method because callers name this operation, and because a
        caller that reaches it without an outbox row (an old store, mid-upgrade)
        should still get its fact projected rather than silently skipped.
        """
        fact_id = getattr(fact, "fact_id", None)
        if fact_id:
            projection_outbox.enqueue(
                self._db, fact_id, getattr(fact, "profile_id", None) or "default",
            )
        self._drain.notify()

    def sync_deleted_fact(self, fact_id: str) -> None:
        """Signal that a deleted fact must leave the projections.

        A forgotten memory still present in the graph or the vector index is
        still recallable, so the removal is queued with the same durability as
        the delete itself.
        """
        if fact_id:
            projection_outbox.enqueue_for_fact(
                self._db, fact_id, projection_outbox.OP_DELETE,
            )
        self._drain.notify()

    def sync_changed_fact(self, fact_id: str) -> None:
        """Refresh projections after an authorized canonical fact update."""
        if fact_id:
            projection_outbox.enqueue_for_fact(self._db, fact_id)
        self._drain.notify()

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

        outbox = self.outbox_health()
        result["projection_queue"] = outbox
        if outbox["stalled"]:
            result["warnings"].append(
                f"{outbox['stalled']} memory/memories could not be projected into "
                "the graph or vector store. Run `slm doctor` for the ids."
            )

        return result

    # ------------------------------------------------------------------
    # Projection queue
    # ------------------------------------------------------------------

    def drain_projections(self, limit: int = 200) -> dict[str, Any]:
        """Apply queued facts now, on the calling thread.

        For the CLI, for a repair pass, and for any caller that needs the
        projections current before it reads them rather than a few milliseconds
        later. Ordinary writes do not need this — they signal the worker.
        """
        return self._drain.drain_once(limit=limit).as_dict()

    def outbox_health(self) -> dict[str, Any]:
        """Queue depth and stalled count, for the status surfaces.

        ``draining`` is true only while the worker runs AND a projection is
        open: with none open (Local Core) rows wait as the catch-up record for
        a later promotion and nothing is being drained.
        """
        health = projection_outbox.health(self._db)
        is_open = self._drain.has_target()
        health["projection_open"] = is_open
        health["draining"] = bool(self._drain.running and is_open)
        return health

    def stop(self) -> None:
        """Stop the drain worker and release the native backends.

        Closing the backends is the part that was missing. ``close()`` on the
        LanceDB backend existed and had no caller anywhere in ``src/``, so even
        an orderly shutdown left its tokio runtime resident. Idempotent,
        because shutdown races call this twice.
        """
        try:
            self._drain.stop()
        except Exception:
            logger.debug("drain stop failed", exc_info=True)
        for attr in ("_lancedb", "_cozo"):
            backend = getattr(self, attr, None)
            if backend is None:
                continue
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("%s close failed", attr, exc_info=True)
            setattr(self, attr, None)

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
