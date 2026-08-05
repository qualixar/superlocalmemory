# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Main Memory Engine (Facade).

Thin orchestrator that delegates to extracted pipeline modules:
  - store_pipeline   (store, store_fact_direct, close_session, enrich_fact)
  - recall_pipeline  (recall, adaptive ranking)
  - engine_wiring    (embedder init, encoding init, retrieval init, hooks)

Single entry point for all memory operations.
Profile-scoped. Mode-aware (A/B/C).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT, SLMConfig
from superlocalmemory.core.engine_capabilities import Capabilities, CapabilityError
from superlocalmemory.core.modes import get_capabilities
from superlocalmemory.storage.models import (
    AtomicFact, FactType, MemoryRecord, Mode, RecallResponse,
)

logger = logging.getLogger(__name__)

from superlocalmemory.core.hooks import HookRegistry


def _verify_ingestion_schema(memory_db: Path) -> bool:
    """Verify M018 using an explicitly owned SQLite connection."""
    import sqlite3

    from superlocalmemory.storage.migrations import M018_ingestion_operations

    connection = sqlite3.connect(str(memory_db))
    try:
        return bool(M018_ingestion_operations.verify(connection))
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Workstream D (3.8.4) — warm-guard sync embed helpers
# ---------------------------------------------------------------------------

def _is_remote_embedder(embedder: object) -> bool:
    """Return True if *embedder* makes remote HTTP calls (cloud / OpenAI-compatible).

    Remote embedders (100–400ms round-trip) must never block the store_fast()
    write path synchronously — they stay on the async materializer path.

    Local embedders:
      - EmbeddingService (subprocess, local ONNX/sentence-transformers): has
        ``_config`` with ``is_cloud=False`` and ``is_openai_compatible=False``.
      - OllamaEmbedder (localhost HTTP, ~73ms): has NO ``_config`` attribute.

    Remote embedders:
      - EmbeddingService with ``_config.is_cloud=True`` (Azure, etc.)
      - EmbeddingService with ``_config.is_openai_compatible=True``
    """
    cfg = getattr(embedder, "_config", None)
    if cfg is None:
        return False  # OllamaEmbedder — local
    return bool(
        getattr(cfg, "is_cloud", False)
        or getattr(cfg, "is_openai_compatible", False)
    )


class MemoryEngine:
    """Main orchestrator for the SuperLocalMemory V3 memory system.

    Wires encoding (fact extraction, entity resolution, graph building,
    consolidation) with retrieval (five candidate producers -> RRF fusion,
    reranking, entity-graph post-fusion enhancement) and all supporting
    layers (trust, learning, compliance).

    Usage::

        config = SLMConfig.for_mode(Mode.A)
        engine = MemoryEngine(config)
        engine.store("Alice went to Paris last summer", session_id="s1")
        response = engine.recall("Where did Alice go?")
    """

    def __init__(
        self,
        config: SLMConfig,
        capabilities: Capabilities = Capabilities.FULL,
    ) -> None:
        self._config = config
        self._caps = get_capabilities(config.mode)
        self._capabilities = capabilities
        self._profile_id = config.active_profile
        self._initialized = False

        self._db = None
        self._embedder = None
        self._llm = None
        self._fact_extractor = None
        self._entity_resolver = None
        self._temporal_parser = None
        self._type_router = None
        self._graph_builder = None
        self._consolidator = None
        self._observation_builder = None
        self._scene_builder = None
        self._entropy_gate = None
        self._retrieval_engine = None
        self._trust_scorer = None
        self._ann_index = None
        self._sheaf_checker = None
        self._provenance = None
        self._adaptive_learner = None
        self._compliance_checker = None
        self._vector_store = None
        self._access_log = None
        self._context_generator = None
        self._temporal_validator = None
        self._auto_invoker = None
        self._auto_linker = None
        self._graph_analyzer = None
        self._consolidation_engine = None
        self._maintenance_scheduler = None
        self._hooks = HookRegistry()
        # Workstream D (3.8.4): single-worker pool reused across store_fast() calls.
        # Lazy-created on first warm-guard attempt; avoids per-call thread churn.
        self._store_fast_embed_pool: object | None = None
        # Lock guards the lazy-init to prevent TOCTOU race on concurrent first calls.
        self._store_fast_embed_pool_lock = threading.Lock()

    # -- Public properties (Phase 2+ access) --------------------------------

    @property
    def db(self):
        """Database manager (read-only access for Phase 2+)."""
        return self._db

    @property
    def trust_scorer(self):
        """Trust scorer (read-only access for Phase 2+)."""
        return self._trust_scorer

    @property
    def embedder(self):
        """Embedding service (read-only access for Phase 2+)."""
        return self._embedder

    @property
    def capabilities(self) -> Capabilities:
        """Capability level chosen at construction (LIGHT or FULL)."""
        return self._capabilities

    # -- Initialization -----------------------------------------------------

    def initialize(self) -> None:
        """Initialize all components. Call once before use.

        In LIGHT mode only the DB layer is initialized (SQLite + schema
        migrations + profile bookkeeping). In FULL mode the heavy layer
        (embedder, LLM, encoding, retrieval, hooks, consolidation) follows.
        """
        if self._initialized:
            return

        self._init_db_layer()

        if self._capabilities is Capabilities.FULL:
            self._init_heavy_layer()
        else:
            # V3.5.9: LIGHT mode — try to get embedder from running daemon so that
            # memories stored via MCP have real embeddings (fixes PR #30 NULL embedder).
            self._try_init_proxy()

        self._initialized = True
        logger.info(
            "MemoryEngine initialized: mode=%s profile=%s capabilities=%s",
            self._config.mode.value, self._profile_id,
            self._capabilities.value,
        )

        if self._capabilities is Capabilities.FULL:
            # Replay pending async writes only when heavy layer is available.
            self._process_pending_memories()

    def _init_db_layer(self) -> None:
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.database import DatabaseManager

        self._db = DatabaseManager(self._config.db_path)
        self._db.initialize(schema)

        # V3.4.3: Apply "Unified Brain" schema extensions (mesh, entity compilation, ingestion)
        # Idempotent — safe to call on every init. Skips if already applied.
        try:
            from superlocalmemory.storage.schema_v343 import apply_v343_schema
            apply_v343_schema(str(self._db.db_path))
        except Exception as exc:
            logger.warning("V3.4.3 schema migration failed: %s", exc)

        # V3.4.6: Apply "Connected Brain" mesh enhancements (broadcast, project routing, offline queue)
        try:
            from superlocalmemory.storage.schema_v343 import apply_v346_schema
            apply_v346_schema(str(self._db.db_path))
        except Exception as exc:
            logger.warning("V3.4.6 schema migration failed: %s", exc)

        # V3.4.7: Apply "Learning Brain" schema before deferred migrations.
        # M029 adds composite history indexes to these runtime tables, so the
        # tables must exist before the migration runner records M029 complete.
        try:
            from superlocalmemory.storage.schema_v347 import apply_v347_schema
            apply_v347_schema(str(self._db.db_path))
        except Exception as exc:
            logger.warning("V3.4.7 schema migration failed: %s", exc)

        # v3.6.15: apply ALL pending migrations — including DEFERRED ones like
        # M016 (scope/shared_with columns) — for DIRECT-engine usage: `slm
        # remember --sync`, the Python API, and LangChain/CrewAI integrations.
        # Previously only the daemon lifespan ran apply_deferred, so an existing
        # pre-3.6.15 database used WITHOUT the daemon hit
        # "table memories has no column named scope" on the first scoped write.
        # Idempotent (skips applied migrations). M018 is now a hard runtime
        # prerequisite: allowing initialization to continue without it would
        # make every advertised write path fail after startup.
        try:
            from superlocalmemory.storage.migration_runner import (
                apply_all, apply_deferred,
            )
            _base = self._config.base_dir
            _learning_db = _base / "learning.db"
            _memory_db = self._db.db_path
            # Forward migrations extend the historical learning schema. A
            # fresh explicit data root has no learning.db yet, so bootstrap
            # those base tables before applying M001/M002/M009.
            from superlocalmemory.learning.database import LearningDatabase
            LearningDatabase(_learning_db)
            apply_all(_learning_db, _memory_db)
            apply_deferred(_learning_db, _memory_db)
            if not _verify_ingestion_schema(_memory_db):
                raise RuntimeError(
                    "required M018 canonical-ingestion schema is unavailable"
                )
        except Exception as exc:
            raise RuntimeError(
                "MemoryEngine initialization stopped because the required "
                f"ingestion migration failed: {exc}"
            ) from exc

        # V3.4.10: Apply "Fortress" schema (backup_destinations, entity_blacklist)
        try:
            from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
            apply_v3410_schema(str(self._db.db_path))
        except Exception as exc:
            logger.warning("V3.4.10 schema migration failed: %s", exc)

        # V3.4.11: Apply "Scale-Ready" schema (pinned_facts, backend_status, fact_consolidations)
        try:
            from superlocalmemory.storage.schema_v3411 import apply_v3411_schema
            apply_v3411_schema(str(self._db.db_path))
        except Exception as exc:
            logger.warning("V3.4.11 schema migration failed: %s", exc)

        # DB-only learner — no embedder / LLM dependency. Available in
        # LIGHT so MCP report_feedback and session_init phase counters
        # work on the MCP process without loading the heavy layer.
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        self._adaptive_learner = AdaptiveLearner(self._db)

    def _try_init_proxy(self) -> None:
        """V3.5.9: Attach McpEmbedderProxy when running in LIGHT mode.

        If the daemon is reachable, the proxy delegates embed calls over HTTP
        so that MCP-stored facts have real embeddings (fixes PR #30). Silently
        skips if the daemon is not running — keyword recall still works.
        """
        try:
            from superlocalmemory.core.mcp_embedder_proxy import McpEmbedderProxy
            port = getattr(self._config, "daemon_port", 8765)
            proxy = McpEmbedderProxy(port=port)
            if proxy.is_available():
                self._embedder = proxy
                logger.info("MCP embedder proxy attached (daemon port %d)", port)
            else:
                logger.debug("Daemon not reachable — MCP will run without embedder")
        except Exception as exc:
            logger.debug("McpEmbedderProxy init skipped: %s", exc)

    def _init_heavy_layer(self) -> None:
        from superlocalmemory.llm.backbone import LLMBackbone
        from superlocalmemory.core.engine_wiring import (
            init_embedder, init_encoding, init_retrieval, wire_hooks,
            _init_auto_invoker, _init_consolidation,
        )

        self._embedder = init_embedder(self._config)

        # Rebuild the complete vector projection before VectorStore opens it.
        # This preserves the previous vec0 table until the shadow activation
        # succeeds, including across embedding-dimension changes.
        self._check_embedding_migration()

        if self._caps.llm_fact_extraction:
            self._llm = LLMBackbone(self._config.llm)
            if not self._llm.is_available():
                logger.warning(
                    "LLM not available. Falling back to Mode A extraction.",
                )
                self._llm = None

        # H-03 (3.7.9): surface mode-capability degradation explicitly at
        # startup. validate_mode_config existed but was never called, so Mode B
        # silently using rule-based extraction after Ollama vanished (update,
        # restart, port conflict) went unwarned. Each mode only checks its own
        # capabilities, so passing the single llm-availability signal is safe.
        from superlocalmemory.core.modes import validate_mode_config
        _llm_up = getattr(self, "_llm", None) is not None
        for _warning in validate_mode_config(
            self._config.mode, has_ollama=_llm_up, has_cloud_llm=_llm_up,
        ):
            logger.warning("Mode config: %s", _warning)

        from superlocalmemory.trust.scorer import TrustScorer
        from superlocalmemory.trust.provenance import ProvenanceTracker
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

        self._trust_scorer = TrustScorer(self._db)

        enc = init_encoding(
            self._config, self._db, self._embedder, self._llm,
        )
        self._ann_index = enc["ann_index"]
        self._fact_extractor = enc["fact_extractor"]
        self._entity_resolver = enc["entity_resolver"]
        self._temporal_parser = enc["temporal_parser"]
        self._type_router = enc["type_router"]
        self._graph_builder = enc["graph_builder"]
        self._consolidator = enc["consolidator"]
        self._observation_builder = enc["observation_builder"]
        self._scene_builder = enc["scene_builder"]
        self._entropy_gate = enc["entropy_gate"]
        self._sheaf_checker = enc["sheaf_checker"]
        self._vector_store = enc.get("vector_store")
        self._access_log = enc.get("access_log")
        self._context_generator = enc.get("context_generator")
        self._temporal_validator = enc.get("temporal_validator")
        self._auto_linker = enc.get("auto_linker")
        self._graph_analyzer = enc.get("graph_analyzer")

        self._retrieval_engine = init_retrieval(
            self._config, self._db, self._embedder,
            self._entity_resolver, self._trust_scorer,
            vector_store=self._vector_store,
        )

        self._provenance = ProvenanceTracker(self._db)
        # self._adaptive_learner is initialized in _init_db_layer() because
        # it depends only on the DB (no embedder / LLM); see note there.
        self._compliance_checker = EUAIActChecker()

        hook_result = wire_hooks(
            self._hooks, self._config, self._db,
            self._trust_scorer, self._profile_id,
        )
        self._signal_recorder = hook_result["signal_recorder"]
        self._audit_chain = hook_result["audit_chain"]

        # V3.2: AutoInvoker (Phase 2) -- multi-signal auto-recall
        self._auto_invoker = _init_auto_invoker(
            self._config, self._db, self._vector_store,
            self._trust_scorer, self._embedder,
        )

        # V3.2: ConsolidationEngine (Phase 5) -- sleep-time consolidation
        from superlocalmemory.core.summarizer import Summarizer
        summarizer = Summarizer(self._config)
        # P1-5 (core-promotion-01): wire a real behavioral store. Previously
        # hardcoded None → _compile_behavioral_block always returned the
        # "No behavioral patterns detected yet." placeholder, so behavioral
        # patterns never reached the always-injected core block (dead feature).
        from superlocalmemory.core.recall_pipeline import _get_behavioral_tracker
        self._consolidation_engine = _init_consolidation(
            self._config, self._db,
            auto_linker=self._auto_linker,
            graph_analyzer=self._graph_analyzer,
            temporal_validator=self._temporal_validator,
            summarizer=summarizer,
            behavioral_store=_get_behavioral_tracker(self._db),
            embedder=self._embedder,  # v3.4.7: for CCQ worker
            llm=getattr(self, "_llm", None),  # v3.4.7: for CCQ worker
        )

        # Lifecycle/tier evaluation, bounded housekeeping, and backup checks
        # must continue even when optional forgetting/math maintenance is
        # disabled. The scheduler itself gates those optional calculations.
        try:
            from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
            self._maintenance_scheduler = MaintenanceScheduler(
                self._db, self._config, self._profile_id,
                embedder=self._embedder,  # v3.8.2: periodic NULL-embedding self-heal
            )
            self._maintenance_scheduler.start()
        except Exception as exc:
            logger.debug("Maintenance scheduler init failed: %s", exc)

    def _process_pending_memories(self) -> None:
        """Process pending memories from store-first async pattern.

        Called on initialize(). If pending.db doesn't exist or has no items,
        returns immediately (~0ms). If items exist, processes them through the
        normal store() pipeline and marks them done/failed.
        """
        try:
            from superlocalmemory.cli.pending_store import (
                get_pending, mark_done, mark_failed,
            )
        except ImportError:
            return

        base_dir = self._config.base_dir
        # Only drain items enqueued under THIS engine's profile — a queued
        # memory must never materialize under a profile it was not written for.
        pending = get_pending(base_dir, limit=20, profile_id=self.profile_id)
        if not pending:
            return

        logger.info("Processing %d pending memories from async store", len(pending))
        for item in pending:
            try:
                # v3.6.15 multi-scope: the pending row carries a metadata JSON
                # blob that may hold scope/shared_with (written by the async
                # /remember path). Replay them so a queued ``--scope global``
                # write lands as global, not silently downgraded to personal.
                import json as _json
                meta = item.get("metadata")
                if isinstance(meta, str):
                    try:
                        meta = _json.loads(meta) if meta else {}
                    except (ValueError, TypeError):
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                _scope = meta.pop("scope", None) or "personal"
                _shared = meta.pop("shared_with", None)
                _source_type = str(
                    meta.pop("_slm_source_type", "legacy-pending")
                )
                _idempotency_key = str(
                    meta.pop("_slm_idempotency_key", f"pending:{item['id']}")
                )
                if item.get("tags"):
                    meta.setdefault("tags", item["tags"])
                from superlocalmemory.core.engine_ingestion import (
                    canonical_store,
                    local_trusted_actor_id,
                )
                canonical_store(
                    self,
                    item["content"],
                    source_type=_source_type,
                    trusted_actor_id=local_trusted_actor_id("engine-startup"),
                    metadata=meta or None,
                    scope=_scope,
                    shared_with=_shared,
                    session_id=str(meta.get("session_id") or ""),
                    idempotency_key=_idempotency_key,
                )
                mark_done(item["id"], base_dir)
            except Exception as exc:
                logger.warning("Pending memory %d failed: %s", item["id"], exc)
                mark_failed(item["id"], str(exc), base_dir)

    # -- Store operations ---------------------------------------------------

    def store(
        self,
        content: str,
        session_id: str = "",
        session_date: str | None = None,
        speaker: str = "",
        role: str = "user",
        metadata: dict[str, Any] | None = None,
        *,
        scope: str = "personal",
        shared_with: list[str] | None = None,
    ) -> list[str]:
        """Store content and extract structured facts. Returns fact_ids.

        Multi-scope: ``scope`` sets the visibility (personal/shared/global).
        ``shared_with`` is a list of profile_ids for shared scope.
        """
        self._require_full("store")
        self._ensure_init()

        from superlocalmemory.core.engine_ingestion import (
            canonical_store,
            local_trusted_actor_id,
        )
        return canonical_store(
            self,
            content,
            source_type="python-api",
            trusted_actor_id=local_trusted_actor_id("python-api"),
            metadata=metadata,
            scope=scope,
            shared_with=shared_with,
            session_id=session_id,
            session_date=session_date,
            speaker=speaker,
            role=role,
            require_complete=False,
        )

    def store_fact_direct(self, fact: AtomicFact) -> str:
        """Durably store a pre-built fact with full enrichment."""
        self._require_full("store_fact_direct")
        self._ensure_init()

        from superlocalmemory.core.engine_ingestion import (
            canonical_store_fact,
            local_trusted_actor_id,
        )
        return canonical_store_fact(
            self,
            fact,
            trusted_actor_id=local_trusted_actor_id("python-api-prebuilt"),
        )

    def store_fast(
        self, content: str, metadata: dict[str, Any] | None = None,
        *, scope: str = "personal", shared_with: list[str] | None = None,
        session_date: str | None = None, speaker: str = "", role: str = "user",
        index_external: bool = True,
    ) -> list[str]:
        """v3.5.5 WRITE-THROUGH: synchronous verbatim insert for IMMEDIATE recall.

        Full ``store()`` blocks 30-180s on LLM fact-extraction + Ollama embedding
        + graph building. That created a recall window: a memory stored via the
        async path sat in pending.db, unrecallable, until the background
        materializer caught up. An agent storing a decision then immediately
        recalling it (same session, or a parallel/next session) would miss it.

        store_fast inserts a verbatim AtomicFact (+ memory row) synchronously.
        The FTS5 ``atomic_facts_fts`` trigger auto-populates on INSERT, so the
        memory is **keyword/BM25-recallable the instant this returns** (~ms, no
        LLM, no embedding). Embedding + entities + graph are enriched async by
        the materializer (which detects facts with NULL embedding).

        Returns real fact_ids immediately. Quality gate rejects template junk.
        """
        self._require_full("store_fast")
        self._ensure_init()
        import re as _re
        import uuid as _uuid
        from datetime import datetime, timezone
        from superlocalmemory.core.engine_ingestion import content_passes_admission
        if not content_passes_admission(content):
            return []
        # v3.6.6 ingest gate: reject 1MB monsters + prompt-template pollution;
        # clamp the searchable FACT copy (head+tail) while the memories row keeps
        # the FULL original. Embedding/BM25 use the clamped copy so a 167KB paste
        # never produces a garbage vector. Env kill-switch: SLM_INGEST_NO_GATE=1.
        fact_text = content
        try:
            from superlocalmemory.core.ingest_gate import apply_ingest_gate
            _sc = getattr(self._config, "store", None)
            gate = apply_ingest_gate(
                content,
                max_verbatim_chars=getattr(_sc, "max_verbatim_chars", 24000),
                max_ingest_bytes=getattr(_sc, "max_ingest_bytes", 1_048_576),
            )
            if gate.rejected:
                logger.debug("store_fast ingest gate rejected: %s", gate.rejection_reason)
                return []
            fact_text = gate.fact_content
        except ImportError:
            pass  # gate module missing → store verbatim (never block a write)
        now = datetime.now(timezone.utc).isoformat()
        record = MemoryRecord(
            profile_id=self._profile_id, content=content,
            session_date=session_date or now[:10],
            session_id=(metadata or {}).get("session_id", ""),
            speaker=speaker,
            role=role,
            metadata=metadata or {},
            scope=scope, shared_with=shared_with,
        )
        self._db.store_memory(record)
        # Lightweight regex entities (matches store_pipeline verbatim path) so
        # the entity_graph channel has something to work with before enrichment.
        ents = sorted(
            {m.group(1) for m in _re.finditer(
                r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3})\b", fact_text)}
            | {m.group(1) for m in _re.finditer(r"\b([A-Z]{2,})\b", fact_text)}
        )
        # Workstream D (3.8.4) — warm-guard synchronous embed.
        #
        # Original contract (3.8.2): queryable admission NEVER acquires the
        # embedding worker lock. On a clean Mode-A install the background model
        # load owns that lock for up to 180s, turning the receipt-first path into
        # a hidden synchronous wait. The canonical materializer promotes this
        # fact with its full pipeline (embedding, Fisher, entities, graph edges).
        #
        # 3.8.4 extension: when the embedder is PROVABLY warm (_available is True)
        # AND is a local embedder (not a remote cloud/OpenAI endpoint), compute the
        # embedding synchronously with a hard 500ms cap. On timeout or any
        # exception, fall through to emb=None — the materializer fills it async.
        # This preserves the 3.8.2 invariant for cold start while eliminating the
        # semantic-channel blind spot on warm daemons (the top UX complaint).
        emb = None
        fmean = fvar = None
        _embedder_ref = self._embedder
        if (
            _embedder_ref is not None
            and getattr(_embedder_ref, "_available", None) is True
            and not _is_remote_embedder(_embedder_ref)
        ):
            import concurrent.futures as _cf
            # Lazy-init the pool once per engine instance — avoids per-call
            # thread churn and the associated resource leak from discard-on-exit.
            # Double-checked locking guards against TOCTOU on concurrent first calls.
            if self._store_fast_embed_pool is None:
                with self._store_fast_embed_pool_lock:
                    if self._store_fast_embed_pool is None:
                        self._store_fast_embed_pool = _cf.ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="slm-sg-embed",
                        )
            try:
                _timeout_s = int(os.environ.get("SLM_STORE_FAST_EMBED_TIMEOUT_MS", 500)) / 1000.0
            except (ValueError, TypeError):
                _timeout_s = 0.5  # default 500 ms
            try:
                def _best_effort_embed():
                    from superlocalmemory.core.recall_gate import background_work
                    with background_work():
                        return _embedder_ref.embed(fact_text)

                _future = self._store_fast_embed_pool.submit(_best_effort_embed)
                try:
                    emb = _future.result(timeout=_timeout_s)
                    if emb:
                        fmean, fvar = _embedder_ref.compute_fisher_params(emb)
                except _cf.TimeoutError:
                    logger.debug(
                        "store_fast: warm-guard embed timed out (>%.0fms) — deferring to materializer",
                        _timeout_s * 1000,
                    )
                    emb = None
                except Exception as _exc:
                    logger.debug(
                        "store_fast: warm-guard embed failed (%s) — deferring to materializer",
                        _exc,
                    )
                    emb = None
            except Exception as _exc:
                logger.debug("store_fast: warm-guard pool submit failed (%s)", _exc)
                emb = None
        fact = AtomicFact(
            fact_id=_uuid.uuid4().hex[:16], memory_id=record.memory_id,
            profile_id=self._profile_id, content=fact_text,
            fact_type=FactType.EPISODIC, entities=ents,
            observation_date=session_date or now[:10],
            confidence=0.7, importance=0.5,
            embedding=emb, fisher_mean=fmean, fisher_variance=fvar,
            created_at=now,
            scope=scope, shared_with=shared_with,
        )
        self._db.store_fact(fact)  # FTS5 trigger → immediately BM25-recallable
        # Upsert to vector store so the semantic channel finds it now.
        if index_external:
            try:
                vs = getattr(self, "_vector_store", None)
                if emb and vs and getattr(vs, "available", False):
                    vs.upsert(fact.fact_id, self._profile_id, emb)
            except Exception:
                pass
        # Persist BM25 tokens too (covers the in-memory rank_bm25 fallback path).
        if index_external:
            try:
                bm25 = getattr(self._retrieval_engine, "_bm25", None)
                if bm25:
                    bm25.add(fact.fact_id, fact_text, self._profile_id)
            except Exception:
                pass
        return [fact.fact_id]

    # -- Recall operations --------------------------------------------------

    def recall(
        self, query: str, profile_id: str | None = None,
        mode: Mode | None = None, limit: int = CANONICAL_RECALL_LIMIT,
        agent_id: str = "unknown",
        session_id: str | None = None,
        fast: bool | None = None,
        *,
        include_global: bool | None = None,
        include_shared: bool | None = None,
        window: str | tuple[str, str] | None = None,
        as_of: str | None = None,
    ) -> RecallResponse:
        """Recall relevant facts for a query.

        S9-DASH-02: when ``session_id`` is provided, the recall is
        non-blockingly enqueued to the outcome queue so downstream
        hooks (PostToolUse, Stop) can attach engagement signals.
        Zero additional latency on the hot path — enqueue is a
        ``put_nowait`` and the actual ``pending_outcomes`` INSERT runs
        on a background worker.

        ``fast`` controls only the internal agentic verification round; all six
        local retrieval channels + reranker run regardless. ``fast=None`` (the
        default) resolves to the client-driven-agentic policy: the agent hot
        path skips the internal round (equivalent to ``fast=True``) and lets the
        calling LLM drive query refinement using the returned confidence signals.
        Pass ``fast=False`` to force the internal agentic round when no smart
        client sits in front of SLM. See ``recall_pipeline.resolve_hot_path_fast``.

        Multi-scope: ``include_global`` / ``include_shared`` control which
        scopes participate in retrieval. ``None`` (the default) means "use the
        configured ScopeConfig default", which ships OFF — shared memory is
        opt-in (v3.6.15). Personal facts are ALWAYS returned regardless, so a
        config of False reproduces 3.6.14 pure-isolation behaviour exactly.
        This is the single policy chokepoint: every recall path (CLI, MCP,
        daemon HTTP, in-process adapter) flows through here, so a caller that
        forgets to thread the flag still gets the safe configured default.
        """
        self._require_full("recall")
        self._ensure_init()

        # Resolve None → configured ScopeConfig default (shared-off by default).
        _scope_cfg = getattr(self._config, "scope", None)
        if include_global is None:
            include_global = bool(getattr(_scope_cfg, "recall_include_global", False))
        if include_shared is None:
            include_shared = bool(getattr(_scope_cfg, "recall_include_shared", False))

        pid = profile_id or self._profile_id

        from superlocalmemory.core.recall_pipeline import run_recall
        try:
            response = run_recall(
                query, pid, mode=mode, limit=limit, agent_id=agent_id,
                config=self._config,
                retrieval_engine=self._retrieval_engine,
                trust_scorer=self._trust_scorer,
                embedder=self._embedder,
                db=self._db, llm=self._llm,
                hooks=self._hooks,
                access_log=self._access_log,
                auto_linker=self._auto_linker,
                fast=fast,
                include_global=include_global,
                include_shared=include_shared,
                window=window,
                as_of=as_of,
            )
        except Exception:
            # Diagnostics are intentionally not recorded here.  A recall is a
            # read command; diagnostics, outcomes, and implicit feedback must
            # be submitted through explicit write commands.
            raise

        return response

    # -- Session operations -------------------------------------------------

    def create_speaker_entities(
        self, speaker_a: str, speaker_b: str,
    ) -> None:
        """Pre-create canonical entities for conversation speakers."""
        self._require_full("create_speaker_entities")
        self._ensure_init()
        if self._entity_resolver:
            self._entity_resolver.create_speaker_entities(
                speaker_a, speaker_b, self._profile_id,
            )

    def close_session(self, session_id: str) -> int:
        """Create session-level temporal summary."""
        self._ensure_init()

        from superlocalmemory.core.store_pipeline import run_close_session
        return run_close_session(
            session_id, self._profile_id, db=self._db,
        )

    # -- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release engine-owned resources without waiting for model workers.

        Daemon shutdown must be a bounded operation.  In particular, a
        ``store_fast`` embed submitted just before shutdown may be blocked in a
        model runtime forever; waiting for its executor here used to make the
        service manager SIGKILL the daemon and leave its children behind.
        References are cleared before invoking each cleanup hook, so a second
        close is safe even when one optional cleanup hook fails.
        """
        scheduler = getattr(self, "_maintenance_scheduler", None)
        self._maintenance_scheduler = None
        if scheduler is not None:
            try:
                scheduler.stop()
            except Exception:
                logger.warning("engine cleanup: maintenance scheduler stop failed", exc_info=True)

        embed_pool = getattr(self, "_store_fast_embed_pool", None)
        embed_pool_lock = getattr(self, "_store_fast_embed_pool_lock", None)
        if embed_pool_lock is not None:
            with embed_pool_lock:
                embed_pool, self._store_fast_embed_pool = self._store_fast_embed_pool, None
        else:
            self._store_fast_embed_pool = None
        if embed_pool is not None:
            try:
                embed_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.warning(
                    "engine cleanup: store-fast embed pool shutdown failed",
                    exc_info=True,
                )

        retrieval = getattr(self, "_retrieval_engine", None)
        self._retrieval_engine = None
        if retrieval is not None:
            reranker = getattr(retrieval, "_reranker", None)
            try:
                if reranker is not None:
                    shutdown = getattr(reranker, "shutdown", None)
                    if callable(shutdown):
                        shutdown(timeout=1.0)
                    else:
                        unload = getattr(reranker, "unload", None)
                        if callable(unload):
                            unload()
            except Exception:
                logger.warning("engine cleanup: reranker shutdown failed", exc_info=True)
            try:
                retrieval.close(wait=False)
            except TypeError:
                # Compatibility for plugins with the legacy no-argument
                # close hook.  Built-in RetrievalEngine accepts ``wait``.
                try:
                    retrieval.close()
                except Exception:
                    logger.warning(
                        "engine cleanup: legacy retrieval shutdown failed",
                        exc_info=True,
                    )
            except Exception:
                logger.warning("engine cleanup: retrieval executor shutdown failed", exc_info=True)

        embedder = getattr(self, "_embedder", None)
        self._embedder = None
        if embedder is not None:
            try:
                shutdown = getattr(embedder, "shutdown", None)
                if callable(shutdown):
                    shutdown(timeout=1.0)
                else:
                    unload = getattr(embedder, "unload", None)
                    if not callable(unload):
                        unload = None
                if not callable(shutdown) and callable(unload):
                    try:
                        unload(timeout=1.0)
                    except TypeError:
                        # Ollama and third-party embedders may still expose
                        # the legacy no-argument unload hook.
                        unload()
            except Exception:
                logger.warning("engine cleanup: embedder unload failed", exc_info=True)

        db = getattr(self, "_db", None)
        self._db = None
        if db is not None:
            try:
                from superlocalmemory.core.recall_pipeline import release_recall_resources

                release_recall_resources(db)
            except Exception:
                logger.warning("engine cleanup: recall resources release failed", exc_info=True)
            try:
                db.close()
            except Exception:
                logger.warning("engine cleanup: database close failed", exc_info=True)
        self._initialized = False

    @property
    def profile_id(self) -> str:
        return self._profile_id

    @profile_id.setter
    def profile_id(self, value: str) -> None:
        self._profile_id = value

    @property
    def fact_count(self) -> int:
        self._ensure_init()
        return self._db.get_fact_count(self._profile_id)

    # -- Internal -----------------------------------------------------------

    def _check_embedding_migration(self) -> None:
        """Detect embedding model change and re-index if needed."""
        try:
            from superlocalmemory.storage.embedding_migrator import (
                check_embedding_migration,
                run_embedding_migration,
            )
            if check_embedding_migration(self._config):
                count = run_embedding_migration(
                    self._config, self._db, self._embedder,
                )
                if count > 0:
                    logger.info(
                        "Embedding migration: %d facts re-embedded", count,
                    )
        except Exception as exc:
            logger.warning("Embedding migration check failed: %s", exc)

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.initialize()

    def _require_full(self, operation: str) -> None:
        if self._capabilities is not Capabilities.FULL:
            raise CapabilityError(
                f"{operation} requires a FULL MemoryEngine but this instance "
                f"is LIGHT; route through WorkerPool (pool.{operation}) or "
                f"construct MemoryEngine(config, capabilities=Capabilities.FULL)."
            )
