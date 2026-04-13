# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
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
from typing import Any

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.modes import get_capabilities
from superlocalmemory.storage.models import (
    AtomicFact, MemoryRecord, Mode, RecallResponse,
)

logger = logging.getLogger(__name__)

from superlocalmemory.core.hooks import HookRegistry


class MemoryEngine:
    """Main orchestrator for the SuperLocalMemory V3 memory system.

    Wires encoding (fact extraction, entity resolution, graph building,
    consolidation) with retrieval (4-channel search, RRF fusion,
    reranking) and all supporting layers (trust, learning, compliance).

    Usage::

        config = SLMConfig.for_mode(Mode.A)
        engine = MemoryEngine(config)
        engine.store("Alice went to Paris last summer", session_id="s1")
        response = engine.recall("Where did Alice go?")
    """

    def __init__(self, config: SLMConfig) -> None:
        self._config = config
        self._caps = get_capabilities(config.mode)
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

    # -- Initialization -----------------------------------------------------

    def initialize(self) -> None:
        """Initialize all components. Call once before use."""
        if self._initialized:
            return

        from superlocalmemory.storage import schema
        from superlocalmemory.storage.database import DatabaseManager
        from superlocalmemory.llm.backbone import LLMBackbone
        from superlocalmemory.core.engine_wiring import (
            init_embedder, init_encoding, init_retrieval, wire_hooks,
            _init_auto_invoker, _init_consolidation,
        )

        self._db = DatabaseManager(self._config.db_path)
        self._db.initialize(schema)

        # V3.4.3: Apply "Unified Brain" schema extensions (mesh, entity compilation, ingestion)
        # Idempotent — safe to call on every init. Skips if already applied.
        try:
            from superlocalmemory.storage.schema_v343 import apply_v343_schema
            apply_v343_schema(str(self._db.db_path))
        except Exception as exc:
            logger.debug("V3.4.3 schema migration: %s", exc)

        # V3.4.6: Apply "Connected Brain" mesh enhancements (broadcast, project routing, offline queue)
        try:
            from superlocalmemory.storage.schema_v343 import apply_v346_schema
            apply_v346_schema(str(self._db.db_path))
        except Exception as exc:
            logger.debug("V3.4.6 schema migration: %s", exc)

        self._embedder = init_embedder(self._config)

        if self._caps.llm_fact_extraction:
            self._llm = LLMBackbone(self._config.llm)
            if not self._llm.is_available():
                logger.warning(
                    "LLM not available. Falling back to Mode A extraction.",
                )
                self._llm = None

        from superlocalmemory.trust.scorer import TrustScorer
        from superlocalmemory.trust.provenance import ProvenanceTracker
        from superlocalmemory.learning.adaptive import AdaptiveLearner
        from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

        self._trust_scorer = TrustScorer(self._db)

        # Encoding components
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

        # Retrieval engine
        self._retrieval_engine = init_retrieval(
            self._config, self._db, self._embedder,
            self._entity_resolver, self._trust_scorer,
            vector_store=self._vector_store,
        )

        self._provenance = ProvenanceTracker(self._db)
        self._adaptive_learner = AdaptiveLearner(self._db)
        self._compliance_checker = EUAIActChecker()

        # Wire lifecycle hooks
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
        self._consolidation_engine = _init_consolidation(
            self._config, self._db,
            auto_linker=self._auto_linker,
            graph_analyzer=self._graph_analyzer,
            temporal_validator=self._temporal_validator,
            summarizer=summarizer,
            behavioral_store=None,
        )

        # V3.3: Check for embedding model migration on mode switch
        self._check_embedding_migration()

        # V3.3.13: Background maintenance scheduler (Langevin/Ebbinghaus/Sheaf)
        if self._config.forgetting.enabled:
            try:
                from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
                self._maintenance_scheduler = MaintenanceScheduler(
                    self._db, self._config, self._profile_id,
                )
                self._maintenance_scheduler.start()
            except Exception as exc:
                logger.debug("Maintenance scheduler init failed: %s", exc)

        self._initialized = True
        logger.info(
            "MemoryEngine initialized: mode=%s profile=%s",
            self._config.mode.value, self._profile_id,
        )

        # V3.3.21: Process any pending memories from failed async remember.
        # Zero cost if no pending.db exists. Backward compatible.
        self._process_pending_memories()

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
        pending = get_pending(base_dir, limit=20)
        if not pending:
            return

        logger.info("Processing %d pending memories from async store", len(pending))
        for item in pending:
            try:
                self.store(item["content"])
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
    ) -> list[str]:
        """Store content and extract structured facts. Returns fact_ids."""
        self._ensure_init()

        from superlocalmemory.core.store_pipeline import run_store
        return run_store(
            content, self._profile_id,
            session_id=session_id, session_date=session_date,
            speaker=speaker, role=role, metadata=metadata,
            config=self._config, db=self._db,
            embedder=self._embedder,
            fact_extractor=self._fact_extractor,
            entity_resolver=self._entity_resolver,
            temporal_parser=self._temporal_parser,
            type_router=self._type_router,
            graph_builder=self._graph_builder,
            consolidator=self._consolidator,
            observation_builder=self._observation_builder,
            scene_builder=self._scene_builder,
            entropy_gate=self._entropy_gate,
            ann_index=self._ann_index,
            sheaf_checker=self._sheaf_checker,
            retrieval_engine=self._retrieval_engine,
            provenance=self._provenance,
            hooks=self._hooks,
            vector_store=self._vector_store,
            context_generator=self._context_generator,
            temporal_validator=self._temporal_validator,
            auto_linker=self._auto_linker,
            consolidation_engine=self._consolidation_engine,
        )

    def store_fact_direct(self, fact: AtomicFact) -> str:
        """Store a pre-built fact with full enrichment."""
        self._ensure_init()

        from superlocalmemory.core.store_pipeline import run_store_fact_direct
        return run_store_fact_direct(
            fact, self._profile_id,
            db=self._db, embedder=self._embedder,
            entity_resolver=self._entity_resolver,
            ann_index=self._ann_index,
            graph_builder=self._graph_builder,
            retrieval_engine=self._retrieval_engine,
            vector_store=self._vector_store,
        )

    # -- Recall operations --------------------------------------------------

    def recall(
        self, query: str, profile_id: str | None = None,
        mode: Mode | None = None, limit: int = 20,
        agent_id: str = "unknown",
    ) -> RecallResponse:
        """Recall relevant facts for a query."""
        self._ensure_init()

        pid = profile_id or self._profile_id

        from superlocalmemory.core.recall_pipeline import run_recall
        return run_recall(
            query, pid, mode=mode, limit=limit, agent_id=agent_id,
            config=self._config,
            retrieval_engine=self._retrieval_engine,
            trust_scorer=self._trust_scorer,
            embedder=self._embedder,
            db=self._db, llm=self._llm,
            hooks=self._hooks,
            access_log=self._access_log,
            auto_linker=self._auto_linker,
        )

    # -- Session operations -------------------------------------------------

    def create_speaker_entities(
        self, speaker_a: str, speaker_b: str,
    ) -> None:
        """Pre-create canonical entities for conversation speakers."""
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
        if self._maintenance_scheduler is not None:
            self._maintenance_scheduler.stop()
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
