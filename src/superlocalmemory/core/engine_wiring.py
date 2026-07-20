# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Engine wiring — extracted free functions for MemoryEngine initialization.

Direction: engine.py imports this module. This module NEVER imports engine.py.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.hooks import HookRegistry
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


def _log_reranker_warmup_status(reranker: Any) -> None:
    """Record non-blocking reranker warmup state without alarming first-run users."""
    ready = reranker.warmup_sync(timeout=180)
    if ready:
        logger.info("Cross-encoder reranker warm and ready")
        return
    try:
        from superlocalmemory.retrieval.reranker import _is_reranker_worker_alive

        if _is_reranker_worker_alive():
            logger.info(
                "Cross-encoder reranker worker held by another process "
                "(machine-wide singleton — usually the unified daemon); "
                "this process will route reranking through that worker"
            )
            return
    except Exception:
        pass
    logger.info(
        "Cross-encoder reranker did not become ready during background warmup; "
        "recalls use fallback scoring. Run 'slm doctor' for diagnostics."
    )


# ---------------------------------------------------------------------------
# init_embedder  (was MemoryEngine._init_embedder + helpers)
# ---------------------------------------------------------------------------

def _try_ollama_embedder(emb_cfg: Any) -> Any | None:
    """Try to create an OllamaEmbedder. Returns it or None."""
    try:
        from superlocalmemory.core.ollama_embedder import OllamaEmbedder
        emb = OllamaEmbedder(
            model=emb_cfg.ollama_model,
            base_url=emb_cfg.ollama_base_url,
            dimension=emb_cfg.dimension,
        )
        if emb.is_available:
            logger.info("Using Ollama embeddings (%s)", emb_cfg.ollama_model)
            return emb
        logger.warning(
            "Ollama embedder not available (model=%s). Falling back.",
            emb_cfg.ollama_model,
        )
    except Exception as exc:
        logger.warning("OllamaEmbedder init failed: %s", exc)
    return None


def _try_service_embedder(cls: type, emb_cfg: Any) -> Any | None:
    """Try to create an EmbeddingService. Returns it or None."""
    try:
        emb = cls(emb_cfg)
        if emb.is_available:
            return emb
        logger.warning("EmbeddingService not available. BM25-only mode. Run 'slm doctor' to diagnose.")
    except Exception as exc:
        logger.warning("Embeddings unavailable (%s). BM25-only mode. Run 'slm doctor' to diagnose.", exc)
    return None


def init_embedder(config: SLMConfig) -> Any | None:
    """Initialize the best available embedding provider.

    Priority order:
    1. Explicit provider in config (ollama / cloud / sentence-transformers)
    2. Auto-detect: Ollama first (lightweight), then sentence-transformers
       subprocess (NEVER in-process for Mode A/B)
    3. If nothing works -> None (BM25-only mode)

    Memory safety: Mode A/B NEVER load sentence-transformers in-process.
    EmbeddingService uses subprocess isolation — the main process stays
    at ~60MB and never imports torch.
    """
    from superlocalmemory.core.embeddings import EmbeddingService
    from superlocalmemory.storage.models import Mode

    emb_cfg = config.embedding
    provider = emb_cfg.provider

    # v3.4.55: When provider is ollama, use Ollama's embedding API as
    # PRIMARY. The stored vectors were created by Ollama's nomic-embed-text;
    # sentence-transformers' nomic-embed-text-v1.5 produces different vectors.
    # Mixing them degrades semantic recall. ST subprocess is the fallback,
    # not the primary — this also eliminates the 2-3 minute cold-start
    # where the ST model loads from disk (~2.1 GB) on every daemon restart.
    if provider == "ollama":
        result = _try_ollama_embedder(emb_cfg)
        if result is not None:
            logger.info("Using Ollama embeddings (nomic-embed-text, local)")
            return result
        st_emb = _try_service_embedder(EmbeddingService, emb_cfg)
        if st_emb is not None:
            logger.warning("Ollama unavailable; falling back to sentence-transformers subprocess")
            return st_emb
        return None

    # --- V3.4.24: Explicit OpenAI-compatible provider ---
    if provider == "openai" and emb_cfg.is_openai_compatible:
        logger.info(
            "Using OpenAI-compatible embedding endpoint: %s (model=%s, dim=%d)",
            emb_cfg.api_endpoint, emb_cfg.model_name, emb_cfg.dimension,
        )
        return _try_service_embedder(EmbeddingService, emb_cfg)

    # --- Explicit cloud provider ---
    if provider == "cloud" or emb_cfg.is_cloud:
        return _try_service_embedder(EmbeddingService, emb_cfg)

    # --- Explicit sentence-transformers (subprocess-isolated) ---
    if provider == "sentence-transformers":
        return _try_service_embedder(EmbeddingService, emb_cfg)

    # --- Auto-detect: try Ollama first (lightweight, <1s) ---
    ollama_emb = _try_ollama_embedder(emb_cfg)
    if ollama_emb is not None:
        logger.info("Auto-detected Ollama embeddings (fast path)")
        return ollama_emb

    # --- Fallback: sentence-transformers subprocess ---
    # EmbeddingService ALWAYS uses subprocess isolation (see embeddings.py).
    # The main process never imports torch — safe for Mode A/B.
    return _try_service_embedder(EmbeddingService, emb_cfg)


# ---------------------------------------------------------------------------
# init_encoding  (was MemoryEngine._init_encoding)
# ---------------------------------------------------------------------------

def init_encoding(
    config: SLMConfig,
    db: DatabaseManager,
    embedder: Any,
    llm: Any,
) -> dict[str, Any]:
    """Create all encoding components. Returns dict of components."""
    from superlocalmemory.encoding.fact_extractor import FactExtractor
    from superlocalmemory.encoding.entity_resolver import EntityResolver
    from superlocalmemory.encoding.temporal_parser import TemporalParser
    from superlocalmemory.encoding.type_router import TypeRouter
    from superlocalmemory.encoding.graph_builder import GraphBuilder
    from superlocalmemory.encoding.consolidator import MemoryConsolidator
    from superlocalmemory.encoding.observation_builder import ObservationBuilder
    from superlocalmemory.encoding.scene_builder import SceneBuilder
    from superlocalmemory.encoding.entropy_gate import EntropyGate
    from superlocalmemory.retrieval.ann_index import ANNIndex

    ann_index = ANNIndex(dimension=config.embedding.dimension)
    fact_extractor = FactExtractor(
        config=config.encoding, llm=llm,
        embedder=embedder, mode=config.mode,
    )
    entity_resolver = EntityResolver(db, llm)
    temporal_parser = TemporalParser()
    type_router = TypeRouter(
        mode=config.mode, embedder=embedder, llm=llm,
    )
    graph_builder = GraphBuilder(db, ann_index)
    consolidator = MemoryConsolidator(
        db, embedder, llm, config.encoding,
    )
    observation_builder = ObservationBuilder(db)
    scene_builder = SceneBuilder(db, embedder)
    entropy_gate = EntropyGate(
        embedder, config.encoding.entropy_threshold,
    )

    sheaf_checker = None
    if config.math.sheaf_at_encoding:
        from superlocalmemory.math.sheaf import SheafConsistencyChecker
        sheaf_checker = SheafConsistencyChecker(
            db, config.math.sheaf_contradiction_threshold,
        )

    # V3.2: VectorStore (Phase 1) -- sqlite-vec KNN
    vector_store = _init_vector_store(config)

    # V3.2: AccessLog (Phase 1) -- fact access tracking
    access_log = _init_access_log(db)

    # V3.2: ContextGenerator (Phase 2) -- contextual descriptions for facts
    context_generator = _init_context_generator(config, llm)

    # V3.2: TemporalValidator (Phase 4) -- bi-temporal fact invalidation
    temporal_validator = _init_temporal(config, db, sheaf_checker, llm)

    # V3.2: Phase 3 -- Association Graph components
    auto_linker = _init_auto_linker(db, vector_store, context_generator, config)
    graph_analyzer = _init_graph_analyzer(db)

    return {
        "ann_index": ann_index,
        "fact_extractor": fact_extractor,
        "entity_resolver": entity_resolver,
        "temporal_parser": temporal_parser,
        "type_router": type_router,
        "graph_builder": graph_builder,
        "consolidator": consolidator,
        "observation_builder": observation_builder,
        "scene_builder": scene_builder,
        "entropy_gate": entropy_gate,
        "sheaf_checker": sheaf_checker,
        "vector_store": vector_store,
        "access_log": access_log,
        "context_generator": context_generator,
        "temporal_validator": temporal_validator,
        "auto_linker": auto_linker,
        "graph_analyzer": graph_analyzer,
    }


def _init_vector_store(config: SLMConfig) -> Any | None:
    """Create VectorStore if sqlite-vec is available. Returns None on failure."""
    try:
        from superlocalmemory.retrieval.vector_store import (
            VectorStore, VectorStoreConfig,
        )
        vec_config = VectorStoreConfig(
            dimension=config.embedding.dimension,
            enabled=True,
        )
        vs = VectorStore(config.db_path, vec_config)
        if vs.available:
            logger.info("VectorStore initialized (sqlite-vec KNN enabled)")
            return vs
        logger.debug("VectorStore unavailable; using ANNIndex fallback")
    except Exception as exc:
        logger.debug("VectorStore init failed: %s", exc)
    return None


def _init_access_log(db: DatabaseManager) -> Any | None:
    """Create AccessLog for fact access tracking."""
    try:
        from superlocalmemory.storage.access_log import AccessLog
        return AccessLog(db)
    except Exception as exc:
        logger.debug("AccessLog init failed: %s", exc)
        return None


def _init_context_generator(config: SLMConfig, llm: Any) -> Any | None:
    """Create ContextGenerator for Phase 2 contextual descriptions."""
    try:
        from superlocalmemory.encoding.context_generator import ContextGenerator
        if hasattr(config, "auto_invoke") and config.auto_invoke.enabled:
            return ContextGenerator(llm=llm)
        # Still create the generator (rules-only) even when disabled,
        # so store pipeline can generate context when called directly.
        return ContextGenerator(llm=llm)
    except Exception as exc:
        logger.debug("ContextGenerator init failed: %s", exc)
        return None


def _init_temporal(
    config: SLMConfig,
    db: DatabaseManager,
    sheaf_checker: Any,
    llm: Any,
) -> Any | None:
    """Create TemporalValidator for Phase 4 temporal intelligence."""
    if not config.temporal_validator.enabled:
        return None

    try:
        from superlocalmemory.encoding.temporal_validator import TemporalValidator
        from superlocalmemory.trust.scorer import TrustScorer

        trust_scorer = TrustScorer(db)
        tv = TemporalValidator(
            db=db,
            sheaf_checker=sheaf_checker,
            trust_scorer=trust_scorer,
            llm=llm,
            config=config.temporal_validator,
        )
        logger.info("TemporalValidator initialized (mode=%s)", config.temporal_validator.mode)
        return tv
    except Exception as exc:
        logger.debug("TemporalValidator init failed: %s", exc)
        return None


def _init_auto_linker(
    db: DatabaseManager,
    vector_store: Any,
    context_generator: Any,
    config: SLMConfig,
) -> Any | None:
    """Create AutoLinker for Phase 3 association graph."""
    try:
        from superlocalmemory.encoding.auto_linker import AutoLinker
        return AutoLinker(
            db=db,
            vector_store=vector_store,
            context_generator=context_generator,
            config=config,
        )
    except Exception as exc:
        logger.debug("AutoLinker init failed: %s", exc)
        return None


def _init_graph_analyzer(db: DatabaseManager) -> Any | None:
    """Create GraphAnalyzer for Phase 3 structural importance."""
    try:
        from superlocalmemory.core.graph_analyzer import GraphAnalyzer
        return GraphAnalyzer(db=db)
    except Exception as exc:
        logger.debug("GraphAnalyzer init failed: %s", exc)
        return None


def _init_consolidation(
    config: SLMConfig,
    db: DatabaseManager,
    auto_linker: Any,
    graph_analyzer: Any,
    temporal_validator: Any,
    summarizer: Any,
    behavioral_store: Any,
    embedder: Any = None,
    llm: Any = None,
) -> Any | None:
    """Create ConsolidationEngine for Phase 5 sleep-time consolidation."""
    try:
        from superlocalmemory.core.consolidation_engine import ConsolidationEngine

        # v3.4.7: Create CCQWorker so Step 7 (cognitive consolidation) actually runs.
        # Previously, ccq_worker was always None → Step 7 returned {"enabled": False}.
        ccq_worker = None
        try:
            from superlocalmemory.core.config import CCQConfig
            from superlocalmemory.encoding.cognitive_consolidator import CognitiveConsolidator
            from superlocalmemory.learning.consolidation_quantization_worker import CCQWorker
            ccq_config = getattr(config, "ccq", CCQConfig())
            consolidator = CognitiveConsolidator(
                db=db, embedder=embedder, llm=llm, config=ccq_config,
            )
            ccq_worker = CCQWorker(consolidator=consolidator, config=ccq_config)
            logger.info("CCQ worker initialized (enabled=%s)", ccq_config.enabled)
        except Exception as exc:
            logger.debug("CCQ worker init failed (non-fatal): %s", exc)

        return ConsolidationEngine(
            db=db,
            config=config.consolidation,
            summarizer=summarizer,
            behavioral_store=behavioral_store,
            auto_linker=auto_linker,
            graph_analyzer=graph_analyzer,
            temporal_validator=temporal_validator,
            slm_config=config,
            ccq_worker=ccq_worker,
        )
    except Exception as exc:
        logger.debug("ConsolidationEngine init failed: %s", exc)
        return None


def _init_spreading_activation(
    db: DatabaseManager,
    vector_store: Any,
) -> Any | None:
    """Create SpreadingActivation for Phase 3 5th retrieval channel."""
    try:
        from superlocalmemory.retrieval.spreading_activation import (
            SpreadingActivation,
            SpreadingActivationConfig,
        )
        sa_config = SpreadingActivationConfig(
            enabled=True,
            use_pagerank_bias=True,      # v3.4.1: PageRank-weighted propagation
            community_boost=0.15,        # v3.4.1: 15% boost for same-community nodes
        )
        return SpreadingActivation(
            db=db, vector_store=vector_store, config=sa_config,
        )
    except Exception as exc:
        logger.debug("SpreadingActivation init failed: %s", exc)
        return None


def _init_auto_invoker(
    config: SLMConfig,
    db: DatabaseManager,
    vector_store: Any,
    trust_scorer: Any,
    embedder: Any,
) -> Any | None:
    """Create AutoInvoker for Phase 2 multi-signal retrieval."""
    if not hasattr(config, "auto_invoke") or not config.auto_invoke.enabled:
        return None
    try:
        from superlocalmemory.hooks.auto_invoker import AutoInvoker
        return AutoInvoker(
            db=db,
            vector_store=vector_store,
            trust_scorer=trust_scorer,
            embedder=embedder,
            config=config.auto_invoke,
        )
    except Exception as exc:
        logger.debug("AutoInvoker init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# init_retrieval  (was MemoryEngine._init_retrieval)
# ---------------------------------------------------------------------------

def _init_hopfield_channel(
    db: DatabaseManager,
    vector_store: Any,
    config: SLMConfig,
) -> Any | None:
    """Create HopfieldChannel for Phase G 6th retrieval channel."""
    if not config.hopfield.enabled:
        return None
    try:
        from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel
        return HopfieldChannel(
            db=db, vector_store=vector_store, config=config.hopfield,
        )
    except Exception as exc:
        logger.debug("HopfieldChannel init failed: %s", exc)
        return None


def _init_quantization_aware_search(
    vector_store: Any,
    db: DatabaseManager,
    config: SLMConfig,
) -> Any | None:
    """Create QuantizationAwareSearch if quantized store is available.

    Returns None on failure — SemanticChannel falls back to VectorStore KNN.
    Stateless wrapper: zero memory overhead, no workers, no threads.
    """
    if vector_store is None:
        return None
    try:
        from superlocalmemory.retrieval.quantization_aware_search import QuantizationAwareSearch
        from superlocalmemory.storage.quantized_store import QuantizedEmbeddingStore
        from superlocalmemory.math.polar_quant import PolarQuantEncoder
        from superlocalmemory.math.qjl import QJLEncoder

        polar = PolarQuantEncoder(dimension=config.embedding.dimension)
        qjl: QJLEncoder | None = None
        try:
            qjl = QJLEncoder(dimension=config.embedding.dimension)
        except Exception:
            pass  # QJL is optional (HR-07)

        q_store = QuantizedEmbeddingStore(
            db=db, polar=polar, qjl=qjl, config=config.quantization,
        )
        qas = QuantizationAwareSearch(
            vector_store=vector_store,
            quantized_store=q_store,
            config=config.quantization,
        )
        logger.info("QuantizationAwareSearch initialized (3-tier quantization pipeline)")
        return qas
    except Exception as exc:
        logger.debug("QuantizationAwareSearch init failed (non-fatal): %s", exc)
        return None


def init_retrieval(
    config: SLMConfig,
    db: DatabaseManager,
    embedder: Any,
    entity_resolver: Any,
    trust_scorer: Any,
    vector_store: Any = None,
) -> Any:
    """Create the RetrievalEngine — five candidate producers (semantic, BM25,
    temporal, spreading_activation, hopfield) plus the entity graph used for
    post-fusion score enhancement. Returns it."""
    from superlocalmemory.retrieval.engine import RetrievalEngine
    from superlocalmemory.retrieval.semantic_channel import SemanticChannel
    from superlocalmemory.retrieval.bm25_channel import BM25Channel
    from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
    from superlocalmemory.retrieval.temporal_channel import TemporalChannel
    from superlocalmemory.retrieval.reranker import CrossEncoderReranker
    from superlocalmemory.retrieval.profile_channel import ProfileChannel
    from superlocalmemory.retrieval.bridge_discovery import BridgeDiscovery

    # V3.3.19: TurboQuant 3-tier search (stateless, zero memory overhead)
    qas = _init_quantization_aware_search(vector_store, db, config)

    channels: dict = {
        "semantic": SemanticChannel(
            db,
            fisher_temperature=config.math.fisher_temperature,
            embedder=embedder,
            fisher_mode=config.math.fisher_mode,
            vector_store=vector_store,
            quantization_aware_search=qas,
        ),
        "bm25": BM25Channel(db),
        "entity_graph": EntityGraphChannel(db, entity_resolver),
        "temporal": TemporalChannel(db),
    }

    # Phase 3: Register SpreadingActivation as 5th channel
    sa_channel = _init_spreading_activation(db, vector_store)
    if sa_channel is not None:
        channels["spreading_activation"] = sa_channel

    # Phase G: Register Hopfield as 6th channel
    hopfield_channel = _init_hopfield_channel(db, vector_store, config)
    if hopfield_channel is not None:
        channels["hopfield"] = hopfield_channel

    reranker = None
    if config.retrieval.use_cross_encoder:
        reranker = CrossEncoderReranker(
            config.retrieval.cross_encoder_model,
            backend=config.retrieval.cross_encoder_backend,
        )

    profile_ch = ProfileChannel(db)
    bridge = BridgeDiscovery(db)

    engine = RetrievalEngine(
        db=db, config=config.retrieval, channels=channels,
        embedder=embedder, reranker=reranker,
        base_weights=config.channel_weights,
        profile_channel=profile_ch,
        bridge_discovery=bridge,
        trust_scorer=trust_scorer,
    )

    # V3.3.13: Ensure reranker warmup is in progress.
    # The CrossEncoderReranker constructor starts background warmup, but
    # callers can also call warmup_sync() to block until ready.
    # Here we just log warmup status — benchmark scripts call warmup_sync() explicitly.
    #
    # v3.4.42: Distinguish the legitimate "another process owns the
    # reranker worker" case (machine-wide singleton — usually the unified
    # daemon) from a real warmup failure. Before this fix, any CLI process
    # that wired an Engine while the daemon was up would log
    # "reranker warmup failed — recalls will use fallback scoring" even
    # though the daemon's reranker was healthy and serving fine. The
    # warning was a false positive that masked real failures and eroded
    # trust in slm health / slm doctor output.
    if reranker is not None:
        import threading
        t = threading.Thread(
            target=_log_reranker_warmup_status,
            args=(reranker,),
            daemon=True,
            name="ce-init-warmup",
        )
        t.start()

    # Phase A: Register forgetting filter into the channel registry
    try:
        from superlocalmemory.retrieval.forgetting_filter import register_forgetting_filter
        register_forgetting_filter(engine._registry, db, config.forgetting)
    except Exception as exc:
        logger.debug("Forgetting filter registration failed: %s", exc)

    return engine


# ---------------------------------------------------------------------------
# wire_hooks  (was MemoryEngine._wire_hooks)
# ---------------------------------------------------------------------------

def wire_hooks(
    hooks: HookRegistry,
    config: SLMConfig,
    db: DatabaseManager,
    trust_scorer: Any,
    profile_id: str,
) -> dict[str, Any]:
    """Wire trust, compliance, and event bus hooks into engine lifecycle.

    Returns dict of created components (signal_recorder, audit_chain)
    so engine can hold references.
    """
    result: dict[str, Any] = {"signal_recorder": None, "audit_chain": None}

    # -- Pre-store hooks (synchronous, can reject) --
    if trust_scorer:
        from superlocalmemory.trust.gate import TrustGate
        gate = TrustGate(trust_scorer)
        hooks.register_pre("store", lambda ctx: gate.check_write(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id)))
        hooks.register_pre("delete", lambda ctx: gate.check_delete(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id)))
        hooks.register_pre("update", lambda ctx: gate.check_write(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id)))

    # -- Post-store hooks (async, never block) --
    if trust_scorer:
        hooks.register_post("store", lambda ctx: trust_scorer.record_signal(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id), "store_success"))
        hooks.register_post("recall", lambda ctx: trust_scorer.record_signal(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id), "recall_hit"))

    # -- Burst detection via SignalRecorder --
    try:
        from superlocalmemory.trust.signals import SignalRecorder
        signal_recorder = SignalRecorder(db)
        hooks.register_post("store", lambda ctx: signal_recorder.record(
            ctx.get("agent_id", "unknown"), ctx.get("profile_id", profile_id), "store_success"))
        result["signal_recorder"] = signal_recorder
    except Exception:
        pass

    # -- Tamper-proof audit chain (all operations logged with hash chain) --
    try:
        from superlocalmemory.compliance.audit import AuditChain
        audit_path = config.db_path.parent / "audit_chain.db"
        audit_chain = AuditChain(audit_path)
        for op in ("store", "recall", "delete"):
            hooks.register_post(op, lambda ctx, _op=op: audit_chain.log(
                operation=_op,
                agent_id=ctx.get("agent_id", "unknown"),
                profile_id=ctx.get("profile_id", profile_id),
                content_hash=ctx.get("content_hash", ""),
            ))
        result["audit_chain"] = audit_chain
    except Exception:
        pass

    return result
