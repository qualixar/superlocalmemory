# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Configuration.

Unified configuration with Mode A/B/C capability matrix.
Clean — zero dead options, every config has a consumer.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path.home() / ".superlocalmemory"
DEFAULT_DB_NAME = "memory.db"
DEFAULT_PROFILES_FILE = "profiles.json"


# ---------------------------------------------------------------------------
# Embedding Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration per mode."""

    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    dimension: int = 768
    # Provider: "" = auto-detect, "sentence-transformers", "ollama", "cloud"
    provider: str = ""
    # Ollama settings (used when provider="ollama" or auto-detected)
    ollama_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    # Azure / cloud settings (Mode C only)
    api_endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-02-01"
    deployment_name: str = ""

    @property
    def is_cloud(self) -> bool:
        return bool(self.api_endpoint) or self.provider == "cloud"

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"


# ---------------------------------------------------------------------------
# LLM Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration per mode."""

    provider: str = ""             # "" = no LLM, "ollama", "azure", "openai", "anthropic"
    model: str = ""                # Model name/deployment
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0       # Deterministic by default
    max_tokens: int = 4096
    timeout_seconds: float = 60.0

    @property
    def is_available(self) -> bool:
        return bool(self.provider)


# ---------------------------------------------------------------------------
# Channel Weights
# ---------------------------------------------------------------------------

@dataclass
class ChannelWeights:
    """Retrieval channel weights — 5 channels, query-adaptive."""

    # Entity-linked facts are high-precision matches that rank above BM25.
    semantic: float = 1.2
    bm25: float = 1.0
    entity_graph: float = 1.3
    temporal: float = 1.0
    spreading_activation: float = 1.0  # Phase 3: 5th channel (BC-08: default value)

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "bm25": self.bm25,
            "entity_graph": self.entity_graph,
            "temporal": self.temporal,
            "spreading_activation": self.spreading_activation,
        }


# ---------------------------------------------------------------------------
# Encoding Config
# ---------------------------------------------------------------------------

@dataclass
class EncodingConfig:
    """Configuration for the encoding (memory creation) pipeline."""

    # Fact extraction
    chunk_size: int = 10           # Conversation turns per extraction chunk
    max_facts_per_chunk: int = 5   # Max facts extracted per chunk
    min_fact_confidence: float = 0.3

    # Entity resolution
    entity_similarity_threshold: float = 0.85
    max_entity_candidates: int = 10

    # Graph construction
    semantic_edge_top_k: int = 5   # Top-K semantic edges per new fact
    temporal_edge_window_hours: int = 168  # 1 week

    # Consolidation
    consolidation_similarity_threshold: float = 0.85
    max_consolidation_candidates: int = 5

    # Entropy gate
    entropy_threshold: float = 0.95


# ---------------------------------------------------------------------------
# Retrieval Config
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """Configuration for the retrieval (recall) pipeline."""

    # Fusion
    rrf_k: int = 60               # RRF smoothing constant (D116: k=60 for diversity)
    top_k: int = 20               # Final results to return

    # Per-channel
    semantic_top_k: int = 50      # ANN pre-filter candidates
    bm25_top_k: int = 50
    entity_graph_max_hops: int = 3
    temporal_proximity_days: int = 30

    # Reranking
    use_cross_encoder: bool = True
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"

    # Agentic (Mode C only)
    agentic_max_rounds: int = 3
    agentic_confidence_threshold: float = 0.3

    # Spreading activation
    spreading_activation_decay: float = 0.7
    spreading_activation_threshold: float = 0.1

    # Trust weighting — apply Bayesian trust scores to retrieval ranking.
    # When enabled, each fact's score is multiplied by a trust weight in [0.5, 1.5].
    # Low-trust facts are demoted; high-trust facts are promoted.
    # Default trust = 1.0 (no effect when no trust data exists).
    use_trust_weighting: bool = True

    # Ablation channel control for experiments.
    # List of channel names to SKIP during retrieval (e.g., ["bm25", "entity_graph"]).
    # Used by s19_runner for ablation experiments. Empty = all channels active.
    disabled_channels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Math Config
# ---------------------------------------------------------------------------

@dataclass
class MathConfig:
    """Configuration for mathematical layers."""

    # Fisher-Rao
    fisher_temperature: float = 15.0
    fisher_bayesian_update: bool = True
    # "simplified" = local Mahalanobis-like (fast, existing behaviour)
    # "full"       = Atkinson-Mitchell geodesic from FisherRaoMetric class
    fisher_mode: str = "simplified"

    # Langevin
    langevin_dt: float = 0.005
    langevin_temperature: float = 0.3
    langevin_persist_positions: bool = True
    langevin_weight_range: tuple[float, float] = (0.0, 1.0)

    # Hopfield

    # Sheaf (at encoding time, NOT retrieval)
    sheaf_at_encoding: bool = True
    sheaf_contradiction_threshold: float = 0.45
    # Max edges to check per fact during sheaf consistency.
    # At 18K+ edges, coboundary computation becomes O(N*dim^2) and hangs.
    # Facts with more edges than this skip sheaf check (still get contradiction
    # detection via consolidator UPDATE/SUPERSEDE path).
    sheaf_max_edges_per_check: int = 200

    # Rate-Distortion (production only, disabled for benchmarks)


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsolidationConfig:
    """Configuration for sleep-time consolidation (Phase 5).

    Feature-flagged: starts enabled=False (Rule 12).
    """

    enabled: bool = False                   # Feature flag (Rule 12)
    step_count_trigger: int = 50            # Lightweight consolidation every N stores (L7)
    session_trigger: bool = True            # Run on session end
    idle_timeout_seconds: int = 300         # 5 min inactivity
    scheduled_sessions: int = 5             # Full consolidation every N sessions
    core_memory_char_limit: int = 2000      # Total chars across all blocks
    block_char_limit: int = 500             # Per-block character limit
    compression_similarity: float = 0.85    # Dedup threshold for compression
    promotion_min_access: int = 3           # Min access count for promotion
    promotion_min_trust: float = 0.5        # Min trust for promotion
    decay_days_threshold: int = 30          # Edge decay after N days


@dataclass(frozen=True)
class TemporalValidatorConfig:
    """Configuration for temporal intelligence (Phase 4).

    Feature-flagged: starts enabled=False (Rule 12).
    """

    enabled: bool = False
    mode: str = "a"                              # "a" (sheaf), "b"/"c" (LLM)

    # Sheaf contradiction threshold
    contradiction_threshold: float = 0.45        # Mode A threshold (768d)

    # LLM pre-filter threshold (lower to catch more candidates)
    llm_prefilter_threshold: float = 0.30

    # Max LLM checks per new fact (cost control)
    max_llm_checks: int = 5

    # Trust penalty for expired facts
    expiration_trust_penalty: float = -0.2

    # Include expired facts in historical queries
    include_expired_in_history: bool = True


@dataclass(frozen=True)
class AutoInvokeConfig:
    """Configuration for the Auto-Invoke Engine (Phase 2).

    Feature-flagged: starts enabled=False, graduated to default on
    after MRR validation passes (Rule 12).

    References:
      - SYNAPSE: FOK gating (fok_threshold = 0.12)
      - ACT-R: base-level activation (act_r_decay = 0.5)
      - Zep/Hindsight: multi-signal ranking consensus
    """

    enabled: bool = False                      # Feature flag (Rule 12)
    profile_id: str = "default"

    # Scoring weights (4-signal default) -- must sum to 1.0
    weights: dict = field(default_factory=lambda: {
        "similarity": 0.40,
        "recency": 0.25,
        "frequency": 0.20,
        "trust": 0.15,
    })

    # ACT-R mode (3-signal alternative) -- must sum to 1.0
    use_act_r: bool = False
    act_r_weights: dict = field(default_factory=lambda: {
        "similarity": 0.40,
        "base_level": 0.35,
        "trust": 0.25,
    })
    act_r_decay: float = 0.5                   # Power-law decay exponent

    # FOK gating (Feeling-of-Knowing)
    fok_threshold: float = 0.12                # SYNAPSE minimum score gate

    # Retrieval limits
    max_memories_injected: int = 10
    candidate_multiplier: int = 3              # candidates = limit * multiplier

    # Mode A degradation weights -- must sum to 1.0
    mode_a_weights: dict = field(default_factory=lambda: {
        "similarity": 0.00,
        "recency": 0.40,
        "frequency": 0.35,
        "trust": 0.25,
    })

    # Behavioral
    include_archived: bool = False
    relevance_threshold: float = 0.3           # Legacy compat with AutoRecall


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class SLMConfig:
    """Master configuration for SuperLocalMemory V3.

    Create via SLMConfig.for_mode(Mode.A) for mode-specific defaults.
    """

    mode: Mode = Mode.A
    base_dir: Path = DEFAULT_BASE_DIR
    db_path: Path | None = None    # Computed from base_dir if None
    active_profile: str = "default"

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    channel_weights: ChannelWeights = field(default_factory=ChannelWeights)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    math: MathConfig = field(default_factory=MathConfig)
    temporal_validator: TemporalValidatorConfig = field(
        default_factory=TemporalValidatorConfig,
    )
    auto_invoke: AutoInvokeConfig = field(default_factory=AutoInvokeConfig)
    consolidation: ConsolidationConfig = field(
        default_factory=ConsolidationConfig,
    )

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.base_dir / DEFAULT_DB_NAME

    @classmethod
    def load(cls, config_path: Path | None = None) -> SLMConfig:
        """Load config from JSON file. Returns default Mode A if file doesn't exist."""
        path = config_path or (DEFAULT_BASE_DIR / "config.json")
        if not path.exists():
            return cls.for_mode(Mode.A)
        import json
        data = json.loads(path.read_text())
        mode = Mode(data.get("mode", "a"))
        llm_data = data.get("llm", {})
        emb_data = data.get("embedding", {})
        config = cls.for_mode(
            mode,
            llm_provider=llm_data.get("provider", ""),
            llm_model=llm_data.get("model", ""),
            llm_api_key=llm_data.get("api_key", ""),
            llm_api_base=llm_data.get("base_url", ""),
            embedding_provider=emb_data.get("provider", ""),
            embedding_endpoint=emb_data.get("api_endpoint", ""),
            embedding_key=emb_data.get("api_key", ""),
            embedding_deployment=emb_data.get("deployment_name", ""),
        )
        config.active_profile = data.get("active_profile", "default")
        return config

    def save(self, config_path: Path | None = None) -> None:
        """Save config to JSON file."""
        import json
        path = config_path or (self.base_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode.value,
            "active_profile": self.active_profile,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "api_key": self.llm.api_key,
                "base_url": self.llm.api_base,
            },
            "embedding": {
                "model_name": self.embedding.model_name,
                "dimension": self.embedding.dimension,
                "provider": self.embedding.provider,
                "api_endpoint": self.embedding.api_endpoint,
                "api_key": self.embedding.api_key,
                "deployment_name": self.embedding.deployment_name,
            },
        }
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def provider_presets() -> dict[str, dict[str, str]]:
        """Provider presets for setup wizard."""
        return {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4.1-mini",
                "embedding_model": "text-embedding-3-large",
                "env_key": "OPENAI_API_KEY",
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "model": "claude-sonnet-4-6",
                "embedding_model": "",
                "env_key": "ANTHROPIC_API_KEY",
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3.2",
                "embedding_model": "nomic-embed-text",
                "env_key": "",
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-4.1-mini",
                "embedding_model": "",
                "env_key": "OPENROUTER_API_KEY",
            },
        }

    @classmethod
    def default(cls) -> SLMConfig:
        """Create default Mode A configuration."""
        return cls.for_mode(Mode.A)

    @classmethod
    def for_mode(
        cls,
        mode: Mode,
        base_dir: Path | None = None,
        *,
        llm_provider: str = "",
        llm_model: str = "",
        llm_api_key: str = "",
        llm_api_base: str = "",
        embedding_provider: str = "",
        embedding_endpoint: str = "",
        embedding_key: str = "",
        embedding_deployment: str = "",
    ) -> SLMConfig:
        """Create config with mode-appropriate defaults."""
        _base = base_dir or DEFAULT_BASE_DIR

        if mode == Mode.A:
            return cls(
                mode=mode,
                base_dir=_base,
                embedding=EmbeddingConfig(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    dimension=768,
                    provider=embedding_provider,
                ),
                llm=LLMConfig(),  # No LLM
                retrieval=RetrievalConfig(
                    use_cross_encoder=True,
                ),
                math=MathConfig(
                    sheaf_contradiction_threshold=0.45,  # 768d threshold
                ),
            )

        if mode == Mode.B:
            return cls(
                mode=mode,
                base_dir=_base,
                embedding=EmbeddingConfig(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    dimension=768,
                    provider=embedding_provider,
                ),
                llm=LLMConfig(
                    provider=llm_provider or "ollama",
                    model=llm_model or "llama3.2",
                    api_base=llm_api_base or "http://localhost:11434",
                    api_key=llm_api_key or "",
                ),
                retrieval=RetrievalConfig(use_cross_encoder=True),
            )

        # Mode C — FULL POWER, UNRESTRICTED
        return cls(
            mode=mode,
            base_dir=_base,
            embedding=EmbeddingConfig(
                model_name="text-embedding-3-large",
                dimension=3072,
                api_endpoint=embedding_endpoint,
                api_key=embedding_key,
                deployment_name=embedding_deployment,
            ),
            llm=LLMConfig(
                provider=llm_provider or "azure",
                model=llm_model or "gpt-4.1-mini",
                api_key=llm_api_key,
                api_base=llm_api_base,
            ),
            channel_weights=ChannelWeights(
                semantic=1.5,
                bm25=1.2,
                entity_graph=1.3,
                temporal=1.0,
                spreading_activation=1.2,  # Phase 3: SA boost in Mode C
            ),
            retrieval=RetrievalConfig(
                use_cross_encoder=True,
                semantic_top_k=80,
                agentic_max_rounds=2,  # EverMemOS 2-round
            ),
            math=MathConfig(
                sheaf_contradiction_threshold=0.65,  # Higher for 3072d embeddings
            ),
        )
