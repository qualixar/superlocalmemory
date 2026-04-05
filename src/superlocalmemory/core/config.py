# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
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

    # Semantic should dominate for conversational retrieval (paraphrase matters most).
    semantic: float = 1.5
    bm25: float = 1.0
    entity_graph: float = 1.0
    temporal: float = 1.0
    spreading_activation: float = 1.0  # Phase 3: 5th channel (BC-08: default value)
    hopfield: float = 0.8  # Phase G: 6th channel (Hopfield associative memory)

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "bm25": self.bm25,
            "entity_graph": self.entity_graph,
            "temporal": self.temporal,
            "spreading_activation": self.spreading_activation,
            "hopfield": self.hopfield,
        }


# ---------------------------------------------------------------------------
# Encoding Config
# ---------------------------------------------------------------------------

@dataclass
class EncodingConfig:
    """Configuration for the encoding (memory creation) pipeline."""

    # Fact extraction
    chunk_size: int = 10           # Conversation turns per extraction chunk
    max_facts_per_chunk: int = 10  # V3.3.11: increased from 5 to preserve more details
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
    rrf_k: int = 15               # RRF smoothing constant (k=15 for candidate pools of 50-200)
    top_k: int = 20               # Final results to return

    # Per-channel
    semantic_top_k: int = 50      # ANN pre-filter candidates
    bm25_top_k: int = 50
    entity_graph_max_hops: int = 3
    temporal_proximity_days: int = 30

    # Reranking (V3.3.2: ONNX backend enabled for all modes)
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    cross_encoder_backend: str = ""  # "" = PyTorch (~500MB stable), "onnx" = ONNX (leaks on ARM64 CoreML)

    # Agentic (Mode C only)
    agentic_max_rounds: int = 3
    agentic_confidence_threshold: float = 0.3

    # Spreading activation
    spreading_activation_decay: float = 0.7
    spreading_activation_threshold: float = 0.1

    # Hopfield (Phase G: 6th channel)
    hopfield_top_k: int = 50

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

    Ships enabled by default. Users can disable via slm config.
    """

    enabled: bool = True
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
class ForgettingConfig:
    """Ebbinghaus forgetting configuration."""

    enabled: bool = True
    # Strength coefficients
    alpha: float = 2.0              # Access frequency weight (log scale)
    beta: float = 1.5               # Importance weight (PageRank)
    gamma: float = 1.0              # Confirmation count weight
    delta: float = 0.5              # Emotional salience weight
    # Strength bounds
    min_strength: float = 0.1       # Floor (prevents instant forgetting)
    max_strength: float = 100.0     # Ceiling (numerical stability)
    # Zone thresholds
    archive_threshold: float = 0.2  # Below this -> ARCHIVE
    forget_threshold: float = 0.05  # Below this -> FORGOTTEN
    # Spaced repetition
    learning_rate: float = 1.0      # eta in spaced repetition update
    # Coupling
    forgetting_drift_scale: float = 0.5  # How strongly forgetting affects Langevin drift
    # Trust-weighted forgetting (Paper 3, Section 5.5)
    trust_kappa: float = 2.0  # Sensitivity: lambda_eff = lambda * (1 + trust_kappa * (1 - tau))
    # Scheduler
    scheduler_interval_minutes: int = 30  # How often to recompute retentions
    # Immunity
    core_memory_immune: bool = True  # Core Memory blocks never forget


@dataclass(frozen=True)
class HopfieldConfig:
    """Modern Continuous Hopfield Network configuration (Ramsauer et al., 2020).

    Energy: E(xi) = -log(sum_i exp(B * xi' * x_i)) + B/2 * ||xi||^2
    Update: xi_new = X' @ softmax(B * X @ xi)
    Beta:   B = 1/sqrt(d) where d = dimension
    Storage capacity: O(e^{d/2}) -- exponential in dimension.
    """

    enabled: bool = True
    dimension: int = 768
    max_iterations: int = 1
    convergence_epsilon: float = 1e-6
    prefilter_threshold: int = 10_000
    prefilter_candidates: int = 1000
    skip_threshold: int = 100_000
    cache_ttl_seconds: float = 60.0


@dataclass(frozen=True)
class ReaperConfig:
    """Process health & stale reaper configuration (Phase H0).

    Prevents zombie SLM processes from exhausting RAM.
    """

    enabled: bool = True
    heartbeat_interval_seconds: int = 60
    orphan_age_threshold_hours: float = 4.0
    pid_file_path: str = ""
    graceful_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class PolarQuantConfig:
    """PolarQuant embedding quantization configuration.

    Random orthogonal rotation + recursive polar + scalar quantization.
    Reference: TurboQuant (ICLR 2026), PolarQuant (arXiv 2502.02617).
    """

    dimension: int = 768
    rotation_matrix_path: str = ""  # empty = ~/.superlocalmemory/polar_rotation.npy
    seed: int = 42                  # reproducible rotation matrix
    codebook_method: str = "turbo"  # "turbo" (default) or "polar_legacy"


@dataclass(frozen=True)
class QJLConfig:
    """QJL 1-bit residual correction configuration.

    Random projection + sign-bit quantization for asymmetric IP estimation.
    Reference: QJL (AAAI 2025, arXiv 2406.03482).
    """

    projection_dim: int = 128
    seed: int = 43  # separate from PolarQuant


@dataclass(frozen=True)
class QuantizationConfig:
    """Memory-aware embedding quantization (EAP + LP2E).

    Couples Ebbinghaus retention to embedding precision.
    """

    enabled: bool = True
    polar: PolarQuantConfig = field(default_factory=PolarQuantConfig)
    qjl: QJLConfig = field(default_factory=QJLConfig)
    default_bit_width: int = 32
    eap_enabled: bool = True
    keep_float32_backup: bool = True
    auto_compact_interval_hours: int = 6
    polar_search_penalty: float = 0.97  # V3.3.8: 0.95→0.97, TurboQuant has lower MSE


@dataclass(frozen=True)
class CCQConfig:
    """Cognitive Consolidation Quantization configuration (Phase E).

    Ships enabled by default. CCQ runs as Step 7 of the consolidation cycle.
    Biological analogy: sleep-time hippocampal-neocortical transfer.
    """

    enabled: bool = True

    # Candidate identification
    retention_threshold: float = 0.5
    max_candidates_per_run: int = 200

    # Clustering
    min_entity_overlap: int = 2
    temporal_window_days: int = 7
    min_cluster_size: int = 3
    max_cluster_size: int = 20

    # Gist extraction
    use_llm_gist: bool = True
    max_gist_chars: int = 500
    min_entity_coverage: float = 0.5

    # Embedding compression
    target_bit_width: int = 2
    compress_embeddings: bool = True

    # Scheduling
    store_count_trigger: int = 100
    run_on_session_end: bool = True

    # Safety
    core_memory_immune: bool = True


@dataclass(frozen=True)
class SAGQConfig:
    """Spreading Activation-Guided Quantization configuration.

    Centrality formula:
        centrality(i) = w_pagerank * pr_norm + w_degree * deg_norm + w_sa_freq * sa_freq_norm

    SAGQ precision:
        sagq_bw = b_min + (b_max - b_min) * centrality, snapped to valid_bit_widths

    Combined precision (with Phase A EAP):
        final_bw = max(eap_bw, sagq_bw)
    """

    enabled: bool = True

    # Centrality weights (MUST sum to 1.0 -- validated in __post_init__)
    w_pagerank: float = 0.5      # PageRank structural importance
    w_degree: float = 0.3        # Degree centrality (connection count)
    w_sa_freq: float = 0.2       # Spreading activation frequency (7-day window)

    # Bit-width range
    b_min: int = 2               # Minimum bit-width (most aggressive quantization)
    b_max: int = 32              # Maximum bit-width (full float32 precision)

    # Valid bit-widths (snapping targets) -- must be sorted ascending
    valid_bit_widths: tuple[int, ...] = (2, 4, 8, 32)

    # SA frequency window (days to look back in activation_cache)
    sa_frequency_window_days: int = 7

    # Scheduler
    scheduler_interval_hours: float = 6.0   # How often to run combined scheduler

    def __post_init__(self) -> None:
        weight_sum = self.w_pagerank + self.w_degree + self.w_sa_freq
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"SAGQConfig centrality weights must sum to 1.0, got {weight_sum:.6f}"
            )
        if not self.valid_bit_widths:
            raise ValueError("SAGQConfig.valid_bit_widths must not be empty")
        if self.b_min < 1:
            raise ValueError(f"SAGQConfig.b_min must be >= 1, got {self.b_min}")
        if self.b_max < self.b_min:
            raise ValueError(
                f"SAGQConfig.b_max ({self.b_max}) must be >= b_min ({self.b_min})"
            )


@dataclass(frozen=True)
class ParameterizationConfig:
    """Soft prompt parameterization configuration (Phase F: The Learning Brain).

    Controls pattern extraction, prompt generation, injection, and lifecycle.
    Ships enabled by default. Pure text soft prompts — no LoRA, no weights.
    """

    enabled: bool = True

    # Pattern extraction
    min_confidence: float = 0.7        # Minimum pattern confidence [0.3, 1.0]
    min_evidence: int = 5              # Minimum evidence count for behavioral/workflow
    cross_project_boost: float = 1.2   # 20% confidence boost for cross-project patterns

    # Prompt generation
    max_prompt_tokens: int = 500       # Token budget for soft prompts
    max_memory_tokens: int = 1500      # Token budget for regular memories
    categories_enabled: tuple[str, ...] = (
        "identity", "tech_preference", "communication_style",
        "workflow_pattern", "project_context", "decision_history",
        "avoidance",
    )

    # Lifecycle
    refresh_interval_hours: float = 24.0   # Min hours between parameterization runs
    effectiveness_tracking: bool = True     # Track prompt effectiveness via feedback


@dataclass(frozen=True)
class TemporalValidatorConfig:
    """Configuration for temporal intelligence (Phase 4).

    Ships enabled by default. Users can disable via slm config.
    """

    enabled: bool = True
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

    Ships enabled by default. Users can disable via slm config.

    References:
      - SYNAPSE: FOK gating (fok_threshold = 0.12)
      - ACT-R: base-level activation (act_r_decay = 0.5)
      - Zep/Hindsight: multi-signal ranking consensus
    """

    enabled: bool = True
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
    forgetting: ForgettingConfig = field(default_factory=ForgettingConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    reaper: ReaperConfig = field(default_factory=ReaperConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    sagq: SAGQConfig = field(default_factory=SAGQConfig)
    ccq: CCQConfig = field(default_factory=CCQConfig)
    parameterization: ParameterizationConfig = field(
        default_factory=ParameterizationConfig,
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

        # V3.3 config fields (additive — defaults work if missing from JSON)
        fg = data.get("forgetting", {})
        if fg:
            config.forgetting = ForgettingConfig(**{
                k: v for k, v in fg.items()
                if k in ForgettingConfig.__dataclass_fields__
            })

        rt = data.get("retrieval", {})
        if rt:
            # V3.3.2 migration: add ONNX cross-encoder backend field.
            # Pre-3.3.2 configs lacked cross_encoder_backend. Add it,
            # but NEVER override an explicit use_cross_encoder setting.
            # The user's explicit choice always wins.
            if "cross_encoder_backend" not in rt:
                rt.setdefault("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                rt["cross_encoder_backend"] = ""  # V3.3.18: PyTorch (ONNX CoreML leaks on ARM64)
                # Only auto-enable if user didn't explicitly set the field
                rt.setdefault("use_cross_encoder", True)
            config.retrieval = RetrievalConfig(**{
                k: v for k, v in rt.items()
                if k in RetrievalConfig.__dataclass_fields__
            })

        return config

    def save(self, config_path: Path | None = None) -> None:
        """Save config to JSON file."""
        import json
        path = config_path or (self.base_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Read existing config to preserve V3.3 fields not in this save
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

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
            "retrieval": {
                "use_cross_encoder": self.retrieval.use_cross_encoder,
                "cross_encoder_model": self.retrieval.cross_encoder_model,
                "cross_encoder_backend": self.retrieval.cross_encoder_backend,
            },
        }

        # Preserve existing V3.3 config sections that aren't in for_mode()
        for key in ("forgetting", "quantization", "sagq", "embedding_signature", "auto_invoke"):
            if key in existing:
                data[key] = existing[key]

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
                    # Mode A: sentence-transformers in SUBPROCESS (never in-process)
                    provider=embedding_provider or "sentence-transformers",
                ),
                llm=LLMConfig(),  # No LLM
                retrieval=RetrievalConfig(
                    # V3.3.2: ONNX cross-encoder enabled for all modes (~200MB)
                    use_cross_encoder=True,
                    # V3.3.19: Enable 1 round of rule-based query decomposition.
                    # The enhanced _heuristic_expand generates entity+action
                    # sub-queries that dramatically improve multi-hop retrieval.
                    agentic_max_rounds=1,
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
                    # Mode B: Ollama HTTP API (zero PyTorch in-process)
                    provider=embedding_provider or "ollama",
                ),
                llm=LLMConfig(
                    provider=llm_provider or "ollama",
                    model=llm_model or "llama3.2",
                    api_base=llm_api_base or "http://localhost:11434",
                    api_key=llm_api_key or "",
                ),
                retrieval=RetrievalConfig(
                    # V3.3.2: ONNX cross-encoder enabled for all modes (~200MB)
                    use_cross_encoder=True,
                ),
            )

        # Mode C — FULL POWER, UNRESTRICTED
        # Don't carry over local-only providers (ollama) to cloud mode
        c_provider = llm_provider if llm_provider not in ("ollama", "") else "openrouter"
        c_model = llm_model if llm_provider not in ("ollama", "") else "anthropic/claude-sonnet-4"
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
                provider=c_provider,
                model=c_model,
                api_key=llm_api_key,
                api_base=llm_api_base,
            ),
            channel_weights=ChannelWeights(
                semantic=1.5,
                bm25=1.2,
                entity_graph=1.3,
                temporal=1.0,
                spreading_activation=1.2,  # Phase 3: SA boost in Mode C
                hopfield=1.0,  # Phase G: Hopfield in Mode C
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
