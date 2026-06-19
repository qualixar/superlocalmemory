# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Configuration.

Unified configuration with Mode A/B/C capability matrix.
Clean — zero dead options, every config has a consumer.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from superlocalmemory.storage.models import Mode


# ---------------------------------------------------------------------------
# Canonical limits (WP-02 — single source of truth across all surfaces)
# ---------------------------------------------------------------------------

#: Default number of results returned by recall across MCP, CLI, daemon, and
#: engine.  All surfaces bind their ``limit`` defaults to this constant so a
#: single change is sufficient to keep the full stack in sync.
CANONICAL_RECALL_LIMIT: int = 20


# ---------------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path.home() / ".superlocalmemory"
DEFAULT_DB_NAME = "memory.db"
DEFAULT_PROFILES_FILE = "profiles.json"
CURRENT_MODE_FILE = "current_mode"
# Populated lazily in _get_mode_config_path() to avoid circular imports
_MODE_CONFIG_NAMES: dict | None = None


# ---------------------------------------------------------------------------
# Embedding Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration per mode."""

    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    dimension: int = 768
    # Provider: "" = auto-detect, "sentence-transformers", "ollama", "cloud",
    # "openai" (V3.4.24: any OpenAI-compatible /v1/embeddings endpoint)
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
        if self.provider == "openai":
            return False
        return bool(self.api_endpoint) or self.provider == "cloud"

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def is_openai_compatible(self) -> bool:
        """V3.4.24: True when using a custom OpenAI-compatible endpoint."""
        return self.provider == "openai" and bool(self.api_endpoint)


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
# Scope Weights
# ---------------------------------------------------------------------------

@dataclass
class ScopeWeights:
    """RRF fusion weights for multi-scope retrieval.

    Personal scope has highest weight (most relevant to current profile).
    Shared scope has medium weight (team/group memories).
    Global scope has lowest weight (public/common knowledge).
    """

    personal: float = 1.0
    shared: float = 0.7
    global_: float = 0.5  # trailing underscore avoids Python keyword

    def __post_init__(self) -> None:
        for name in ("personal", "shared", "global_"):
            val = getattr(self, name)
            if val < 0:
                raise ValueError(f"ScopeWeights values must be non-negative, got {name}={val}")

    def as_dict(self) -> dict[str, float]:
        return {"personal": self.personal, "shared": self.shared, "global": self.global_}


# ---------------------------------------------------------------------------
# Scope Config (multi-scope memory behaviour defaults)
# ---------------------------------------------------------------------------

_VALID_SCOPES = ("personal", "shared", "global")


@dataclass
class ScopeConfig:
    """User-facing defaults for multi-scope (shared) memory.

    SHARED MEMORY IS OPT-IN — NOT a default feature (v3.6.15 product decision).
    The defaults below make a fresh / unconfigured install behave EXACTLY like
    3.6.14: every write is ``personal`` and recall returns only this profile's
    own facts. Another profile's ``global``/``shared`` facts never leak into
    recall until the user explicitly turns sharing on.

      - ``default_scope='personal'``      → writes stay private by default;
      - ``recall_include_global=False``   → don't surface other profiles' global;
      - ``recall_include_shared=False``   → don't surface facts shared *to* me.

    Turning it on is a deliberate act, done either per-call (``--scope`` /
    ``--include-global`` / MCP args) or persistently by editing config.json /
    mode_a|b|c.json (the installer can also write the choice). The CLI/MCP
    boundary passes ``None`` ("not specified") so the engine resolves these
    config values as the effective default.
    """

    default_scope: str = "personal"         # scope assigned to new memories
    recall_include_global: bool = False     # surface scope='global' facts in recall
    recall_include_shared: bool = False     # surface scope='shared' facts in recall

    def __post_init__(self) -> None:
        if self.default_scope not in _VALID_SCOPES:
            raise ValueError(
                f"default_scope must be one of {_VALID_SCOPES}, got {self.default_scope!r}"
            )

    def as_dict(self) -> dict:
        return {
            "default_scope": self.default_scope,
            "recall_include_global": self.recall_include_global,
            "recall_include_shared": self.recall_include_shared,
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
    # V3.4.2: Tested gte-reranker-modernbert-base (8K context) — REGRESSED
    # LoCoMo from 68.4% to 64.1%. Reverted to MiniLM-L-12-v2. The 512-token
    # limit is acceptable because SLM's 6-channel retrieval pre-filters
    # relevant facts before reranking. See bench-v342-locomo.md.
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

    # v3.6.6: Evidence floor — gate on per-channel scores, not fused/RRF score.
    # Nonsense queries earn 0.0 on every primary channel; real matches earn
    # semantic >= 0.85 or bm25 > 0. The discriminator is earned channel evidence.
    # Env kill-switch: SLM_RECALL_NO_FLOOR=1 disables without release.
    evidence_floor_enabled: bool = True
    min_semantic_evidence: float = 0.60   # Minimum cosine similarity to keep a result

    # v3.6.6: Recall output budget — protect consuming agents from 585KB responses.
    recall_per_fact_max_chars: int = 2400  # ~600 tokens; head 70% + tail 30%
    recall_total_max_chars: int = 12000    # ~3K tokens; stubs beyond this


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
# Store Config (v3.6.6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoreConfig:
    """Configuration for the remember/store write path (v3.6.6).

    Ingest gate: protect the DB from oversized facts and prompt-template
    pollution. Defaults ON; env kill-switch SLM_INGEST_NO_GATE=1.
    """

    # Max chars for the FACT content stored in atomic_facts.content.
    # Content above this is clamped to head 70% + tail 30% + truncation marker.
    # The FULL original is preserved in the memories table row. Set high (24K
    # ≈ 6K tokens) so only pathological pastes are touched; normal dense
    # session-handoff memories (6-15K chars) are stored 100% intact.
    max_verbatim_chars: int = 24000

    # Hard upper bound in bytes. Content above this is rejected outright.
    # Nobody's "memory" is a megabyte (MCP: success=False, HTTP: 413).
    max_ingest_bytes: int = 1_048_576  # 1 MB


# ---------------------------------------------------------------------------
# Context Injection (v3.4.65)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionConfig:
    """Context-injection budgets & framing (v3.4.65).

    Single source of truth for what reaches the agent at session_init,
    prestage_context, and the recall hooks. Replaces scattered char caps.

    Budgets are in *estimated tokens* (chars/4 heuristic; see
    core/injection.estimate_tokens). Mode-aware: A=fast/cheap, C=rich.
    """
    enabled: bool = True

    # Mode-aware TOTAL budget (core block + recall), in estimated tokens.
    total_budget_tokens_a: int = 2000
    total_budget_tokens_b: int = 4000
    total_budget_tokens_c: int = 8000

    # Per-memory ceiling (estimated tokens). Whole-memory inclusion until
    # the total budget is hit; this only clamps a single oversized memory.
    per_memory_max_tokens: int = 600

    # Core Memory Block (Letta pattern).
    core_block_enabled: bool = True
    core_block_max_facts: int = 5
    core_block_max_tokens: int = 1000
    core_block_importance_min: float = 0.8
    core_block_min_access_count: int = 2

    # Lost-in-the-middle: place strongest at top & bottom edges.
    edge_ordering: bool = True

    # Trust framing. False (shipped) = cautious "reference only" wrapper.
    # True (Varun personal) = clean "memory context" framing.
    # redact_secrets ALWAYS runs regardless.
    trust_first_party: bool = False

    # prestage_context response byte cap (was hardcoded 16 KB).
    prestage_max_response_bytes: int = 64 * 1024


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
class EvolutionConfig:
    """Configuration for Skill Evolution Engine (v3.4.10).

    OFF by default — opt in via `slm setup` (interactive) or
    `slm config set evolution.enabled true` (CLI).

    Backend auto-detection priority:
      1. `claude` CLI available → spawn `claude --model haiku` (ECC pattern, free)
      2. Ollama running → use Ollama (free, local)
      3. API key set → use Anthropic/OpenAI API (paid)
      4. Nothing → dashboard-only (show candidates, manual evolution)
    """

    enabled: bool = False                        # OFF by default, opt-in
    backend: str = "auto"                        # auto, claude, ollama, anthropic, openai
    max_evolutions_per_cycle: int = 3            # Budget cap per consolidation


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
# Health Config (v3.6.9 BUG-A)
# ---------------------------------------------------------------------------

@dataclass
class HealthConfig:
    """Health-monitor tuning knobs.

    All values have safe defaults so an empty ``"health": {}`` JSON section
    silently inherits them. The ``global_rss_budget_mb`` default is computed
    at runtime (40% of physical RAM, floor 2500 MB) so low-RAM boxes keep the
    old behaviour while large machines are never accidentally throttled.
    """
    global_rss_budget_mb: int = 0          # 0 = compute at runtime (40% RAM, floor 2500)
    heartbeat_timeout_sec: int = 60
    health_check_interval_sec: int = 15
    enable_structured_logging: bool = True


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
    scope_weights: ScopeWeights = field(default_factory=ScopeWeights)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
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
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    # v3.5.0: scaling backends — "sqlite" / "cozo" / "auto" / "lancedb" / "sqlite-vec" / "auto".
    graph_backend: str = "auto"       # "auto" = cozo if pycozo installed, else sqlite
    vector_backend: str = "auto"      # "auto" = lancedb if installed, else sqlite-vec
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    # v3.4.3: Daemon configuration
    daemon_idle_timeout: int = 0       # 0 = 24/7 (no auto-kill). >0 = seconds before auto-kill.
    daemon_port: int = 8765            # Primary daemon port
    daemon_legacy_port: int = 8767     # Backward-compat redirect port
    daemon_enable_legacy_port: bool = True  # Set False to disable 8767 redirect

    # v3.4.3: Entity compilation
    entity_compilation_enabled: bool = True
    entity_compilation_retrieval_boost: float = 1.0  # 1.0 = disabled. >1.0 = boost score.

    # v3.4.3: Mesh
    mesh_enabled: bool = True

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = self.base_dir / DEFAULT_DB_NAME

    @classmethod
    def load(cls, config_path: Path | None = None) -> SLMConfig:
        """Load config from JSON file. Returns default Mode A if file doesn't exist."""
        # WP-07: resolve base dir through slm_home() at call time so all 3 env
        # aliases (SLM_DATA_DIR → SL_MEMORY_PATH → SLM_HOME) are honoured.
        try:
            from superlocalmemory.cli._lazy_init import slm_home as _slm_home
            _runtime_base = _slm_home()
        except Exception:
            _runtime_base = DEFAULT_BASE_DIR
        path = config_path or (_runtime_base / "config.json")
        if not path.exists():
            return cls.for_mode(Mode.A)
        import json
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # An already-corrupt/truncated config.json must NOT brick every `slm`
            # call — degrade to the Mode-A default and warn rather than raise.
            logger.warning(
                "config.json unreadable/corrupt (%s) — using Mode A default", exc
            )
            return cls.for_mode(Mode.A)
        mode = Mode(data.get("mode", "a"))
        llm_data = data.get("llm", {})
        emb_data = data.get("embedding", {})
        # V3.5.9: read base_dir before constructing config so for_mode() builds
        # db_path from the user's directory, not DEFAULT_BASE_DIR.
        raw_base_dir = Path(data.get("base_dir", str(_runtime_base)))
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
            embedding_model_name=emb_data.get("model_name", ""),
            embedding_dimension=int(emb_data.get("dimension", 0) or 0),
            base_dir=raw_base_dir,
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

        # V3.4.3 config fields (additive — missing keys get dataclass defaults)
        config.daemon_idle_timeout = data.get("daemon_idle_timeout", 0)
        config.daemon_port = data.get("daemon_port", 8765)
        config.daemon_legacy_port = data.get("daemon_legacy_port", 8767)
        config.daemon_enable_legacy_port = data.get("daemon_enable_legacy_port", True)
        config.entity_compilation_enabled = data.get("entity_compilation_enabled", True)
        config.entity_compilation_retrieval_boost = data.get(
            "entity_compilation_retrieval_boost", 1.0,
        )
        config.mesh_enabled = data.get("mesh_enabled", True)

        # V3.4.10: Evolution config
        evo = data.get("evolution", {})
        if evo:
            config.evolution = EvolutionConfig(**{
                k: v for k, v in evo.items()
                if k in EvolutionConfig.__dataclass_fields__
            })

        # V3.6.9: Health monitor config (BUG-A — previously silently ignored)
        hlth = data.get("health", {})
        if hlth:
            config.health = HealthConfig(**{
                k: v for k, v in hlth.items()
                if k in HealthConfig.__dataclass_fields__
            })

        # V3.4.65: Injection config (additive — defaults if missing from JSON)
        inj = data.get("injection", {}) or {}
        config.injection = InjectionConfig(
            enabled=bool(inj.get("enabled", True)),
            total_budget_tokens_a=int(inj.get("total_budget_tokens_a", 2000)),
            total_budget_tokens_b=int(inj.get("total_budget_tokens_b", 4000)),
            total_budget_tokens_c=int(inj.get("total_budget_tokens_c", 8000)),
            per_memory_max_tokens=int(inj.get("per_memory_max_tokens", 600)),
            core_block_enabled=bool(inj.get("core_block_enabled", True)),
            core_block_max_facts=int(inj.get("core_block_max_facts", 5)),
            core_block_max_tokens=int(inj.get("core_block_max_tokens", 1000)),
            core_block_importance_min=float(inj.get("core_block_importance_min", 0.8)),
            core_block_min_access_count=int(inj.get("core_block_min_access_count", 2)),
            edge_ordering=bool(inj.get("edge_ordering", True)),
            trust_first_party=bool(inj.get("trust_first_party", False)),
            prestage_max_response_bytes=int(inj.get("prestage_max_response_bytes", 64 * 1024)),
        )

        # Multi-scope memory: scope weights
        sw = data.get("scope_weights", {})
        if sw:
            config.scope_weights = ScopeWeights(**{
                k: v for k, v in sw.items()
                if k in ScopeWeights.__dataclass_fields__
            })

        # Multi-scope memory: behaviour defaults (default scope + recall visibility)
        sc = data.get("scope", {})
        if sc:
            config.scope = ScopeConfig(**{
                k: v for k, v in sc.items()
                if k in ScopeConfig.__dataclass_fields__
            })

        return config

    def save(
        self,
        config_path: Path | None = None,
        *,
        mode_change: bool = False,
    ) -> None:
        """Save config to JSON file.

        v3.4.34: mode protection. If the existing config.json has a mode
        that differs from ``self.mode`` and the caller did NOT pass
        ``mode_change=True``, the EXISTING mode is preserved.  This
        prevents accidental mode resets when code creates a fresh
        ``SLMConfig()`` (defaults to Mode A) and calls ``save()`` to
        persist an unrelated field change.

        Callers that intentionally switch mode (``slm mode b``, the MCP
        ``set_mode`` tool, the dashboard PUT ``/api/v3/mode``) MUST pass
        ``mode_change=True``.
        """
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

        # v3.4.34: mode protection — preserve user's mode unless explicitly changing
        effective_mode = self.mode.value
        existing_mode = existing.get("mode")
        if existing_mode and existing_mode != effective_mode and not mode_change:
            logger.warning(
                "SLMConfig.save(): mode change blocked (%s → %s). "
                "Pass mode_change=True to override. Preserving '%s'.",
                existing_mode, effective_mode, existing_mode,
            )
            effective_mode = existing_mode

        data = {
            "mode": effective_mode,
            "active_profile": self.active_profile,
            "base_dir": str(self.base_dir),  # V3.5.9: persist so load() can restore custom paths
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

        # V3.4.11: Persist evolution config (C-CONFIGSAVE fix)
        data["evolution"] = {
            "enabled": self.evolution.enabled,
            "backend": self.evolution.backend,
            "max_evolutions_per_cycle": self.evolution.max_evolutions_per_cycle,
        }

        # V3.4.65: Persist injection config
        data["injection"] = {
            "enabled": self.injection.enabled,
            "total_budget_tokens_a": self.injection.total_budget_tokens_a,
            "total_budget_tokens_b": self.injection.total_budget_tokens_b,
            "total_budget_tokens_c": self.injection.total_budget_tokens_c,
            "per_memory_max_tokens": self.injection.per_memory_max_tokens,
            "core_block_enabled": self.injection.core_block_enabled,
            "core_block_max_facts": self.injection.core_block_max_facts,
            "core_block_max_tokens": self.injection.core_block_max_tokens,
            "core_block_importance_min": self.injection.core_block_importance_min,
            "core_block_min_access_count": self.injection.core_block_min_access_count,
            "edge_ordering": self.injection.edge_ordering,
            "trust_first_party": self.injection.trust_first_party,
            "prestage_max_response_bytes": self.injection.prestage_max_response_bytes,
        }

        # Multi-scope memory: scope weights
        data["scope_weights"] = {
            "personal": self.scope_weights.personal,
            "shared": self.scope_weights.shared,
            "global_": self.scope_weights.global_,
        }

        # Multi-scope memory: behaviour defaults
        data["scope"] = self.scope.as_dict()

        # Preserve existing V3.3 config sections that aren't in for_mode()
        for key in ("forgetting", "quantization", "sagq", "embedding_signature", "auto_invoke"):
            if key in existing:
                data[key] = existing[key]

        # Atomic write: a crash mid-write must NOT leave a truncated/corrupt
        # config.json (which would make every subsequent `slm` call fail to load).
        import os as _os
        _tmp = path.with_suffix(path.suffix + ".tmp")
        _tmp.write_text(json.dumps(data, indent=2))
        _os.replace(_tmp, path)

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
        embedding_model_name: str = "",
        embedding_dimension: int = 0,
    ) -> SLMConfig:
        """Create config with mode-appropriate defaults."""
        # WP-07: resolve base dir via slm_home() at call time when not explicit.
        if base_dir is None:
            try:
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _base = _slm_home()
            except Exception:
                _base = DEFAULT_BASE_DIR
        else:
            _base = base_dir

        if mode == Mode.A:
            # V3.4.24: If user chose "openai" provider, honour their custom
            # endpoint/model/dimension. Otherwise use local defaults.
            _a_provider = embedding_provider or "sentence-transformers"
            if _a_provider == "openai" and embedding_endpoint:
                _a_emb = EmbeddingConfig(
                    model_name=embedding_model_name or "nomic-ai/nomic-embed-text-v1.5",
                    dimension=embedding_dimension or 768,
                    provider="openai",
                    api_endpoint=embedding_endpoint,
                    api_key=embedding_key,
                )
            else:
                _a_emb = EmbeddingConfig(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    dimension=768,
                    provider=_a_provider,
                )
            return cls(
                mode=mode,
                base_dir=_base,
                embedding=_a_emb,
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
            # V3.4.24: If user chose "openai" provider with a custom endpoint
            # (e.g. local vLLM, LiteLLM, Ollama /v1), honour it.
            _b_provider = embedding_provider or "ollama"
            if _b_provider == "openai" and embedding_endpoint:
                _b_emb = EmbeddingConfig(
                    model_name=embedding_model_name or "nomic-ai/nomic-embed-text-v1.5",
                    dimension=embedding_dimension or 768,
                    provider="openai",
                    api_endpoint=embedding_endpoint,
                    api_key=embedding_key,
                )
            else:
                _b_emb = EmbeddingConfig(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    dimension=768,
                    provider=_b_provider,
                )
            return cls(
                mode=mode,
                base_dir=_base,
                embedding=_b_emb,
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
        # V3.4.24: If user chose "openai" provider, honour it in Mode C too.
        _c_emb_provider = embedding_provider or ""
        if _c_emb_provider == "openai" and embedding_endpoint:
            _c_emb = EmbeddingConfig(
                model_name=embedding_model_name or "text-embedding-3-large",
                dimension=embedding_dimension or 3072,
                provider="openai",
                api_endpoint=embedding_endpoint,
                api_key=embedding_key,
            )
        else:
            _c_emb = EmbeddingConfig(
                model_name="text-embedding-3-large",
                dimension=3072,
                api_endpoint=embedding_endpoint,
                api_key=embedding_key,
                deployment_name=embedding_deployment,
            )
        return cls(
            mode=mode,
            base_dir=_base,
            embedding=_c_emb,
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

    # ------------------------------------------------------------------
    # 3-Mode config system (v3.4.54)
    # ------------------------------------------------------------------

    @staticmethod
    def _mode_config_path(base_dir: Path, mode: "Mode") -> Path:
        """Return the per-mode config file path."""
        from superlocalmemory.storage.models import Mode as _M
        _names = {
            _M.A: "mode_a.json",
            _M.B: "mode_b.json",
            _M.C: "mode_c.json",
        }
        return base_dir / _names.get(mode, "mode_a.json")

    @staticmethod
    def read_current_mode(base_dir: Path | None = None) -> str:
        """Read the current mode from the ``current_mode`` file.

        Returns ``\"b\"`` (the default) if the file doesn't exist.
        """
        # WP-07: resolve via slm_home() at call time when base_dir not explicit.
        if base_dir is None:
            try:
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _base = _slm_home()
            except Exception:
                _base = DEFAULT_BASE_DIR
        else:
            _base = base_dir
        _f = _base / CURRENT_MODE_FILE
        try:
            return _f.read_text(encoding="utf-8").strip().lower() or "b"
        except (OSError, FileNotFoundError):
            return "b"

    @staticmethod
    def write_current_mode(mode: str, base_dir: Path | None = None) -> None:
        """Write the current mode letter to ``current_mode``."""
        # WP-07: resolve via slm_home() at call time when base_dir not explicit.
        if base_dir is None:
            try:
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _base = _slm_home()
            except Exception:
                _base = DEFAULT_BASE_DIR
        else:
            _base = base_dir
        _base.mkdir(parents=True, exist_ok=True)
        (_base / CURRENT_MODE_FILE).write_text(
            mode.lower().strip(), encoding="utf-8",
        )

    @classmethod
    def switch_mode(
        cls,
        new_mode: str,
        base_dir: Path | None = None,
    ) -> "SLMConfig":
        """Switch to a different mode, preserving ALL non-mode settings.

        v3.4.58 fix: Only ``config.mode`` changes. The previous implementation
        called ``for_mode()`` (which resets embedding/LLM/retrieval to mode
        defaults) whenever the target mode's per-mode file didn't exist.
        This silently clobbered custom embeddings, endpoints, and LLM config.

        Correct behavior:
          - Load the current (old) config from config.json.
          - Save it to mode_{old}.json for the 3-mode system.
          - If mode_{new}.json exists, load it (user had a prior config for that mode).
          - If mode_{new}.json does NOT exist: start from the OLD config, change
            only the mode field. This preserves embedding, retrieval, forgetting, etc.
          - Exception: if user has NO LLM provider AND is switching to B/C, populate
            sensible LLM defaults so the daemon doesn't start dead.
          - Write the result to config.json (backward compat) and current_mode.

        Returns the new active config.
        """
        import copy
        import dataclasses

        from superlocalmemory.storage.models import Mode as _M
        # WP-07: resolve via slm_home() at call time when base_dir not explicit.
        if base_dir is None:
            try:
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _base = _slm_home()
            except Exception:
                _base = DEFAULT_BASE_DIR
        else:
            _base = base_dir
        _base.mkdir(parents=True, exist_ok=True)

        old_config = cls.load(_base / "config.json")
        old_mode = old_config.mode.value.lower()
        new_mode_val = _M(new_mode.lower())

        # 1. Save current config to its per-mode file (preserve customizations)
        if old_mode != new_mode.lower():
            old_path = cls._mode_config_path(_base, old_config.mode)
            old_config.save(old_path)

        # 2. Determine new config:
        #    a) mode_{new}.json exists → user had prior config for this mode, use it
        #    b) otherwise → copy old config, change only mode
        new_path = cls._mode_config_path(_base, new_mode_val)
        if new_path.exists():
            try:
                new_config = cls.load(new_path)
                if new_config.mode != new_mode_val:
                    # Corrupt/mismatched mode file — rebuild preserving old values
                    new_config = copy.copy(old_config)
                    new_config.mode = new_mode_val
            except Exception:
                new_config = copy.copy(old_config)
                new_config.mode = new_mode_val
        else:
            # First time switching to this mode: start from old config, change mode only.
            # Use dataclasses.replace() to produce a new frozen-compatible object.
            new_config = dataclasses.replace(old_config, mode=new_mode_val)

        # 3. LLM default population — ONLY if user has no provider AND switching to B/C.
        #    Never overwrites an existing provider.
        if new_mode_val in (_M.B, _M.C) and not new_config.llm.provider:
            if new_mode_val == _M.B:
                new_config = dataclasses.replace(
                    new_config,
                    llm=LLMConfig(
                        provider="ollama",
                        model="llama3.2",
                        api_base="http://localhost:11434",
                    ),
                )
            else:  # Mode C
                new_config = dataclasses.replace(
                    new_config,
                    llm=LLMConfig(
                        provider="openrouter",
                        model="anthropic/claude-sonnet-4",
                    ),
                )

        # 4. Save as active config.json (backward compat)
        new_config.save(_base / "config.json", mode_change=True)

        # 5. Write current_mode
        cls.write_current_mode(new_mode, _base)

        return new_config

    @classmethod
    def migrate_to_3mode(cls, base_dir: Path | None = None) -> bool:
        """One-time migration: config.json → 3-mode system.

        Called on daemon boot. Idempotent — if ``current_mode`` already
        exists, this is a no-op. Returns True if migration was performed.
        """
        # WP-07: resolve via slm_home() at call time when base_dir not explicit.
        if base_dir is None:
            try:
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _base = _slm_home()
            except Exception:
                _base = DEFAULT_BASE_DIR
        else:
            _base = base_dir
        _current = _base / CURRENT_MODE_FILE
        if _current.exists():
            return False  # already migrated

        legacy = _base / "config.json"
        if not legacy.exists():
            # No config at all — write defaults and current_mode
            from superlocalmemory.storage.models import Mode as _M
            _def = cls.for_mode(_M.B, base_dir=_base)
            _def.save(legacy)
            cls.write_current_mode("b", _base)
            for _m in (_M.A, _M.B, _M.C):
                _mp = cls._mode_config_path(_base, _m)
                if not _mp.exists():
                    try:
                        _mc = cls.for_mode(_m, base_dir=_base)
                        _mc.save(_mp)
                    except Exception:
                        pass
            return True

        # Migrate existing config.json → mode_{current}.json
        try:
            config = cls.load(legacy)
            current = config.mode.value.lower()
            cls.write_current_mode(current, _base)

            # Save current config to its mode file
            mode_path = cls._mode_config_path(_base, config.mode)
            config.save(mode_path)

            # Generate other mode files from defaults
            from superlocalmemory.storage.models import Mode as _M
            for _m in (_M.A, _M.B, _M.C):
                _mp = cls._mode_config_path(_base, _m)
                if not _mp.exists():
                    try:
                        _mc = cls.for_mode(_m, base_dir=_base)
                        _mc.save(_mp)
                    except Exception:
                        pass

            logger.info(
                "3-mode config system migrated: config.json → mode_%s.json. "
                "All three mode configs generated.", current,
            )
            return True
        except Exception:
            return False
