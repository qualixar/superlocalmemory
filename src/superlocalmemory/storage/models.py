# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Data Models.

All dataclasses for the memory system. Profile-scoped by design.
Every model that persists to DB includes profile_id.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactType(str, Enum):
    """Memory fact classification — ENGRAM-inspired typed stores."""

    EPISODIC = "episodic"      # Events: who did what when
    SEMANTIC = "semantic"      # World knowledge: X is Y
    OPINION = "opinion"        # Subjective with confidence
    TEMPORAL = "temporal"      # Time-bounded events with intervals


class EdgeType(str, Enum):
    """Knowledge graph edge types."""

    ENTITY = "entity"          # Shared entity link
    TEMPORAL = "temporal"      # Chronological ordering
    SEMANTIC = "semantic"      # Embedding similarity
    CAUSAL = "causal"          # Cause-effect relationship
    CONTRADICTION = "contradiction"  # Conflicting facts
    SUPERSEDES = "supersedes"  # Newer fact replaces older


class ConsolidationActionType(str, Enum):
    """Mem0-style consolidation actions."""

    ADD = "add"                # New information, store as new fact
    UPDATE = "update"          # Refines existing fact
    SUPERSEDE = "supersede"    # Contradicts existing fact
    NOOP = "noop"              # Duplicate, skip storage


class MemoryLifecycle(str, Enum):
    """Memory lifecycle states — coupled with Langevin dynamics."""

    ACTIVE = "active"          # Frequently accessed, near origin
    WARM = "warm"              # Moderate access, mid-radius
    COLD = "cold"              # Rarely accessed, near boundary
    ARCHIVED = "archived"      # Beyond boundary, deep storage


class SignalType(str, Enum):
    """V2-compatible signal inference types."""

    FACTUAL = "factual"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    OPINION = "opinion"
    REQUEST = "request"
    SOCIAL = "social"


class Mode(str, Enum):
    """Operating modes — EU AI Act alignment."""

    A = "a"  # Local Guardian: zero LLM, EU AI Act FULL
    B = "b"  # Smart Local: local Ollama LLM, EU AI Act FULL
    C = "c"  # Full Power: UNRESTRICTED, best models, 90%+ target


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Profile:
    """Memory profile — first-class isolation boundary.

    Every table is scoped by profile_id. Instant switching.
    Ported from V2.8 columnar profile isolation pattern.
    """

    profile_id: str
    name: str
    description: str = ""
    personality: str = ""          # Profile personality description
    mode: Mode = Mode.A            # Default operating mode for this profile
    created_at: str = field(default_factory=_now)
    last_used: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """Primary memory entry — a conversation turn or message.

    This is the RAW input. Facts are extracted FROM this during encoding.
    """

    memory_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    scope: str = "personal"
    shared_with: list[str] | None = None
    content: str = ""
    session_id: str = ""
    speaker: str = ""              # Who said this
    role: str = ""                 # user / assistant / system
    session_date: str | None = None   # Parsed ISO-8601 date
    created_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtomicFact:
    """Structured fact extracted from memory — the PRIMARY retrieval unit.

    This is what the encoding engine produces. This is what retrieval searches.
    Typed stores (ENGRAM pattern): episodic / semantic / opinion / temporal.
    """

    fact_id: str = field(default_factory=_new_id)
    memory_id: str = ""            # Source memory this was extracted from
    profile_id: str = "default"
    scope: str = "personal"
    shared_with: list[str] | None = None
    content: str = ""              # Atomic fact statement
    fact_type: FactType = FactType.SEMANTIC

    # Entities
    entities: list[str] = field(default_factory=list)
    canonical_entities: list[str] = field(default_factory=list)

    # Temporal (3-date model — Mastra pattern)
    observation_date: str | None = None    # When the conversation happened
    referenced_date: str | None = None     # Date mentioned in content
    interval_start: str | None = None      # Event start (for duration events)
    interval_end: str | None = None        # Event end

    # Quality
    confidence: float = 1.0
    importance: float = 0.5        # Priority: 0.0 (low) to 1.0 (critical)
    evidence_count: int = 1        # Incremented on UPDATE consolidation
    access_count: int = 0

    # Source tracing
    source_turn_ids: list[str] = field(default_factory=list)
    session_id: str = ""

    # Embeddings (populated by encoding pipeline)
    embedding: list[float] | None = None
    fisher_mean: list[float] | None = None
    fisher_variance: list[float] | None = None

    # Lifecycle (Langevin-coupled)
    lifecycle: MemoryLifecycle = MemoryLifecycle.ACTIVE
    langevin_position: list[float] | None = None

    # Emotional (VADER)
    emotional_valence: float = 0.0   # -1.0 negative to +1.0 positive
    emotional_arousal: float = 0.0   # 0.0 calm to 1.0 intense

    # Signal type (V2 compatible)
    signal_type: SignalType = SignalType.FACTUAL

    # v3.4.65: Core Memory Block explicit pin (M015)
    pinned: bool = False

    created_at: str = field(default_factory=_now)


@dataclass
class CanonicalEntity:
    """Resolved canonical entity — the single identity for an entity.

    Aliases point here. Entity profiles accumulate knowledge about this entity.
    """

    entity_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    scope: str = "personal"
    shared_with: list[str] | None = None
    canonical_name: str = ""
    entity_type: str = ""          # person / place / org / concept / event
    first_seen: str = field(default_factory=_now)
    last_seen: str = field(default_factory=_now)
    fact_count: int = 0


@dataclass
class EntityAlias:
    """Alias mapping to a canonical entity."""

    alias_id: str = field(default_factory=_new_id)
    entity_id: str = ""            # FK to CanonicalEntity
    alias: str = ""
    confidence: float = 1.0
    source: str = ""               # How this alias was discovered


@dataclass
class EntityProfile:
    """Accumulated knowledge about a canonical entity.

    Built during encoding by ObservationBuilder. Updated on each new fact.
    """

    profile_entry_id: str = field(default_factory=_new_id)
    entity_id: str = ""            # FK to CanonicalEntity
    profile_id: str = "default"
    knowledge_summary: str = ""    # Running summary of all facts about entity
    fact_ids: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_now)


@dataclass
class MemoryScene:
    """Clustered group of related facts — EverMemOS MemScene pattern.

    Scenes provide contextual retrieval: related facts come together.
    """

    scene_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    theme: str = ""                # Scene description / topic
    fact_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    last_updated: str = field(default_factory=_now)


@dataclass
class TemporalEvent:
    """Per-entity timeline entry with 3-date model.

    Enables temporal retrieval: "What happened to Alice in March?"
    """

    event_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    scope: str = "personal"
    shared_with: list[str] | None = None
    entity_id: str = ""            # FK to CanonicalEntity
    fact_id: str = ""              # FK to AtomicFact
    observation_date: str | None = None
    referenced_date: str | None = None
    interval_start: str | None = None
    interval_end: str | None = None
    description: str = ""


@dataclass
class GraphEdge:
    """Knowledge graph edge between facts or entities."""

    edge_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    scope: str = "personal"
    shared_with: list[str] | None = None
    source_id: str = ""            # Fact ID or Entity ID
    target_id: str = ""            # Fact ID or Entity ID
    edge_type: EdgeType = EdgeType.ENTITY
    weight: float = 1.0
    created_at: str = field(default_factory=_now)


@dataclass
class ConsolidationAction:
    """Log of consolidation decisions (ADD/UPDATE/SUPERSEDE/NOOP)."""

    action_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    action_type: ConsolidationActionType = ConsolidationActionType.ADD
    new_fact_id: str = ""          # The incoming fact
    existing_fact_id: str = ""     # The matched existing fact (if any)
    reason: str = ""               # Why this action was chosen
    timestamp: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Trust & Provenance (Ported from V2.8)
# ---------------------------------------------------------------------------

@dataclass
class TrustScore:
    """Bayesian trust score per source / entity / fact.

    Updated on access, contradiction, confirmation. Profile-scoped.
    """

    trust_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    target_type: str = ""          # "entity" / "source" / "fact"
    target_id: str = ""            # ID of the entity/source/fact
    trust_score: float = 0.5       # Bayesian: 0.0 = untrusted, 1.0 = fully trusted
    evidence_count: int = 0
    last_updated: str = field(default_factory=_now)


@dataclass
class ProvenanceRecord:
    """Provenance tracking — who/what created this memory and how."""

    provenance_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    fact_id: str = ""
    source_type: str = ""          # "conversation" / "import" / "consolidation"
    source_id: str = ""            # Session ID or import batch ID
    created_by: str = ""           # Agent or user who created this
    timestamp: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Learning & Behavioral (Ported from V2.8)
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    """User feedback on retrieval results — drives adaptive learning."""

    feedback_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    query: str = ""
    fact_id: str = ""
    feedback_type: str = ""        # "relevant" / "irrelevant" / "partial"
    dwell_time_ms: int = 0
    timestamp: str = field(default_factory=_now)


@dataclass
class BehavioralPattern:
    """Learned behavioral pattern — query habits, topic preferences."""

    pattern_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    pattern_type: str = ""         # "query_topic" / "time_of_day" / "entity_pref"
    pattern_key: str = ""          # The pattern identifier
    pattern_value: str = ""        # Serialized pattern data
    confidence: float = 0.0
    observation_count: int = 0
    last_updated: str = field(default_factory=_now)


@dataclass
class ActionOutcome:
    """Outcome tracking for learning — did the retrieved facts help?"""

    outcome_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    query: str = ""
    fact_ids: list[str] = field(default_factory=list)
    outcome: str = ""              # "success" / "partial" / "failure"
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Compliance (EU AI Act, GDPR, Lifecycle)
# ---------------------------------------------------------------------------

@dataclass
class ComplianceAuditEntry:
    """Audit trail for compliance (GDPR right-to-know, EU AI Act)."""

    audit_id: str = field(default_factory=_new_id)
    profile_id: str = "default"
    action: str = ""               # "store" / "retrieve" / "delete" / "export"
    target_type: str = ""          # "fact" / "memory" / "profile"
    target_id: str = ""
    details: str = ""
    timestamp: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Retrieval Result (Not persisted — runtime only)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieval result with scores and evidence chain."""

    fact: AtomicFact
    score: float = 0.0
    channel_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    # Score Contract v2. Legacy ``score`` and ``confidence`` remain one-release
    # aliases of relevance_score and memory_confidence respectively.
    relevance_score: float | None = None
    ranking_score: float | None = None
    memory_confidence: float | None = None
    rank_position: int = 0
    evidence_chain: list[str] = field(default_factory=list)
    trust_score: float = 0.5
    # LLD-00 §3 + P0.4: HMAC marker emitted during recall so post-tool hooks
    # can validate that a fact_id observed in tool output really came from
    # this install. Empty string by default preserves backward-compat.
    marker: str = ""


@dataclass
class RecallResponse:
    """Complete recall response with ranked results and metadata."""

    query: str = ""
    mode: Mode = Mode.A
    results: list[RetrievalResult] = field(default_factory=list)
    query_type: str = ""           # factual / temporal / opinion / multi-hop
    channel_weights: dict[str, float] = field(default_factory=dict)
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    # v3.6.6: Evidence floor. True when floor gates out ALL results.
    # Additive field — backward compatible (defaults to False).
    no_confident_match: bool = False
    score_contract_version: str = "2"
    calibration_status: str = "uncalibrated"
    calibration_id: str | None = None
    answer_confidence: float | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    # Cross-encoder execution is observable; cold/busy/error fallback must not
    # be mistaken for a reranked response.
    reranker_applied: bool = False
    reranker_status: str = "not_configured"
