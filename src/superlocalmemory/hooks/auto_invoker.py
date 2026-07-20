# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.2 | https://qualixar.com

"""Auto-Invoke Engine -- multi-signal memory retrieval.

Drop-in replacement for AutoRecall. Surfaces memories automatically
when context triggers them, using 4-signal (or 3-signal ACT-R) scoring.

NEVER imports core/engine.py (Rule 06).
Components received via __init__, not the engine itself.

References:
  - SYNAPSE: FOK gating (tau_gate = 0.12)
  - ACT-R: base-level activation (Anderson & Lebiere 1998)
  - Zep/Hindsight: multi-signal ranking consensus
  - A-MEM: contextual description enrichment

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import math

from superlocalmemory.core.config import AutoInvokeConfig
from superlocalmemory.core.injection import InjectableMemory, render_context

logger = logging.getLogger(__name__)


class AutoInvoker:
    """Multi-signal memory auto-invocation.

    Replaces AutoRecall with richer scoring. Maintains exact same
    public interface for backward compatibility (Rule 16 / AI-04).

    Components received via __init__ (NOT the engine itself -- Rule 06):
    - db: DatabaseManager (for access_log, fact_context queries)
    - vector_store: VectorStore or None (for KNN similarity)
    - trust_scorer: TrustScorer (for per-fact trust)
    - embedder: EmbeddingService (for query encoding)
    - config: AutoInvokeConfig
    - prompt_injector: PromptInjector or None (V3.3 soft prompt injection)
    """

    # Lifecycle zones that are excluded from auto-invoke results.
    # "archived" was always skipped; "forgotten" added in V3.3 for
    # forgetting-aware auto-invoke (Phase A integration).
    _EXCLUDED_ZONES: frozenset[str] = frozenset({"archived", "forgotten"})

    def __init__(
        self,
        db,                    # DatabaseManager
        vector_store=None,     # VectorStore (Phase 1) or None
        trust_scorer=None,     # TrustScorer (existing)
        embedder=None,         # EmbeddingService for query encoding
        config=None,           # AutoInvokeConfig
        prompt_injector=None,  # PromptInjector (V3.3 soft prompt injection)
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._trust_scorer = trust_scorer
        self._embedder = embedder
        self._config = config or AutoInvokeConfig()
        self._prompt_injector = prompt_injector

    # ------------------------------------------------------------------
    # Public API: AutoRecall-compatible interface (Rule 16 / AI-04)
    # ------------------------------------------------------------------

    def get_session_context(self, project_path: str = "", query: str = "") -> str:
        """Get relevant context for a session or query.

        EXACT same signature as AutoRecall.get_session_context().
        Returns a formatted string of relevant memories suitable
        for injection into an AI's system prompt.

        V3.3: If a PromptInjector is wired, soft prompts are prepended
        to the memory context with priority (soft prompts first).
        """
        if not self._config.enabled:
            return ""

        try:
            results = self.invoke(
                query=query or f"project context {project_path}",
                profile_id=self._config.profile_id,
                limit=self._config.max_memories_injected,
            )

            soft_prompt_text = self._get_soft_prompt_text()
            if soft_prompt_text:
                results = [
                    {
                        "fact_id": "",
                        "content": soft_prompt_text,
                        "fact_type": "behavioral-pattern",
                        "score": 0.0,
                        "contextual_description": "",
                    },
                    *results,
                ]
            return self.format_for_injection(results) if results else ""
        except Exception as exc:
            logger.debug("Auto-invoke failed: %s", exc)
            return ""

    def get_query_context(self, query: str) -> list[dict]:
        """Get relevant memories for a specific query.

        EXACT same signature as AutoRecall.get_query_context().
        Returns structured data for MCP tools.
        """
        if not self._config.enabled:
            return []

        try:
            results = self.invoke(
                query=query,
                profile_id=self._config.profile_id,
                limit=self._config.max_memories_injected,
            )
            return [
                {
                    "fact_id": r["fact_id"],
                    "content": r["content"][:300],
                    "score": round(r["score"], 3),
                    "context": r.get("contextual_description", ""),
                }
                for r in results
            ]
        except Exception as exc:
            logger.debug("Auto-invoke query failed: %s", exc)
            return []

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def enable(self) -> None:
        """Enable auto-invoke. Creates new frozen config."""
        fields = {
            k: v for k, v in self._config.__dict__.items() if k != "enabled"
        }
        self._config = AutoInvokeConfig(enabled=True, **fields)

    def disable(self) -> None:
        """Disable auto-invoke. Creates new frozen config."""
        fields = {
            k: v for k, v in self._config.__dict__.items() if k != "enabled"
        }
        self._config = AutoInvokeConfig(enabled=False, **fields)

    # ------------------------------------------------------------------
    # Core: invoke() -- multi-signal scoring
    # ------------------------------------------------------------------

    def invoke(
        self,
        query: str,
        profile_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """Run multi-signal scoring to find relevant memories.

        Algorithm:
        1. Get candidate facts via VectorStore KNN (or BM25 fallback).
        2. For each candidate, compute multi-signal score.
        3. Apply FOK gating (reject scores < threshold).
        4. Return top-K ranked results with contextual descriptions.

        Returns list of dicts: {fact_id, content, score, signals, contextual_description}
        """
        # Step 1: Get candidates
        candidates = self._get_candidates(
            query, profile_id, top_k=limit * self._config.candidate_multiplier,
        )

        if not candidates:
            return []

        # Step 2: Score each candidate
        scored = []
        for fact_id, similarity in candidates:
            signals = self._compute_signals(fact_id, profile_id, similarity)
            score = self._combine_signals(signals)
            scored.append((fact_id, score, signals))

        # Step 3: Sort by combined score descending
        scored.sort(key=lambda t: t[1], reverse=True)

        # Step 4: FOK gating -- reject below threshold
        gated = [
            (fid, score, signals)
            for fid, score, signals in scored
            if score >= self._config.fok_threshold
        ]

        # Step 5: Enrich with fact content and contextual description
        results = []
        for fact_id, score, signals in gated[:limit]:
            fact_data = self._enrich_result(fact_id, score, signals, profile_id)
            if fact_data:
                results.append(fact_data)

        return results

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _get_candidates(
        self, query: str, profile_id: str, top_k: int = 30,
    ) -> list[tuple[str, float]]:
        """Get candidate facts via VectorStore KNN or text search fallback.

        Returns list of (fact_id, similarity_score).
        """
        # Primary: VectorStore KNN (Phase 1)
        if self._vector_store is not None and self._embedder is not None:
            try:
                query_embedding = self._embedder.embed(query)
                if query_embedding:
                    return self._vector_store.search(
                        query_embedding, top_k=top_k, profile_id=profile_id,
                    )
            except Exception as exc:
                logger.debug("VectorStore search failed: %s", exc)

        # Fallback: text search for candidates (Mode A degradation)
        # V3.3: Exclude archived/forgotten facts from candidates
        try:
            rows = self._db.execute(
                "SELECT fact_id FROM atomic_facts "
                "WHERE profile_id = ? AND content LIKE ? "
                "AND COALESCE(lifecycle, 'active') NOT IN ('archived', 'forgotten') "
                "ORDER BY access_count DESC LIMIT ?",
                (profile_id, f"%{query[:50]}%", top_k),
            )
            # Text fallback: similarity=0 (per Mode A weights)
            return [(dict(r)["fact_id"], 0.0) for r in rows]
        except Exception as exc:
            logger.debug("Text search fallback failed for profile %s: %s", profile_id, exc)
            return []

    def _compute_signals(
        self, fact_id: str, profile_id: str, similarity: float,
    ) -> dict[str, float]:
        """Compute all signals for a single fact.

        Returns dict with keys: similarity, recency, frequency, trust,
        and optionally base_level (ACT-R mode).
        """
        signals: dict[str, float] = {"similarity": similarity}

        # Recency: from fact_access_log.MAX(accessed_at) [H1 fix / AI-17]
        signals["recency"] = self._compute_recency(fact_id, profile_id)

        # Frequency: from atomic_facts.access_count
        signals["frequency"] = self._compute_frequency(fact_id, profile_id)

        # Trust: from TrustScorer Bayesian Beta distribution
        signals["trust"] = self._compute_trust(fact_id, profile_id)

        # Optional: ACT-R base-level activation (combines recency + frequency)
        if self._config.use_act_r:
            signals["base_level"] = self._compute_act_r_base_level(
                fact_id, profile_id,
            )

        return signals

    def _compute_recency(self, fact_id: str, profile_id: str) -> float:
        """Recency from fact_access_log.MAX(accessed_at) [AI-17].

        Formula: exp(-0.01 * seconds_since_last_access)
        Returns 0.1 if never accessed (cold start default).
        Range: [0.01, 1.0]
        """
        try:
            rows = self._db.execute(
                "SELECT MAX(accessed_at) as last_access "
                "FROM fact_access_log "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
            if rows:
                last_access = dict(rows[0]).get("last_access")
                if last_access:
                    from datetime import UTC, datetime
                    last_dt = datetime.fromisoformat(last_access)
                    now = datetime.now(UTC)
                    seconds_since = max(
                        0, (now - last_dt.replace(tzinfo=UTC)).total_seconds(),
                    )
                    return max(0.01, math.exp(-0.01 * seconds_since))
        except Exception as exc:
            logger.debug("Recency computation failed for fact %s: %s", fact_id, exc)

        return 0.1  # Cold start default

    def _compute_frequency(self, fact_id: str, profile_id: str) -> float:
        """Frequency from access_count, normalized via log1p.

        Formula: log1p(access_count) / log1p(max_access_count_in_profile)
        Range: [0.0, 1.0]
        """
        try:
            rows = self._db.execute(
                "SELECT access_count FROM atomic_facts WHERE fact_id = ?",
                (fact_id,),
            )
            count = dict(rows[0])["access_count"] if rows else 0

            max_rows = self._db.execute(
                "SELECT MAX(access_count) as max_count "
                "FROM atomic_facts WHERE profile_id = ?",
                (profile_id,),
            )
            max_count = dict(max_rows[0]).get("max_count", 1) if max_rows else 1
            max_count = max(max_count, 1)

            return math.log1p(count) / math.log1p(max_count)
        except Exception as exc:
            logger.debug("Frequency computation failed for fact %s: %s", fact_id, exc)
            return 0.0

    def _compute_trust(self, fact_id: str, profile_id: str) -> float:
        """Trust score from TrustScorer Bayesian Beta distribution.

        Default: 0.5 (uniform prior).
        Range: [0.0, 1.0]
        """
        if self._trust_scorer is None:
            return 0.5
        try:
            return self._trust_scorer.get_fact_trust(fact_id, profile_id)
        except Exception:
            return 0.5

    def _compute_act_r_base_level(
        self, fact_id: str, profile_id: str,
    ) -> float:
        """ACT-R base-level activation [L21 fix].

        Formula: B_i = ln(SUM_k (t_k)^(-d))
        Where t_k = seconds since k-th access, d = decay parameter.

        Combines recency AND frequency into a single signal.
        Returns 0.0 if no access history.

        Reference: Anderson & Lebiere 1998, "The Atomic Components of Thought"
        """
        try:
            rows = self._db.execute(
                "SELECT accessed_at FROM fact_access_log "
                "WHERE fact_id = ? AND profile_id = ? "
                "ORDER BY accessed_at DESC LIMIT 100",
                (fact_id, profile_id),
            )
            if not rows:
                return 0.0

            from datetime import UTC, datetime
            now = datetime.now(UTC)
            decay = self._config.act_r_decay

            summation = 0.0
            for row in rows:
                accessed = dict(row)["accessed_at"]
                try:
                    t_dt = datetime.fromisoformat(accessed)
                    t_seconds = max(
                        1.0, (now - t_dt.replace(tzinfo=UTC)).total_seconds(),
                    )
                    summation += t_seconds ** (-decay)
                except (ValueError, TypeError):
                    continue

            if summation <= 0:
                return 0.0

            # ln(sum) can be negative for small sums; normalize to [0, 1]
            raw = math.log(summation)
            # Sigmoid normalization to [0, 1] range
            return 1.0 / (1.0 + math.exp(-raw))

        except Exception as exc:
            logger.debug(
                "ACT-R base-level computation failed for fact %s: %s",
                fact_id, exc,
            )
            return 0.0

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def _combine_signals(self, signals: dict[str, float]) -> float:
        """Combine signals into a single score using configured weights.

        4-signal mode (default):
          score = 0.40*sim + 0.25*rec + 0.20*freq + 0.15*trust

        3-signal ACT-R mode (optional):
          score = 0.40*sim + 0.35*base_level + 0.25*trust

        Mode A degradation (no embeddings):
          score = 0.00*sim + 0.40*rec + 0.35*freq + 0.25*trust
        """
        # Mode A degradation: no embeddings available
        if (
            signals.get("similarity", 0.0) == 0.0
            and self._vector_store is None
        ):
            weights = self._config.mode_a_weights
        elif self._config.use_act_r and "base_level" in signals:
            # 3-signal ACT-R mode
            weights = self._config.act_r_weights
            return (
                weights.get("similarity", 0.40) * signals.get("similarity", 0.0)
                + weights.get("base_level", 0.35) * signals.get("base_level", 0.0)
                + weights.get("trust", 0.25) * signals.get("trust", 0.5)
            )
        else:
            weights = self._config.weights

        # 4-signal default mode (or Mode A degradation)
        return (
            weights.get("similarity", 0.40) * signals.get("similarity", 0.0)
            + weights.get("recency", 0.25) * signals.get("recency", 0.0)
            + weights.get("frequency", 0.20) * signals.get("frequency", 0.0)
            + weights.get("trust", 0.15) * signals.get("trust", 0.5)
        )

    # ------------------------------------------------------------------
    # Result enrichment
    # ------------------------------------------------------------------

    def _enrich_result(
        self, fact_id: str, score: float, signals: dict, profile_id: str,
    ) -> dict | None:
        """Load full fact + contextual description for a scored result."""
        try:
            fact_rows = self._db.execute(
                "SELECT fact_id, content, fact_type, lifecycle "
                "FROM atomic_facts WHERE fact_id = ?",
                (fact_id,),
            )
            if not fact_rows:
                return None
            fact_data = dict(fact_rows[0])

            # Skip archived/forgotten facts unless config allows (V3.3: forgetting-aware)
            lifecycle = fact_data.get("lifecycle", "")
            if lifecycle in self._EXCLUDED_ZONES and not self._config.include_archived:
                return None

            # Get contextual description
            ctx = self._db.get_fact_context(fact_id)
            ctx_desc = ctx["contextual_description"] if ctx else ""

            return {
                "fact_id": fact_id,
                "content": fact_data["content"],
                "fact_type": fact_data.get("fact_type", "semantic"),
                "score": score,
                "signals": signals,
                "contextual_description": ctx_desc,
            }
        except Exception as exc:
            logger.debug("Result enrichment failed for %s: %s", fact_id, exc)
            return None

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def format_for_injection(self, results: list[dict]) -> str:
        """Format results as bounded, untrusted evidence with provenance."""
        if not results:
            return ""

        memories: list[InjectableMemory] = []
        for r in results:
            content = str(r.get("content", ""))
            ctx = r.get("contextual_description", "")
            if ctx:
                content += f"\nContext: {ctx}"
            memories.append(InjectableMemory(
                content=content,
                score=float(r.get("score", 0.0) or 0.0),
                fact_id=str(r.get("fact_id", "")),
                source_type=str(r.get("fact_type", "auto-invoke")),
                source_id=f"fok-threshold:{self._config.fok_threshold}",
            ))
        return render_context(memories, mode="B", cfg=None, wrap=True)

    # ------------------------------------------------------------------
    # V3.3: Soft prompt injection
    # ------------------------------------------------------------------

    def _get_soft_prompt_text(self) -> str:
        """Retrieve soft prompt text via PromptInjector (V3.3).

        Returns assembled soft prompt text, or "" if injector is not
        wired or no active soft prompts exist. Errors are logged and
        swallowed -- soft prompt failure MUST NOT block auto-invoke.
        """
        if self._prompt_injector is None:
            return ""
        try:
            return self._prompt_injector.get_injection_context(
                self._config.profile_id,
            )
        except Exception as exc:
            logger.debug("Soft prompt injection failed: %s", exc)
            return ""
