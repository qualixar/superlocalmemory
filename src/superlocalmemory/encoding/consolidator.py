# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Memory Consolidator.

Mem0-style ADD/UPDATE/SUPERSEDE/NOOP logic for incoming facts.
V1 was append-only (never updated, never deleted, never merged).
This module gives a ~26% uplift by deduplicating, updating, and
resolving contradictions at encoding time.

Mode A: keyword-based contradiction detection (zero LLM).
Mode B/C: LLM-assisted contradiction detection when available.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: Elastic-2.0
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.encoding.policy import MergeAction, MergePolicy
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    ConsolidationAction,
    ConsolidationActionType,
    EdgeType,
    GraphEdge,
    MemoryLifecycle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols (avoid tight coupling to concrete classes)
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that produces an embedding vector from text."""

    def encode(self, text: str) -> list[float]: ...


class LLM(Protocol):
    """Anything that can generate text from a prompt."""

    def generate(self, prompt: str, system: str = "") -> str: ...

    def is_available(self) -> bool: ...


class TrustScorerProtocol(Protocol):
    """Anything that can look up an agent's Bayesian trust score tau(s)."""

    def get_agent_trust(self, agent_id: str, profile_id: str) -> float: ...


# ---------------------------------------------------------------------------
# Negation patterns for Mode A keyword contradiction detection
# ---------------------------------------------------------------------------

_NEGATION_MARKERS: frozenset[str] = frozenset({
    "not", "no longer", "never", "stopped", "quit", "left",
    "changed", "moved", "divorced", "fired", "resigned",
    "broke up", "ended", "cancelled", "dropped", "switched",
    "former", "ex-", "previously", "used to", "no more",
})

# Score thresholds (match EncodingConfig defaults)
_NOOP_THRESHOLD = 0.95
_MATCH_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# MemoryConsolidator
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Decides ADD/UPDATE/SUPERSEDE/NOOP for each incoming fact.

    For each new fact the consolidator:
    1. Finds candidate matches via entity overlap + semantic similarity.
    2. Scores each candidate (Jaccard entity + cosine embedding).
    3. Classifies the relationship and executes the action.
    4. Logs every decision in ``consolidation_log``.

    Thread-safe — delegates all DB ops to DatabaseManager which holds a lock.
    """

    def __init__(
        self,
        db: DatabaseManager,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: EncodingConfig | None = None,
        trust_scorer: TrustScorerProtocol | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._cfg = config or EncodingConfig()
        self._trust_scorer = trust_scorer
        self._policy = MergePolicy(self._is_contradicting, self._cfg)

    # -- Public API ---------------------------------------------------------

    def consolidate(
        self, new_fact: AtomicFact, profile_id: str, agent_id: str = "unknown",
    ) -> ConsolidationAction:
        """Consolidate *new_fact* against existing knowledge.

        Formalizes ``pi(a_t | o_t, r_t)`` (Stage 3 of the belief-update
        framework): *new_fact* is the observation ``o_t``, the candidates
        found below are ``r_t`` (belief-aware retrieval doubling as the
        write path's candidate lookup — a pending/quarantined fact can
        itself surface here, which is how independent corroboration on
        first mention is detected). ``MergePolicy`` selects the action;
        this method executes it via the ``_execute_*`` primitives.

        Returns a ``ConsolidationAction`` describing what was done.
        """
        new_fact.source_agent_id = new_fact.source_agent_id or agent_id
        candidates = self._find_candidates(new_fact, profile_id)
        tau = (
            self._trust_scorer.get_agent_trust(agent_id, profile_id)
            if self._trust_scorer is not None else 0.5
        )
        decision = self._policy.select(new_fact, candidates, tau)

        if decision.action == MergeAction.ADD:
            return self._execute_add(new_fact, profile_id, reason=decision.reason)
        if decision.action == MergeAction.ADD_QUARANTINE:
            return self._execute_add_quarantine(new_fact, profile_id, reason=decision.reason)
        if decision.action == MergeAction.UPDATE:
            return self._execute_update(
                new_fact, decision.best_fact, profile_id, reason=decision.reason,
            )
        if decision.action == MergeAction.UPDATE_QUARANTINE:
            return self._execute_quarantine_merge(
                new_fact, decision.best_fact, profile_id, reason=decision.reason,
            )
        if decision.action == MergeAction.CORROBORATE:
            return self._execute_corroborate(
                new_fact, decision.best_fact, profile_id, reason=decision.reason,
            )
        if decision.action == MergeAction.STILL_PENDING:
            return self._execute_still_pending(
                new_fact, decision.best_fact, profile_id, reason=decision.reason,
            )
        if decision.action == MergeAction.SUPERSEDE:
            return self._execute_supersede(
                new_fact, decision.best_fact, profile_id, reason=decision.reason,
            )
        return self._execute_noop(
            new_fact, decision.best_fact, profile_id, reason=decision.reason,
        )

    def get_consolidation_history(
        self, profile_id: str, limit: int = 50,
    ) -> list[ConsolidationAction]:
        """Recent consolidation actions for audit/debugging."""
        rows = self._db.execute(
            "SELECT * FROM consolidation_log WHERE profile_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [
            ConsolidationAction(
                action_id=(d := dict(r))["action_id"],
                profile_id=d["profile_id"],
                action_type=ConsolidationActionType(d["action_type"]),
                new_fact_id=d["new_fact_id"],
                existing_fact_id=d["existing_fact_id"],
                reason=d["reason"],
                timestamp=d["timestamp"],
            )
            for r in rows
        ]

    # -- Candidate search ---------------------------------------------------

    def _find_candidates(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> list[tuple[AtomicFact, float]]:
        """Find and score candidate matches from existing facts.

        Uses two signals:
        1. Entity overlap — ``get_facts_by_entity`` for each canonical entity.
        2. Semantic similarity — cosine of embedding vectors.

        Returns sorted list of (fact, combined_score), descending.
        """
        seen_ids: set[str] = set()
        candidate_facts: list[AtomicFact] = []

        # --- entity-based candidates ---
        for entity in new_fact.canonical_entities:
            for fact in self._db.get_facts_by_entity(entity, profile_id):
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    candidate_facts.append(fact)

        # --- semantic candidates (top-K by embedding) ---
        if new_fact.embedding is not None and self._embedder is not None:
            all_facts = self._db.get_all_facts(profile_id)
            semantic_scored: list[tuple[AtomicFact, float]] = []
            for fact in all_facts:
                if fact.fact_id in seen_ids:
                    continue
                sim = _compute_similarity(new_fact.embedding, fact.embedding)
                if sim > 0.5:
                    semantic_scored.append((fact, sim))
            semantic_scored.sort(key=lambda t: t[1], reverse=True)
            for fact, _ in semantic_scored[: self._cfg.max_consolidation_candidates]:
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    candidate_facts.append(fact)

        # --- score all candidates ---
        scored: list[tuple[AtomicFact, float]] = []
        for cand in candidate_facts:
            entity_overlap = _jaccard(
                set(new_fact.canonical_entities),
                set(cand.canonical_entities),
            )
            semantic_sim = _compute_similarity(
                new_fact.embedding, cand.embedding,
            )
            combined = 0.4 * entity_overlap + 0.6 * semantic_sim
            scored.append((cand, combined))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    # -- Contradiction detection --------------------------------------------

    def _is_contradicting(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Detect if *fact_a* contradicts *fact_b*.

        Mode B/C: delegates to LLM for nuanced judgment.
        Mode A: keyword-based negation detection.
        """
        if self._llm is not None and self._llm.is_available():
            return self._llm_contradiction_check(fact_a, fact_b)
        return self._keyword_contradiction_check(fact_a, fact_b)

    def _keyword_contradiction_check(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Heuristic: check negation markers in either fact's content."""
        text_a = fact_a.content.lower()
        text_b = fact_b.content.lower()
        for marker in _NEGATION_MARKERS:
            if marker in text_a and marker not in text_b:
                return True
            if marker in text_b and marker not in text_a:
                return True

        # Opposing emotional valence with same entities
        shared_entities = set(fact_a.canonical_entities) & set(fact_b.canonical_entities)
        if shared_entities:
            valence_diff = abs(fact_a.emotional_valence - fact_b.emotional_valence)
            if valence_diff > 1.2:
                return True

        return False

    def _llm_contradiction_check(
        self, fact_a: AtomicFact, fact_b: AtomicFact,
    ) -> bool:
        """Ask the LLM whether two facts contradict each other."""
        assert self._llm is not None  # guarded by caller
        prompt = (
            "Do these two statements contradict each other?\n\n"
            f"Statement A: {fact_a.content}\n"
            f"Statement B: {fact_b.content}\n\n"
            "Answer ONLY 'yes' or 'no'."
        )
        response = self._llm.generate(
            prompt, system="You are a precise fact-checker.",
        )
        return response.strip().lower().startswith("yes")

    # -- Action executors ---------------------------------------------------

    def _execute_add(
        self, new_fact: AtomicFact, profile_id: str, *, reason: str,
    ) -> ConsolidationAction:
        """Commit a brand-new fact as a trusted belief (single-source bootstrap).

        Confidence is clamped into ``bootstrap_confidence_range`` — the
        diagram's "Commit — bootstrap p in [0.7, 0.9]" — rather than
        overwritten outright, so extraction-quality signal already on the
        fact is preserved where it already falls in range.
        """
        lo, hi = self._cfg.bootstrap_confidence_range
        new_fact.confidence = min(max(new_fact.confidence, lo), hi)
        new_fact.pending_corroboration = False
        self._db.store_fact(new_fact)
        self._create_semantic_edges(new_fact, profile_id)
        action = self._log_action(
            ConsolidationActionType.ADD, new_fact.fact_id, "", profile_id, reason,
        )
        logger.debug("ADD fact %s (confidence=%.3f): %s", new_fact.fact_id, new_fact.confidence, reason)
        return action

    def _execute_add_quarantine(
        self, new_fact: AtomicFact, profile_id: str, *, reason: str,
    ) -> ConsolidationAction:
        """Store a brand-new fact quarantined, awaiting independent corroboration."""
        new_fact.pending_corroboration = True
        new_fact.corroboration_agents = (
            [new_fact.source_agent_id] if new_fact.source_agent_id else []
        )
        self._db.store_fact(new_fact)
        self._create_semantic_edges(new_fact, profile_id)
        action = self._log_action(
            ConsolidationActionType.NOOP, new_fact.fact_id, "", profile_id, reason,
        )
        logger.debug("ADD_QUARANTINE fact %s: %s", new_fact.fact_id, reason)
        return action

    def _execute_update(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Trust-gated noisy-OR merge: commit new evidence into the existing belief.

        Only reached once ``MergePolicy`` has already confirmed
        ``tau(s) * delta`` clears ``merge_trust_threshold`` — this method
        performs the commit unconditionally.
        """
        delta = new_fact.confidence
        new_evidence = existing.evidence_count + 1
        new_confidence = min(1 - (1 - existing.confidence) * (1 - delta), 0.99)
        agents = list(existing.corroboration_agents)
        if new_fact.source_agent_id and new_fact.source_agent_id not in agents:
            agents.append(new_fact.source_agent_id)
        updates: dict[str, Any] = {
            "evidence_count": new_evidence,
            "confidence": new_confidence,
            "corroboration_agents_json": agents,
        }

        # If LLM available, merge content for a richer fact
        if self._llm is not None and self._llm.is_available():
            merged = self._merge_facts(existing.content, new_fact.content)
            if merged:
                updates["content"] = merged

        self._db.update_fact(existing.fact_id, updates)
        action = self._log_action(
            ConsolidationActionType.UPDATE,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug(
            "UPDATE fact %s (evidence=%d, confidence=%.3f <- noisy-OR delta=%.3f): %s",
            existing.fact_id, new_evidence, new_confidence, delta, reason,
        )
        return action

    def _execute_quarantine_merge(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Reject an untrusted/uncorroborated merge — quarantine, leave the belief untouched."""
        self._db.update_fact(existing.fact_id, {"pending_corroboration": 1})
        action = self._log_action(
            ConsolidationActionType.NOOP,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug("UPDATE_QUARANTINE %s (merge rejected): %s", existing.fact_id, reason)
        return action

    def _execute_corroborate(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Independent corroboration received — commit the pending fact."""
        lo, hi = self._cfg.bootstrap_confidence_range
        commit_confidence = (lo + hi) / 2.0
        agents = list(existing.corroboration_agents)
        if new_fact.source_agent_id and new_fact.source_agent_id not in agents:
            agents.append(new_fact.source_agent_id)
        self._db.update_fact(existing.fact_id, {
            "pending_corroboration": 0,
            "confidence": commit_confidence,
            "evidence_count": existing.evidence_count + 1,
            "corroboration_agents_json": agents,
        })
        action = self._log_action(
            ConsolidationActionType.UPDATE,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug(
            "CORROBORATE %s -> confidence=%.3f: %s",
            existing.fact_id, commit_confidence, reason,
        )
        return action

    def _execute_still_pending(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Same source repeats an uncorroborated claim — no state change, stays quarantined."""
        self._db.update_fact(existing.fact_id, {"access_count": existing.access_count + 1})
        action = self._log_action(
            ConsolidationActionType.NOOP,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug("STILL_PENDING %s: %s", existing.fact_id, reason)
        return action

    def _execute_supersede(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Archive old fact, store new, create contradiction edge."""
        # Archive old fact (keep for history but deprioritize in retrieval)
        self._db.update_fact(
            existing.fact_id,
            {"lifecycle": MemoryLifecycle.ARCHIVED},
        )
        # Store new fact
        self._db.store_fact(new_fact)
        # Create contradiction + supersedes edges
        self._db.store_edge(GraphEdge(
            profile_id=profile_id,
            source_id=new_fact.fact_id,
            target_id=existing.fact_id,
            edge_type=EdgeType.CONTRADICTION,
            weight=1.0,
        ))
        self._db.store_edge(GraphEdge(
            profile_id=profile_id,
            source_id=new_fact.fact_id,
            target_id=existing.fact_id,
            edge_type=EdgeType.SUPERSEDES,
            weight=1.0,
        ))
        action = self._log_action(
            ConsolidationActionType.SUPERSEDE,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug(
            "SUPERSEDE %s → %s: %s",
            new_fact.fact_id, existing.fact_id, reason,
        )
        return action

    def _execute_noop(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        profile_id: str,
        *,
        reason: str,
    ) -> ConsolidationAction:
        """Near-duplicate — just bump access count on existing."""
        self._db.update_fact(
            existing.fact_id,
            {"access_count": existing.access_count + 1},
        )
        action = self._log_action(
            ConsolidationActionType.NOOP,
            new_fact.fact_id, existing.fact_id,
            profile_id, reason,
        )
        logger.debug("NOOP for %s (dup of %s): %s", new_fact.fact_id, existing.fact_id, reason)
        return action

    # -- Helpers ------------------------------------------------------------

    def _create_semantic_edges(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> None:
        """Link new fact to top-K most similar existing facts."""
        if new_fact.embedding is None:
            return
        all_facts = self._db.get_all_facts(profile_id)
        scored: list[tuple[str, float]] = []
        for fact in all_facts:
            if fact.fact_id == new_fact.fact_id:
                continue
            sim = _compute_similarity(new_fact.embedding, fact.embedding)
            if sim > 0.5:
                scored.append((fact.fact_id, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        for target_id, weight in scored[: self._cfg.semantic_edge_top_k]:
            self._db.store_edge(GraphEdge(
                profile_id=profile_id,
                source_id=new_fact.fact_id,
                target_id=target_id,
                edge_type=EdgeType.SEMANTIC,
                weight=weight,
            ))

    def _merge_facts(self, existing_text: str, new_text: str) -> str:
        """Use LLM to merge two fact statements into one richer fact."""
        assert self._llm is not None
        prompt = (
            "Merge these two statements about the same topic into one "
            "concise, accurate fact. Keep all unique information.\n\n"
            f"Existing: {existing_text}\n"
            f"New: {new_text}\n\n"
            "Merged statement:"
        )
        result = self._llm.generate(prompt, system="You are a precise editor.")
        return result.strip() if result.strip() else ""

    def _log_action(
        self,
        action_type: ConsolidationActionType,
        new_fact_id: str,
        existing_fact_id: str,
        profile_id: str,
        reason: str,
    ) -> ConsolidationAction:
        """Persist action to consolidation_log and return the model."""
        action = ConsolidationAction(
            profile_id=profile_id,
            action_type=action_type,
            new_fact_id=new_fact_id,
            existing_fact_id=existing_fact_id,
            reason=reason,
        )
        self._db.execute(
            "INSERT INTO consolidation_log "
            "(action_id, profile_id, action_type, new_fact_id, "
            " existing_fact_id, reason, timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                action.action_id, action.profile_id,
                action.action_type.value, action.new_fact_id,
                action.existing_fact_id, action.reason, action.timestamp,
            ),
        )
        return action


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no side effects)
# ---------------------------------------------------------------------------

def _compute_similarity(
    emb_a: list[float] | None, emb_b: list[float] | None,
) -> float:
    """Cosine similarity between two embedding vectors.

    Returns 0.0 if either embedding is None or empty.
    """
    if not emb_a or not emb_b:
        return 0.0
    if len(emb_a) != len(emb_b):
        return 0.0

    dot = sum(a * b for a, b in zip(emb_a, emb_b))
    norm_a = math.sqrt(sum(a * a for a in emb_a))
    norm_b = math.sqrt(sum(b * b for b in emb_b))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity of two string sets. Returns 0.0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
