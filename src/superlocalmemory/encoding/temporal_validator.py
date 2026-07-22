# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Temporal Intelligence -- contradiction detection and fact invalidation.

Implements full bi-temporal validity tracking with 4 timestamps (L8 fix).
Contradiction detection via sheaf cohomology (Mode A: pure math) or
LLM verification (Mode B/C).

References:
  - Zep/Graphiti: bi-temporal model (t_valid, t_invalid, t_created, t_expired)
  - SLM sheaf.py: coboundary norm for contradiction severity
  - Mem0 consolidator: SUPERSEDE action pattern

NEVER imports core/engine.py (Rule 06).
Receives components via __init__.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import TemporalValidatorConfig
    from superlocalmemory.math.sheaf import SheafConsistencyChecker
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.trust.scorer import TrustScorer

from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)


class TemporalValidator:
    """Validates temporal consistency and manages fact invalidation.

    Components received via __init__ (NOT the engine -- Rule 06):
    - db: DatabaseManager
    - sheaf_checker: SheafConsistencyChecker (existing, for Mode A)
    - trust_scorer: TrustScorer (existing, for trust penalty)
    - llm: LLMBackbone or None (for Mode B/C verification)
    - config: TemporalValidatorConfig
    """

    # Trust penalty for expired facts
    EXPIRATION_TRUST_PENALTY: float = -0.2

    def __init__(
        self,
        db: DatabaseManager,
        sheaf_checker: SheafConsistencyChecker | None = None,
        trust_scorer: TrustScorer | None = None,
        llm: Any | None = None,
        config: TemporalValidatorConfig | None = None,
    ) -> None:
        self._db = db
        self._sheaf_checker = sheaf_checker
        self._trust_scorer = trust_scorer
        self._llm = llm
        if config is None:
            from superlocalmemory.core.config import TemporalValidatorConfig as _TVC
            config = _TVC()
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_and_invalidate(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> list[dict]:
        """Check new fact for contradictions and invalidate old facts.

        Algorithm:
        1. Detect contradictions (sheaf or LLM).
        2. For each contradiction: invalidate old fact (set valid_until + system_expired_at).
        3. Apply trust penalty to invalidated facts.
        4. Return list of invalidation actions.

        Returns list of dicts: {old_fact_id, new_fact_id, reason, severity}
        """
        contradictions = self.detect_contradiction(new_fact, profile_id)

        if not contradictions:
            return []

        actions: list[dict] = []
        for contradiction in contradictions:
            old_fact_id = contradiction["fact_id_b"]
            severity = contradiction["severity"]
            reason = contradiction["description"]

            # Step 1: Invalidate the old fact (bi-temporal)
            self.invalidate_fact(
                fact_id=old_fact_id,
                invalidated_by=new_fact.fact_id,
                reason=reason,
            )

            # Step 2: Apply trust penalty
            self._apply_trust_penalty(old_fact_id, profile_id)

            actions.append({
                "old_fact_id": old_fact_id,
                "new_fact_id": new_fact.fact_id,
                "reason": reason,
                "severity": severity,
            })

        logger.info(
            "Temporal: invalidated %d facts due to new fact %s",
            len(actions), new_fact.fact_id,
        )
        return actions

    def detect_contradiction(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> list[dict]:
        """Detect contradictions between new fact and existing facts.

        Mode A: Sheaf consistency (pure math, no LLM).
        Mode B/C: LLM verification with sheaf as pre-filter.

        Returns list of dicts: {fact_id_a, fact_id_b, severity, edge_type, description}
        """
        mode = self._config.mode

        if mode == "a" or self._llm is None or not self._llm.is_available():
            return self._sheaf_contradiction(new_fact, profile_id)
        else:
            return self._llm_contradiction(new_fact, profile_id)

    def invalidate_fact(
        self,
        fact_id: str,
        invalidated_by: str,
        reason: str,
    ) -> None:
        """Set valid_until and system_expired_at for a fact.

        BI-TEMPORAL INTEGRITY: BOTH timestamps set in same operation.
        NEVER deletes the fact (Rule 17: immutability).
        Double invalidation is idempotent (TI-17).
        """
        try:
            # Check if temporal record exists
            existing = self._db.get_temporal_validity(fact_id)
            if existing and existing.get("valid_until") is not None:
                # Already invalidated -- skip (idempotent)
                logger.debug("Fact %s already invalidated, skipping", fact_id)
                return

            if existing:
                # Update existing record
                self._db.invalidate_fact_temporal(
                    fact_id=fact_id,
                    invalidated_by=invalidated_by,
                    invalidation_reason=reason,
                )
            else:
                # Create record then invalidate
                profile_rows = self._db.execute(
                    "SELECT profile_id FROM atomic_facts WHERE fact_id = ?",
                    (fact_id,),
                )
                if not profile_rows:
                    logger.debug("Fact %s not found, cannot invalidate", fact_id)
                    return
                pid = dict(profile_rows[0])["profile_id"]
                self._db.store_temporal_validity(fact_id, pid)
                self._db.invalidate_fact_temporal(
                    fact_id=fact_id,
                    invalidated_by=invalidated_by,
                    invalidation_reason=reason,
                )

            logger.debug(
                "Invalidated fact %s by %s: %s", fact_id, invalidated_by, reason,
            )
        except Exception as exc:
            logger.debug(
                "Fact invalidation failed for %s: %s", fact_id, exc,
            )

    def is_temporally_valid(self, fact_id: str, profile_id: str = "") -> bool:
        """Check if a fact is currently temporally valid.

        A fact is valid if:
        - No temporal record exists (assumed valid), OR
        - valid_until IS NULL AND system_expired_at IS NULL

        Args:
            fact_id: The fact to check.
            profile_id: Profile scope (accepted for API consistency with Rule 01,
                        but fact_id is PK so lookup is unambiguous).

        NOTE: Phase 5 calls this as is_temporally_valid(fact_id, profile_id).
        Both params are REQUIRED in the call site. Do NOT rename to is_valid().
        """
        try:
            # profile_id or None: an empty-string default must NOT become a
            # `profile_id = ''` filter (which never matches → an expired fact
            # would be reported valid). None falls back to the unscoped lookup.
            tv = self._db.get_temporal_validity(fact_id, profile_id or None)
            if tv is None:
                return True  # No temporal record = assumed valid
            return (
                tv.get("valid_until") is None
                and tv.get("system_expired_at") is None
            )
        except Exception:
            return True  # Fail open -- assume valid

    def get_facts_valid_at(
        self, profile_id: str, event_time: str,
    ) -> list[str]:
        """Get fact_ids that were valid at a specific event time.

        Queries: valid_from <= event_time AND (valid_until IS NULL OR valid_until > event_time)
        """
        try:
            rows = self._db.execute(
                "SELECT f.fact_id FROM atomic_facts f "
                "JOIN fact_temporal_validity tv ON f.fact_id = tv.fact_id "
                "WHERE f.profile_id = ? "
                "  AND (tv.valid_from IS NULL OR tv.valid_from <= ?) "
                "  AND (tv.valid_until IS NULL OR tv.valid_until > ?)",
                (profile_id, event_time, event_time),
            )
            return [dict(r)["fact_id"] for r in rows]
        except Exception as exc:
            logger.debug(
                "Temporal query failed for profile %s at %s: %s",
                profile_id, event_time, exc,
            )
            return []

    def get_system_knowledge_at(
        self, profile_id: str, transaction_time: str,
    ) -> list[str]:
        """Get fact_ids that the system knew about at a specific transaction time.

        Queries: system_created_at <= T AND (system_expired_at IS NULL OR system_expired_at > T)
        """
        try:
            rows = self._db.execute(
                "SELECT f.fact_id FROM atomic_facts f "
                "JOIN fact_temporal_validity tv ON f.fact_id = tv.fact_id "
                "WHERE f.profile_id = ? "
                "  AND tv.system_created_at <= ? "
                "  AND (tv.system_expired_at IS NULL OR tv.system_expired_at > ?)",
                (profile_id, transaction_time, transaction_time),
            )
            return [dict(r)["fact_id"] for r in rows]
        except Exception as exc:
            logger.debug(
                "System knowledge query failed for profile %s at %s: %s",
                profile_id, transaction_time, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Contradiction detection: Mode A (sheaf, pure math)
    # ------------------------------------------------------------------

    def _sheaf_contradiction(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> list[dict]:
        """Detect contradictions via sheaf coboundary norm.

        Uses existing SheafConsistencyChecker.check_consistency().
        No LLM needed -- pure linear algebra.
        """
        if self._sheaf_checker is None:
            return []

        try:
            results = self._sheaf_checker.check_consistency(new_fact, profile_id)
            return [
                {
                    "fact_id_a": r.fact_id_a,
                    "fact_id_b": r.fact_id_b,
                    "severity": r.severity,
                    "edge_type": r.edge_type,
                    "description": r.description,
                }
                for r in results
                if r.severity > self._config.contradiction_threshold
            ]
        except Exception as exc:
            logger.debug(
                "Sheaf contradiction check failed for fact %s: %s",
                new_fact.fact_id, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Contradiction detection: Mode B/C (LLM with sheaf pre-filter)
    # ------------------------------------------------------------------

    def _llm_contradiction(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> list[dict]:
        """Detect contradictions via LLM verification.

        Two-stage pipeline:
        1. Sheaf pre-filter: find candidates with coboundary > 0.3
        2. LLM verification: ask LLM to confirm each candidate.
        """
        contradictions: list[dict] = []

        # Stage 1: Sheaf pre-filter (get candidates)
        candidates: list[Any] = []
        if self._sheaf_checker is not None:
            try:
                results = self._sheaf_checker.check_consistency(
                    new_fact, profile_id,
                )
                candidates = [
                    r for r in results
                    if r.severity > self._config.llm_prefilter_threshold
                ]
            except Exception as exc:
                logger.debug(
                    "Sheaf pre-filter failed for fact %s: %s",
                    new_fact.fact_id, exc,
                )

        # If no sheaf results, find candidates by entity overlap
        if not candidates:
            candidates = self._entity_based_candidates(new_fact, profile_id)

        # Stage 2: LLM verification
        for candidate in candidates[: self._config.max_llm_checks]:
            other_fact_id = (
                candidate.fact_id_b
                if hasattr(candidate, "fact_id_b")
                else candidate
            )

            other_content = self._get_fact_content(other_fact_id)
            if not other_content:
                continue

            is_contradiction = self._llm_verify_contradiction(
                new_fact.content, other_content,
            )

            if is_contradiction:
                severity = getattr(candidate, "severity", 0.8)
                fact_b = (
                    other_fact_id
                    if isinstance(other_fact_id, str)
                    else candidate.fact_id_b
                )
                contradictions.append({
                    "fact_id_a": new_fact.fact_id,
                    "fact_id_b": fact_b,
                    "severity": severity,
                    "edge_type": getattr(
                        candidate, "edge_type", "llm_detected",
                    ),
                    "description": (
                        f"LLM-verified contradiction "
                        f"(sheaf pre-filter severity: {severity:.3f})"
                    ),
                })

        return contradictions

    def _llm_verify_contradiction(
        self, content_a: str, content_b: str,
    ) -> bool:
        """Ask LLM whether two statements contradict each other."""
        if self._llm is None or not self._llm.is_available():
            return False

        try:
            prompt = (
                "Do these two statements contradict each other? "
                "A contradiction means they cannot both be true "
                "at the same time.\n\n"
                f"Statement A: {content_a}\n"
                f"Statement B: {content_b}\n\n"
                "Answer ONLY 'yes' or 'no'."
            )
            response = self._llm.generate(
                prompt, system="You are a precise fact-checker.",
            )
            return response.strip().lower().startswith("yes")
        except Exception as exc:
            logger.debug("LLM contradiction check failed: %s", exc)
            return False

    def _entity_based_candidates(
        self, new_fact: AtomicFact, profile_id: str,
    ) -> list[str]:
        """Find contradiction candidates by entity overlap when sheaf unavailable."""
        candidates: list[str] = []
        try:
            for entity in new_fact.canonical_entities[:5]:
                rows = self._db.execute(
                    "SELECT DISTINCT fact_id FROM atomic_facts "
                    "WHERE profile_id = ? AND fact_id != ? "
                    "AND canonical_entities_json LIKE ?",
                    (profile_id, new_fact.fact_id, f"%{entity}%"),
                )
                for row in rows:
                    candidates.append(dict(row)["fact_id"])
        except Exception as exc:
            logger.debug(
                "Entity candidate search failed for fact %s: %s",
                new_fact.fact_id, exc,
            )
        return list(set(candidates))[:10]

    def _get_fact_content(self, fact_id: str) -> str | None:
        """Get fact content by ID."""
        try:
            rows = self._db.execute(
                "SELECT content FROM atomic_facts WHERE fact_id = ?",
                (fact_id,),
            )
            return dict(rows[0])["content"] if rows else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Trust penalty
    # ------------------------------------------------------------------

    def _apply_trust_penalty(self, fact_id: str, profile_id: str) -> None:
        """Apply trust penalty to an expired/invalidated fact.

        Uses TrustScorer.update_on_contradiction() which adds +3.0
        to beta parameter, reducing trust score.
        """
        if self._trust_scorer is None:
            return

        try:
            self._trust_scorer.update_on_contradiction(
                target_type="fact",
                target_id=fact_id,
                profile_id=profile_id,
            )
            logger.debug("Trust penalty applied to expired fact %s", fact_id)
        except Exception as exc:
            logger.debug(
                "Trust penalty failed for fact %s: %s", fact_id, exc,
            )


# ------------------------------------------------------------------
# Temporal validity filter (registered via ChannelRegistry)
# ------------------------------------------------------------------

def temporal_validity_filter_impl(
    channel_results: dict[str, list],
    profile_id: str,
    db: DatabaseManager,
    include_expired: bool = False,
) -> dict[str, list]:
    """Filter expired facts from all channel results.

    Handles both tuple format (fact_id, score) and dict format from channels.
    Called via closure wrapper registered in engine_wiring.py.
    """
    if include_expired:
        return channel_results

    try:
        # Get all expired fact_ids for this profile (single query)
        expired_rows = db.execute(
            "SELECT fact_id FROM fact_temporal_validity "
            "WHERE profile_id = ? "
            "  AND (valid_until IS NOT NULL OR system_expired_at IS NOT NULL)",
            (profile_id,),
        )
        expired_ids = {dict(r)["fact_id"] for r in expired_rows}
    except Exception:
        return channel_results

    if not expired_ids:
        return channel_results

    # Filter each channel's results
    filtered: dict[str, list] = {}
    for channel_name, results in channel_results.items():
        filtered[channel_name] = [
            item for item in results
            if _extract_fact_id(item) not in expired_ids
        ]

    return filtered


def _extract_fact_id(item: Any) -> str:
    """Extract fact_id from channel result item (tuple or dict)."""
    if isinstance(item, tuple):
        return item[0]  # (fact_id, score)
    if isinstance(item, dict):
        return item.get("fact_id", "")
    return str(item)
