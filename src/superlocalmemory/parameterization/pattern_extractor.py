# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3.3

"""PatternExtractor — Mine patterns from 4 sources for soft prompt generation.

Sources: core memory blocks, behavioral patterns, cross-project preferences,
workflow sequences. Includes deduplication (independence assumption) and
contradiction resolution (temporal ordering + close-confidence alternate marking).

[AUDIT FIX F-1] Sheaf removed — heuristic key-value comparison for contradictions.
[AUDIT FIX F-4] source_ids as tuple for true immutability.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.learning.behavioral import BehavioralPatternStore
    from superlocalmemory.learning.cross_project import CrossProjectAggregator
    from superlocalmemory.learning.workflows import WorkflowMiner
    from superlocalmemory.core.config import ParameterizationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# Keyword sets for classification heuristics
_IDENTITY_KEYWORDS = frozenset({
    "role", "title", "senior", "architect", "engineer", "manager",
    "developer", "designer", "expertise", "lead", "director", "analyst",
})
_TECH_KEYWORDS = frozenset({
    "typescript", "python", "react", "vue", "angular", "node", "java",
    "rust", "go", "framework", "library", "language", "database",
    "postgres", "mongodb", "redis", "aws", "azure", "gcp", "docker",
    "kubernetes", "tool", "sdk", "api",
})
_STYLE_KEYWORDS = frozenset({
    "prefer", "style", "tone", "format", "concise", "verbose",
    "detailed", "brief", "formal", "casual",
})
_AVOIDANCE_KEYWORDS = frozenset({
    "avoid", "never", "don't", "dont", "stop", "hate", "dislike",
    "not use", "refuse",
})
_DECISION_KEYWORDS = frozenset({
    "decided", "chose", "selected", "picked", "switched", "migrated",
    "adopted", "dropped",
})

# Behavioral pattern type -> PatternCategory mapping
_BEHAVIORAL_TYPE_MAP: dict[str, str] = {
    "entity_pref": "tech_preference",
    "query_type": "workflow_pattern",
    "time_of_day": "workflow_pattern",
    "refinement": "communication_style",
    "interest": "tech_preference",
    "archival": "avoidance",
}

# Cross-project key -> PatternCategory mapping
_CROSS_PROJECT_KEY_MAP: dict[str, str] = {
    "frontend_framework": "tech_preference",
    "backend_framework": "tech_preference",
    "language": "tech_preference",
    "database": "tech_preference",
    "cloud": "tech_preference",
    "testing": "tech_preference",
}


class PatternCategory(str, Enum):
    """Categories for extracted pattern assertions."""

    IDENTITY = "identity"
    TECH_PREFERENCE = "tech_preference"
    COMMUNICATION_STYLE = "communication_style"
    WORKFLOW_PATTERN = "workflow_pattern"
    PROJECT_CONTEXT = "project_context"
    DECISION_HISTORY = "decision_history"
    AVOIDANCE = "avoidance"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PatternAssertion:
    """Single extracted pattern assertion with provenance."""

    category: PatternCategory
    key: str
    value: str
    confidence: float
    evidence_count: int
    source: str  # "core_memory" | "behavioral" | "cross_project" | "workflow"
    source_ids: tuple[str, ...] = ()
    cross_project_validated: bool = False
    created_at: str = ""


# ---------------------------------------------------------------------------
# PatternExtractor class
# ---------------------------------------------------------------------------

class PatternExtractor:
    """Extract patterns from 4 SLM sources for soft prompt generation.

    Sources:
        1. Core Memory blocks (user_profile, behavioral_patterns, learned_preferences)
        2. Behavioral pattern store (entity_pref, query_type, etc.)
        3. Cross-project aggregator (transferable preferences)
        4. Workflow miner (action sequences)
    """

    def __init__(
        self,
        db: DatabaseManager,
        behavioral_store: BehavioralPatternStore,
        cross_project: CrossProjectAggregator,
        workflow_miner: WorkflowMiner,
        config: ParameterizationConfig,
    ) -> None:
        if not (0.3 <= config.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0.3, 1.0], got {config.min_confidence}"
            )
        self._db = db
        self._behavioral_store = behavioral_store
        self._cross_project = cross_project
        self._workflow_miner = workflow_miner
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, profile_id: str) -> list[PatternAssertion]:
        """Master extraction pipeline across all 4 sources.

        Args:
            profile_id: Profile to extract patterns for.

        Returns:
            Deduplicated, contradiction-resolved list sorted by confidence DESC.
        """
        all_patterns: list[PatternAssertion] = []
        all_patterns.extend(self._extract_from_core_memory(profile_id))
        all_patterns.extend(self._extract_from_behavioral(profile_id))
        all_patterns.extend(self._extract_from_cross_project(profile_id))
        all_patterns.extend(self._extract_from_workflows(profile_id))
        # v3.4.7: Extract from behavioral assertions (learned patterns)
        all_patterns.extend(self._extract_from_assertions(profile_id))

        if not all_patterns:
            return []

        deduped = self._deduplicate(all_patterns)
        resolved = self._check_contradictions(deduped, profile_id)
        resolved.sort(key=lambda p: p.confidence, reverse=True)
        return resolved

    # ------------------------------------------------------------------
    # Source extractors
    # ------------------------------------------------------------------

    def _extract_from_core_memory(
        self, profile_id: str,
    ) -> list[PatternAssertion]:
        """Extract patterns from core memory blocks."""
        rows = self._db.execute(
            "SELECT block_id, block_type, content, source_fact_ids "
            "FROM core_memory_blocks "
            "WHERE profile_id = ? AND block_type IN "
            "('user_profile', 'behavioral_patterns', 'learned_preferences')",
            (profile_id,),
        )
        patterns: list[PatternAssertion] = []
        for row in rows:
            block_id = row["block_id"]
            content = row["content"]
            raw_ids = row["source_fact_ids"]
            try:
                fact_ids = json.loads(raw_ids) if raw_ids else []
            except (json.JSONDecodeError, TypeError):
                fact_ids = []

            evidence = len(fact_ids)
            assertions = self._split_assertions(content)
            for text in assertions:
                category = self._classify_text(text)
                confidence = min(evidence / 10.0, 1.0)
                if confidence < self._config.min_confidence:
                    continue
                key = self._extract_key(text)
                value = text.strip()[:200]
                patterns.append(PatternAssertion(
                    category=category,
                    key=key,
                    value=value,
                    confidence=confidence,
                    evidence_count=evidence,
                    source="core_memory",
                    source_ids=(block_id,),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))
        return patterns

    def _extract_from_behavioral(
        self, profile_id: str,
    ) -> list[PatternAssertion]:
        """Extract patterns from behavioral pattern store."""
        raw_patterns = self._behavioral_store.get_patterns(
            profile_id, min_confidence=self._config.min_confidence,
        )
        patterns: list[PatternAssertion] = []
        for bp in raw_patterns:
            if isinstance(bp, dict):
                ev_count = bp.get("evidence_count", 0)
                if ev_count < self._config.min_evidence:
                    continue
                p_type = bp.get("pattern_type", "")
                cat_str = _BEHAVIORAL_TYPE_MAP.get(p_type, "custom")
                category = PatternCategory(cat_str)
                p_key = bp.get("pattern_key", "")
                p_value = bp.get("pattern_value", p_key)
                conf = bp.get("confidence", 0.0)
                p_id = str(bp.get("pattern_id", ""))
                patterns.append(PatternAssertion(
                    category=category,
                    key=p_key,
                    value=p_value,
                    confidence=conf,
                    evidence_count=ev_count,
                    source="behavioral",
                    source_ids=(p_id,),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))
            else:
                # BehavioralPattern object
                ev_count = getattr(bp, "evidence_count", 0)
                if ev_count < self._config.min_evidence:
                    continue
                p_type = getattr(bp, "pattern_type", "")
                cat_str = _BEHAVIORAL_TYPE_MAP.get(p_type, "custom")
                category = PatternCategory(cat_str)
                p_key = getattr(bp, "pattern_key", "")
                p_value = getattr(bp, "pattern_value", p_key)
                conf = getattr(bp, "confidence", 0.0)
                p_id = str(getattr(bp, "pattern_id", ""))
                patterns.append(PatternAssertion(
                    category=category,
                    key=p_key,
                    value=p_value,
                    confidence=conf,
                    evidence_count=ev_count,
                    source="behavioral",
                    source_ids=(p_id,),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))
        return patterns

    def _extract_from_cross_project(
        self, profile_id: str = "default"
    ) -> list[PatternAssertion]:
        """Extract patterns from cross-project aggregator, scoped to *profile_id*."""
        preferences = self._cross_project.get_preferences(
            profile_id=profile_id,
            min_confidence=self._config.min_confidence,
        )
        patterns: list[PatternAssertion] = []
        for key, data in preferences.items():
            cat_str = _CROSS_PROJECT_KEY_MAP.get(key, "custom")
            category = PatternCategory(cat_str)
            raw_conf = data.get("confidence", 0.0)
            boosted = min(1.0, raw_conf * self._config.cross_project_boost)
            ev_count = data.get("evidence_count", 0)
            patterns.append(PatternAssertion(
                category=category,
                key=key,
                value=data.get("value", ""),
                confidence=boosted,
                evidence_count=ev_count,
                source="cross_project",
                source_ids=(),
                cross_project_validated=True,
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
        return patterns

    def _extract_from_workflows(
        self, profile_id: str,
    ) -> list[PatternAssertion]:
        """Extract patterns from workflow miner."""
        raw = self._workflow_miner.mine(profile_id, min_support=0.3)
        patterns: list[PatternAssertion] = []
        for wp in raw:
            count = wp.get("count", 0)
            if count < self._config.min_evidence:
                continue
            sequence = wp.get("sequence", [])
            value = " -> ".join(sequence)
            confidence = wp.get("support", 0.0)
            if confidence < self._config.min_confidence:
                continue
            key = f"workflow_{len(sequence)}gram"
            patterns.append(PatternAssertion(
                category=PatternCategory.WORKFLOW_PATTERN,
                key=key,
                value=value,
                confidence=confidence,
                evidence_count=count,
                source="workflow",
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
        return patterns

    # ------------------------------------------------------------------
    # v3.4.7: Behavioral assertions → soft prompts
    # ------------------------------------------------------------------

    def _extract_from_assertions(
        self, profile_id: str,
    ) -> list[PatternAssertion]:
        """Extract high-confidence behavioral assertions for soft prompt injection."""
        patterns: list[PatternAssertion] = []
        try:
            rows = self._db.execute(
                "SELECT trigger_condition, action, confidence, evidence_count, "
                "created_at FROM behavioral_assertions "
                "WHERE profile_id = ? AND confidence >= 0.5 "
                "ORDER BY confidence DESC LIMIT 10",
                (profile_id,),
            )
            for row in rows:
                r = dict(row)
                patterns.append(PatternAssertion(
                    category=PatternCategory.WORKFLOW_PATTERN,
                    key=f"behavioral:{r['trigger_condition'][:30]}",
                    value=f"When {r['trigger_condition']}, {r['action']}",
                    confidence=r["confidence"],
                    evidence_count=r["evidence_count"],
                    source="behavioral_assertion",
                    created_at=r.get("created_at", ""),
                ))
        except Exception as exc:
            logger.debug("Behavioral assertion extraction: %s", exc)
        return patterns

    # ------------------------------------------------------------------
    # Deduplication & contradiction resolution
    # ------------------------------------------------------------------

    def _deduplicate(
        self, patterns: list[PatternAssertion],
    ) -> list[PatternAssertion]:
        """Merge patterns sharing (category, key) using independence assumption.

        Merged confidence: c_merged = 1 - product(1 - c_i)
        """
        groups: dict[tuple[str, str], list[PatternAssertion]] = {}
        for p in patterns:
            gkey = (p.category.value, p.key)
            groups.setdefault(gkey, []).append(p)

        result: list[PatternAssertion] = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
                continue

            # Independence-assumption merge
            c_complement = 1.0
            for p in group:
                c_complement *= (1.0 - p.confidence)
            merged_confidence = 1.0 - c_complement

            total_evidence = sum(p.evidence_count for p in group)
            # Value from highest-confidence pattern
            best = max(group, key=lambda p: p.confidence)
            # Collect all source_ids
            all_ids: list[str] = []
            for p in group:
                all_ids.extend(p.source_ids)
            cross_validated = any(p.cross_project_validated for p in group)
            # Use latest created_at
            timestamps = [p.created_at for p in group if p.created_at]
            latest_ts = max(timestamps) if timestamps else ""

            result.append(PatternAssertion(
                category=best.category,
                key=best.key,
                value=best.value,
                confidence=merged_confidence,
                evidence_count=total_evidence,
                source=best.source,
                source_ids=tuple(all_ids),
                cross_project_validated=cross_validated,
                created_at=latest_ts,
            ))
        return result

    def _check_contradictions(
        self,
        patterns: list[PatternAssertion],
        profile_id: str,
    ) -> list[PatternAssertion]:
        """Resolve contradictions within same category+key.

        Resolution: temporal ordering (newer wins). For tech_preference with
        close confidence (both >= 0.8, diff <= 0.1), keep both with _alternate.
        """
        # Group by category
        cat_groups: dict[str, list[PatternAssertion]] = {}
        for p in patterns:
            cat_groups.setdefault(p.category.value, []).append(p)

        resolved: list[PatternAssertion] = []
        for cat_value, group in cat_groups.items():
            # Group by key within category
            key_groups: dict[str, list[PatternAssertion]] = {}
            for p in group:
                key_groups.setdefault(p.key, []).append(p)

            for key, key_patterns in key_groups.items():
                if len(key_patterns) == 1:
                    resolved.append(key_patterns[0])
                    continue

                # Check pairwise for contradictions
                surviving = list(key_patterns)
                to_remove: set[int] = set()

                for i in range(len(surviving)):
                    for j in range(i + 1, len(surviving)):
                        if i in to_remove or j in to_remove:
                            continue
                        p_a = surviving[i]
                        p_b = surviving[j]

                        if p_a.value == p_b.value:
                            continue  # Same value = not a contradiction

                        # Check close confidence for tech_preference
                        if (
                            cat_value == "tech_preference"
                            and p_a.confidence >= 0.8
                            and p_b.confidence >= 0.8
                            and abs(p_a.confidence - p_b.confidence) <= 0.1
                        ):
                            # Keep both — mark lower as alternate
                            if p_a.confidence >= p_b.confidence:
                                surviving[j] = PatternAssertion(
                                    category=p_b.category,
                                    key=f"{p_b.key}_alternate",
                                    value=p_b.value,
                                    confidence=p_b.confidence,
                                    evidence_count=p_b.evidence_count,
                                    source=p_b.source,
                                    source_ids=p_b.source_ids,
                                    cross_project_validated=p_b.cross_project_validated,
                                    created_at=p_b.created_at,
                                )
                            else:
                                surviving[i] = PatternAssertion(
                                    category=p_a.category,
                                    key=f"{p_a.key}_alternate",
                                    value=p_a.value,
                                    confidence=p_a.confidence,
                                    evidence_count=p_a.evidence_count,
                                    source=p_a.source,
                                    source_ids=p_a.source_ids,
                                    cross_project_validated=p_a.cross_project_validated,
                                    created_at=p_a.created_at,
                                )
                            logger.warning(
                                "Close confidence conflict in %s: '%s' vs '%s' "
                                "for key '%s'. Both kept as alternate.",
                                cat_value, p_a.value, p_b.value, key,
                            )
                            continue

                        # Temporal resolution
                        if p_a.created_at and p_b.created_at:
                            if p_a.created_at > p_b.created_at:
                                to_remove.add(j)
                                resolution = "temporal (newer wins)"
                            elif p_b.created_at > p_a.created_at:
                                to_remove.add(i)
                                resolution = "temporal (newer wins)"
                            else:
                                # Same timestamp — keep higher confidence
                                if p_a.confidence >= p_b.confidence:
                                    to_remove.add(j)
                                else:
                                    to_remove.add(i)
                                resolution = "confidence (higher wins)"
                        else:
                            # No timestamps — keep higher confidence
                            if p_a.confidence >= p_b.confidence:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                            resolution = "confidence (higher wins)"

                        logger.warning(
                            "Contradiction in %s: '%s' vs '%s' for key '%s'. "
                            "Resolved by %s.",
                            cat_value, p_a.value, p_b.value, key, resolution,
                        )

                for idx, p in enumerate(surviving):
                    if idx not in to_remove:
                        resolved.append(p)

        return resolved

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_assertions(content: str) -> list[str]:
        """Split block content into atomic assertions."""
        import re
        parts = re.split(r"\n[-*\.]\s+", content)
        result = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                # Remove leading bullet markers from first element
                stripped = re.sub(r"^[-*\.]\s*", "", stripped)
                if stripped:
                    result.append(stripped)
        return result

    @staticmethod
    def _classify_text(text: str) -> PatternCategory:
        """Classify assertion text into a PatternCategory using keywords."""
        lower = text.lower()
        words = set(lower.split())

        if words & _AVOIDANCE_KEYWORDS:
            return PatternCategory.AVOIDANCE
        if words & _DECISION_KEYWORDS:
            return PatternCategory.DECISION_HISTORY
        if words & _IDENTITY_KEYWORDS:
            return PatternCategory.IDENTITY
        if words & _TECH_KEYWORDS:
            return PatternCategory.TECH_PREFERENCE
        if words & _STYLE_KEYWORDS:
            return PatternCategory.COMMUNICATION_STYLE
        return PatternCategory.CUSTOM

    @staticmethod
    def _extract_key(text: str) -> str:
        """Extract a key from assertion text (first significant phrase)."""
        # Take first few words as key, normalized
        words = text.strip().split()[:4]
        key = "_".join(w.lower().strip(",.;:") for w in words if w)
        return key or "unknown"
