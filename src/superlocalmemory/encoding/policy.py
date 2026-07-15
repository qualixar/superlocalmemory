# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Merge Policy (Stage 3: Policy + Action Selection).

Formalizes ``pi(a_t | o_t, r_t)``: given a new fact (the observation ``o_t``)
and the candidates retrieved for it (``r_t``, via
``MemoryConsolidator._find_candidates`` — belief-aware retrieval doubles as
the write path's candidate lookup), selects one discrete maintenance action.
``MemoryConsolidator`` then executes the chosen action via its existing
``_execute_*`` primitives — "Execute a_t" in the reference framework,
reusing execution logic that already existed rather than inventing new
side effects SLM has no business performing as a memory backend.

This is deliberately scoped to SLM's own maintenance operations (the only
actions a memory backend can safely/meaningfully execute autonomously),
not a generic external actuator.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from superlocalmemory.core.config import EncodingConfig
from superlocalmemory.storage.models import AtomicFact

# Score thresholds (match EncodingConfig defaults; kept here so the policy is
# self-contained and consolidator.py's own copies stay in sync by construction).
_NOOP_THRESHOLD = 0.95
_MATCH_THRESHOLD = 0.85

ContradictionCheck = Callable[[AtomicFact, AtomicFact], bool]


class MergeAction(str, Enum):
    """The discrete action space a_t for the merge policy."""

    ADD = "add"                            # brand-new attribute, committed
    ADD_QUARANTINE = "add_quarantine"      # brand-new attribute, awaiting corroboration
    UPDATE = "update"                      # trust-gated merge, committed (noisy-OR)
    UPDATE_QUARANTINE = "update_quarantine"  # trust-gated merge, rejected
    CORROBORATE = "corroborate"            # matched a pending fact, different source -> commit both
    STILL_PENDING = "still_pending"        # matched a pending fact, same source -> no change
    SUPERSEDE = "supersede"                # contradiction detected
    NOOP = "noop"                          # near-duplicate


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: MergeAction
    best_fact: AtomicFact | None
    best_score: float
    tau: float
    delta: float
    reason: str


class MergePolicy:
    """Selects a_t ~ pi(. | o_t, r_t) for an incoming fact."""

    def __init__(self, is_contradicting: ContradictionCheck, cfg: EncodingConfig) -> None:
        self._is_contradicting = is_contradicting
        self._cfg = cfg

    def select(
        self,
        new_fact: AtomicFact,
        candidates: list[tuple[AtomicFact, float]],
        tau: float,
    ) -> PolicyDecision:
        delta = new_fact.confidence

        if not candidates:
            return self._decide_add(new_fact, tau, delta, best_score=0.0)

        best_fact, best_score = candidates[0]

        if best_score > _NOOP_THRESHOLD:
            if self._is_contradicting(new_fact, best_fact):
                return PolicyDecision(
                    MergeAction.SUPERSEDE, best_fact, best_score, tau, delta,
                    f"contradiction detected (score={best_score:.3f})",
                )
            return PolicyDecision(
                MergeAction.NOOP, best_fact, best_score, tau, delta,
                f"near-duplicate (score={best_score:.3f})",
            )

        if best_score > _MATCH_THRESHOLD:
            if self._is_contradicting(new_fact, best_fact):
                return PolicyDecision(
                    MergeAction.SUPERSEDE, best_fact, best_score, tau, delta,
                    f"contradiction detected (score={best_score:.3f})",
                )
            if new_fact.fact_type == best_fact.fact_type:
                return self._decide_update(new_fact, best_fact, best_score, tau, delta)

        return self._decide_add(new_fact, tau, delta, best_score=best_score)

    # -- Sub-decisions --------------------------------------------------

    def _decide_update(
        self,
        new_fact: AtomicFact,
        existing: AtomicFact,
        score: float,
        tau: float,
        delta: float,
    ) -> PolicyDecision:
        if getattr(existing, "pending_corroboration", False):
            if existing.source_agent_id and existing.source_agent_id != new_fact.source_agent_id:
                return PolicyDecision(
                    MergeAction.CORROBORATE, existing, score, tau, delta,
                    f"independent corroboration from '{new_fact.source_agent_id}' "
                    f"(previously '{existing.source_agent_id}')",
                )
            return PolicyDecision(
                MergeAction.STILL_PENDING, existing, score, tau, delta,
                f"same source '{new_fact.source_agent_id}' repeats an "
                "uncorroborated claim — still awaiting an independent source",
            )

        if new_fact.intent_flagged or tau * delta < self._cfg.merge_trust_threshold:
            reason = (
                f"tau*delta={tau * delta:.3f} below threshold "
                f"{self._cfg.merge_trust_threshold:.3f}"
            )
            if new_fact.intent_flagged:
                reason += " (intent_flagged: classified as query/directive)"
            return PolicyDecision(
                MergeAction.UPDATE_QUARANTINE, existing, score, tau, delta,
                f"QUARANTINE: {reason}",
            )

        return PolicyDecision(
            MergeAction.UPDATE, existing, score, tau, delta,
            f"refines existing (score={score:.3f}, tau*delta={tau * delta:.3f})",
        )

    def _decide_add(
        self, new_fact: AtomicFact, tau: float, delta: float, *, best_score: float,
    ) -> PolicyDecision:
        if new_fact.intent_flagged:
            return PolicyDecision(
                MergeAction.ADD_QUARANTINE, None, best_score, tau, delta,
                "QUARANTINE: intent_flagged (classified as query/directive) — "
                "never committed as a trusted belief on first mention",
            )
        if self._cfg.deployment_mode == "multi_source":
            return PolicyDecision(
                MergeAction.ADD_QUARANTINE, None, best_score, tau, delta,
                "multi-source deployment — awaiting independent corroboration "
                "before first commit",
            )
        return PolicyDecision(
            MergeAction.ADD, None, best_score, tau, delta,
            "ungated-bootstrap (single-source deployment — trust math dormant "
            "on first mention, no quorum check possible)",
        )
