# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Bayesian Trust Scorer (Beta Distribution).

Per-agent and per-fact trust using conjugate Beta(alpha, beta) priors.
Trust = alpha / (alpha + beta). Prior decay toward uniform when idle.

Encoding into existing schema (no ALTER TABLE):
  trust_score  = alpha / (alpha + beta)
  evidence_count = round(alpha + beta - 2)   (subtract the default prior)
  Reconstruct:  alpha = trust_score * (evidence_count + 2)
                beta  = (1 - trust_score) * (evidence_count + 2)

Backward-compatible: update_on_confirmation / contradiction / access
delegate to record_signal internally.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from superlocalmemory.storage.models import TrustScore

logger = logging.getLogger(__name__)

# Default Beta(1,1) = uniform prior => trust 0.5
_DEFAULT_ALPHA = 1.0
_DEFAULT_BETA = 1.0
_DEFAULT_TRUST = 0.5

# Signal strengths (how much each event shifts alpha/beta)
_SIGNAL_WEIGHTS: dict[str, tuple[float, float]] = {
    # (delta_alpha, delta_beta)
    "store_success": (1.0, 0.0),
    "store_rejected": (0.0, 2.0),
    "recall_hit": (0.5, 0.0),
    "contradiction": (0.0, 3.0),
    "deletion": (0.0, 1.5),
}

# Prior decay: days of inactivity before decay starts, and rate per day
_DECAY_IDLE_DAYS = 30
_DECAY_RATE = 0.05  # 5% per day toward prior


class TrustScorer:
    """Bayesian trust scoring with Beta distribution.

    Trust(agent) = alpha / (alpha + beta).
    - Positive evidence increments alpha.
    - Negative evidence increments beta.
    - Idle decay shrinks both toward 1.0 (uniform prior).

    Facts inherit their source agent's trust, penalized by contradictions.
    """

    def __init__(self, db) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Public API: per-agent trust
    # ------------------------------------------------------------------

    def get_agent_trust(self, agent_id: str, profile_id: str) -> float:
        """Get trust score for an agent. Returns 0.5 if unknown."""
        alpha, beta = self._get_beta_params("agent", agent_id, profile_id)
        return self._compute_trust(alpha, beta)

    def get_fact_trust(self, fact_id: str, profile_id: str) -> float:
        """Get trust for a fact.

        When a direct trust record exists for the fact, use it.
        Otherwise inherit the trust of the agent that created the fact,
        as recorded in the provenance table.
        """
        rows = self._db.execute(
            "SELECT trust_score, evidence_count FROM trust_scores "
            "WHERE target_type = ? AND target_id = ? AND profile_id = ?",
            ("fact", fact_id, profile_id),
        )
        if rows:
            d = dict(rows[0])
            alpha, beta = self._decode_beta(d["trust_score"], d["evidence_count"])
            return self._compute_trust(alpha, beta)

        # No direct record — inherit from the provenance source agent.
        prov = self._db.execute(
            "SELECT created_by FROM provenance "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        if prov:
            agent_id = dict(prov[0]).get("created_by", "")
            if agent_id:
                return self.get_agent_trust(agent_id, profile_id)

        return _DEFAULT_TRUST

    def get_entity_trust(self, entity_id: str, profile_id: str) -> float:
        """Convenience: get trust for an entity."""
        return self.get_trust("entity", entity_id, profile_id)

    def get_trust(
        self, target_type: str, target_id: str, profile_id: str
    ) -> float:
        """Generic trust lookup — backward compatible with V3 Task 3."""
        alpha, beta = self._get_beta_params(target_type, target_id, profile_id)
        return self._compute_trust(alpha, beta)

    # ------------------------------------------------------------------
    # Public API: record signals
    # ------------------------------------------------------------------

    def record_signal(
        self, agent_id: str, profile_id: str, signal_type: str
    ) -> float:
        """Record a trust signal and return updated trust score.

        Args:
            agent_id: The agent whose trust is being updated.
            profile_id: Active profile scope.
            signal_type: One of store_success, store_rejected, recall_hit,
                         contradiction, deletion.

        Returns:
            Updated trust score for the agent.
        """
        weights = _SIGNAL_WEIGHTS.get(signal_type, (0.0, 0.0))
        delta_alpha, delta_beta = weights

        alpha, beta = self._get_beta_params("agent", agent_id, profile_id)
        new_alpha = alpha + delta_alpha
        new_beta = beta + delta_beta

        self._persist_beta("agent", agent_id, profile_id, new_alpha, new_beta)
        score = self._compute_trust(new_alpha, new_beta)
        logger.debug(
            "trust signal: agent=%s signal=%s trust=%.3f (a=%.1f b=%.1f)",
            agent_id, signal_type, score, new_alpha, new_beta,
        )
        return score

    # ------------------------------------------------------------------
    # Public API: propagation
    # ------------------------------------------------------------------

    def propagate_recall_trust(
        self, agent_id: str, profile_id: str
    ) -> float:
        """Boost trust for agents whose memories are recalled.

        A recall-hit means the system found the agent's data useful.
        """
        return self.record_signal(agent_id, profile_id, "recall_hit")

    # ------------------------------------------------------------------
    # Public API: direct set (testing & migration)
    # ------------------------------------------------------------------

    def set_trust(
        self,
        agent_id: str,
        profile_id: str,
        alpha: float = _DEFAULT_ALPHA,
        beta_param: float = _DEFAULT_BETA,
    ) -> float:
        """Set trust directly via Beta params. Primarily for testing."""
        alpha = max(0.01, alpha)
        beta_param = max(0.01, beta_param)
        self._persist_beta("agent", agent_id, profile_id, alpha, beta_param)
        return self._compute_trust(alpha, beta_param)

    # ------------------------------------------------------------------
    # Public API: bulk query
    # ------------------------------------------------------------------

    def get_all_scores(self, profile_id: str) -> list[dict]:
        """Get all trust scores for a profile as dicts.

        Returns list of {target_type, target_id, trust_score,
        alpha, beta_param, evidence_count, last_updated}.
        """
        rows = self._db.execute(
            "SELECT * FROM trust_scores WHERE profile_id = ?",
            (profile_id,),
        )
        results: list[dict] = []
        for r in rows:
            d = dict(r)
            alpha, beta = self._decode_beta(
                d["trust_score"], d["evidence_count"]
            )
            results.append({
                "trust_id": d["trust_id"],
                "target_type": d["target_type"],
                "target_id": d["target_id"],
                "trust_score": d["trust_score"],
                "alpha": alpha,
                "beta_param": beta,
                "evidence_count": d["evidence_count"],
                "last_updated": d["last_updated"],
            })
        return results

    # ------------------------------------------------------------------
    # Backward-compatible delegators
    # ------------------------------------------------------------------

    def update_on_confirmation(
        self, target_type: str, target_id: str, profile_id: str
    ) -> float:
        """V3-compat: confirmation -> store_success signal."""
        alpha, beta = self._get_beta_params(target_type, target_id, profile_id)
        new_alpha = alpha + 1.0
        self._persist_beta(target_type, target_id, profile_id, new_alpha, beta)
        return self._compute_trust(new_alpha, beta)

    def update_on_contradiction(
        self, target_type: str, target_id: str, profile_id: str
    ) -> float:
        """V3-compat: contradiction -> contradiction signal."""
        alpha, beta = self._get_beta_params(target_type, target_id, profile_id)
        new_beta = beta + 3.0
        self._persist_beta(target_type, target_id, profile_id, alpha, new_beta)
        return self._compute_trust(alpha, new_beta)

    def update_on_access(
        self, target_type: str, target_id: str, profile_id: str
    ) -> float:
        """V3-compat: access -> recall_hit signal (small boost)."""
        alpha, beta = self._get_beta_params(target_type, target_id, profile_id)
        new_alpha = alpha + 0.5
        self._persist_beta(
            target_type, target_id, profile_id, new_alpha, beta
        )
        return self._compute_trust(new_alpha, beta)

    # ------------------------------------------------------------------
    # Prior decay
    # ------------------------------------------------------------------

    def apply_prior_decay(self, profile_id: str) -> int:
        """Decay idle trust scores toward uniform prior Beta(1,1).

        Called periodically (e.g., daily maintenance).
        Returns number of scores decayed.
        """
        cutoff = datetime.now(UTC).isoformat()
        rows = self._db.execute(
            "SELECT * FROM trust_scores WHERE profile_id = ?",
            (profile_id,),
        )
        decayed = 0
        for r in rows:
            d = dict(r)
            last = d.get("last_updated", "")
            if not last:
                continue
            try:
                last_dt = datetime.fromisoformat(last)
                idle_days = (
                    datetime.now(UTC) - last_dt.replace(tzinfo=UTC)
                ).days
            except (ValueError, TypeError):
                continue

            if idle_days < _DECAY_IDLE_DAYS:
                continue

            alpha, beta = self._decode_beta(
                d["trust_score"], d["evidence_count"]
            )
            excess_days = idle_days - _DECAY_IDLE_DAYS
            decay_factor = max(0.0, 1.0 - _DECAY_RATE * excess_days)

            new_alpha = _DEFAULT_ALPHA + (alpha - _DEFAULT_ALPHA) * decay_factor
            new_beta = _DEFAULT_BETA + (beta - _DEFAULT_BETA) * decay_factor

            new_alpha = max(0.01, new_alpha)
            new_beta = max(0.01, new_beta)

            score = self._compute_trust(new_alpha, new_beta)
            ev = max(0, round(new_alpha + new_beta - 2))
            self._db.execute(
                "UPDATE trust_scores SET trust_score = ?, evidence_count = ?, "
                "last_updated = ? WHERE trust_id = ?",
                (score, ev, cutoff, d["trust_id"]),
            )
            decayed += 1
        return decayed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trust(alpha: float, beta: float) -> float:
        """Trust = alpha / (alpha + beta), clamped to [0, 1]."""
        total = alpha + beta
        if total <= 0:
            return _DEFAULT_TRUST
        return max(0.0, min(1.0, alpha / total))

    @staticmethod
    def _decode_beta(
        trust_score: float, evidence_count: int
    ) -> tuple[float, float]:
        """Reconstruct Beta params from encoded trust_score + evidence_count.

        Encoding: trust_score = a/(a+b), evidence_count = round(a+b-2).
        """
        total = float(evidence_count) + 2.0
        alpha = trust_score * total
        beta = (1.0 - trust_score) * total
        return max(0.01, alpha), max(0.01, beta)

    def _get_beta_params(
        self, target_type: str, target_id: str, profile_id: str
    ) -> tuple[float, float]:
        """Load Beta(alpha, beta) from DB, or return default prior."""
        rows = self._db.execute(
            "SELECT trust_score, evidence_count FROM trust_scores "
            "WHERE target_type = ? AND target_id = ? AND profile_id = ?",
            (target_type, target_id, profile_id),
        )
        if rows:
            d = dict(rows[0])
            return self._decode_beta(d["trust_score"], d["evidence_count"])
        return _DEFAULT_ALPHA, _DEFAULT_BETA

    def _persist_beta(
        self,
        target_type: str,
        target_id: str,
        profile_id: str,
        alpha: float,
        beta: float,
    ) -> None:
        """Encode and persist Beta params into existing schema."""
        score = self._compute_trust(alpha, beta)
        ev = max(0, round(alpha + beta - 2))
        now = datetime.now(UTC).isoformat()

        existing = self._db.execute(
            "SELECT trust_id FROM trust_scores "
            "WHERE target_type = ? AND target_id = ? AND profile_id = ?",
            (target_type, target_id, profile_id),
        )
        if existing:
            tid = dict(existing[0])["trust_id"]
            self._db.execute(
                "UPDATE trust_scores SET trust_score = ?, evidence_count = ?, "
                "last_updated = ? WHERE trust_id = ?",
                (score, ev, now, tid),
            )
        else:
            from superlocalmemory.storage.models import _new_id

            tid = _new_id()
            self._db.execute(
                "INSERT INTO trust_scores "
                "(trust_id, profile_id, target_type, target_id, trust_score, "
                "evidence_count, last_updated) VALUES (?,?,?,?,?,?,?)",
                (tid, profile_id, target_type, target_id, score, ev, now),
            )
