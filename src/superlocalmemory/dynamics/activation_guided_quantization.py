# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Spreading Activation-Guided Quantization (SAGQ) engine.

Novel contribution: uses graph centrality (PageRank + degree + SA frequency)
to allocate embedding precision. Well-connected memories keep higher precision
because they serve as hubs for spreading activation retrieval.

Core formula:
    centrality(i) = w_pr * pr_norm + w_deg * deg_norm + w_sa * sa_freq_norm
    sagq_bw = b_min + (b_max - b_min) * centrality, ceil-snapped to valid set

Conflict resolution with Phase A EAP:
    final_bw = max(eap_bw, sagq_bw) -- safety first, never over-quantize

HR-02: Conflict resolution is ALWAYS max().
HR-03: Centrality scores always in [0.0, 1.0].
HR-04: Bit-width always from valid_bit_widths.
HR-05: All SQL uses parameterized queries.
HR-07: No-op when config.enabled=False.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.core.config import SAGQConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes (frozen -- all immutable, Rule 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CentralityScore:
    """Computed centrality for a single memory node."""

    fact_id: str
    pagerank_norm: float       # pr_norm(i) in [0, 1]
    degree_norm: float         # deg_norm(i) in [0, 1]
    sa_freq_norm: float        # sa_freq(i) in [0, 1]
    combined_centrality: float  # weighted sum in [0, 1]


@dataclass(frozen=True)
class SAGQPrecision:
    """SAGQ precision recommendation for a single memory."""

    fact_id: str
    centrality: float    # combined centrality in [0, 1]
    sagq_bit_width: int  # recommended bit-width from SAGQ signal
    eap_bit_width: int   # recommended bit-width from EAP signal (Phase A)
    final_bit_width: int  # max(sagq_bit_width, eap_bit_width)
    current_bit_width: int  # current bit-width from embedding_metadata
    action: str          # "upgrade" | "downgrade" | "skip"


# ---------------------------------------------------------------------------
# ActivationGuidedQuantizer
# ---------------------------------------------------------------------------


class ActivationGuidedQuantizer:
    """SAGQ engine: maps graph centrality to embedding precision.

    Reads from fact_importance (PageRank, degree) and activation_cache
    (spreading activation frequency) to compute centrality per memory.
    Maps centrality to bit-width via linear interpolation + ceiling snap.
    """

    def __init__(self, db: Any, config: Any) -> None:
        """Initialize SAGQ quantizer. No side effects."""
        self._db = db
        self._config = config

    def compute_centrality_batch(
        self, profile_id: str,
    ) -> list[CentralityScore]:
        """Compute centrality for all facts in a profile.

        Returns list of CentralityScore, each with combined_centrality in [0, 1].
        Returns empty list if disabled or no data.
        """
        if not self._config.enabled:
            return []

        # Step 1: Query fact_importance (Q1)
        try:
            rows = self._db.execute(
                "SELECT fact_id, pagerank_score, degree_centrality "
                "FROM fact_importance "
                "WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.warning("SAGQ: fact_importance query failed: %s", exc)
            return []

        if not rows:
            logger.info("SAGQ: no importance data yet for profile %s, skipping", profile_id)
            return []

        # Step 3: Compute max values for normalization (avoid division by zero)
        fact_data = [(dict(r)["fact_id"], dict(r)["pagerank_score"], dict(r)["degree_centrality"]) for r in rows]

        max_pr = max((pr for _, pr, _ in fact_data), default=0.0)
        max_deg = max((deg for _, _, deg in fact_data), default=0.0)
        if max_pr == 0.0:
            max_pr = 1.0
        if max_deg == 0.0:
            max_deg = 1.0

        # Step 4: Query activation_cache for SA frequency (Q2)
        sa_window = f"-{self._config.sa_frequency_window_days} days"
        try:
            sa_rows = self._db.execute(
                "SELECT node_id, COUNT(*) as activation_count "
                "FROM activation_cache "
                "WHERE profile_id = ? "
                "AND created_at > datetime('now', ?) "
                "GROUP BY node_id",
                (profile_id, sa_window),
            )
        except Exception as exc:
            logger.debug("SAGQ: activation_cache query failed: %s", exc)
            sa_rows = []

        # Step 5: Build SA frequency map
        sa_freq_map: dict[str, int] = {}
        for sa_row in sa_rows:
            d = dict(sa_row)
            sa_freq_map[d["node_id"]] = int(d["activation_count"])

        # Step 6: max SA frequency
        max_sa = max(sa_freq_map.values()) if sa_freq_map else 1

        # Step 7: Compute centrality for each fact
        cfg = self._config
        result: list[CentralityScore] = []
        for fact_id, pr_score, deg_score in fact_data:
            pr_norm = pr_score / max_pr
            deg_norm = deg_score / max_deg
            sa_freq = sa_freq_map.get(fact_id, 0)
            sa_freq_norm = sa_freq / max_sa

            combined = (
                cfg.w_pagerank * pr_norm
                + cfg.w_degree * deg_norm
                + cfg.w_sa_freq * sa_freq_norm
            )

            # NaN safety (HR-03)
            if math.isnan(combined):
                logger.warning("SAGQ: NaN centrality for %s, defaulting to 0.0", fact_id)
                combined = 0.0

            # Clamp to [0.0, 1.0]
            combined = max(0.0, min(1.0, combined))

            result.append(CentralityScore(
                fact_id=fact_id,
                pagerank_norm=pr_norm,
                degree_norm=deg_norm,
                sa_freq_norm=sa_freq_norm,
                combined_centrality=combined,
            ))

        return result

    def centrality_to_bit_width(self, centrality: float) -> int:
        """Map centrality score to SAGQ bit-width.

        Linear interpolation from [0,1] to [b_min, b_max], then ceiling-snap
        to nearest valid bit-width. SAGQ is a preservation signal -- always
        round UP (Decision D1).

        Returns one of valid_bit_widths.
        """
        cfg = self._config

        # Clamp input
        centrality = max(0.0, min(1.0, centrality))

        # Linear mapping
        raw_bw = cfg.b_min + (cfg.b_max - cfg.b_min) * centrality

        # Ceiling snap to nearest valid bit-width (smallest >= raw_bw)
        for vbw in cfg.valid_bit_widths:
            if vbw >= raw_bw:
                return vbw

        # raw_bw exceeds all valid values -- cap at maximum
        return cfg.valid_bit_widths[-1]

    def compute_sagq_precision_batch(
        self,
        profile_id: str,
        eap_precision_fn: Callable[[str], int],
    ) -> list[SAGQPrecision]:
        """Compute combined SAGQ + EAP precision for all facts in a profile.

        Args:
            profile_id: Profile to process.
            eap_precision_fn: Callable(fact_id) -> EAP bit-width (from Phase A).

        Returns:
            List of SAGQPrecision with action ("upgrade"/"downgrade"/"skip").
        """
        if not self._config.enabled:
            return []

        # Step 1: Compute centrality
        centrality_scores = self.compute_centrality_batch(profile_id)
        if not centrality_scores:
            return []

        # Step 3: Batch-fetch current bit-widths (Q3)
        try:
            bw_rows = self._db.execute(
                "SELECT fact_id, COALESCE(bit_width, 32) as bit_width "
                "FROM embedding_metadata "
                "WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.warning("SAGQ: embedding_metadata query failed: %s", exc)
            bw_rows = []

        current_bw_map: dict[str, int] = {}
        for row in bw_rows:
            d = dict(row)
            current_bw_map[d["fact_id"]] = int(d["bit_width"])

        # Step 5: Compute precision for each fact
        result: list[SAGQPrecision] = []
        for cs in centrality_scores:
            # SAGQ signal
            sagq_bw = self.centrality_to_bit_width(cs.combined_centrality)

            # EAP signal
            eap_bw = eap_precision_fn(cs.fact_id)

            # HR-02: Conflict resolution -- ALWAYS max()
            final_bw = max(sagq_bw, eap_bw)

            # Current bit-width
            current_bw = current_bw_map.get(cs.fact_id, 32)

            # Determine action
            if final_bw < current_bw:
                action = "downgrade"
            elif final_bw > current_bw:
                action = "upgrade"
            else:
                action = "skip"

            result.append(SAGQPrecision(
                fact_id=cs.fact_id,
                centrality=cs.combined_centrality,
                sagq_bit_width=sagq_bw,
                eap_bit_width=eap_bw,
                final_bit_width=final_bw,
                current_bit_width=current_bw,
                action=action,
            ))

        return result

    def get_centrality_for_fact(
        self, fact_id: str, profile_id: str,
    ) -> float:
        """Get centrality for a single fact. Returns 0.0 if not found.

        Used by Phase E (CCQ) for centrality-aware consolidation.
        """
        if not self._config.enabled:
            return 0.0

        # Step 1: Query single fact importance (Q11)
        try:
            rows = self._db.execute(
                "SELECT pagerank_score, degree_centrality "
                "FROM fact_importance "
                "WHERE fact_id = ? AND profile_id = ?",
                (fact_id, profile_id),
            )
        except Exception as exc:
            logger.debug("SAGQ: single fact query failed: %s", exc)
            return 0.0

        if not rows:
            return 0.0

        d = dict(rows[0])
        pr_score = float(d["pagerank_score"])
        deg_score = float(d["degree_centrality"])

        # Step 3: Normalization (Q4)
        try:
            max_rows = self._db.execute(
                "SELECT "
                "  COALESCE(MAX(pagerank_score), 0.0) as max_pr, "
                "  COALESCE(MAX(degree_centrality), 0.0) as max_deg "
                "FROM fact_importance "
                "WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("SAGQ: max query failed: %s", exc)
            return 0.0

        md = dict(max_rows[0])
        max_pr = max(float(md["max_pr"]), 1e-8)
        max_deg = max(float(md["max_deg"]), 1e-8)

        pr_norm = pr_score / max_pr
        deg_norm = deg_score / max_deg

        # Step 6-7: SA frequency for this fact (Q6)
        sa_window = f"-{self._config.sa_frequency_window_days} days"
        try:
            sa_rows = self._db.execute(
                "SELECT COUNT(*) as cnt "
                "FROM activation_cache "
                "WHERE node_id = ? AND profile_id = ? "
                "AND created_at > datetime('now', ?)",
                (fact_id, profile_id, sa_window),
            )
            sa_cnt = int(dict(sa_rows[0])["cnt"]) if sa_rows else 0
        except Exception:
            sa_cnt = 0

        # Step 7: Max SA frequency (Q5)
        try:
            max_sa_rows = self._db.execute(
                "SELECT COALESCE(MAX(cnt), 1) as max_cnt FROM ("
                "  SELECT COUNT(*) as cnt "
                "  FROM activation_cache "
                "  WHERE profile_id = ? "
                "  AND created_at > datetime('now', ?) "
                "  GROUP BY node_id"
                ")",
                (profile_id, sa_window),
            )
            max_sa = int(dict(max_sa_rows[0])["max_cnt"]) if max_sa_rows else 1
        except Exception:
            max_sa = 1

        sa_norm = sa_cnt / max(max_sa, 1)

        # Step 9: Weighted combination
        cfg = self._config
        combined = cfg.w_pagerank * pr_norm + cfg.w_degree * deg_norm + cfg.w_sa_freq * sa_norm

        # NaN safety
        if math.isnan(combined):
            logger.warning("SAGQ: NaN centrality for %s, defaulting to 0.0", fact_id)
            return 0.0

        # Step 10: Clamp
        return max(0.0, min(1.0, combined))
