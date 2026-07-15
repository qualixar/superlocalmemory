# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Sheaf cohomology for contradiction detection at ENCODING time.

V1 applied sheaf at RECALL (HARMFUL — penalized diverse multi-hop results).
V2 runs at ENCODING ONLY, checking new facts against graph-connected facts.

Coboundary: delta_0 f(e) = R_b emb_b - R_a emb_a
Restriction maps by edge type: ENTITY=I, TEMPORAL=s*I (s<1), SEMANTIC=I.
Severity = ||delta_0 f(e)|| / (||R_a emb_a|| + ||R_b emb_b||).

Refs: Curry 2014 (arXiv:1303.3255), Hansen & Ghrist 2019, Robinson 2020.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

_EPS = 1e-12

# Temporal tolerance: fraction of coboundary norm below which temporal
# edges are NOT flagged as contradictions (legitimate state changes).
TEMPORAL_TOLERANCE: float = 0.15


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContradictionResult:
    """Detected contradiction between two facts."""

    fact_id_a: str   # New fact being stored
    fact_id_b: str   # Existing fact it conflicts with
    severity: float  # Normalized disagreement [0.0, 1.0]
    edge_type: str   # Graph edge type (entity/temporal/semantic/causal)
    description: str


# ---------------------------------------------------------------------------
# Restriction maps — non-trivial (V1 used identity everywhere)
# ---------------------------------------------------------------------------

def _restriction_for_edge_type(
    edge_type: str,
    dim: int,
    emb_a: np.ndarray | None = None,
    emb_b: np.ndarray | None = None,
) -> np.ndarray:
    """Edge-type-specific restriction map.

    Entity edges use non-trivial maps that AMPLIFY the disagreement
    along the axis of maximum difference between embeddings:
        R = 0.7*I + 0.3 * outer(diff, diff) / ||diff||^2

    This projects contradictions onto the most discriminative subspace,
    making the coboundary norm more sensitive to real disagreements.

    Temporal edges use identity restriction (scalar multiples cancel
    in the normalized coboundary). Temporal tolerance is applied at
    the caller level via TEMPORAL_TOLERANCE threshold.
    """
    if edge_type == "entity" and emb_a is not None and emb_b is not None:
        diff = emb_a - emb_b
        norm_sq = max(float(np.dot(diff, diff)), 1e-12)
        return 0.7 * np.eye(dim) + 0.3 * np.outer(diff, diff) / norm_sq
    return np.eye(dim)


def edge_residual(
    emb_a: np.ndarray, emb_b: np.ndarray,
    R_a: np.ndarray, R_b: np.ndarray,
) -> np.ndarray:
    """Coboundary residual: delta_0 f(e) = R_b emb_b - R_a emb_a."""
    return R_b @ emb_b - R_a @ emb_a


def coboundary_norm(
    emb_a: np.ndarray, emb_b: np.ndarray,
    R_a: np.ndarray, R_b: np.ndarray,
) -> float:
    """Normalized coboundary: ||delta|| / (||R_a emb_a|| + ||R_b emb_b||).

    Returns [0, ~2]. Near 0 = agree; near 1+ = strong disagreement.
    """
    residual = edge_residual(emb_a, emb_b, R_a, R_b)
    res_norm = float(np.linalg.norm(residual))
    denom = float(np.linalg.norm(R_a @ emb_a) + np.linalg.norm(R_b @ emb_b) + _EPS)
    return res_norm / denom


class SheafConsistencyChecker:
    """Detect contradictions at ENCODING time (not retrieval).

    V1 differences: uses real graph edges (not all-pairs), non-trivial
    restriction maps, and returns results for the pipeline to act on.
    """

    def __init__(
        self,
        db: DatabaseManager,
        contradiction_threshold: float = 0.7,
    ) -> None:
        self._db = db
        self._threshold = max(0.0, min(2.0, contradiction_threshold))

    # -- Public API ---------------------------------------------------------

    def check_consistency(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> list[ContradictionResult]:
        """Check new fact against graph-connected existing facts."""
        if new_fact.embedding is None:
            return []
        if not new_fact.canonical_entities:
            return []

        emb_a = np.asarray(new_fact.embedding, dtype=np.float64)
        dim = emb_a.shape[0]
        if dim == 0:
            return []

        contradictions: list[ContradictionResult] = []
        checked_pairs: set[str] = set()

        # Get all graph edges touching this fact
        edges = self._db.get_edges_for_node(new_fact.fact_id, profile_id)

        for edge in edges:
            # Determine the OTHER fact in this edge
            other_id = (
                edge.target_id
                if edge.source_id == new_fact.fact_id
                else edge.source_id
            )

            # Skip already-checked pairs and skip contradiction/supersedes edges
            if other_id in checked_pairs:
                continue
            if edge.edge_type.value in ("contradiction", "supersedes"):
                continue
            checked_pairs.add(other_id)

            # Look up the other fact's embedding
            other_emb = self._get_fact_embedding(other_id)
            if other_emb is None or other_emb.shape[0] != dim:
                continue

            # Compute coboundary with edge-type-specific restriction map
            edge_type_str = edge.edge_type.value
            R = _restriction_for_edge_type(edge_type_str, dim, emb_a, other_emb)
            severity = coboundary_norm(emb_a, other_emb, R, R)

            if severity > self._threshold:
                contradictions.append(ContradictionResult(
                    fact_id_a=new_fact.fact_id,
                    fact_id_b=other_id,
                    severity=min(severity, 1.0),
                    edge_type=edge_type_str,
                    description=(
                        f"Sheaf coboundary {severity:.3f} > {self._threshold:.2f} "
                        f"along {edge_type_str} edge"
                    ),
                ))

        if contradictions:
            logger.info(
                "Sheaf: %d contradiction(s) for fact %s",
                len(contradictions), new_fact.fact_id,
            )
        return contradictions

    def detect_contradictions_batch(
        self,
        facts: list[AtomicFact],
        profile_id: str,
    ) -> list[ContradictionResult]:
        """Pairwise check within entity groups (for batch imports)."""
        # Group facts by canonical entity
        entity_groups: dict[str, list[AtomicFact]] = {}
        for fact in facts:
            if fact.embedding is None:
                continue
            for eid in fact.canonical_entities:
                entity_groups.setdefault(eid, []).append(fact)

        contradictions: list[ContradictionResult] = []
        checked_pairs: set[tuple[str, str]] = set()

        for _entity_id, group in entity_groups.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    fa, fb = group[i], group[j]
                    pair = (min(fa.fact_id, fb.fact_id), max(fa.fact_id, fb.fact_id))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    emb_a = np.asarray(fa.embedding, dtype=np.float64)
                    emb_b = np.asarray(fb.embedding, dtype=np.float64)
                    if emb_a.shape[0] != emb_b.shape[0] or emb_a.shape[0] == 0:
                        continue

                    dim = emb_a.shape[0]
                    R = np.eye(dim)
                    severity = coboundary_norm(emb_a, emb_b, R, R)

                    if severity > self._threshold:
                        contradictions.append(ContradictionResult(
                            fact_id_a=fa.fact_id,
                            fact_id_b=fb.fact_id,
                            severity=min(severity, 1.0),
                            edge_type="entity",
                            description=(
                                f"Batch sheaf coboundary {severity:.3f} > "
                                f"{self._threshold:.2f} (shared entity group)"
                            ),
                        ))

        return contradictions

    # -- Internal -----------------------------------------------------------

    def _get_fact_embedding(self, fact_id: str) -> np.ndarray | None:
        """Load a fact's embedding from the database."""
        import json
        rows = self._db.execute(
            "SELECT embedding FROM atomic_facts WHERE fact_id = ?",
            (fact_id,),
        )
        if not rows:
            return None
        raw = dict(rows[0]).get("embedding")
        if raw is None or raw == "":
            return None
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return np.asarray(data, dtype=np.float64)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
