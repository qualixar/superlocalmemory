# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A memory unrelated to the question never counts as semantic evidence (Q3).

The semantic channel reports (cosine + 1) / 2, so an unrelated memory (cosine
0) scores 0.5, and about 0.556 with the Fisher blend. Fusion is rank-based, so
that offset never reorders anything; the one place an absolute semantic score
decides an outcome is the evidence floor. Its 0.60 threshold is a raw cosine
and is converted to this scale (0.80) before the comparison; comparing it raw
let cosine 0.2 through, so recall never abstained. This pins that an unrelated
memory cannot pass the floor on semantic score alone.
"""

from __future__ import annotations

import numpy as np

from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.retrieval.semantic_channel import _cosine_similarity


def _fr(fid: str, **scores: float) -> FusionResult:
    return FusionResult(fact_id=fid, fused_score=0.77, channel_scores=scores)


def test_an_orthogonal_vector_scores_one_half_on_the_channel_scale() -> None:
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert (_cosine_similarity(a, b) + 1.0) / 2.0 == 0.5


def test_a_zero_cosine_memory_is_not_evidence_but_a_related_one_is() -> None:
    kept = RetrievalEngine._apply_evidence_floor(
        [_fr("unrelated", semantic=0.556, spreading_activation=0.9, hopfield=0.8),
         _fr("related", semantic=0.82),
         _fr("keyword", semantic=0.50, bm25=1.2)],
        facts={}, min_semantic=0.60)
    assert [k.fact_id for k in kept] == ["related", "keyword"]
