"""Top-k ranking, reciprocal-rank fusion and the BM25 smoke scorer."""
from __future__ import annotations

import re

import numpy as np

K_TOP = 100


def rank_topk(scores: np.ndarray, doc_ids: list[str], k: int = K_TOP) -> list[tuple[str, float]]:
    """Top-k (doc_id, score); ties broken by doc id (deterministic)."""
    order = np.argsort(np.asarray(doc_ids))
    s = np.asarray(scores)[order]
    top = np.argsort(-s, kind="stable")[:k]
    return [(doc_ids[order[i]], float(s[i])) for i in top]


MEDIA_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def rrf_fuse(rankings: list[list[str]], k: int = 60, top: int = K_TOP,
             weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion of ranked doc-id lists (rank is 1-based).

    Ties in fused score go to the doc whose best-weighted contribution comes
    from the earlier channel (channel 0 is the priority channel), then doc id.
    """
    weights = weights or [1.0] * len(rankings)
    acc: dict[str, float] = {}
    first: dict[str, int] = {}
    for ch, (ranking, w) in enumerate(zip(rankings, weights)):
        if w == 0:
            continue
        for pos, doc in enumerate(ranking, start=1):
            acc[doc] = acc.get(doc, 0.0) + w / (k + pos)
            first.setdefault(doc, ch)
    ordered = sorted(acc.items(), key=lambda kv: (-kv[1], first[kv[0]], kv[0]))
    return ordered[:top]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rank(doc_ids: list[str], doc_texts: list[str], queries: list[str],
              k: int = K_TOP) -> list[list[tuple[str, float]]]:
    """Model-free smoke system: BM25Okapi over doc text (incl. OCR text)."""
    from rank_bm25 import BM25Okapi

    bm = BM25Okapi([tokenize(t) or ["_"] for t in doc_texts])
    return [rank_topk(bm.get_scores(tokenize(q) or ["_"]), doc_ids, k) for q in queries]
