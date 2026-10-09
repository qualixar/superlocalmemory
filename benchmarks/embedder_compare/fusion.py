"""S3 fusion: weighted RRF of the text channel and the media channel.

The media-channel weight is tuned on the dev split only (maximise answerable
recall@5 over all strata); ties go to the smaller weight, i.e. the text channel.
"""
from __future__ import annotations

import numpy as np

import ranking
import stats


def fuse_one(text_top: list, media_top: list, weight: float) -> list[tuple[str, float]]:
    return ranking.rrf_fuse([[d for d, _ in text_top], [d for d, _ in media_top]],
                            weights=[1.0, weight])


def abstain_score(fused: list, text_top: list, media_top: list, media_ids: set[str]) -> float:
    """Raw cosine of the fused top-1 doc in the channel it belongs to.

    Media docs use the media-channel cosine, text docs the text-channel cosine
    (RRF scores carry no confidence). Falls back to the other channel's score.
    """
    if not fused:
        return 0.0
    doc = fused[0][0]
    text, media = dict(text_top), dict(media_top)
    own, other = (media, text) if doc in media_ids else (text, media)
    return float(own.get(doc, other.get(doc, 0.0)))


def fuse_all(text_tops: list, media_tops: list, media_ids: set[str],
             weight: float) -> tuple[list, list]:
    fused = [fuse_one(t, m, weight) for t, m in zip(text_tops, media_tops)]
    abst = [abstain_score(f, t, m, media_ids) for f, t, m in zip(fused, text_tops, media_tops)]
    return fused, abst


def dev_recall(fused: list, qids: list[str], dev_qrels: dict) -> float:
    """Mean answerable recall@5 over the dev queries."""
    index = {q: i for i, q in enumerate(qids)}
    run, qrels = {}, {}
    for qid, rel in dev_qrels.items():
        if rel and qid in index:
            qrels[qid] = rel
            top = fused[index[qid]]
            run[qid] = {d: float(len(top) - i) for i, (d, _) in enumerate(top)}
    vals = list(stats.per_query(qrels, run, "recall@5").values())
    return float(np.mean(vals)) if vals else 0.0


def tune_weight(text_tops: list, media_tops: list, media_ids: set[str], qids: list[str],
                dev_qrels: dict, weights=ranking.MEDIA_WEIGHTS) -> tuple[float, dict]:
    """Return (chosen weight, {weight: dev recall@5}); dev labels only."""
    table = {}
    for w in weights:
        fused, _ = fuse_all(text_tops, media_tops, media_ids, w)
        table[w] = dev_recall(fused, qids, dev_qrels)
    best = max(weights, key=lambda w: (table[w], -weights.index(w)))
    return best, table
