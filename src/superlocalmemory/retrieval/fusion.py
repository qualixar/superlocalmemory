# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Weighted Reciprocal Rank Fusion.

Single-pass RRF with k=15 for sharp rank discrimination on small candidate pools.
V1 had triple re-fusion which destroyed rankings — fixed in V2.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusionResult:
    """Single fused result with per-channel provenance."""
    fact_id: str
    fused_score: float
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)


def weighted_rrf(
    channels: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    k: int = 15,
    max_rank_penalty: int = 1000,
) -> list[FusionResult]:
    """Fuse ranked lists via Weighted Reciprocal Rank Fusion.

    Args:
        channels: channel_name -> [(fact_id, score)] sorted desc.
        weights: channel_name -> weight multiplier.
        k: RRF smoothing constant (60 for diverse retrieval, D116).
        max_rank_penalty: Rank assigned to absent documents.

    Returns:
        FusionResult list sorted by fused_score descending.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    rank_maps: dict[str, dict[str, int]] = {}
    score_maps: dict[str, dict[str, float]] = {}

    for ch, ranked in channels.items():
        ranks: dict[str, int] = {}
        scores: dict[str, float] = {}
        for i, (fid, sc) in enumerate(ranked):
            ranks[fid] = i + 1
            scores[fid] = sc
        rank_maps[ch] = ranks
        score_maps[ch] = scores

    all_ids: set[str] = set()
    for ranked in channels.values():
        for fid, _ in ranked:
            all_ids.add(fid)

    results: list[FusionResult] = []
    for fid in all_ids:
        fused = 0.0
        ch_ranks: dict[str, int] = {}
        ch_scores: dict[str, float] = {}
        for ch in channels:
            w = weights.get(ch, 1.0)
            rank = rank_maps[ch].get(fid, max_rank_penalty)
            ch_ranks[ch] = rank
            ch_scores[ch] = score_maps[ch].get(fid, 0.0)
            fused += w / (k + rank)
        results.append(FusionResult(fid, fused, ch_ranks, ch_scores))

    results.sort(key=lambda r: r.fused_score, reverse=True)
    return results
