# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §5.1

"""Static 40-arm catalog for the contextual Thompson bandit.

LLD reference: ``.backup/active-brain/lld/LLD-03-contextual-bandit-and-ensemble.md``
Section 5.1 — arm = (semantic, bm25, entity_graph, temporal,
cross_encoder_bias) weight bundle drawn from a 7-point canonical grid.

Pure-data module — zero imports from the rest of the codebase. Audit-friendly
diff. Bumping the catalog requires bumping ``__version__`` and emitting a
migration in LLD-07.

Hard rules enforced here:
  - B3: ``len(ARM_CATALOG) == 40`` — asserted at import time.
  - B3: every weight in every arm belongs to ``_WEIGHT_GRID``.
"""

from __future__ import annotations

__version__ = "1"

# Single source of truth for the discrete weight grid (LLD-03 §3.1).
_WEIGHT_GRID: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 2.0)


# 40-arm catalog. Grouped by regime for auditability — DO NOT reorder.
ARM_CATALOG: dict[str, dict[str, float]] = {
    # Balanced anchors (4)
    "balanced_1_0":          {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "balanced_1_2":          {"semantic": 1.2, "bm25": 1.2, "entity_graph": 1.2, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "balanced_1_5":          {"semantic": 1.5, "bm25": 1.3, "entity_graph": 1.3, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "balanced_2_0":          {"semantic": 2.0, "bm25": 1.5, "entity_graph": 1.5, "temporal": 1.0, "cross_encoder_bias": 1.0},
    # Semantic-heavy (5)
    "semantic_heavy_1":      {"semantic": 1.5, "bm25": 0.5, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "semantic_heavy_2":      {"semantic": 2.0, "bm25": 0.5, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "semantic_heavy_3":      {"semantic": 2.0, "bm25": 1.0, "entity_graph": 0.5, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "semantic_rerank_boost": {"semantic": 2.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.5},
    "semantic_pure":         {"semantic": 2.0, "bm25": 0.5, "entity_graph": 0.5, "temporal": 0.5, "cross_encoder_bias": 1.3},
    # BM25-heavy (5)
    "bm25_heavy_1":          {"semantic": 0.5, "bm25": 2.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "bm25_heavy_2":          {"semantic": 1.0, "bm25": 2.0, "entity_graph": 0.5, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "bm25_temporal":         {"semantic": 0.5, "bm25": 1.5, "entity_graph": 1.0, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "bm25_pure":             {"semantic": 0.5, "bm25": 2.0, "entity_graph": 0.5, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "bm25_rerank_strong":    {"semantic": 1.0, "bm25": 1.5, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.5},
    # Entity-heavy (6)
    "entity_heavy_1":        {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.5, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "entity_heavy_2":        {"semantic": 1.0, "bm25": 1.0, "entity_graph": 2.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "entity_semantic":       {"semantic": 1.5, "bm25": 0.5, "entity_graph": 2.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "entity_pure":           {"semantic": 0.5, "bm25": 0.5, "entity_graph": 2.0, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "entity_graph_boost":    {"semantic": 1.0, "bm25": 0.5, "entity_graph": 2.0, "temporal": 1.0, "cross_encoder_bias": 1.3},
    "entity_kg_multi":       {"semantic": 1.3, "bm25": 1.0, "entity_graph": 1.5, "temporal": 1.0, "cross_encoder_bias": 1.2},
    # Temporal-heavy (6)
    "temporal_heavy_1":      {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "temporal_heavy_2":      {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 2.0, "cross_encoder_bias": 1.0},
    "temporal_semantic":     {"semantic": 1.5, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "temporal_recent":       {"semantic": 1.0, "bm25": 1.2, "entity_graph": 0.5, "temporal": 2.0, "cross_encoder_bias": 1.0},
    "temporal_bm25":         {"semantic": 0.5, "bm25": 1.5, "entity_graph": 0.5, "temporal": 2.0, "cross_encoder_bias": 1.0},
    "temporal_entity":       {"semantic": 1.0, "bm25": 0.5, "entity_graph": 1.5, "temporal": 1.5, "cross_encoder_bias": 1.0},
    # Diagonal / exploratory (8)
    "sem_bm25_diag":         {"semantic": 1.5, "bm25": 1.5, "entity_graph": 0.5, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "sem_entity_diag":       {"semantic": 1.5, "bm25": 0.5, "entity_graph": 1.5, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "sem_temporal_diag":     {"semantic": 1.5, "bm25": 0.5, "entity_graph": 0.5, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "bm25_entity_diag":      {"semantic": 0.5, "bm25": 1.5, "entity_graph": 1.5, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "bm25_temporal_diag":    {"semantic": 0.5, "bm25": 1.5, "entity_graph": 0.5, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "entity_temporal_diag":  {"semantic": 0.5, "bm25": 0.5, "entity_graph": 1.5, "temporal": 1.5, "cross_encoder_bias": 1.0},
    "three_axis_high":       {"semantic": 1.5, "bm25": 1.5, "entity_graph": 1.5, "temporal": 1.0, "cross_encoder_bias": 1.0},
    "all_high":              {"semantic": 1.5, "bm25": 1.5, "entity_graph": 1.5, "temporal": 1.5, "cross_encoder_bias": 1.0},
    # Conservative / fallback (6)
    "conservative_low":      {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.3},
    "conservative_high":     {"semantic": 1.3, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.5},
    "rerank_only":           {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 2.0},
    "light_bm25_only":       {"semantic": 0.5, "bm25": 1.3, "entity_graph": 0.5, "temporal": 0.5, "cross_encoder_bias": 1.0},
    "mid_semantic":          {"semantic": 1.2, "bm25": 0.8, "entity_graph": 0.8, "temporal": 0.8, "cross_encoder_bias": 1.0},
    "fallback_default":      {"semantic": 1.0, "bm25": 1.0, "entity_graph": 1.0, "temporal": 1.0, "cross_encoder_bias": 1.0},
}


# B3 — module-level invariant. Raises ImportError if someone edits the catalog
# without keeping the count at 40.
assert len(ARM_CATALOG) == 40, (
    f"ARM_CATALOG size drift: {len(ARM_CATALOG)} (expected 40)"
)


# B3 — every weight belongs to the canonical grid. Checked at import so any
# off-grid weight fails loudly during CI / daemon startup.
_GRID_SET: frozenset[float] = frozenset(_WEIGHT_GRID)
for _name, _weights in ARM_CATALOG.items():
    for _channel, _w in _weights.items():
        if _w not in _GRID_SET:  # pragma: no cover — invariant
            raise AssertionError(
                f"arm {_name!r} channel {_channel!r} weight {_w} not in grid"
            )
del _name, _weights, _channel, _w  # don't leak loop vars as module globals


__all__ = ("ARM_CATALOG", "_WEIGHT_GRID", "__version__")
