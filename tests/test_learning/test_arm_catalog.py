# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §7.1

"""Tests for ``learning/arm_catalog.py`` — static 40-arm catalog.

Covers hard rule B3 (size == 40) and the canonical weight-grid invariant.
"""

from __future__ import annotations

import pytest


def test_catalog_size_is_40():
    """B3: catalog size is locked at exactly 40 arms."""
    from superlocalmemory.learning.arm_catalog import ARM_CATALOG

    assert len(ARM_CATALOG) == 40, (
        f"ARM_CATALOG must have exactly 40 entries, got {len(ARM_CATALOG)}"
    )


def test_catalog_keys_unique():
    """Names are dict keys — duplicates are impossible, but we confirm.

    The assertion also catches accidental copy-paste where two arms end up
    with the same name (silently overriding the first in the dict literal).
    Counting via ``len`` against the known expected count is the guard.
    """
    from superlocalmemory.learning.arm_catalog import ARM_CATALOG

    names = list(ARM_CATALOG.keys())
    assert len(names) == len(set(names)) == 40


def test_catalog_weights_in_grid():
    """B3: every weight in every arm belongs to the canonical grid."""
    from superlocalmemory.learning.arm_catalog import ARM_CATALOG, _WEIGHT_GRID

    grid = set(_WEIGHT_GRID)
    for name, weights in ARM_CATALOG.items():
        for channel, w in weights.items():
            assert w in grid, (
                f"arm {name!r} channel {channel!r} weight {w} not in {grid}"
            )


def test_catalog_channels_are_exact_5():
    """Each arm must define exactly the 5 retrieval channels."""
    from superlocalmemory.learning.arm_catalog import ARM_CATALOG

    expected = {"semantic", "bm25", "entity_graph", "temporal",
                "cross_encoder_bias"}
    for name, weights in ARM_CATALOG.items():
        assert set(weights.keys()) == expected, (
            f"arm {name!r} channels mismatch: got {set(weights.keys())}"
        )


def test_weight_grid_is_canonical():
    """The canonical grid is immutable and exactly 7 points."""
    from superlocalmemory.learning.arm_catalog import _WEIGHT_GRID

    assert _WEIGHT_GRID == (0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 2.0)


def test_fallback_default_present():
    """``fallback_default`` must exist — used by error-matrix fallback path."""
    from superlocalmemory.learning.arm_catalog import ARM_CATALOG

    assert "fallback_default" in ARM_CATALOG
    weights = ARM_CATALOG["fallback_default"]
    # Fallback is the neutral all-1.0 arm.
    assert all(w == 1.0 for w in weights.values())
