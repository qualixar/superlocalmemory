# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Released Optimize configuration reference must match the typed schema."""

from __future__ import annotations

from pathlib import Path


_DOC = Path(__file__).resolve().parents[2] / "docs" / "optimize-config.md"


def test_optimize_config_reference_matches_current_compression_contract() -> None:
    """Removed knobs must not be advertised; safe lossless compression is on by default."""
    text = _DOC.read_text(encoding="utf-8")

    assert "| `compress_enabled` | bool | `true`" in text
    assert "| `compress_protect_recent` | int | `4`" in text
    for removed in ("compress_code", "compress_json", "compress_ccr", "compress_align"):
        assert f"`{removed}`" not in text


def test_optimize_config_reference_matches_current_master_and_semantic_defaults() -> None:
    """The page calls itself a reference, so its quoted defaults are executable truth."""
    text = _DOC.read_text(encoding="utf-8")

    for expected in (
        "| `enabled` | bool | `false`",
        "| `config_version` | int | `1`",
        "| `semantic_boundary_ceiling` | float | `0.995`",
        "| `semantic_centroid_distance_floor` | float | `0.15`",
        "| `semantic_pad_latency_ms` | float | `0`",
    ):
        assert expected in text


def test_optimize_config_reference_explains_actual_activation_boundaries() -> None:
    """The reference must not promise live proxy mounting from a config edit."""
    text = _DOC.read_text(encoding="utf-8")

    assert "SDK adapter enablement" in text
    assert "daemon startup" in text
    assert "config file changes do not mount proxy routes" in text
    assert "does not trigger hot-reload" in text
    assert '`"safe"` (lossless whitespace normalization)' in text
