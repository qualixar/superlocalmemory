"""LLD-00 §10.5 — OptimizeConfig tests."""

from __future__ import annotations

import dataclasses

import pytest

from superlocalmemory.optimize.config.schema import OptimizeConfig, TTLConfig


def test_from_dict_empty_dict_uses_defaults() -> None:
    cfg = OptimizeConfig.from_dict({})
    assert cfg.cache_enabled is True
    assert cfg.compress_mode == "safe"
    assert cfg.ttl.exact_seconds == 86400


def test_from_dict_ignores_unknown_keys() -> None:
    cfg = OptimizeConfig.from_dict({"future_feature": True, "cache_enabled": False})
    assert cfg.cache_enabled is False


def test_from_dict_partial_ttl() -> None:
    cfg = OptimizeConfig.from_dict({"ttl": {"exact_seconds": 7200}})
    assert cfg.ttl.exact_seconds == 7200
    assert cfg.ttl.semantic_seconds == 3600


def test_as_dict_round_trip() -> None:
    original = OptimizeConfig.from_dict({
        "cache_enabled": True,
        "compress_mode": "aggressive",
        "ttl": {"exact_seconds": 7200},
    })
    round_tripped = OptimizeConfig.from_dict(original.as_dict())
    assert round_tripped.cache_enabled == original.cache_enabled
    assert round_tripped.compress_mode == original.compress_mode
    assert round_tripped.ttl.exact_seconds == original.ttl.exact_seconds


def test_validate_rejects_invalid_compress_mode() -> None:
    cfg = OptimizeConfig.from_dict({"compress_mode": "ultra_lossy"})
    with pytest.raises(ValueError, match="compress_mode"):
        cfg.validate()


def test_validate_rejects_zero_ttl() -> None:
    cfg = OptimizeConfig.from_dict({"ttl": {"exact_seconds": 0}})
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_rejects_out_of_range_threshold() -> None:
    cfg = OptimizeConfig.from_dict({"semantic_return_threshold": 1.5})
    with pytest.raises(ValueError, match="semantic_return_threshold"):
        cfg.validate()


def test_optimize_config_is_immutable() -> None:
    """frozen=True prevents mutation (LLD-00 CRIT-3)."""
    cfg = OptimizeConfig.from_dict({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # must fail

