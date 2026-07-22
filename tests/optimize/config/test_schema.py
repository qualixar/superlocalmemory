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


def test_validate_rejects_invalid_compress_mode_when_constructed_directly() -> None:
    """validate() stays strict for programmatic construction (bypassing
    from_dict's fail-open normalization)."""
    base = OptimizeConfig.from_dict({})
    cfg = dataclasses.replace(base, compress_mode="ultra_lossy")
    with pytest.raises(ValueError, match="compress_mode"):
        cfg.validate()


# ---- C1/C2: nested config groups + fail-open mode normalization ----

def test_from_dict_accepts_nested_compress_block() -> None:
    """C1: real-world optimize.json nests compress settings; from_dict must
    honor them instead of silently leaving compression OFF."""
    cfg = OptimizeConfig.from_dict({"compress": {"enabled": True, "prose": True}})
    assert cfg.compress_enabled is True
    assert cfg.compress_prose is True


def test_from_dict_accepts_nested_cache_block() -> None:
    cfg = OptimizeConfig.from_dict(
        {"cache": {"enabled": False, "ttl_seconds": 3600, "semantic": True}}
    )
    assert cfg.cache_enabled is False
    assert cfg.ttl_seconds == 3600
    assert cfg.semantic_enabled is True


def test_from_dict_accepts_nested_proxy_block() -> None:
    cfg = OptimizeConfig.from_dict({"proxy": {"enabled": True}})
    assert cfg.proxy_enabled is True


def test_from_dict_flat_key_wins_over_nested() -> None:
    cfg = OptimizeConfig.from_dict(
        {"compress_enabled": False, "compress": {"enabled": True}}
    )
    assert cfg.compress_enabled is False


def test_c2_invalid_nested_mode_does_not_crash_boot() -> None:
    """C2: real optimize.json had compress.mode='fast' (invalid enum).
    Parse + validate must succeed (fail-open), never raise at daemon boot."""
    cfg = OptimizeConfig.from_dict(
        {"enabled": True, "compress": {"enabled": True, "mode": "fast"}}
    )
    cfg.validate()  # must NOT raise
    assert cfg.compress_mode == "safe"  # 'fast' is a legacy alias for safe
    assert cfg.compress_enabled is True


def test_from_dict_normalizes_unknown_mode_to_safe() -> None:
    cfg = OptimizeConfig.from_dict({"compress_mode": "ludicrous"})
    cfg.validate()  # must NOT raise
    assert cfg.compress_mode == "safe"


def test_from_dict_preserves_valid_aggressive_mode() -> None:
    cfg = OptimizeConfig.from_dict({"compress_mode": "aggressive"})
    assert cfg.compress_mode == "aggressive"


def test_from_dict_does_not_mutate_input() -> None:
    src = {"compress": {"enabled": True}}
    OptimizeConfig.from_dict(src)
    assert src == {"compress": {"enabled": True}}  # unchanged


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

