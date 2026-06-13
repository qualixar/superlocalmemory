"""LLD-02 §8.1 — KeyBuilder tests."""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.key_builder import CacheConfig, KeyBuilder


def _tenant(n: int) -> str:
    """64-char lowercase hex (canonical tenant_id format)."""
    return f"{n:064x}"


def test_two_tenants_same_prompt_produce_different_keys() -> None:
    kb = KeyBuilder()
    args_a = dict(
        tenant_id=_tenant(1), model_id="claude-sonnet-4-6",
        model_version="20250219", system="You are helpful.",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        raw_params={"max_tokens": 100},
    )
    args_b = dict(args_a); args_b["tenant_id"] = _tenant(2)
    key_a = kb.build(**args_a)
    key_b = kb.build(**args_b)
    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b
    assert _tenant(1) in key_a
    assert _tenant(2) in key_b


def test_identical_inputs_produce_identical_key() -> None:
    kb = KeyBuilder()
    kwargs = dict(
        tenant_id=_tenant(1), model_id="gpt-4o", model_version="",
        system="", messages=[{"role": "user", "content": "Hello"}],
        raw_params={"max_tokens": 50},
    )
    assert kb.build(**kwargs) == kb.build(**kwargs)


def test_model_version_change_produces_different_key() -> None:
    kb = KeyBuilder()
    base = dict(
        tenant_id=_tenant(1), model_id="claude-sonnet-4-6",
        system="", messages=[{"role": "user", "content": "Hello"}],
        raw_params={},
    )
    key_old = kb.build(**{**base, "model_version": "20250101"})
    key_new = kb.build(**{**base, "model_version": "20250219"})
    assert key_old != key_new


def test_system_prompt_change_produces_different_key() -> None:
    kb = KeyBuilder()
    base = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v1",
        messages=[{"role": "user", "content": "Hi"}], raw_params={},
    )
    key_a = kb.build(**{**base, "system": "You are a pirate."})
    key_b = kb.build(**{**base, "system": "You are a lawyer."})
    assert key_a != key_b


def test_temperature_gt_zero_returns_none_by_default() -> None:
    kb = KeyBuilder()
    result = kb.build(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[], raw_params={"temperature": 0.7},
    )
    assert result is None


def test_temperature_zero_returns_key() -> None:
    kb = KeyBuilder()
    result = kb.build(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[], raw_params={"temperature": 0},
    )
    assert result is not None


def test_temperature_gt_zero_allowed_when_opted_in() -> None:
    kb = KeyBuilder(CacheConfig(allow_nonzero_temperature_cache=True))
    result = kb.build(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[], raw_params={"temperature": 0.7},
    )
    assert result is not None


def test_excluded_params_do_not_affect_key() -> None:
    kb = KeyBuilder()
    base = dict(
        tenant_id=_tenant(1), model_id="m", model_version="v",
        system="", messages=[{"role": "user", "content": "x"}],
    )
    key_a = kb.build(**base, raw_params={"max_tokens": 100, "stream": True, "user": "u1"})
    key_b = kb.build(**base, raw_params={"max_tokens": 100, "stream": False, "user": "u2"})
    assert key_a == key_b


def test_empty_tenant_raises() -> None:
    kb = KeyBuilder()
    with pytest.raises(ValueError, match="tenant_id must be a 64-char"):
        kb.build(
            tenant_id="", model_id="m", model_version="v",
            system="", messages=[], raw_params={},
        )


def test_short_tenant_raises() -> None:
    kb = KeyBuilder()
    with pytest.raises(ValueError, match="tenant_id must be a 64-char"):
        kb.build(
            tenant_id="a" * 32, model_id="m", model_version="v",
            system="", messages=[], raw_params={},
        )


def test_full_message_array_in_key() -> None:
    kb = KeyBuilder()
    base = dict(tenant_id=_tenant(1), model_id="m", model_version="v", system="", raw_params={})
    key_a = kb.build(**base, messages=[{"role": "user", "content": "a"}])
    key_b = kb.build(**base, messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert key_a != key_b


def test_tenant_tag() -> None:
    kb = KeyBuilder()
    assert kb.tenant_tag(_tenant(1)).startswith("tenant:")


def test_model_tag() -> None:
    kb = KeyBuilder()
    assert kb.model_tag("claude-sonnet-4-6") == "model:claude-sonnet-4-6"


def test_c04_temperature_skip_increments_calls_skipped() -> None:
    """C-04: build() with non-zero temperature must increment MetricsCollector.calls_skipped."""
    from superlocalmemory.optimize.metrics.counters import MetricsCollector

    collector = MetricsCollector.get_instance()
    collector.reset()

    kb = KeyBuilder()
    key = kb.build(
        tenant_id=_tenant(1),
        model_id="claude",
        model_version="",
        system="",
        messages=[{"role": "user", "content": "temp test"}],
        raw_params={"temperature": 0.7},
    )

    assert key is None, "non-zero temperature with default config must return None"
    snap = collector.snapshot()
    assert snap.calls_skipped >= 1, (
        f"C-04: calls_skipped not incremented on temperature skip; got {snap.calls_skipped}"
    )
