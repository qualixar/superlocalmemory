"""TDD tests for context_key.py — multi-turn context fingerprint.

LLD-03 §4.1. RED-first: every test asserts a specific property the
implementer MUST satisfy.
"""

from __future__ import annotations

import pytest

from superlocalmemory.optimize.cache.context_key import ContextKeyBuilder


def test_build_returns_16_hex_chars():
    """Fingerprint must be exactly 16 hex chars (64 bits — A-22 fix)."""
    builder = ContextKeyBuilder(window_turns=3)
    fp = builder.build(
        messages=[{"role": "user", "content": "hi"}],
        tenant_id="t1",
    )
    assert isinstance(fp, str)
    assert len(fp) == 16
    assert all(c in "0123456789abcdef" for c in fp)


def test_same_conversation_same_fingerprint():
    """Deterministic — same inputs produce same fingerprint."""
    builder = ContextKeyBuilder(window_turns=3)
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+5?"},
    ]
    fp1 = builder.build(msgs, tenant_id="t1")
    fp2 = builder.build(msgs, tenant_id="t1")
    assert fp1 == fp2


def test_different_tenants_different_fingerprints():
    """Cross-tenant context isolation — fingerprints differ per tenant."""
    builder = ContextKeyBuilder(window_turns=3)
    msgs = [{"role": "user", "content": "hi"}]
    fp_a = builder.build(msgs, tenant_id="tenant_a")
    fp_b = builder.build(msgs, tenant_id="tenant_b")
    assert fp_a != fp_b


def test_different_contexts_different_fingerprints():
    """Different prior turns → different fingerprint."""
    builder = ContextKeyBuilder(window_turns=3)
    msgs_a = [
        {"role": "user", "content": "capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "and Germany?"},
    ]
    msgs_b = [
        {"role": "user", "content": "weather today?"},
        {"role": "assistant", "content": "Sunny"},
        {"role": "user", "content": "and tomorrow?"},
    ]
    assert builder.build(msgs_a, "t") != builder.build(msgs_b, "t")


def test_single_turn_stable_sentinel():
    """Single-turn (no prior context) returns a stable sentinel."""
    builder = ContextKeyBuilder(window_turns=3)
    msgs = [{"role": "user", "content": "hi"}]
    fp1 = builder.build(msgs, "t1")
    fp2 = builder.build(msgs, "t1")
    assert fp1 == fp2
    assert isinstance(fp1, str) and len(fp1) == 16


def test_turn_count_counts_assistant_messages():
    """turn_count returns number of completed assistant turns."""
    builder = ContextKeyBuilder(window_turns=3)
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "c"},
    ]
    assert builder.turn_count(msgs) == 2


def test_window_turns_validates():
    """window_turns must be >= 1."""
    with pytest.raises(ValueError):
        ContextKeyBuilder(window_turns=0)
    with pytest.raises(ValueError):
        ContextKeyBuilder(window_turns=-1)
