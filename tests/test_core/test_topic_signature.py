# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-01 §4.2

"""Tests for superlocalmemory.core.topic_signature.

Covers LLD-01 §6.2 test matrix.
RED→GREEN→REFACTOR — written before implementation.
"""

from __future__ import annotations

import time
import unicodedata

import pytest

from superlocalmemory.core import topic_signature as ts


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_signature_deterministic_same_input() -> None:
    sig_a = ts.compute_topic_signature("implement the context cache module")
    sig_b = ts.compute_topic_signature("implement the context cache module")
    assert sig_a == sig_b


def test_signature_empty_input_is_zeros() -> None:
    assert ts.compute_topic_signature("") == "0" * 16


def test_signature_is_exactly_16_hex_chars() -> None:
    sig = ts.compute_topic_signature("hello world this is a test")
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


def test_signature_returns_string_for_whitespace_only() -> None:
    sig = ts.compute_topic_signature("   \n\t   ")
    assert isinstance(sig, str)
    assert len(sig) == 16


# ---------------------------------------------------------------------------
# NFC / Unicode
# ---------------------------------------------------------------------------


def test_signature_nfc_normalizes_codepoints() -> None:
    # e + combining acute (NFD) vs precomposed é (NFC) must hash the same.
    nfc = "caf\u00e9"
    nfd = "cafe\u0301"
    assert unicodedata.normalize("NFC", nfd) == nfc
    assert ts.compute_topic_signature(nfc) == ts.compute_topic_signature(nfd)


def test_signature_handles_emoji_and_rtl() -> None:
    sig = ts.compute_topic_signature("\U0001F600 hello \u05D0\u05D1\u05D2 world")
    assert len(sig) == 16


def test_signature_handles_mixed_scripts() -> None:
    sig = ts.compute_topic_signature("hello \u4e2d\u6587 world \u0645\u0631\u062d\u0628\u0627")
    assert len(sig) == 16


# ---------------------------------------------------------------------------
# Distinctness / collision resistance
# ---------------------------------------------------------------------------


def test_different_inputs_give_different_sigs() -> None:
    a = ts.compute_topic_signature("implement context cache fast path")
    b = ts.compute_topic_signature("implement topic signature hashing")
    assert a != b


def test_case_is_normalized_for_word_tokens() -> None:
    # Lowercasing word tokens collapses case for lowercase-vs-capitalized
    # inputs. ALL-CAPS intentionally preserves identifier semantics so
    # "API" != "api" (constants are meaningful). Verify that plain
    # sentence-case is normalized.
    a = ts.compute_topic_signature("hello world goodbye everyone")
    b = ts.compute_topic_signature("Hello World Goodbye Everyone")
    assert a == b


def test_bigram_resists_stopword_collision() -> None:
    # Two sentences with mostly stopwords but different content words:
    # the bigram feature must keep them distinct even after stopword filter.
    a = ts.compute_topic_signature("to be or not to be implement cache")
    b = ts.compute_topic_signature("to be or not to be implement signature")
    assert a != b


# ---------------------------------------------------------------------------
# ReDoS / truncation
# ---------------------------------------------------------------------------


def test_signature_truncates_over_limit_inputs() -> None:
    big = "word " * 5000  # 25000 chars
    sig = ts.compute_topic_signature(big)
    assert len(sig) == 16


def test_signature_nonbacktracking_on_pathological_input() -> None:
    # Attempt classic catastrophic-backtracking bait strings; since our
    # patterns are O(n) non-backtracking, this must complete quickly.
    pathological = ("/" + "a" * 100) * 40  # nested-looking path
    start = time.perf_counter()
    sig = ts.compute_topic_signature(pathological)
    elapsed = time.perf_counter() - start
    assert len(sig) == 16
    # Generous budget — the only failure mode we want to catch is pathological
    # blow-up into seconds. 0.5s is already 100x our target.
    assert elapsed < 0.5, f"took {elapsed:.3f}s"


def test_signature_perf_under_budget_on_2000_chars() -> None:
    text = "the quick brown fox jumps over the lazy dog " * 45  # ~2000 chars
    # Warm-up
    ts.compute_topic_signature(text)
    start = time.perf_counter()
    for _ in range(20):
        ts.compute_topic_signature(text)
    elapsed = (time.perf_counter() - start) / 20
    # Budget: <5 ms p95 per LLD-01 R7. Generous wall-clock cap to avoid CI flake.
    assert elapsed < 0.02, f"avg {elapsed*1000:.2f} ms"


# ---------------------------------------------------------------------------
# Structural tokens — URLs / paths / identifiers / quotes
# ---------------------------------------------------------------------------


def test_urls_contribute_to_signature() -> None:
    a = ts.compute_topic_signature("look at https://example.com/foo and fix")
    b = ts.compute_topic_signature("look at https://example.com/bar and fix")
    assert a != b


def test_paths_contribute_to_signature() -> None:
    a = ts.compute_topic_signature("edit /src/core/context_cache.py line 42")
    b = ts.compute_topic_signature("edit /src/core/topic_signature.py line 42")
    assert a != b


def test_identifiers_contribute_to_signature() -> None:
    a = ts.compute_topic_signature("rename ContextCache to ContextStore everywhere")
    b = ts.compute_topic_signature("rename TopicSignature to TopicHasher everywhere")
    assert a != b


def test_quoted_strings_contribute_to_signature() -> None:
    a = ts.compute_topic_signature('set mode to "fast" in config')
    b = ts.compute_topic_signature('set mode to "slow" in config')
    assert a != b


# ---------------------------------------------------------------------------
# Type safety — no raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["", " ", "a", "ab", "\x00\x01\x02"])
def test_signature_never_raises_on_short_inputs(payload: str) -> None:
    sig = ts.compute_topic_signature(payload)
    assert isinstance(sig, str)
    assert len(sig) == 16
