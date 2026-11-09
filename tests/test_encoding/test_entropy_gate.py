# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.entropy_gate.

Covers:
  - Content-based filtering (short text, low-info patterns)
  - Similarity-based deduplication
  - Window management (size, reset)
  - First content always passes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.entropy_gate import EntropyGate, _cosine

# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self) -> None:
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self) -> None:
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Content-based filtering
# ---------------------------------------------------------------------------

class TestContentFiltering:
    def test_too_short(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("hi") is False
        assert gate.should_pass("ok") is False
        assert gate.should_pass("a") is False

    def test_empty_string(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("") is False

    def test_whitespace_only(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("   ") is False

    def test_low_info_patterns(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("ok") is False
        assert gate.should_pass("sure") is False
        assert gate.should_pass("thanks") is False
        assert gate.should_pass("thank you") is False
        assert gate.should_pass("got it") is False
        assert gate.should_pass("yes") is False
        assert gate.should_pass("yeah") is False
        assert gate.should_pass("nice") is False

    def test_low_info_with_punctuation(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("okay.") is False
        assert gate.should_pass("sure!") is False
        assert gate.should_pass("thanks!") is False

    def test_meaningful_content_passes(self) -> None:
        gate = EntropyGate()
        assert gate.should_pass("Alice works at Google as a software engineer") is True

    def test_borderline_length(self) -> None:
        gate = EntropyGate()
        # Exactly 10 chars = passes length check
        assert gate.should_pass("1234567890") is True
        # 9 chars = fails
        assert gate.should_pass("123456789") is False


# ---------------------------------------------------------------------------
# Similarity-based filtering
# ---------------------------------------------------------------------------

class TestSimilarityFiltering:
    def _mock_embedder(self, embeddings: list[list[float]]) -> MagicMock:
        embedder = MagicMock()
        call_count = [0]
        def embed(text: str) -> list[float]:
            idx = min(call_count[0], len(embeddings) - 1)
            call_count[0] += 1
            return embeddings[idx]
        embedder.embed = embed
        return embedder

    def test_first_content_always_passes(self) -> None:
        embedder = self._mock_embedder([[1.0, 0.0, 0.0]])
        gate = EntropyGate(embedder=embedder, similarity_threshold=0.95)
        assert gate.should_pass("Alice works at Google") is True

    def test_near_duplicate_blocked(self) -> None:
        embedder = self._mock_embedder([
            [1.0, 0.0, 0.0],  # First call (first content)
            [1.0, 0.0, 0.0],  # Second call (near-duplicate)
        ])
        gate = EntropyGate(embedder=embedder, similarity_threshold=0.95)
        gate.should_pass("Alice works at Google")  # First, passes
        assert gate.should_pass("Alice works at Google") is False  # Near-duplicate

    def test_different_content_passes(self) -> None:
        embedder = self._mock_embedder([
            [1.0, 0.0, 0.0],  # First content
            [0.0, 1.0, 0.0],  # Different content
        ])
        gate = EntropyGate(embedder=embedder, similarity_threshold=0.95)
        gate.should_pass("Alice works at Google")
        assert gate.should_pass("Bob likes swimming") is True

    def test_no_embedder_skips_similarity(self) -> None:
        gate = EntropyGate(embedder=None, similarity_threshold=0.95)
        assert gate.should_pass("Alice works at Google") is True
        assert gate.should_pass("Alice works at Google") is True  # No dedup

    def test_window_size_respected(self) -> None:
        idx = [0]
        embeddings = [[float(i)] for i in range(10)]
        embedder = MagicMock()
        def embed(text: str) -> list[float]:
            r = embeddings[min(idx[0], 9)]
            idx[0] += 1
            return r
        embedder.embed = embed

        gate = EntropyGate(embedder=embedder, window_size=3)
        for i in range(5):
            gate.should_pass(f"Content number {i} which is long enough")
        # Window should only hold last 3 embeddings
        assert len(gate._recent_embeddings) <= 3


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_window(self) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]
        gate = EntropyGate(embedder=embedder)
        gate.should_pass("Alice works at Google")
        assert len(gate._recent_embeddings) == 1
        gate.reset()
        assert len(gate._recent_embeddings) == 0

    def test_after_reset_first_content_passes(self) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]
        gate = EntropyGate(embedder=embedder, similarity_threshold=0.95)
        gate.should_pass("Alice works at Google")
        gate.reset()
        # After reset, same content should pass again (first content)
        assert gate.should_pass("Alice works at Google") is True
