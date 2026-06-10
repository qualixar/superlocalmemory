# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — F-2 Recall Budget tests

"""F-2: Recall output budget + per-fact clamp.

TDD RED suite. Tests for:
- per-fact content clamped to recall_per_fact_max_chars (default 2400)
  with head 70% + "...truncated N chars..." + tail 30%
- total budget: stubs beyond recall_total_max_chars (default 12000)
- full=True bypasses clamping
- truncated: True additive field on clamped results
- Config fields on RetrievalConfig
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.server.recall_serializer import (
    clamp_fact_content,
    apply_recall_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _long_content(n: int) -> str:
    """Create deterministic content of n chars."""
    chunk = "abcdefghijklmnopqrstuvwxyz0123456789"
    return (chunk * (n // len(chunk) + 1))[:n]


def _make_result(fact_id: str, content: str, score: float = 0.8) -> dict:
    return {"fact_id": fact_id, "content": content, "score": score}


# ---------------------------------------------------------------------------
# F-2-A: Config fields
# ---------------------------------------------------------------------------

class TestRecallBudgetConfig:
    """RetrievalConfig recall budget fields."""

    def test_has_recall_per_fact_max_chars(self) -> None:
        """RetrievalConfig.recall_per_fact_max_chars defaults to 2400."""
        cfg = RetrievalConfig()
        assert hasattr(cfg, "recall_per_fact_max_chars")
        assert cfg.recall_per_fact_max_chars == 2400

    def test_has_recall_total_max_chars(self) -> None:
        """RetrievalConfig.recall_total_max_chars defaults to 12000."""
        cfg = RetrievalConfig()
        assert hasattr(cfg, "recall_total_max_chars")
        assert cfg.recall_total_max_chars == 12000


# ---------------------------------------------------------------------------
# F-2-B: Per-fact clamp
# ---------------------------------------------------------------------------

class TestPerFactClamp:
    """Per-fact content clamping."""

    def test_short_content_unchanged(self) -> None:
        """Content under limit is returned unchanged, no truncation flag."""
        content = "short content"
        result, truncated = clamp_fact_content(content, max_chars=2400)
        assert result == content
        assert truncated is False

    def test_long_content_clamped(self) -> None:
        """Content over limit is clamped."""
        content = _long_content(3000)
        result, truncated = clamp_fact_content(content, max_chars=2400)
        assert len(result) < len(content)
        assert truncated is True

    def test_clamp_head_70_tail_30(self) -> None:
        """Head is 70%, tail is 30% of allowed content."""
        content = _long_content(3000)
        result, truncated = clamp_fact_content(content, max_chars=2400)
        assert truncated is True
        # Head: first 70% of 2400 = 1680 chars from start
        head_len = int(2400 * 0.70)
        tail_len = 2400 - head_len
        assert result.startswith(content[:head_len])
        assert result.endswith(content[-tail_len:])

    def test_clamp_has_truncation_marker(self) -> None:
        """Truncated result contains a [truncated N chars] marker."""
        content = _long_content(3000)
        result, _ = clamp_fact_content(content, max_chars=2400)
        assert "[truncated" in result.lower() or "truncated" in result.lower()

    def test_content_exactly_at_limit_unchanged(self) -> None:
        """Content exactly at limit is NOT truncated."""
        content = _long_content(2400)
        result, truncated = clamp_fact_content(content, max_chars=2400)
        assert result == content
        assert truncated is False


# ---------------------------------------------------------------------------
# F-2-C: Total budget — stub remaining results
# ---------------------------------------------------------------------------

class TestTotalBudget:
    """Total recall budget with stubs."""

    def test_within_total_budget_no_stubs(self) -> None:
        """Results fitting within total budget are returned in full."""
        results = [_make_result(f"f{i}", _long_content(100)) for i in range(5)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000)
        # All short results fit — no stubs
        for r in out:
            assert r.get("stub", False) is False

    def test_exceeding_total_budget_generates_stubs(self) -> None:
        """Results beyond total budget are returned as stubs."""
        # Each result is 3000 chars (clamped to ~2400), 5 results = ~12K total
        # 6th result pushes over
        results = [_make_result(f"f{i}", _long_content(3000)) for i in range(6)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000)
        # Some must be stubs
        stubs = [r for r in out if r.get("stub")]
        assert len(stubs) >= 1, "Expected at least one stub result beyond budget"

    def test_stub_shape(self) -> None:
        """Stub results have fact_id, score, and first 120 chars of content."""
        results = [_make_result(f"f{i}", _long_content(3000)) for i in range(8)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000)
        stubs = [r for r in out if r.get("stub")]
        assert stubs, "Need at least one stub to verify shape"
        stub = stubs[0]
        assert "fact_id" in stub
        assert "score" in stub
        assert "content" in stub
        assert stub.get("stub") is True
        assert len(stub["content"]) <= 120 + len("…")

    def test_full_param_bypasses_clamping(self) -> None:
        """full=True returns complete content without clamping."""
        content = _long_content(5000)
        results = [_make_result("f1", content)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000, full=True)
        assert out[0]["content"] == content
        assert out[0].get("truncated", False) is False

    def test_full_param_bypasses_total_budget(self) -> None:
        """full=True also bypasses the total budget stub mechanism."""
        results = [_make_result(f"f{i}", _long_content(3000)) for i in range(10)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000, full=True)
        stubs = [r for r in out if r.get("stub")]
        assert stubs == [], "full=True must not generate stubs"

    def test_truncated_field_on_clamped_results(self) -> None:
        """Clamped results carry truncated=True field."""
        content = _long_content(3000)
        results = [_make_result("f1", content)]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000)
        assert out[0].get("truncated") is True

    def test_short_content_no_truncated_field(self) -> None:
        """Short content results do NOT get truncated=True."""
        results = [_make_result("f1", "short content")]
        out = apply_recall_budget(results, per_fact_max=2400, total_max=12000)
        assert out[0].get("truncated", False) is False
