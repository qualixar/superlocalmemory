"""LLD-05 §12.5 — MCP prestage_context tool tests."""

from __future__ import annotations

import json
import time

import pytest

from superlocalmemory.mcp.tools_context import (
    MAX_CALLS_PER_MINUTE,
    MAX_RESPONSE_BYTES,
    _RateLimiter,
    prestage_context,
)


def _fake_recall(_q: str, _limit: int, _pid: str) -> list[dict]:
    return [
        {"id": "m1", "text": "the assistant should use Qualixar",
         "score": 0.8, "source": "memory"},
        {"id": "m2", "text": "remember the 2026 ecosystem",
         "score": 0.7, "source": "memory"},
    ]


def test_tool_returns_topk_with_scores():
    out = prestage_context("qualixar", limit=2, recall_fn=_fake_recall)
    assert "memories" in out
    assert len(out["memories"]) == 2
    assert all("score" in m for m in out["memories"])
    assert all(0.0 <= m["score"] <= 1.0 for m in out["memories"])


def test_rate_limit_30_per_minute():
    limiter = _RateLimiter(max_calls=MAX_CALLS_PER_MINUTE, window=60.0)
    for _ in range(MAX_CALLS_PER_MINUTE):
        out = prestage_context("q", recall_fn=_fake_recall,
                               limiter=limiter, session_id="s1")
        assert "error" not in out
    over = prestage_context("q", recall_fn=_fake_recall,
                             limiter=limiter, session_id="s1")
    assert over.get("error") == "rate_limit_exceeded"


def test_response_capped_16kb():
    def _huge(_q, _l, _p):
        # Each memory ~2 KB post-cap
        return [{"id": f"m{i}", "text": "x" * 8000, "score": 0.5}
                for i in range(20)]
    out = prestage_context("huge", limit=20, recall_fn=_huge)
    encoded = json.dumps(out).encode("utf-8")
    assert len(encoded) <= MAX_RESPONSE_BYTES
    assert out.get("truncated_count", 0) >= 0


def test_redaction_applied_to_memories():
    def _secrets(_q, _l, _p):
        return [{"id": "x", "text": "sk-" + "a" * 40, "score": 0.5}]
    out = prestage_context("any", recall_fn=_secrets)
    assert "REDACTED" in out["memories"][0]["text"]


def test_empty_query_rejected():
    out = prestage_context("", recall_fn=_fake_recall)
    assert out.get("error") == "empty_query"


def test_recall_exception_returns_error():
    def _boom(_q, _l, _p):
        raise RuntimeError("engine dead")
    out = prestage_context("q", recall_fn=_boom)
    assert out.get("error") == "recall_error"


def test_limit_clamped():
    out = prestage_context("q", limit=10_000, recall_fn=_fake_recall)
    assert out["limit"] == 50  # max


def test_limit_minimum_is_one():
    out = prestage_context("q", limit=0, recall_fn=_fake_recall)
    assert out["limit"] == 1


def test_rate_limiter_reset_window():
    clock = [0.0]
    limiter = _RateLimiter(max_calls=2, window=10.0, now_fn=lambda: clock[0])
    assert limiter.allow("s") is True
    assert limiter.allow("s") is True
    assert limiter.allow("s") is False
    clock[0] = 11.0
    assert limiter.allow("s") is True


def test_rate_limiter_reset_method():
    limiter = _RateLimiter(max_calls=1, window=60.0)
    limiter.allow("s")
    assert limiter.allow("s") is False
    limiter.reset()
    assert limiter.allow("s") is True


def test_non_dict_rows_filtered():
    def _mixed(_q, _l, _p):
        return [{"id": "m1", "text": "ok", "score": 0.5}, "not-a-dict"]
    out = prestage_context("q", recall_fn=_mixed)
    assert len(out["memories"]) == 1


def test_register_tool_smoke():
    """Tool registration should not raise — smoke check."""
    class _FakeServer:
        def __init__(self):
            self.registered = []

        def tool(self, *args, **kwargs):
            # v3.4.26 Phase 1: ignore ToolAnnotations kwargs.
            def deco(fn):
                self.registered.append(fn.__name__)
                return fn
            return deco

    from superlocalmemory.mcp.tools_context import register_prestage_tool
    fs = _FakeServer()
    register_prestage_tool(fs, _fake_recall)
    assert fs.registered == ["prestage_context"]
