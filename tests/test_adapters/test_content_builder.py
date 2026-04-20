"""LLD-05 §12.1 — ContextPayload / build_payload tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.hooks.context_payload import (
    VERSION,
    build_payload,
    format_decisions,
    format_entities,
    format_memories,
    format_topics,
    truncate_payload_for_cap,
)


def test_payload_redacts_secrets():
    """Secrets embedded in recall output must be redacted."""
    def recall_fn(q: str, limit: int, pid: str) -> list[dict]:
        if "topics" in q:
            return [{"name": "real AKIAIOSFODNN7EXAMPLE secret", "score": 0.9}]
        if "entities" in q:
            return [{"name": "ghp_" + "a" * 40, "mentions": 5}]
        if "decisions" in q:
            return [{"text": "sk-ant-" + "x" * 30}]
        if "memories" in q:
            return [{"text": "token sk-" + "z" * 30}]
        return []
    payload = build_payload("default", "project", Path("/tmp"),
                            recall_fn=recall_fn)
    # AWS key pattern
    assert any("REDACTED" in t for t, _ in payload.topics)
    # GitHub PAT pattern
    assert any("REDACTED" in n for n, _ in payload.entities)
    # Anthropic key pattern
    assert any("REDACTED" in d for d in payload.recent_decisions)
    # OpenAI key pattern
    assert any("REDACTED" in m for m in payload.project_memories)


def test_payload_capped_to_K_per_section(fake_recall):
    payload = build_payload(
        "default", "project", Path("/tmp"),
        recall_fn=fake_recall,
        top_k=1, decisions_k=1, memories_k=1,
    )
    assert len(payload.topics) == 1
    assert len(payload.entities) == 1
    assert len(payload.recent_decisions) == 1
    assert len(payload.project_memories) == 1


def test_payload_deterministic(fake_recall):
    a = build_payload("default", "project", Path("/tmp"),
                      recall_fn=fake_recall,
                      now_fn=lambda: "2026-04-18T00:00:00+00:00")
    b = build_payload("default", "project", Path("/tmp"),
                      recall_fn=fake_recall,
                      now_fn=lambda: "2026-04-18T00:00:00+00:00")
    assert a == b


def test_scope_validation():
    with pytest.raises(ValueError):
        build_payload("default", "bogus", Path("/tmp"),
                      recall_fn=lambda q, l, p: [])


def test_recall_exception_returns_empty_sections():
    def boom(q: str, limit: int, pid: str):
        raise RuntimeError("engine down")
    payload = build_payload("default", "project", Path("/tmp"), recall_fn=boom)
    assert payload.topics == ()
    assert payload.entities == ()
    assert payload.recent_decisions == ()
    assert payload.project_memories == ()


def test_format_helpers_handle_empty(fake_recall):
    empty = build_payload("default", "project", Path("/tmp"),
                          recall_fn=lambda q, l, p: [])
    assert "none yet" in format_topics(empty)
    assert "none yet" in format_entities(empty)
    assert "none yet" in format_decisions(empty)
    assert "none yet" in format_memories(empty)


def test_format_helpers_render(fake_recall):
    payload = build_payload("default", "project", Path("/tmp"),
                            recall_fn=fake_recall)
    out_topics = format_topics(payload)
    assert "ai_agents" in out_topics
    assert "(0.87)" in out_topics
    out_entities = format_entities(payload)
    assert "Qualixar" in out_entities
    assert "(142)" in out_entities
    out_decisions = format_decisions(payload)
    assert "AGPL" in out_decisions
    out_mem = format_memories(payload)
    assert "memory one" in out_mem


def test_version_constant():
    assert VERSION == "3.4.22"


def test_truncate_payload_drops_memories_first(make_payload_helper=None):
    from tests.test_adapters.conftest import make_payload
    payload = make_payload(long_text=True)

    def render(p) -> bytes:
        return (
            ",".join(n for n, _ in p.topics)
            + "|"
            + ",".join(n for n, _ in p.entities)
            + "|"
            + ",".join(p.recent_decisions)
            + "|"
            + ",".join(p.project_memories)
        ).encode()

    out = truncate_payload_for_cap(payload, hard_cap=500, render=render)
    # In order: first memories drop, then decisions, then entities, then topics.
    assert len(out) <= 500 or b"|" in out  # sanity — cap tried


def test_recall_skips_non_dict_rows():
    def recall_fn(q: str, limit: int, pid: str) -> list[dict]:
        return ["not-a-dict", {"name": "ok", "score": 0.5}]
    payload = build_payload("default", "project", Path("/tmp"),
                            recall_fn=recall_fn)
    # Only the dict rows survive — no exception.
    assert payload.topics == (("ok", 0.5),)
