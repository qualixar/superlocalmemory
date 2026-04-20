"""Shared fixtures for cross-platform adapter tests (LLD-05)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from superlocalmemory.hooks.context_payload import ContextPayload


def make_payload(
    *, topics_n: int = 3, entities_n: int = 3,
    decisions_n: int = 3, memories_n: int = 3,
    long_text: bool = False,
) -> ContextPayload:
    topic_text = "t" * (800 if long_text else 6)
    entity_text = "e" * (800 if long_text else 6)
    decision_text = "d" * (800 if long_text else 20)
    memory_text = "m" * (800 if long_text else 20)
    return ContextPayload(
        profile_id="default",
        topics=tuple((f"{topic_text}_{i}", 0.9 - i * 0.05)
                     for i in range(topics_n)),
        entities=tuple((f"{entity_text}_{i}", 100 - i)
                       for i in range(entities_n)),
        recent_decisions=tuple(f"{decision_text}_{i}"
                               for i in range(decisions_n)),
        project_memories=tuple(f"{memory_text}_{i}"
                                for i in range(memories_n)),
        generated_at="2026-04-18T00:00:00+00:00",
        version="3.4.22",
    )


@pytest.fixture
def fresh_memory_db(tmp_path: Path) -> Path:
    """Empty sqlite DB. adapter_base auto-creates the sync-log table."""
    return tmp_path / "memory.db"


@pytest.fixture
def fake_recall():
    """A deterministic recall_fn that returns canned memory dicts."""
    def _fn(query: str, limit: int, profile_id: str) -> list[dict]:
        # Produce different shapes based on the query key
        base = {
            "topics": [
                {"name": "ai_agents", "score": 0.87},
                {"name": "llm_evaluation", "score": 0.72},
            ],
            "entities": [
                {"name": "Qualixar", "mentions": 142},
                {"name": "AgentAssert", "mentions": 58},
            ],
            "decisions": [
                {"text": "use AGPL for SLM"},
                {"text": "ship MCP hub v0.1.2"},
            ],
            "memories": [
                {"text": "memory one"},
                {"text": "memory two"},
            ],
        }
        if "topics" in query:
            return base["topics"][:limit]
        if "entities" in query:
            return base["entities"][:limit]
        if "decisions" in query:
            return base["decisions"][:limit]
        if "memories" in query:
            return base["memories"][:limit]
        return []
    return _fn


@pytest.fixture(autouse=True)
def _isolate_force_envs(monkeypatch):
    """Start every test with adapter-force env vars cleared so detection
    tests aren't contaminated by the developer's local environment."""
    for key in (
        "SLM_CURSOR_FORCE", "SLM_CURSOR_DISABLED",
        "SLM_ADAPTER_FORCE_CURSOR",
        "SLM_ANTIGRAVITY_FORCE", "SLM_ANTIGRAVITY_DISABLED",
        "SLM_ADAPTER_FORCE_ANTIGRAVITY",
        "SLM_COPILOT_FORCE", "SLM_COPILOT_DISABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
