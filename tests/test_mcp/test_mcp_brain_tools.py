"""Portable MCP boundary tests for v4.0.2 Brain receipts."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from superlocalmemory.mcp.tools_brain import register_brain_tools
from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040


class _Server:
    def __init__(self) -> None:
        self.tools = {}

    def tool(self, *args, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn
        return register


class _Engine:
    profile_id = "alpha"


def _experience(profile_id: str = "alpha") -> dict:
    return {
        "experience_id": "experience-1",
        "profile_id": profile_id,
        "occurred_at": "2026-08-15T00:00:00+00:00",
        "task_class": "code",
        "project_scope": "project-digest",
        "route": {
            "harness": "codex", "provider": "openai", "model": "gpt",
            "effort": "high", "machine": "m",
        },
        "verification": {"authority": "deterministic_gate", "evidence_digest": "a" * 64},
        "producer_claim": "success",
        "terminal_status": "succeeded",
    }


def _turn(profile_id: str = "alpha") -> dict:
    return {
        "receipt_id": "turn-1", "task_id": "task-1", "profile_id": profile_id,
        "project_scope": "project-digest", "query_digest": "b" * 64,
        "fact_decisions": {"fact-1": "used"}, "state": "open",
    }


def test_brain_tools_are_profile_scoped_and_have_honest_busy_contract(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    with sqlite3.connect(tmp_path / "learning.db") as conn:
        m040.apply(conn)
    server = _Server()
    register_brain_tools(server, lambda: _Engine())

    assert {
        "get_brain_evidence_status", "record_agent_experience",
        "record_cognitive_turn", "finalize_cognitive_turn",
    } <= set(server.tools)
    assert asyncio.run(server.tools["record_agent_experience"](_experience())) == {
        "success": True, "durable": True, "created": True
    }
    denied = asyncio.run(server.tools["record_agent_experience"](_experience("beta")))
    assert denied == {
        "success": False, "durable": False,
        "error": "profile_id must equal the active MCP profile",
    }
    assert asyncio.run(server.tools["record_cognitive_turn"](_turn())) == {
        "success": True, "durable": True, "created": True
    }
    status = asyncio.run(server.tools["get_brain_evidence_status"]())
    assert status["profile_id"] == "alpha"
    assert status["agent_experience"]["experiences_total"] == 1
    assert status["agent_experience"]["turns_by_state"] == {"open": 1}
    assert status["agent_experience"]["claimed_evidence_experiences"] == 1
