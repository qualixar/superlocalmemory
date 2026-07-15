# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Policy-boundary regressions for legacy MCP mutation tools."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


class _Server:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            return fn

        return register


def _tool(register, name: str, engine: MagicMock):
    server = _Server()
    register(server, MagicMock(return_value=engine))
    return server.tools[name]


def _engine() -> MagicMock:
    engine = MagicMock()
    engine.profile_id = "owner-profile"
    engine.db = engine._db
    engine._db.execute.return_value = []
    engine._db.get_all_facts.return_value = []
    return engine


def test_v28_retention_write_is_rejected_before_database_mutation() -> None:
    from superlocalmemory.mcp.tools_v28 import register_v28_tools

    engine = _engine()
    engine._hooks.run_pre.side_effect = PermissionError("policy denied")
    tool = _tool(register_v28_tools, "set_retention_policy", engine)

    result = asyncio.run(tool(cold_after_days=7, archive_after_days=21))

    assert result["success"] is False
    engine._db.set_config.assert_not_called()


def test_v33_forgetting_is_rejected_before_decay_mutation() -> None:
    from superlocalmemory.mcp.tools_v33 import register_v33_tools

    engine = _engine()
    engine._hooks.run_pre.side_effect = PermissionError("policy denied")
    tool = _tool(register_v33_tools, "forget", engine)

    with (
        patch("superlocalmemory.math.ebbinghaus.EbbinghausCurve"),
        patch(
            "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
        ) as scheduler,
    ):
        result = asyncio.run(tool(dry_run=False))

    assert result["success"] is False
    scheduler.return_value.run_decay_cycle.assert_not_called()


def test_active_core_pin_is_rejected_before_database_mutation() -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine = _engine()
    engine._hooks.run_pre.side_effect = PermissionError("policy denied")
    tool = _tool(register_active_tools, "core_memory", engine)

    result = asyncio.run(tool(action="pin", fact_id="fact-1"))

    assert result["success"] is False
    engine._db.set_pinned.assert_not_called()


def test_session_agent_registration_is_rejected_before_registry_write() -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools
    from superlocalmemory.mcp._pool_adapter import PoolRecallResponse

    engine = _engine()
    engine._adaptive_learner.get_feedback_count.return_value = 0
    engine._hooks.run_pre.side_effect = PermissionError("policy denied")
    tool = _tool(register_active_tools, "session_init", engine)
    rules = MagicMock()
    rules.should_recall.return_value = True
    rules.get_recall_config.return_value = {"relevance_threshold": 0.3}

    with (
        patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=rules),
        patch(
            "superlocalmemory.mcp._pool_adapter.pool_recall",
            return_value=PoolRecallResponse(),
        ),
        patch("superlocalmemory.core.registry.AgentRegistry") as registry,
        patch("superlocalmemory.mcp.tools_active._emit_event"),
    ):
        result = asyncio.run(tool(project_path="/project"))

    assert result["success"] is False
    registry.return_value.register_agent.assert_not_called()

