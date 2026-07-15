# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Policy regressions for MCP memory-control mutations."""

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
    engine._hooks.run_pre.side_effect = PermissionError("policy denied")
    return engine


def test_core_pattern_correction_is_rejected_before_store_write() -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine = _engine()
    tool = _tool(register_core_tools, "correct_pattern", engine)

    with patch(
        "superlocalmemory.learning.behavioral.BehavioralPatternStore"
    ) as store:
        result = asyncio.run(tool("pattern-1", "prefer explicit errors"))

    assert result["success"] is False
    store.return_value.record.assert_not_called()


def test_core_profile_switch_is_rejected_before_global_state_change() -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine = _engine()
    tool = _tool(register_core_tools, "switch_profile", engine)

    result = asyncio.run(tool("other-profile"))

    assert result["success"] is False
    assert engine.profile_id == "owner-profile"


def test_learning_assertion_update_is_rejected_before_sql_write() -> None:
    from superlocalmemory.mcp.tools_learning import register_learning_tools

    engine = _engine()
    engine._db.execute.return_value = [{
        "confidence": 0.8,
        "reinforcement_count": 1,
        "contradiction_count": 0,
    }]
    tool = _tool(register_learning_tools, "reinforce_assertion", engine)

    result = asyncio.run(tool("assertion-1"))

    assert result["success"] is False
    engine._db.execute.assert_not_called()


def test_mode_switch_is_rejected_before_configuration_write() -> None:
    from superlocalmemory.mcp.tools_v3 import register_v3_tools

    engine = _engine()
    tool = _tool(register_v3_tools, "set_mode", engine)

    with patch("superlocalmemory.core.config.SLMConfig.switch_mode") as switch:
        result = asyncio.run(tool("b"))

    assert result["success"] is False
    switch.assert_not_called()

