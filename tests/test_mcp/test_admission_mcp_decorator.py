# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 — @admits decorator integration tests for MCP tools.

Tests that the @admits decorator:
  - Allows mutations in personal mode (frictionless OWNER)
  - Denies in enterprise mode with no principal (returns error dict)
  - Preserves the original tool result on allow
  - Works with async tool functions
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: mock server that captures registered tools
# ---------------------------------------------------------------------------

class _MockServer:
    def __init__(self) -> None:
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


# ---------------------------------------------------------------------------
# @admits decorator tests
# ---------------------------------------------------------------------------

def test_admits_decorator_personal_mode_allows_mutation(tmp_path, monkeypatch):
    """Personal mode → OWNER → @admits passes → tool executes and returns result."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    from superlocalmemory.core.admission import admits
    from superlocalmemory.core.operation_request import OperationKind

    @admits(OperationKind.REMEMBER)
    async def my_tool(content: str) -> dict:
        return {"success": True, "stored": content}

    result = asyncio.run(my_tool(content="test memory"))
    assert result["success"] is True
    assert result["stored"] == "test memory"


def test_admits_decorator_enterprise_anonymous_denies(tmp_path, monkeypatch):
    """Enterprise mode + no principal → ANONYMOUS → @admits denies → error dict."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    from superlocalmemory.core.admission import admits, _resolve_deployment
    from superlocalmemory.core.operation_request import OperationKind
    from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

    @admits(OperationKind.REMEMBER)
    async def my_tool(content: str) -> dict:
        return {"success": True}

    with patch(
        "superlocalmemory.core.admission._resolve_deployment",
        return_value=DEPLOYMENT_ENTERPRISE,
    ):
        result = asyncio.run(my_tool(content="test"))

    assert result["success"] is False
    assert result["error"] == "not_authorized"
    assert "authentication_required" in result["reason"]


def test_admits_decorator_preserves_tool_result(tmp_path, monkeypatch):
    """The decorated tool's full return value passes through on allow."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    from superlocalmemory.core.admission import admits
    from superlocalmemory.core.operation_request import OperationKind

    @admits(OperationKind.RECALL)
    async def my_read_tool(query: str) -> dict:
        return {"success": True, "results": ["fact1", "fact2"], "count": 2}

    result = asyncio.run(my_read_tool(query="python"))
    assert result["success"] is True
    assert result["count"] == 2


def test_admits_decorator_is_async_compatible(tmp_path, monkeypatch):
    """@admits wrapper is itself async and preserves coroutine semantics."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    from superlocalmemory.core.admission import admits
    from superlocalmemory.core.operation_request import OperationKind
    import inspect

    @admits(OperationKind.REMEMBER)
    async def my_tool() -> dict:
        return {"success": True}

    assert inspect.iscoroutinefunction(my_tool)


def test_admits_decorator_with_mock_server_integration(tmp_path, monkeypatch):
    """@admits integrates correctly with the mock-server capture pattern used in tests."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    from superlocalmemory.core.admission import admits
    from superlocalmemory.core.operation_request import OperationKind

    server = _MockServer()

    @server.tool()
    @admits(OperationKind.REMEMBER)
    async def test_remember(content: str) -> dict:
        return {"success": True, "content": content}

    captured = server._tools["test_remember"]
    result = asyncio.run(captured(content="hello"))
    assert result["success"] is True
    assert result["content"] == "hello"
