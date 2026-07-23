"""Idle stdio MCP sessions must not duplicate engines or mesh heartbeats."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _function_calls(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def test_idle_stdio_warmup_uses_shared_daemon_only() -> None:
    calls = _function_calls(
        ROOT / "src/superlocalmemory/mcp/server.py",
        "_eager_warmup",
    )

    assert "ensure_daemon" in calls
    assert "get_engine" not in calls
    assert "auto_register_mesh" not in calls


def test_recall_signal_recording_does_not_open_engine_when_profile_known() -> None:
    source = (
        ROOT / "src/superlocalmemory/mcp/tools_core.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_record_recall_hits"
    )
    profile_guard = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "pid"
    )

    guarded_calls = {
        node.func.id
        for node in ast.walk(profile_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "get_engine" in guarded_calls
