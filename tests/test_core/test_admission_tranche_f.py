"""Tranche F admission tests — RED first.

F1. DYNAMIC MUTATOR COVERAGE SELF-CHECK
    coverage_self_check must accept optional ``server`` parameter.
    When provided, it enumerates server._tool_manager._tools and flags any
    tool NOT in _GATED_MCP_TOOLS where readOnlyHint != True as an ungated
    mutator. Enterprise → raise RuntimeError. Personal → warning only.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers: build a minimal FastMCP-compatible tool registry
# ---------------------------------------------------------------------------

def _make_tool(name: str, read_only: bool | None) -> SimpleNamespace:
    """Create a minimal duck-typed Tool object matching FastMCP's structure."""
    annotations = SimpleNamespace(readOnlyHint=read_only)
    return SimpleNamespace(name=name, annotations=annotations)


def _make_server(tools: dict[str, SimpleNamespace]) -> SimpleNamespace:
    """Create a minimal server with _tool_manager._tools dict."""
    tool_manager = SimpleNamespace(_tools=tools)
    return SimpleNamespace(_tool_manager=tool_manager)


def _ensure_required_gated(monkeypatch) -> set:
    """Pre-populate _GATED_MCP_TOOLS with all required gates so Check 3 passes.

    This isolates the new Check 4 (dynamic server enumeration) from Check 3
    (static inventory), which cannot pass in unit tests without triggering all
    registration functions.
    """
    import superlocalmemory.core.admission as adm
    patched = set(adm._REQUIRED_MCP_GATES)
    monkeypatch.setattr(adm, "_GATED_MCP_TOOLS", patched)
    return patched


# ---------------------------------------------------------------------------
# F1 — coverage_self_check accepts server parameter
# ---------------------------------------------------------------------------

class TestF1CoverageSelfCheckSignature:
    """coverage_self_check must accept the server keyword argument."""

    def test_accepts_server_none(self, tmp_path, monkeypatch):
        """Passing server=None is equivalent to omitting it (current behaviour)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_PERSONAL
        coverage_self_check(DEPLOYMENT_PERSONAL, server=None)

    def test_accepts_server_object(self, tmp_path, monkeypatch):
        """Passing a server with only read-only tools must not raise."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_PERSONAL

        tools = {
            "recall": _make_tool("recall", read_only=True),
            "search": _make_tool("search", read_only=True),
        }
        server = _make_server(tools)
        coverage_self_check(DEPLOYMENT_PERSONAL, server=server)


# ---------------------------------------------------------------------------
# F1 — enterprise raises on ungated mutating tool
# ---------------------------------------------------------------------------

class TestF1EnterpriseRaisesOnUngatedMutator:
    """In enterprise mode, any mutating tool not in _GATED_MCP_TOOLS → RuntimeError."""

    def test_ungated_mutator_enterprise_raises(self, tmp_path, monkeypatch):
        """Mutating tool absent from _GATED_MCP_TOOLS → RuntimeError in enterprise."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

        tools = {
            "sneaky_mutator": _make_tool("sneaky_mutator", read_only=False),
        }
        server = _make_server(tools)

        with pytest.raises(RuntimeError, match="sneaky_mutator"):
            coverage_self_check(DEPLOYMENT_ENTERPRISE, server=server)

    def test_ungated_mutator_no_annotation_enterprise_raises(self, tmp_path, monkeypatch):
        """Tool with annotations=None (unknown) → treated as mutator → RuntimeError."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

        # annotations=None → getattr(..., "readOnlyHint", None) is not True → mutator
        tool = SimpleNamespace(name="bare_tool", annotations=None)
        tools = {"bare_tool": tool}
        server = _make_server(tools)

        with pytest.raises(RuntimeError, match="bare_tool"):
            coverage_self_check(DEPLOYMENT_ENTERPRISE, server=server)

    def test_all_mutators_gated_enterprise_passes(self, tmp_path, monkeypatch):
        """All mutating server tools in _GATED_MCP_TOOLS → no RuntimeError."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gated = _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

        # "remember" is in _REQUIRED_MCP_GATES, so it IS in the patched _GATED_MCP_TOOLS.
        tools = {
            "remember": _make_tool("remember", read_only=False),
            "recall": _make_tool("recall", read_only=True),
        }
        server = _make_server(tools)
        assert "remember" in gated
        coverage_self_check(DEPLOYMENT_ENTERPRISE, server=server)  # must not raise


# ---------------------------------------------------------------------------
# F1 — personal mode warns but does not raise
# ---------------------------------------------------------------------------

class TestF1PersonalWarnsOnUngatedMutator:
    """In personal mode, ungated mutating tool → warning only, no RuntimeError."""

    def test_ungated_mutator_personal_warns_not_raises(self, tmp_path, monkeypatch, caplog):
        """Mutating tool absent from _GATED_MCP_TOOLS → warning, no RuntimeError."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_PERSONAL

        tools = {
            "another_sneaky": _make_tool("another_sneaky", read_only=False),
        }
        server = _make_server(tools)

        with caplog.at_level(logging.WARNING, logger="superlocalmemory.core.admission"):
            coverage_self_check(DEPLOYMENT_PERSONAL, server=server)

        assert any("another_sneaky" in msg for msg in caplog.messages), (
            "Expected warning mentioning 'another_sneaky' in personal mode"
        )

    def test_read_only_tool_not_flagged(self, tmp_path, monkeypatch):
        """Read-only tool (readOnlyHint=True) must NOT be flagged even if ungated."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _ensure_required_gated(monkeypatch)
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

        tools = {
            "pure_query": _make_tool("pure_query", read_only=True),
        }
        server = _make_server(tools)
        # Must not raise — read-only tools don't need @admits
        coverage_self_check(DEPLOYMENT_ENTERPRISE, server=server)
