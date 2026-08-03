"""Tranche E admission tests — RED first.

E1. REMAINING MCP MUTATORS
    10 MCP tools still ungated (no @admits decorator):
    slm_loop_run, set_retention_policy, compact_memories, log_tool_event,
    core_memory, run_maintenance, reap_processes, build_code_graph,
    apply_refactor, link_memory_to_code.
    All must be in _REQUIRED_MCP_GATES AND in _GATED_MCP_TOOLS after registration.

E2. HTTP ROUTE MUTATIONS
    authorize_route_mutation (route_mutations.py) must call admit() before
    engine._hooks.run_pre() — same pattern as shared.py Tranche B fix.
    Enterprise anonymous → HTTPException(403).
"""
from __future__ import annotations

import asyncio
import pathlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_server_passthrough():
    """Mock server where @server.tool() is a passthrough decorator."""
    mock = MagicMock()
    mock.tool.return_value = lambda fn: fn
    return mock


def _trigger_registration(register_fn, get_engine=None):
    """Call a register_* function to populate _GATED_MCP_TOOLS."""
    server = _mock_server_passthrough()
    engine_factory = get_engine or (lambda: None)
    register_fn(server, engine_factory)


# ---------------------------------------------------------------------------
# E1 — REQUIRED_MCP_GATES inventory (static check)
# ---------------------------------------------------------------------------

E1_REQUIRED_TOOLS = {
    "slm_loop_run",
    "set_retention_policy",
    "compact_memories",
    "log_tool_event",
    "core_memory",
    "run_maintenance",
    "reap_processes",
    "build_code_graph",
    "apply_refactor",
    "link_memory_to_code",
}


class TestE1RequiredGatesInventory:
    """All 10 E1 tools must be in _REQUIRED_MCP_GATES."""

    @pytest.mark.parametrize("tool_name", sorted(E1_REQUIRED_TOOLS))
    def test_tool_in_required_gates(self, tool_name):
        from superlocalmemory.core.admission import _REQUIRED_MCP_GATES
        assert tool_name in _REQUIRED_MCP_GATES, (
            f"'{tool_name}' not in _REQUIRED_MCP_GATES — "
            "coverage_self_check cannot catch it if @admits is removed"
        )


# ---------------------------------------------------------------------------
# E1 — _GATED_MCP_TOOLS dynamic check (populated via @admits at registration)
# ---------------------------------------------------------------------------

class TestE1GatedToolsAfterRegistration:
    """After calling register_*, all E1 tools must appear in _GATED_MCP_TOOLS."""

    def _gate_set_after_registration(self) -> set:
        """Trigger all registrations and return current _GATED_MCP_TOOLS."""
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS

        from superlocalmemory.mcp.tools_loops import register_loop_tools
        from superlocalmemory.mcp.tools_v28 import register_v28_tools
        from superlocalmemory.mcp.tools_v33 import register_v33_tools
        from superlocalmemory.mcp.tools_learning import register_learning_tools
        from superlocalmemory.mcp.tools_active import register_active_tools
        from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools

        _trigger_registration(register_loop_tools)
        _trigger_registration(register_v28_tools)
        _trigger_registration(register_v33_tools)
        _trigger_registration(register_learning_tools)
        _trigger_registration(register_active_tools)
        _trigger_registration(register_code_graph_tools)

        return set(_GATED_MCP_TOOLS)

    @pytest.mark.parametrize("tool_name", sorted(E1_REQUIRED_TOOLS))
    def test_tool_gated_after_registration(self, tool_name):
        gated = self._gate_set_after_registration()
        assert tool_name in gated, (
            f"'{tool_name}' not in _GATED_MCP_TOOLS after registration — "
            "@admits not applied"
        )


# ---------------------------------------------------------------------------
# E1 — Enterprise deny: slm_loop_run (representative E1 tool)
# ---------------------------------------------------------------------------

class TestE1EnterpriseAnonymousDenied:
    """After E1 fix, @admits blocks anonymous callers in enterprise mode."""

    def test_slm_loop_run_enterprise_anonymous_denied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        (tmp_path / "config.toml").write_text('[deployment]\nmode = "enterprise"\n')

        from superlocalmemory.mcp.tools_loops import register_loop_tools

        registered: dict = {}

        def capturing_tool(**kw):
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator

        mock_server = MagicMock()
        mock_server.tool.side_effect = capturing_tool

        register_loop_tools(mock_server, lambda: None)
        slm_loop_run = registered.get("slm_loop_run")
        assert slm_loop_run is not None, "slm_loop_run not registered"

        result = asyncio.run(slm_loop_run(name="test", gate_query="done?"))
        assert result.get("success") is False
        assert result.get("error") == "not_authorized", (
            f"expected not_authorized, got: {result}"
        )

    def test_slm_loop_run_personal_owner_allowed(self, tmp_path, monkeypatch):
        """Personal owner must remain frictionless after E1 fix."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        # No config.toml → personal default

        from superlocalmemory.mcp.tools_loops import register_loop_tools

        registered: dict = {}

        def capturing_tool(**kw):
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn
            return decorator

        mock_server = MagicMock()
        mock_server.tool.side_effect = capturing_tool

        register_loop_tools(mock_server, lambda: None)
        slm_loop_run = registered.get("slm_loop_run")

        # Personal OWNER passes admission and hits engine, which will fail
        # (no real engine). We only care that error is NOT not_authorized.
        result = asyncio.run(slm_loop_run(name="test", gate_query="done?"))
        assert result.get("error") != "not_authorized", (
            "personal OWNER should not be blocked by admission gate"
        )


# ---------------------------------------------------------------------------
# E2 — authorize_route_mutation must call admit()
# ---------------------------------------------------------------------------

class TestE2AuthorizeRouteMutationCallsAdmit:
    """route_mutations.py::authorize_route_mutation must call admit() before pre-hook."""

    def test_admit_referenced_in_route_mutations(self):
        """Static check: 'admit' must appear in route_mutations.py source."""
        src = pathlib.Path(
            "/Users/v.pratap.bhardwaj/Documents/varun-world/Agentic_official/"
            "slm-wt-p1/src/superlocalmemory/server/route_mutations.py"
        ).read_text()
        assert "admit(" in src or "admit\n" in src or "from superlocalmemory.core.admission import" in src, (
            "admit() not referenced in route_mutations.py — HTTP routes bypass registry"
        )

    def test_enterprise_anonymous_route_mutation_denied(self, tmp_path, monkeypatch):
        """Enterprise anonymous caller → HTTPException(403) from authorize_route_mutation."""
        from fastapi import HTTPException
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        (tmp_path / "config.toml").write_text('[deployment]\nmode = "enterprise"\n')

        from superlocalmemory.server.route_mutations import authorize_route_mutation

        mock_request = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.daemon_descriptor = None

        mock_engine = MagicMock()
        mock_engine.profile_id = "default"
        mock_engine._hooks = MagicMock()

        # Patch at source module (lazy imports inside authorize_route_mutation body)
        with (
            patch(
                "superlocalmemory.server.write_identity.authenticated_request_actor",
                return_value="",  # anonymous
            ),
            patch(
                "superlocalmemory.server.routes.helpers.get_engine_lazy",
                return_value=mock_engine,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                authorize_route_mutation(
                    mock_request,
                    operation="update",
                    source_agent_id="test-route",
                )
        assert exc_info.value.status_code == 403

    def test_personal_owner_route_mutation_allowed(self, tmp_path, monkeypatch):
        """Personal OWNER → authorize_route_mutation proceeds (no 403)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        # No config.toml → personal mode

        from superlocalmemory.server.route_mutations import authorize_route_mutation

        mock_request = MagicMock()
        mock_request.app.state = MagicMock()
        mock_request.app.state.daemon_descriptor = None

        mock_engine = MagicMock()
        mock_engine.profile_id = "default"
        mock_engine._hooks = MagicMock()
        mock_engine._hooks.run_pre.return_value = None

        # Patch at source module (lazy imports inside authorize_route_mutation body)
        with (
            patch(
                "superlocalmemory.server.write_identity.authenticated_request_actor",
                return_value="local-operator",
            ),
            patch(
                "superlocalmemory.server.routes.helpers.get_engine_lazy",
                return_value=mock_engine,
            ),
        ):
            result = authorize_route_mutation(
                mock_request,
                operation="update",
                source_agent_id="test-route",
            )
        assert result is not None
        assert result.operation == "update"
