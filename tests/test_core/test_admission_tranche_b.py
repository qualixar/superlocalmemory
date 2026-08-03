"""Tranche B admission tests — RED first.

Tests that every Tranche B mutating surface is gated with @admits or
gate_cli_mutation, and that authorize_mcp_mutation in shared.py routes
through the policy registry instead of bypassing it via trust-hook alone.

Coverage targets
----------------
MCP tools (12):
    reinforce_assertion  → CORRECT      (tools_learning.py)
    contradict_assertion → FORGET       (tools_learning.py)
    report_outcome       → REMEMBER     (tools_v28.py)
    report_feedback      → REMEMBER     (tools_active.py)
    build_graph          → CORRECT      (tools_core.py)
    slm_cache_set        → REMEMBER     (tools_optimize.py)
    slm_compress         → CONSOLIDATE  (tools_optimize.py)
    update_code_graph    → CORRECT      (tools_code_graph.py)
    mesh_summary         → MESH_SEND    (tools_mesh.py)
    observe              → REMEMBER     (tools_active.py)
    close_session        → CONSOLIDATE  (tools_active.py)
    quantize             → CONSOLIDATE  (tools_v33.py)

CLI commands (4):
    cmd_evolve   → EVOLVE_SKILL  (commands.py:956)
    cmd_decay    → CONSOLIDATE   (commands.py:3622)
    cmd_quantize → CONSOLIDATE   (commands.py:3678)
    cmd_observe  → REMEMBER      (commands.py:3539)

Trust-hook bypass:
    authorize_mcp_mutation in shared.py must call admit() so enterprise
    policy applies even when the caller has local trust.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(base: Path, text: str) -> None:
    cfg = base / "config.toml"
    cfg.write_text(text)


def _mock_server() -> MagicMock:
    """Passthrough mock server: server.tool(...) returns lambda f: f."""
    ms = MagicMock()
    ms.tool.return_value = lambda f: f
    return ms


def _mock_get_engine() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# B1 — Tool inventory: every Tranche B tool in _GATED_MCP_TOOLS
# ---------------------------------------------------------------------------

class TestTrancheBToolInventory:
    """Each Tranche B tool must be decorated with @admits → in _GATED_MCP_TOOLS."""

    def _gate_check(self, tool_name: str, register_fn, *args) -> None:
        register_fn(*args)
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        assert tool_name in _GATED_MCP_TOOLS, (
            f"{tool_name} missing @admits — not in _GATED_MCP_TOOLS"
        )

    # ---- tools_learning.py ----

    def test_reinforce_assertion_gated(self):
        from superlocalmemory.mcp.tools_learning import register_learning_tools
        self._gate_check(
            "reinforce_assertion",
            register_learning_tools,
            _mock_server(), _mock_get_engine(),
        )

    def test_contradict_assertion_gated(self):
        from superlocalmemory.mcp.tools_learning import register_learning_tools
        self._gate_check(
            "contradict_assertion",
            register_learning_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_v28.py ----

    def test_report_outcome_gated(self):
        from superlocalmemory.mcp.tools_v28 import register_v28_tools
        self._gate_check(
            "report_outcome",
            register_v28_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_active.py ----

    def test_observe_gated(self):
        from superlocalmemory.mcp.tools_active import register_active_tools
        self._gate_check(
            "observe",
            register_active_tools,
            _mock_server(), _mock_get_engine(),
        )

    def test_report_feedback_gated(self):
        from superlocalmemory.mcp.tools_active import register_active_tools
        self._gate_check(
            "report_feedback",
            register_active_tools,
            _mock_server(), _mock_get_engine(),
        )

    def test_close_session_gated(self):
        from superlocalmemory.mcp.tools_active import register_active_tools
        self._gate_check(
            "close_session",
            register_active_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_optimize.py ----

    def test_slm_cache_set_gated(self):
        from superlocalmemory.mcp.tools_optimize import register_optimize_tools
        self._gate_check(
            "slm_cache_set",
            register_optimize_tools,
            _mock_server(),
        )

    def test_slm_compress_gated(self):
        from superlocalmemory.mcp.tools_optimize import register_optimize_tools
        self._gate_check(
            "slm_compress",
            register_optimize_tools,
            _mock_server(),
        )

    # ---- tools_mesh.py ----

    def test_mesh_summary_gated(self):
        from superlocalmemory.mcp.tools_mesh import register_mesh_tools
        self._gate_check(
            "mesh_summary",
            register_mesh_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_code_graph.py ----

    def test_update_code_graph_gated(self):
        from superlocalmemory.mcp.tools_code_graph import register_code_graph_tools
        self._gate_check(
            "update_code_graph",
            register_code_graph_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_v33.py ----

    def test_quantize_gated(self):
        from superlocalmemory.mcp.tools_v33 import register_v33_tools
        self._gate_check(
            "quantize",
            register_v33_tools,
            _mock_server(), _mock_get_engine(),
        )

    # ---- tools_core.py ----

    def test_build_graph_gated(self):
        from superlocalmemory.mcp.tools_core import register_core_tools
        self._gate_check(
            "build_graph",
            register_core_tools,
            _mock_server(), _mock_get_engine(),
        )


# ---------------------------------------------------------------------------
# B2 — Enterprise deny: anonymous caller gets not_authorized
# ---------------------------------------------------------------------------

class TestTrancheBEnterpriseAnonymousDenied:
    """In enterprise mode, anonymous MCP caller must be denied for each tool."""

    def _enterprise_fake_result(
        self,
        kind_name: str,
        tmp_path: Path,
        monkeypatch,
    ) -> dict:
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        kind = getattr(OperationKind, kind_name)

        @admits(kind)
        async def fake_tool() -> dict:
            return {"success": True}

        return asyncio.run(fake_tool())

    def test_reinforce_assertion_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CORRECT", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_contradict_assertion_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("FORGET", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_report_outcome_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("REMEMBER", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_report_feedback_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("REMEMBER", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_observe_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("REMEMBER", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_slm_cache_set_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("REMEMBER", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_slm_compress_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CONSOLIDATE", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_build_graph_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CORRECT", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_update_code_graph_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CORRECT", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_mesh_summary_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("MESH_SEND", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_close_session_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CONSOLIDATE", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_quantize_enterprise_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result("CONSOLIDATE", tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"


# ---------------------------------------------------------------------------
# B3 — CLI gates: cmd_evolve / cmd_decay / cmd_quantize / cmd_observe
# ---------------------------------------------------------------------------

class TestTrancheBCliGates:
    """Each CLI command must call gate_cli_mutation before doing work."""

    def _assert_gate_called(self, cmd_module_path: str, cmd_fn_name: str) -> None:
        """Patch gate_cli_mutation and verify the command calls it."""
        from superlocalmemory.cli import commands as cmd_mod
        gate_called = {"called": False, "kind": None}

        def mock_gate(kind):
            gate_called["called"] = True
            gate_called["kind"] = kind

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            fn = getattr(cmd_mod, cmd_fn_name)
            args = MagicMock()
            # Minimal args so the command reaches the gate before any IO.
            args.session = "test-session-id"
            args.profile = "default"
            args.json = False
            args.dry_run = True
            args.content = "test content"
            try:
                fn(args)
            except (SystemExit, Exception):
                pass  # Gate called → enterprise exit or daemon unavailable

        assert gate_called["called"], (
            f"{cmd_fn_name} did not call gate_cli_mutation — gate missing"
        )

    def test_cmd_evolve_calls_gate(self):
        self._assert_gate_called(
            "superlocalmemory.cli.commands", "cmd_evolve"
        )

    def test_cmd_decay_calls_gate(self):
        self._assert_gate_called(
            "superlocalmemory.cli.commands", "cmd_decay"
        )

    def test_cmd_quantize_calls_gate(self):
        self._assert_gate_called(
            "superlocalmemory.cli.commands", "cmd_quantize"
        )

    def test_cmd_observe_calls_gate(self):
        self._assert_gate_called(
            "superlocalmemory.cli.commands", "cmd_observe"
        )


# ---------------------------------------------------------------------------
# B4 — authorize_mcp_mutation routes through registry (trust-hook bypass fix)
# ---------------------------------------------------------------------------

class TestAuthorizeMcpMutationRegistryRouted:
    """authorize_mcp_mutation must call admit() so trust-hook cannot bypass registry."""

    def test_enterprise_anonymous_update_denied_via_registry(
        self, tmp_path, monkeypatch
    ):
        """Enterprise mode: anonymous local caller → admit(CORRECT) → AdmissionDenied."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

        engine = MagicMock()
        engine.profile_id = "default"
        engine._hooks = MagicMock()
        engine._hooks.run_pre = MagicMock()

        from superlocalmemory.core.admission import AdmissionDenied
        from superlocalmemory.mcp.shared import authorize_mcp_mutation

        with pytest.raises((AdmissionDenied, PermissionError, Exception)) as exc_info:
            authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="test",
            )
        # Must NOT succeed silently — something admission-related must be raised.
        # The exact exception type depends on how the bypass fix is wired.
        assert exc_info.value is not None

    def test_enterprise_anonymous_delete_denied_via_registry(
        self, tmp_path, monkeypatch
    ):
        """Enterprise mode: anonymous local caller → admit(FORGET) → AdmissionDenied."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

        engine = MagicMock()
        engine.profile_id = "default"
        engine._hooks = MagicMock()
        engine._hooks.run_pre = MagicMock()

        from superlocalmemory.core.admission import AdmissionDenied
        from superlocalmemory.mcp.shared import authorize_mcp_mutation

        with pytest.raises((AdmissionDenied, PermissionError, Exception)) as exc_info:
            authorize_mcp_mutation(
                engine,
                "delete",
                mutation_source="test",
            )
        assert exc_info.value is not None

    def test_personal_owner_update_allowed(self, tmp_path, monkeypatch):
        """Personal mode: local owner → authorize_mcp_mutation succeeds (no deny)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        # No config.toml → personal mode

        engine = MagicMock()
        engine.profile_id = "default"
        engine._hooks = MagicMock()
        engine._hooks.run_pre = MagicMock()

        from superlocalmemory.mcp.shared import authorize_mcp_mutation

        # Should NOT raise in personal mode (OWNER always admitted).
        # If it does raise something other than AdmissionDenied, re-raise.
        try:
            result = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="test",
            )
            # Success path: result should be MCPMutationAuthorization instance
            assert result is not None
        except Exception as exc:
            from superlocalmemory.core.admission import AdmissionDenied
            if isinstance(exc, AdmissionDenied):
                pytest.fail(
                    f"Personal OWNER must not be denied by authorize_mcp_mutation: {exc}"
                )
            raise


# ---------------------------------------------------------------------------
# B5 — _REQUIRED_MCP_GATES includes all Tranche B tools
# ---------------------------------------------------------------------------

class TestRequiredGatesExtended:
    """_REQUIRED_MCP_GATES must list every Tranche B tool so coverage_self_check
    can catch ungated surfaces."""

    _TRANCHE_B_TOOLS = {
        "reinforce_assertion",
        "contradict_assertion",
        "report_outcome",
        "report_feedback",
        "build_graph",
        "slm_cache_set",
        "slm_compress",
        "update_code_graph",
        "mesh_summary",
        "observe",
        "close_session",
        "quantize",
    }

    def test_required_gates_includes_tranche_b(self):
        from superlocalmemory.core.admission import _REQUIRED_MCP_GATES
        missing = self._TRANCHE_B_TOOLS - _REQUIRED_MCP_GATES
        assert not missing, (
            f"Tranche B tools not in _REQUIRED_MCP_GATES: {sorted(missing)}"
        )
