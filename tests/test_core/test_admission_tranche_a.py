# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tranche A — P0 admission gateway remediation tests.

Covers:
  - Fail-closed deployment resolution (config present-but-unreadable → enterprise)
  - Config absent → personal OWNER (frictionless)
  - @admits on enterprise corrupt config → denied
  - gate_cli_mutation on enterprise corrupt config → sys.exit(1)
  - delete_memory / update_memory gated via @admits
  - CLI cmd_forget / cmd_delete / cmd_update / cmd_profile gate
  - Non-vacuous coverage_self_check (empty transports, tool inventory)
  - HTTP _admit_http_mutation helper
"""

from __future__ import annotations

import os
import sys
from typing import FrozenSet
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path, content: str) -> None:
    """Write config.toml in tmp_path so _resolve_deployment() finds it."""
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# A1 — Fail-closed deployment resolution
# ---------------------------------------------------------------------------

class TestResolveDeploymentFailClosed:
    def test_config_absent_returns_personal(self, tmp_path, monkeypatch):
        """No config.toml → legitimate personal install → DEPLOYMENT_PERSONAL."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert not deployment.is_enterprise

    def test_config_present_corrupt_returns_enterprise(self, tmp_path, monkeypatch):
        """config.toml present but corrupt TOML → fail-closed → DEPLOYMENT_ENTERPRISE."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, "[[[INVALID TOML SYNTAX{{{}}}]]]")
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert deployment.is_enterprise, "Corrupt config must be fail-closed (enterprise)"

    def test_config_present_unreadable_returns_enterprise(self, tmp_path, monkeypatch):
        """config.toml present but unreadable (permissions) → fail-closed."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        cfg = tmp_path / "config.toml"
        cfg.write_text("[deployment]\nmode = \"personal\"")
        cfg.chmod(0o000)  # remove all permissions
        try:
            from superlocalmemory.core.admission import _resolve_deployment
            deployment = _resolve_deployment()
            assert deployment.is_enterprise, "Unreadable config must be fail-closed"
        finally:
            cfg.chmod(0o644)  # restore

    def test_config_present_valid_enterprise_returns_enterprise(self, tmp_path, monkeypatch):
        """Valid enterprise config.toml → DEPLOYMENT_ENTERPRISE."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert deployment.is_enterprise

    def test_config_present_valid_personal_returns_personal(self, tmp_path, monkeypatch):
        """Valid personal config.toml → DEPLOYMENT_PERSONAL."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "personal"\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert not deployment.is_enterprise


class TestAdmitsDecoratorCorruptConfig:
    """@admits with corrupt config → enterprise → ANONYMOUS → denied."""

    def test_admits_corrupt_config_denies_mutation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, "[[[BAD")

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        @admits(OperationKind.REMEMBER)
        async def fake_tool():
            return {"success": True}

        import asyncio
        result = asyncio.run(fake_tool())
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_gate_cli_corrupt_config_exits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, "[[[BAD")

        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind

        with pytest.raises(SystemExit):
            gate_cli_mutation(OperationKind.REMEMBER)

    def test_config_absent_admits_decorator_allows_owner(self, tmp_path, monkeypatch):
        """Absent config → personal → OWNER → mutation allowed."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        sentinel = {"called": False}

        @admits(OperationKind.REMEMBER)
        async def fake_tool():
            sentinel["called"] = True
            return {"success": True}

        import asyncio
        result = asyncio.run(fake_tool())
        assert result.get("success") is True
        assert sentinel["called"]


# ---------------------------------------------------------------------------
# A2 — delete_memory / update_memory gated
# ---------------------------------------------------------------------------

class TestDestructiveMcpGated:
    """delete_memory and update_memory must be decorated with @admits.

    Tools are nested inside register_core_tools(), so @admits fires only
    when register_core_tools is actually called. We use a passthrough mock
    server to trigger decoration without starting a real server.
    """

    @staticmethod
    def _trigger_core_tools_registration():
        """Call register_core_tools with a passthrough mock server."""
        from unittest.mock import MagicMock
        from superlocalmemory.mcp.tools_core import register_core_tools

        mock_server = MagicMock()
        # Make server.tool(...) a passthrough so @admits decorator fires.
        mock_server.tool.return_value = lambda f: f
        mock_get_engine = MagicMock()
        register_core_tools(mock_server, mock_get_engine)

    def test_delete_memory_has_admits_decorator(self):
        """delete_memory must be decorated with @admits → in _GATED_MCP_TOOLS."""
        self._trigger_core_tools_registration()
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        assert "delete_memory" in _GATED_MCP_TOOLS, (
            "delete_memory must be decorated with @admits — not in _GATED_MCP_TOOLS"
        )

    def test_update_memory_has_admits_decorator(self):
        """update_memory must be decorated with @admits → in _GATED_MCP_TOOLS."""
        self._trigger_core_tools_registration()
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        assert "update_memory" in _GATED_MCP_TOOLS, (
            "update_memory must be decorated with @admits — not in _GATED_MCP_TOOLS"
        )

    def test_delete_memory_enterprise_anonymous_denied(self, tmp_path, monkeypatch):
        """In enterprise mode, anonymous caller → delete_memory returns not_authorized."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')
        # No importlib.reload needed: _resolve_deployment() reads SLM_DATA_DIR at call time.

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        @admits(OperationKind.FORGET)
        async def fake_delete_memory(fact_id: str) -> dict:
            return {"success": True, "deleted": fact_id}

        import asyncio
        result = asyncio.run(fake_delete_memory("fact-123"))
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"

    def test_update_memory_enterprise_anonymous_denied(self, tmp_path, monkeypatch):
        """In enterprise mode, anonymous caller → update_memory returns not_authorized."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        @admits(OperationKind.CORRECT)
        async def fake_update_memory(fact_id: str, content: str) -> dict:
            return {"success": True}

        import asyncio
        result = asyncio.run(fake_update_memory("fact-123", "new content"))
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"


# ---------------------------------------------------------------------------
# A3 — CLI gate: cmd_forget, cmd_delete, cmd_update, cmd_profile
# ---------------------------------------------------------------------------

class TestCliGates:
    """gate_cli_mutation wired at the top of destructive CLI commands.

    Strategy: test gate_cli_mutation directly with enterprise/personal config.
    Then test that cmd_* functions actually CALL gate_cli_mutation by patching
    at module level and verifying the call.
    """

    def _enterprise_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

    # --- Direct unit tests for gate_cli_mutation ---

    def test_gate_cli_forget_enterprise_exits(self, tmp_path, monkeypatch):
        """gate_cli_mutation(FORGET) on enterprise box → sys.exit(1)."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind
        with pytest.raises(SystemExit) as exc_info:
            gate_cli_mutation(OperationKind.FORGET)
        assert exc_info.value.code == 1

    def test_gate_cli_correct_enterprise_exits(self, tmp_path, monkeypatch):
        """gate_cli_mutation(CORRECT) on enterprise box → sys.exit(1)."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind
        with pytest.raises(SystemExit) as exc_info:
            gate_cli_mutation(OperationKind.CORRECT)
        assert exc_info.value.code == 1

    def test_gate_cli_profile_switch_enterprise_exits(self, tmp_path, monkeypatch):
        """gate_cli_mutation(PROFILE_SWITCH) on enterprise box → sys.exit(1)."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind
        with pytest.raises(SystemExit) as exc_info:
            gate_cli_mutation(OperationKind.PROFILE_SWITCH)
        assert exc_info.value.code == 1

    def test_gate_cli_personal_forget_no_exit(self, tmp_path, monkeypatch):
        """gate_cli_mutation(FORGET) on personal box → no exit (OWNER allowed)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        # No config.toml → personal
        from superlocalmemory.core.admission import gate_cli_mutation
        from superlocalmemory.core.operation_request import OperationKind
        gate_cli_mutation(OperationKind.FORGET)  # must not raise

    # --- Wire tests: verify cmd_* calls gate_cli_mutation ---

    def test_cmd_forget_calls_gate_before_daemon(self, tmp_path, monkeypatch):
        """cmd_forget calls gate_cli_mutation before any daemon interaction."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_calls: list = []

        def mock_gate(kind, **kwargs):
            gate_calls.append(kind)
            # Don't exit — let the rest of the command run

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            args = MagicMock()
            args.query = "test"
            args.dry_run = False
            args.json = False
            from superlocalmemory.cli.commands import cmd_forget
            try:
                cmd_forget(args)
            except BaseException:
                pass  # daemon not running — OK, we only care gate was called first

        assert gate_calls, "cmd_forget must call gate_cli_mutation"

    def test_cmd_delete_calls_gate_before_daemon(self, tmp_path, monkeypatch):
        """cmd_delete calls gate_cli_mutation before any daemon interaction."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_calls: list = []

        def mock_gate(kind, **kwargs):
            gate_calls.append(kind)

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            args = MagicMock()
            args.fact_id = "fact-123"
            args.yes = True
            args.json = False
            from superlocalmemory.cli.commands import cmd_delete
            try:
                cmd_delete(args)
            except BaseException:
                pass

        assert gate_calls, "cmd_delete must call gate_cli_mutation"

    def test_cmd_update_calls_gate_before_daemon(self, tmp_path, monkeypatch):
        """cmd_update calls gate_cli_mutation before any daemon interaction."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_calls: list = []

        def mock_gate(kind, **kwargs):
            gate_calls.append(kind)

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            args = MagicMock()
            args.fact_id = "fact-123"
            args.content = "new content"
            args.json = False
            from superlocalmemory.cli.commands import cmd_update
            try:
                cmd_update(args)
            except BaseException:
                pass

        assert gate_calls, "cmd_update must call gate_cli_mutation"

    def test_cmd_profile_switch_calls_gate(self, tmp_path, monkeypatch):
        """cmd_profile switch calls gate_cli_mutation."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_calls: list = []

        def mock_gate(kind, **kwargs):
            gate_calls.append(kind)

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            args = MagicMock()
            args.action = "switch"
            args.name = "other"
            args.json = False
            from superlocalmemory.cli.commands import cmd_profile
            try:
                cmd_profile(args)
            except BaseException:
                pass

        assert gate_calls, "cmd_profile switch must call gate_cli_mutation"

    def test_cmd_profile_create_calls_gate(self, tmp_path, monkeypatch):
        """cmd_profile create calls gate_cli_mutation."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_calls: list = []

        def mock_gate(kind, **kwargs):
            gate_calls.append(kind)

        with patch("superlocalmemory.core.admission.gate_cli_mutation", mock_gate):
            args = MagicMock()
            args.action = "create"
            args.name = "new-profile"
            args.json = False
            from superlocalmemory.cli.commands import cmd_profile
            try:
                cmd_profile(args)
            except BaseException:
                pass

        assert gate_calls, "cmd_profile create must call gate_cli_mutation"


# ---------------------------------------------------------------------------
# A4 — Non-vacuous coverage_self_check
# ---------------------------------------------------------------------------

class TestNonVacuousCoverageCheck:
    """coverage_self_check must catch non-vacuous policy gaps."""

    def test_empty_transports_flagged_personal_warns(self, caplog, tmp_path):
        """Policy with empty allowed_transports triggers a warning in personal mode."""
        import logging

        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_PERSONAL
        from superlocalmemory.core.operation_policy import OperationPolicy
        from superlocalmemory.core.operation_policy_registry import (
            OperationPolicyRegistry, _DEFAULT_REGISTRY,
        )
        from superlocalmemory.core.operation_request import OperationKind
        from superlocalmemory.core.actor_context import ActorRole

        # Override REMEMBER in the default registry to have empty transports.
        base_policies = dict(_DEFAULT_REGISTRY._policies)
        base_policies[OperationKind.REMEMBER] = OperationPolicy(
            kind=OperationKind.REMEMBER,
            required_roles=frozenset({ActorRole.OWNER}),
            allowed_transports=frozenset(),  # empty → bug
        )
        reg = OperationPolicyRegistry(base_policies)

        with caplog.at_level(logging.WARNING):
            coverage_self_check(DEPLOYMENT_PERSONAL, registry=reg)

        assert any(
            "empty_transports" in r.message or "no reachable transport" in r.message
            for r in caplog.records
        ), "coverage_self_check must warn about empty allowed_transports"

    def test_empty_transports_flagged_enterprise_raises(self, tmp_path):
        """Policy with empty allowed_transports raises RuntimeError in enterprise mode.

        Use _DEFAULT_REGISTRY as the base so all kinds are covered; then
        inject a bad policy for REMEMBER via a custom dict that overrides only that.
        """
        from superlocalmemory.core.admission import coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE
        from superlocalmemory.core.operation_policy import OperationPolicy
        from superlocalmemory.core.operation_policy_registry import (
            OperationPolicyRegistry, _DEFAULT_REGISTRY,
        )
        from superlocalmemory.core.operation_request import OperationKind
        from superlocalmemory.core.actor_context import ActorRole

        # Build a registry with all default policies except REMEMBER has empty transports.
        base_policies = dict(_DEFAULT_REGISTRY._policies)
        base_policies[OperationKind.REMEMBER] = OperationPolicy(
            kind=OperationKind.REMEMBER,
            required_roles=frozenset({ActorRole.OWNER}),
            allowed_transports=frozenset(),  # bug
        )
        reg = OperationPolicyRegistry(base_policies)

        with pytest.raises(RuntimeError, match="empty_transports|no reachable"):
            coverage_self_check(DEPLOYMENT_ENTERPRISE, registry=reg)

    def test_ungated_required_tool_flagged_personal_warns(self, caplog, tmp_path, monkeypatch):
        """Tool inventory: a required-gate tool missing from _GATED_MCP_TOOLS warns."""
        import logging

        from superlocalmemory.core.admission import _GATED_MCP_TOOLS, coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_PERSONAL
        from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY

        # Temporarily patch _REQUIRED_MCP_GATES to include a fake ungated tool
        with patch(
            "superlocalmemory.core.admission._REQUIRED_MCP_GATES",
            frozenset(_GATED_MCP_TOOLS | {"__fake_ungated_tool__"}),
        ):
            with caplog.at_level(logging.WARNING):
                coverage_self_check(DEPLOYMENT_PERSONAL)

        assert any(
            "__fake_ungated_tool__" in r.message or "ungated" in r.message
            for r in caplog.records
        ), "coverage_self_check must warn about ungated required tools"

    def test_ungated_required_tool_enterprise_raises(self, tmp_path):
        """Tool inventory: required-gate tool missing → RuntimeError in enterprise mode."""
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS, coverage_self_check
        from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE

        with patch(
            "superlocalmemory.core.admission._REQUIRED_MCP_GATES",
            frozenset(_GATED_MCP_TOOLS | {"__fake_ungated_tool__"}),
        ):
            with pytest.raises(RuntimeError, match="ungated|__fake_ungated_tool__"):
                coverage_self_check(DEPLOYMENT_ENTERPRISE)


# ---------------------------------------------------------------------------
# A5 — HTTP route registry evaluation helper
# ---------------------------------------------------------------------------

class TestAdmitHttpMutation:
    """_admit_http_mutation routes DELETE/PATCH through OperationPolicyRegistry."""

    def _make_request(self, is_enterprise: bool, is_owner: bool = True):
        """Build a minimal mock FastAPI Request."""
        req = MagicMock()
        deployment = MagicMock()
        deployment.is_enterprise = is_enterprise
        req.app.state.deployment = deployment
        return req

    def test_personal_owner_delete_allowed(self, tmp_path, monkeypatch):
        """Personal mode owner → _admit_http_mutation allows delete."""
        from superlocalmemory.server.routes.memories import _admit_http_mutation

        req = self._make_request(is_enterprise=False)
        with patch("superlocalmemory.server.rbac_enforce.resolve_principal",
                   return_value={"kind": "owner", "user_id": "local-operator",
                                 "username": "operator"}):
            with patch("superlocalmemory.server.rbac_enforce.resolve_actor_roles",
                       return_value=frozenset()):  # resolve_actor does personal override
                # Should not raise
                _admit_http_mutation(req, "delete")

    def test_enterprise_anonymous_delete_denied(self, tmp_path, monkeypatch):
        """Enterprise mode, no principal → _admit_http_mutation raises 403."""
        from fastapi import HTTPException
        from superlocalmemory.server.routes.memories import _admit_http_mutation
        from superlocalmemory.core.actor_context import ActorRole

        req = self._make_request(is_enterprise=True)
        with patch("superlocalmemory.server.rbac_enforce.resolve_principal",
                   return_value={"kind": "anonymous", "user_id": "",
                                 "username": ""}):
            with patch("superlocalmemory.server.rbac_enforce.resolve_actor_roles",
                       return_value=frozenset({ActorRole.ANONYMOUS})):
                with pytest.raises(HTTPException) as exc_info:
                    _admit_http_mutation(req, "delete")
                assert exc_info.value.status_code == 403

    def test_enterprise_anonymous_patch_denied(self, tmp_path, monkeypatch):
        """Enterprise mode, no principal → _admit_http_mutation raises 403 for patch."""
        from fastapi import HTTPException
        from superlocalmemory.server.routes.memories import _admit_http_mutation
        from superlocalmemory.core.actor_context import ActorRole

        req = self._make_request(is_enterprise=True)
        with patch("superlocalmemory.server.rbac_enforce.resolve_principal",
                   return_value={"kind": "anonymous", "user_id": "",
                                 "username": ""}):
            with patch("superlocalmemory.server.rbac_enforce.resolve_actor_roles",
                       return_value=frozenset({ActorRole.ANONYMOUS})):
                with pytest.raises(HTTPException) as exc_info:
                    _admit_http_mutation(req, "update")
                assert exc_info.value.status_code == 403
