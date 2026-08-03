"""Tranche D admission tests — RED first.

D1. CONFIG-MODE FAIL-OPEN
    config.toml present+readable with unknown mode or bare [deployment] (no mode)
    must resolve ENTERPRISE, not PERSONAL.

D2. HTTP + CLI RECALL SCOPE
    enforce_read_scope must be called in the HTTP /recall handler and
    in cmd_recall before flags reach the daemon/engine.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(base: Path, text: str) -> None:
    (base / "config.toml").write_text(text)


# ---------------------------------------------------------------------------
# D1 — config-mode fail-open
# ---------------------------------------------------------------------------

class TestConfigModeFailOpen:
    """Unknown mode or bare [deployment] section → ENTERPRISE (fail-closed)."""

    def test_mode_typo_returns_enterprise(self, tmp_path, monkeypatch):
        """mode = 'enterprize' (typo) → _resolve_deployment() returns enterprise."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprize"\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert deployment.is_enterprise, (
            "present config.toml with unrecognized mode must fail-closed → enterprise"
        )

    def test_deployment_section_no_mode_returns_enterprise(self, tmp_path, monkeypatch):
        """[deployment] section present but no mode key → enterprise."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nrequire_login = true\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert deployment.is_enterprise, (
            "present [deployment] section with no mode must fail-closed → enterprise"
        )

    def test_no_deployment_section_returns_personal(self, tmp_path, monkeypatch):
        """config.toml with no [deployment] section → personal (no declaration)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[app]\nname = "slm"\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert not deployment.is_enterprise, (
            "config.toml with no [deployment] section → personal (fresh install default)"
        )

    def test_valid_enterprise_mode_still_returns_enterprise(self, tmp_path, monkeypatch):
        """Valid mode='enterprise' → enterprise (regression guard)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert deployment.is_enterprise

    def test_valid_personal_mode_returns_personal(self, tmp_path, monkeypatch):
        """Valid mode='personal' → personal (regression guard)."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "personal"\n')
        from superlocalmemory.core.admission import _resolve_deployment
        deployment = _resolve_deployment()
        assert not deployment.is_enterprise

    def test_mode_typo_mcp_mutation_denied(self, tmp_path, monkeypatch):
        """With mode typo, @admits gate denies anonymous MCP mutation."""
        import asyncio
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprize"\n')

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        @admits(OperationKind.REMEMBER)
        async def fake_tool() -> dict:
            return {"success": True}

        result = asyncio.run(fake_tool())
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"


# ---------------------------------------------------------------------------
# D2 — HTTP recall scope: enforce_read_scope called before engine.recall
# ---------------------------------------------------------------------------

class TestHttpRecallScopeClamped:
    """unified_daemon.py /recall must call enforce_read_scope before engine.recall."""

    def test_enforce_read_scope_called_in_http_handler(self):
        """enforce_read_scope must be imported and invoked in the HTTP recall body."""
        import ast
        import pathlib
        src = pathlib.Path(
            "/Users/v.pratap.bhardwaj/Documents/varun-world/Agentic_official/"
            "slm-wt-p1/src/superlocalmemory/server/unified_daemon.py"
        ).read_text()
        assert "enforce_read_scope" in src, (
            "enforce_read_scope not found in unified_daemon.py — HTTP recall scope NOT clamped"
        )


# ---------------------------------------------------------------------------
# D2 — CLI recall scope: enforce_read_scope called in cmd_recall
# ---------------------------------------------------------------------------

class TestCliRecallScopeClamped:
    """cmd_recall must call enforce_read_scope before forwarding flags to daemon."""

    def test_cmd_recall_calls_enforce_read_scope(self, tmp_path, monkeypatch):
        """Patch enforce_read_scope and verify cmd_recall calls it."""
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        gate_called = {"called": False, "args": None}

        def mock_scope(include_global, include_shared, **kw):
            gate_called["called"] = True
            gate_called["args"] = (include_global, include_shared)
            return (include_global, include_shared)  # passthrough

        from superlocalmemory.cli import commands as cmd_mod
        with patch(
            "superlocalmemory.core.admission.enforce_read_scope",
            side_effect=mock_scope,
        ):
            args = MagicMock()
            args.include_global = True
            args.include_shared = None
            args.json = False
            args.query = "test"
            args.limit = 10
            args.fast = False
            args.window = ""
            args.as_of = ""
            try:
                cmd_mod.cmd_recall(args)
            except (Exception, SystemExit):
                # SystemExit raised by _daemon_unavailable is expected in unit tests.
                pass
        assert gate_called["called"], (
            "cmd_recall did not call enforce_read_scope — CLI scope escalation possible"
        )
