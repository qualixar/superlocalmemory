"""Tranche C admission tests — RED first.

Covers:
  1. recall and search MCP tools gated with @admits(OperationKind.RECALL)
  2. recall and search in _REQUIRED_MCP_GATES
  3. enforce_read_scope: clamps include_global/include_shared in enterprise mode
     when OperationPolicy(RECALL).allow_cross_profile is False
  4. Personal OWNER: scope flags unchanged
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(base: Path, text: str) -> None:
    (base / "config.toml").write_text(text)


def _mock_server() -> MagicMock:
    ms = MagicMock()
    ms.tool.return_value = lambda f: f
    return ms


def _mock_get_engine() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# C1 — Tool inventory: recall + search in _GATED_MCP_TOOLS
# ---------------------------------------------------------------------------

class TestTrancheCToolInventory:
    """recall and search must carry @admits → appear in _GATED_MCP_TOOLS."""

    @staticmethod
    def _trigger():
        from superlocalmemory.mcp.tools_core import register_core_tools
        ms = _mock_server()
        register_core_tools(ms, _mock_get_engine())

    def test_recall_gated(self):
        self._trigger()
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        assert "recall" in _GATED_MCP_TOOLS, (
            "recall missing @admits — not in _GATED_MCP_TOOLS"
        )

    def test_search_gated(self):
        self._trigger()
        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        assert "search" in _GATED_MCP_TOOLS, (
            "search missing @admits — not in _GATED_MCP_TOOLS"
        )


# ---------------------------------------------------------------------------
# C2 — _REQUIRED_MCP_GATES includes recall + search
# ---------------------------------------------------------------------------

class TestRequiredGatesIncludesReads:
    def test_recall_in_required_gates(self):
        from superlocalmemory.core.admission import _REQUIRED_MCP_GATES
        assert "recall" in _REQUIRED_MCP_GATES

    def test_search_in_required_gates(self):
        from superlocalmemory.core.admission import _REQUIRED_MCP_GATES
        assert "search" in _REQUIRED_MCP_GATES


# ---------------------------------------------------------------------------
# C3 — enforce_read_scope clamping
# ---------------------------------------------------------------------------

class TestEnforceReadScope:
    """enforce_read_scope must clamp cross-scope flags in enterprise mode."""

    def _enterprise_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

    def _personal_env(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        # No config.toml → personal mode

    def test_enterprise_include_global_true_clamped_to_false(
        self, tmp_path, monkeypatch
    ):
        """Enterprise mode + allow_cross_profile=False → include_global=True clamped."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import enforce_read_scope
        clamped_global, clamped_shared = enforce_read_scope(
            include_global=True, include_shared=None
        )
        assert clamped_global is False, (
            "include_global=True must be clamped to False in enterprise mode"
        )

    def test_enterprise_include_shared_true_clamped_to_false(
        self, tmp_path, monkeypatch
    ):
        """Enterprise mode + allow_cross_profile=False → include_shared=True clamped."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import enforce_read_scope
        clamped_global, clamped_shared = enforce_read_scope(
            include_global=None, include_shared=True
        )
        assert clamped_shared is False, (
            "include_shared=True must be clamped to False in enterprise mode"
        )

    def test_enterprise_none_flags_unchanged(self, tmp_path, monkeypatch):
        """None flags in enterprise mode stay None (server default applies)."""
        self._enterprise_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import enforce_read_scope
        clamped_global, clamped_shared = enforce_read_scope(
            include_global=None, include_shared=None
        )
        assert clamped_global is None
        assert clamped_shared is None

    def test_personal_include_global_true_unchanged(self, tmp_path, monkeypatch):
        """Personal mode: include_global=True must not be clamped."""
        self._personal_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import enforce_read_scope
        clamped_global, clamped_shared = enforce_read_scope(
            include_global=True, include_shared=True
        )
        assert clamped_global is True
        assert clamped_shared is True

    def test_personal_false_flags_unchanged(self, tmp_path, monkeypatch):
        """Personal mode: False flags stay False."""
        self._personal_env(tmp_path, monkeypatch)
        from superlocalmemory.core.admission import enforce_read_scope
        clamped_global, clamped_shared = enforce_read_scope(
            include_global=False, include_shared=False
        )
        assert clamped_global is False
        assert clamped_shared is False


# ---------------------------------------------------------------------------
# C4 — Enterprise anonymous recall denied
# ---------------------------------------------------------------------------

class TestTrancheCEnterpriseRecallDenied:
    """In enterprise mode, anonymous caller must be denied for recall/search."""

    def _enterprise_fake_result(self, tmp_path, monkeypatch) -> dict:
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
        _write_config(tmp_path, '[deployment]\nmode = "enterprise"\nrequire_login = true\n')

        from superlocalmemory.core.admission import admits
        from superlocalmemory.core.operation_request import OperationKind

        @admits(OperationKind.RECALL)
        async def fake_recall(query: str) -> dict:
            return {"success": True, "results": []}

        return asyncio.run(fake_recall("test query"))

    def test_recall_enterprise_anonymous_denied(self, tmp_path, monkeypatch):
        result = self._enterprise_fake_result(tmp_path, monkeypatch)
        assert result.get("success") is False
        assert result.get("error") == "not_authorized"
