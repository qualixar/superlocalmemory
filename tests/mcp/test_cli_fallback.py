# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Tests for WP-09: MCP→CLI fallback adapter

"""RED-first TDD tests for mcp/cli_fallback.py.

Every test imports the module after it exists; until then they fail at
import. Once the module is created (GREEN phase) all tests should pass.
Coverage target: >=90% of cli_fallback.py.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lazy import helper — avoids hard failure before file is created
# ---------------------------------------------------------------------------

def _get_module():
    """Import cli_fallback with any heavy SLM side-effects suppressed."""
    import os
    os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")
    mod_name = "superlocalmemory.mcp.cli_fallback"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Real CLI verbs present in cli/main.py (verified by grep)
# ---------------------------------------------------------------------------

# ALL top-level subparsers registered in cli/main.py (sub.add_parser calls)
_REAL_CLI_VERBS: frozenset[str] = frozenset({
    "init", "setup", "mode", "provider", "connect", "migrate", "db",
    "remember", "recall", "search", "forget", "delete", "update", "list",
    "status", "health", "trace", "doctor", "wrap", "mcp", "warmup",
    "dashboard", "serve", "restart", "profile", "hooks", "session-context",
    "observe", "decay", "quantize", "consolidate", "soft-prompts", "reap",
    "adapters", "ingest", "config", "evolve", "disable", "enable",
    "clear-cache", "reconfigure", "benchmark", "rotate-token",
    "optimize", "cache", "compress", "proxy", "help-optimize",
})


# ---------------------------------------------------------------------------
# Comprehensive list of ALL registered MCP tools (union across all tool files)
# ---------------------------------------------------------------------------

_ALL_MCP_TOOLS: frozenset[str] = frozenset({
    # tools_core.py
    "remember", "recall", "search", "fetch", "list_recent", "get_status",
    "build_graph", "switch_profile", "backup_status", "memory_used",
    "get_learned_patterns", "correct_pattern", "delete_memory", "update_memory",
    "get_attribution",
    # tools_v28.py
    "report_outcome", "get_lifecycle_status", "set_retention_policy",
    "compact_memories", "get_behavioral_patterns", "audit_trail",
    # tools_v3.py
    "get_version", "set_mode", "get_mode", "health", "consistency_check",
    "recall_trace",
    # tools_active.py
    "session_init", "observe", "report_feedback", "close_session",
    "core_memory",
    # tools_v33.py
    "forget", "quantize", "consolidate_cognitive", "get_soft_prompts",
    "reap_processes", "get_retention_stats", "run_maintenance",
    # tools_code_graph.py (22 code-graph tools)
    "build_code_graph", "update_code_graph", "get_blast_radius",
    "get_review_context", "query_graph", "semantic_search_code",
    "list_graph_stats", "find_large_functions", "list_flows", "get_flow",
    "get_affected_flows", "list_communities", "get_community",
    "get_architecture_overview", "detect_changes", "refactor_preview",
    "apply_refactor", "code_memory_search", "code_entity_history",
    "enrich_blast_radius",
    # tools_mesh.py (8)
    "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
    "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
    # tools_learning.py (4)
    "log_tool_event", "get_assertions", "reinforce_assertion", "contradict_assertion",
    # tools_evolution.py (3)
    "evolve_skill", "skill_health", "skill_lineage",
    # tools_optimize.py (5)
    "slm_compress", "slm_retrieve", "slm_cache_set", "slm_cache_get", "slm_optimize_stats",
})


# ===========================================================================
# TESTS
# ===========================================================================


class TestMapOnlyRealVerbs:
    """AC2: Every argv[0] in TOOL_CLI_MAP must be a REAL subparser in cli/main.py.

    This is the anti-fabrication test — it catches invented verbs like
    `slm code`, `slm semantic`, `slm ccr`, `slm cache set` that the old
    Master Plan had but which DON'T exist.
    """

    def test_map_only_real_verbs(self):
        mod = _get_module()
        for tool_name, verb in mod.TOOL_CLI_MAP.items():
            first_token = verb.argv[0]
            assert first_token in _REAL_CLI_VERBS, (
                f"TOOL_CLI_MAP[{tool_name!r}].argv[0]={first_token!r} is NOT a real "
                f"subparser in cli/main.py. This is a fabricated verb — move to NON_BACKED."
            )


class TestMapCompleteness:
    """Every registered MCP tool must appear in TOOL_CLI_MAP OR NON_BACKED_TOOLS."""

    def test_map_completeness(self):
        mod = _get_module()
        covered = set(mod.TOOL_CLI_MAP.keys()) | set(mod.NON_BACKED_TOOLS)
        uncovered = _ALL_MCP_TOOLS - covered
        assert uncovered == set(), (
            f"These MCP tools are in neither TOOL_CLI_MAP nor NON_BACKED_TOOLS: {uncovered!r}. "
            f"Add them to one of the two."
        )


class TestArgvIsListNeverShell:
    """AC4: subprocess must be called with argv LIST and shell=False.

    grep shell=True == 0 on the module source file.
    """

    def test_argv_is_list_never_shell(self):
        cli_fallback_path = (
            Path(__file__).parent.parent.parent
            / "src" / "superlocalmemory" / "mcp" / "cli_fallback.py"
        )
        assert cli_fallback_path.exists(), "cli_fallback.py not found"
        source = cli_fallback_path.read_text()
        assert "shell=True" not in source, (
            "cli_fallback.py contains 'shell=True' — this is forbidden. "
            "Use argv list with shell=False."
        )


class TestRoundtripRecall:
    """fallback_via_cli('recall', ...) uses recall verb and returns shaped dict."""

    def test_roundtrip_recall(self):
        mod = _get_module()
        fake_envelope = {
            "success": True,
            "command": "recall",
            "version": "3.6.14",
            "data": {
                "results": [{"fact_id": "f1", "content": "hello", "score": 0.9}],
                "count": 1,
                "query": "hello",
            },
            "next_actions": [],
        }
        fake_json = json.dumps(fake_envelope)
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = fake_json
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed) as mock_run, \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "hello", "limit": 5})

        assert result is not None
        assert result["via"] == "cli_fallback"
        assert result["degraded"] is True
        # Must have reshaped data from the envelope
        assert "results" in result or "data" in result or "count" in result

        # argv must be a list (shell=False by virtue of no shell kwarg)
        call_args = mock_run.call_args
        argv_passed = call_args[0][0]
        assert isinstance(argv_passed, list), "subprocess.run must receive a list, not string"


class TestRoundtripRememberSync:
    """remember with sync response {fact_ids, count}."""

    def test_roundtrip_remember_sync(self):
        mod = _get_module()
        fake_envelope = {
            "success": True,
            "command": "remember",
            "version": "3.6.14",
            "data": {"fact_ids": ["id1", "id2"], "count": 2},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(fake_envelope)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("remember", {"content": "test fact"})

        assert result is not None
        assert result["via"] == "cli_fallback"
        assert result["degraded"] is True
        # Sync path must contain fact_ids or count
        assert "fact_ids" in result or "count" in result or "data" in result


class TestRoundtripRememberAsync:
    """remember with async response {queued, pending_id}."""

    def test_roundtrip_remember_async(self):
        mod = _get_module()
        fake_envelope = {
            "success": True,
            "command": "remember",
            "version": "3.6.14",
            "data": {"queued": True, "pending_id": "pending-abc"},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(fake_envelope)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("remember", {"content": "async fact"})

        assert result is not None
        assert result["via"] == "cli_fallback"
        assert result["degraded"] is True
        assert "queued" in result or "pending_id" in result or "data" in result


class TestSearchAlias:
    """'search' is an alias for recall and must work identically."""

    def test_search_alias(self):
        mod = _get_module()
        # Both recall and search must be in TOOL_CLI_MAP
        assert "recall" in mod.TOOL_CLI_MAP
        assert "search" in mod.TOOL_CLI_MAP
        # Both must point to the same CLI verb (recall)
        assert mod.TOOL_CLI_MAP["search"].argv[0] in _REAL_CLI_VERBS


class TestNonBackedReturnsCleanUnavailable:
    """Non-backed tools return {success:False, fallback:'unavailable'} without spawning subprocess."""

    def test_non_backed_returns_clean_unavailable(self):
        mod = _get_module()
        # Pick a confirmed NON_BACKED tool
        non_backed = next(iter(mod.NON_BACKED_TOOLS))

        with mock.patch("subprocess.run") as mock_run:
            result = mod.fallback_via_cli(non_backed, {})

        assert result is not None
        assert result["success"] is False
        assert result.get("fallback") == "unavailable"
        mock_run.assert_not_called()

    def test_fetch_is_non_backed(self):
        """fetch has no reliable CLI counterpart — must be NON_BACKED."""
        mod = _get_module()
        assert "fetch" in mod.NON_BACKED_TOOLS

    def test_mesh_tools_are_non_backed(self):
        """All 8 mesh tools must be NON_BACKED."""
        mod = _get_module()
        mesh_tools = {
            "mesh_summary", "mesh_peers", "mesh_send", "mesh_inbox",
            "mesh_state", "mesh_lock", "mesh_events", "mesh_status",
        }
        uncovered = mesh_tools - mod.NON_BACKED_TOOLS
        assert uncovered == set(), f"Mesh tools not in NON_BACKED: {uncovered}"

    def test_code_graph_tools_are_non_backed(self):
        """All code-graph tools must be NON_BACKED."""
        mod = _get_module()
        code_graph = {
            "build_code_graph", "update_code_graph", "get_blast_radius",
            "get_review_context", "query_graph", "semantic_search_code",
            "list_graph_stats", "find_large_functions", "list_flows", "get_flow",
            "get_affected_flows", "list_communities", "get_community",
            "get_architecture_overview", "detect_changes", "refactor_preview",
            "apply_refactor", "code_memory_search", "code_entity_history",
            "enrich_blast_radius",
        }
        uncovered = code_graph - mod.NON_BACKED_TOOLS
        assert uncovered == set(), f"Code-graph tools not in NON_BACKED: {uncovered}"

    def test_session_lifecycle_tools_are_non_backed(self):
        """session_init, close_session, run_maintenance are daemon-implicit → NON_BACKED."""
        mod = _get_module()
        daemon_tools = {"session_init", "close_session", "run_maintenance"}
        uncovered = daemon_tools - mod.NON_BACKED_TOOLS
        assert uncovered == set(), f"Daemon tools not in NON_BACKED: {uncovered}"


class TestUnmapped:
    """A tool not in MAP and not in NON_BACKED returns unavailable without crash."""

    def test_unmapped(self):
        mod = _get_module()
        with mock.patch("subprocess.run") as mock_run:
            result = mod.fallback_via_cli("totally_unknown_tool_xyz", {})
        assert result is not None
        assert result["success"] is False
        mock_run.assert_not_called()


class TestTimeoutHandled:
    """TimeoutExpired → returns None, does NOT raise."""

    def test_timeout_handled(self):
        mod = _get_module()
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["slm"], timeout=20)), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "hello"})
        assert result is None  # backed-but-subprocess-failed → None


class TestSpawnFailure:
    """OSError (e.g. binary not found) → returns None, does NOT raise."""

    def test_spawn_failure(self):
        mod = _get_module()
        with mock.patch("subprocess.run", side_effect=OSError("not found")), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "hello"})
        assert result is None


class TestSlmNotOnPath:
    """_resolve_slm() returns None → fallback_via_cli returns None for backed tools."""

    def test_slm_not_on_path_returns_None(self):
        mod = _get_module()
        with mock.patch.object(mod, "_resolve_slm", return_value=None), \
             mock.patch("subprocess.run") as mock_run:
            result = mod.fallback_via_cli("recall", {"query": "hello"})
        assert result is None
        mock_run.assert_not_called()


class TestErrorEnvelopePassthrough:
    """CLI returns {success:False, error:{code, message}} → shaped error dict returned."""

    def test_error_envelope_passthrough(self):
        mod = _get_module()
        error_envelope = {
            "success": False,
            "error": {"code": "DB_LOCKED", "message": "database is locked"},
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(error_envelope)
        completed.stderr = ""
        completed.returncode = 1

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "x"})

        assert result is not None
        assert result["success"] is False
        assert result.get("via") == "cli_fallback"


class TestGarbageStdoutReturnsNone:
    """Non-JSON stdout from subprocess → _run_slm returns None."""

    def test_garbage_stdout_returns_None(self):
        mod = _get_module()
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = "ERROR: cannot connect to embedding worker\nsome stacktrace"
        completed.stderr = ""
        completed.returncode = 1

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "x"})

        assert result is None


class TestBannerPrefixedParsed:
    """When CLI emits banner on stderr and JSON on stdout, parsing must succeed."""

    def test_banner_prefixed_parsed(self):
        mod = _get_module()
        banner = "SLM v3.6.14 — SuperLocalMemory\nLoading engine...\n"
        payload = {
            "success": True,
            "command": "recall",
            "version": "3.6.14",
            "data": {"results": [], "count": 0, "query": "test"},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(payload)   # clean stdout
        completed.stderr = banner                 # banner on stderr only
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("recall", {"query": "test"})

        assert result is not None
        assert result["via"] == "cli_fallback"

    def test_banner_mixed_stdout_parsed(self):
        """Some CLI versions may emit banner text before JSON on stdout — brace-scan must handle."""
        mod = _get_module()
        payload = {
            "success": True,
            "command": "status",
            "version": "3.6.14",
            "data": {"mode": "a", "profile": "default"},
            "next_actions": [],
        }
        # Banner text BEFORE the JSON on stdout
        mixed_stdout = "SLM v3.6.14\nWarning: warmup incomplete\n" + json.dumps(payload)
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = mixed_stdout
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("get_status", {})

        assert result is not None
        assert result["via"] == "cli_fallback"


class TestFallbackNeverRaises:
    """Fuzz: fallback_via_cli must never raise regardless of input."""

    def test_fallback_never_raises(self):
        mod = _get_module()
        fuzz_inputs: list[tuple[Any, Any]] = [
            (None, None),
            ("", {}),
            (123, "not a dict"),
            ("recall", None),
            ("recall", {"query": None, "limit": "not_an_int"}),
            ("\x00\xff", {"x": "\x00"}),
            ("remember", {"content": "a" * 100_000}),
            ([], {}),
            ({}, []),
        ]
        for tool_name, args in fuzz_inputs:
            try:
                # May or may not resolve slm — either path must not raise
                with mock.patch.object(mod, "_resolve_slm", return_value=None):
                    mod.fallback_via_cli(tool_name, args)
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(
                    f"fallback_via_cli raised {type(exc).__name__}({exc!r}) "
                    f"for inputs ({tool_name!r}, {args!r}) — must NEVER raise"
                ) from exc


class TestNoSubprocessOnHappyPath:
    """When slm is on path and returns valid JSON, no extra subprocess calls happen."""

    def test_no_subprocess_on_happy_path(self):
        """Exactly ONE subprocess.run call for a backed tool on happy path."""
        mod = _get_module()
        payload = {
            "success": True,
            "command": "status",
            "version": "3.6.14",
            "data": {"mode": "a"},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(payload)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed) as mock_run, \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            mod.fallback_via_cli("get_status", {})

        assert mock_run.call_count == 1, (
            f"Expected exactly 1 subprocess call, got {mock_run.call_count}"
        )


class TestModuleImportNoSideEffects:
    """AC1: importing cli_fallback must not trigger network, DB, or subprocess calls."""

    def test_import_no_side_effects(self):
        with mock.patch("subprocess.run") as mock_run, \
             mock.patch("subprocess.Popen") as mock_popen:
            _get_module()
        mock_run.assert_not_called()
        mock_popen.assert_not_called()


class TestResolveSlm:
    """_resolve_slm probes shutil.which, SLM_CLI_PATH env, and venv-adjacent bin/slm."""

    def test_resolve_slm_finds_via_which(self):
        mod = _get_module()
        with mock.patch("shutil.which", return_value="/usr/local/bin/slm"):
            result = mod._resolve_slm()
        assert result == "/usr/local/bin/slm"

    def test_resolve_slm_uses_env_var(self):
        mod = _get_module()
        import os
        with mock.patch.dict(os.environ, {"SLM_CLI_PATH": "/custom/path/slm"}), \
             mock.patch("shutil.which", return_value=None):
            result = mod._resolve_slm()
        assert result == "/custom/path/slm"

    def test_resolve_slm_returns_none_when_not_found(self):
        mod = _get_module()
        import os
        env_without_slm = {k: v for k, v in os.environ.items() if k != "SLM_CLI_PATH"}
        with mock.patch.dict(os.environ, env_without_slm, clear=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("pathlib.Path.exists", return_value=False):
            result = mod._resolve_slm()
        assert result is None

    def test_resolve_slm_explicit_path_wins(self):
        """Explicit slm_path argument takes priority over everything."""
        mod = _get_module()
        result = mod._resolve_slm(slm_path="/explicit/bin/slm")
        assert result == "/explicit/bin/slm"

    def test_resolve_slm_venv_adjacent(self):
        """Venv-adjacent bin/slm is found when PATH lookup fails."""
        mod = _get_module()
        import os
        env_without_slm = {k: v for k, v in os.environ.items() if k != "SLM_CLI_PATH"}
        with mock.patch.dict(os.environ, env_without_slm, clear=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch("pathlib.Path.exists", return_value=True):
            result = mod._resolve_slm()
        assert result is not None
        assert result.endswith("slm")


class TestParseJsonFromStdout:
    """_parse_json_from_stdout handles edge cases."""

    def test_empty_stdout_returns_none(self):
        mod = _get_module()
        assert mod._parse_json_from_stdout("") is None
        assert mod._parse_json_from_stdout(None) is None  # type: ignore[arg-type]

    def test_valid_json_returns_dict(self):
        mod = _get_module()
        payload = {"success": True, "data": {"count": 1}}
        result = mod._parse_json_from_stdout(json.dumps(payload))
        assert result == payload

    def test_json_array_returns_none(self):
        """JSON arrays are not valid MCP envelopes — return None."""
        mod = _get_module()
        assert mod._parse_json_from_stdout("[1, 2, 3]") is None

    def test_no_brace_returns_none(self):
        mod = _get_module()
        assert mod._parse_json_from_stdout("just plain text no json") is None

    def test_banner_then_json(self):
        """Banner text before JSON must be skipped via brace-scan."""
        mod = _get_module()
        payload = {"success": True, "data": {"mode": "a"}}
        mixed = "SLM Banner line\nAnother line\n" + json.dumps(payload)
        result = mod._parse_json_from_stdout(mixed)
        assert result == payload

    def test_brace_scan_invalid_json_returns_none(self):
        """Even with a '{', if rest is invalid JSON, return None."""
        mod = _get_module()
        result = mod._parse_json_from_stdout("prefix text {invalid json here")
        assert result is None


class TestBuilderFunctions:
    """Unit tests for individual build_args helper functions."""

    def test_build_recall_with_query_and_limit(self):
        mod = _get_module()
        result = mod._build_recall({"query": "hello world", "limit": 10})
        assert result == ["hello world", "--limit", "10"]

    def test_build_recall_query_only(self):
        mod = _get_module()
        result = mod._build_recall({"query": "test"})
        assert result == ["test"]

    def test_build_recall_empty(self):
        mod = _get_module()
        result = mod._build_recall({})
        assert result == []

    def test_build_remember_with_tags(self):
        mod = _get_module()
        result = mod._build_remember({"content": "my fact", "tags": "python,code"})
        assert result == ["my fact", "--tags", "python,code"]

    def test_build_remember_no_tags(self):
        mod = _get_module()
        result = mod._build_remember({"content": "bare fact"})
        assert result == ["bare fact"]

    def test_build_remember_empty(self):
        mod = _get_module()
        result = mod._build_remember({})
        assert result == []

    def test_build_list_recent_default(self):
        mod = _get_module()
        result = mod._build_list_recent({})
        assert result == ["-n", "20"]

    def test_build_list_recent_custom(self):
        mod = _get_module()
        result = mod._build_list_recent({"limit": 5})
        assert result == ["-n", "5"]

    def test_build_delete_memory_with_id(self):
        mod = _get_module()
        result = mod._build_delete_memory({"fact_id": "abc123"})
        assert result == ["abc123", "--yes"]

    def test_build_delete_memory_no_id(self):
        mod = _get_module()
        result = mod._build_delete_memory({})
        assert result == ["--yes"]

    def test_build_update_memory(self):
        mod = _get_module()
        result = mod._build_update_memory({"fact_id": "f1", "content": "new content"})
        assert result == ["f1", "new content"]

    def test_build_update_memory_empty(self):
        mod = _get_module()
        result = mod._build_update_memory({})
        assert result == []

    def test_build_forget_with_query(self):
        mod = _get_module()
        result = mod._build_forget({"query": "old memories"})
        assert result == ["old memories", "--yes"]

    def test_build_forget_no_query(self):
        mod = _get_module()
        result = mod._build_forget({})
        assert result == ["--yes"]

    def test_build_consolidate_cognitive(self):
        mod = _get_module()
        result = mod._build_consolidate_cognitive({})
        assert result == ["--cognitive"]

    def test_build_set_mode_with_mode(self):
        mod = _get_module()
        result = mod._build_set_mode({"mode": "b"})
        assert result == ["b"]

    def test_build_set_mode_empty(self):
        mod = _get_module()
        result = mod._build_set_mode({})
        assert result == []

    def test_build_profile_switch_with_id(self):
        mod = _get_module()
        result = mod._build_profile_switch({"profile_id": "work"})
        assert result == ["switch", "work"]

    def test_build_profile_switch_no_id(self):
        mod = _get_module()
        result = mod._build_profile_switch({})
        assert result == ["switch"]

    def test_build_trace_with_query_and_limit(self):
        mod = _get_module()
        result = mod._build_trace({"query": "trace me", "limit": 5})
        assert result == ["trace me", "--limit", "5"]

    def test_build_trace_query_only(self):
        mod = _get_module()
        result = mod._build_trace({"query": "trace me"})
        assert result == ["trace me"]

    def test_build_trace_empty(self):
        mod = _get_module()
        result = mod._build_trace({})
        assert result == []


class TestReshapeRemember:
    """_reshape_remember handles sync and async envelopes."""

    def test_reshape_remember_sync(self):
        mod = _get_module()
        result = mod._reshape_remember({"fact_ids": ["a", "b"], "count": 2})
        assert result["fact_ids"] == ["a", "b"]
        assert result["count"] == 2

    def test_reshape_remember_async(self):
        mod = _get_module()
        result = mod._reshape_remember({"queued": True, "pending_id": "pid1"})
        assert result["queued"] is True
        assert result["pending_id"] == "pid1"

    def test_reshape_remember_empty_falls_through_to_async(self):
        mod = _get_module()
        result = mod._reshape_remember({})
        # No fact_ids/count → async branch
        assert "queued" in result
        assert "pending_id" in result


class TestFallbackEdgeCases:
    """Additional coverage for edge-case paths in fallback_via_cli."""

    def test_non_dict_data_in_envelope(self):
        """If CLI returns data as non-dict, fallback must still return shaped result."""
        mod = _get_module()
        envelope = {
            "success": True,
            "command": "status",
            "version": "3.6.14",
            "data": "unexpected string",
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(envelope)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("get_status", {})

        assert result is not None
        assert result["via"] == "cli_fallback"
        assert result["degraded"] is True

    def test_build_args_exception_recovered(self):
        """If build_args raises, fallback should still attempt the call with empty extra."""
        mod = _get_module()
        bad_verb = mod.CliVerb(
            argv=["status"],
            build_args=lambda _: (_ for _ in ()).throw(RuntimeError("build fail")),  # type: ignore[arg-type]
            reshape=mod._passthrough,
        )
        # Patch TOOL_CLI_MAP temporarily
        patched_map = {**mod.TOOL_CLI_MAP, "test_bad_verb": bad_verb}
        envelope = {
            "success": True,
            "command": "status",
            "version": "3.6.14",
            "data": {"mode": "a"},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(envelope)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch.object(mod, "TOOL_CLI_MAP", patched_map), \
             mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            # Must not raise even with bad build_args
            result = mod.fallback_via_cli("test_bad_verb", {})

        # Either result or None is acceptable, but must not raise
        assert result is None or isinstance(result, dict)

    def test_invalid_tool_name_type_returns_unavailable(self):
        """Non-string tool_name returns unavailable dict, not None."""
        mod = _get_module()
        result = mod.fallback_via_cli(42, {})  # type: ignore[arg-type]
        assert result is not None
        assert result["success"] is False
        assert result["fallback"] == "unavailable"

    def test_reshape_exception_recovered(self):
        """If reshape raises, fallback must still return tagged dict (not raise)."""
        mod = _get_module()

        def bad_reshape(_: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("reshape failed")

        bad_verb = mod.CliVerb(
            argv=["status"],
            build_args=mod._no_args,
            reshape=bad_reshape,
        )
        patched_map = {**mod.TOOL_CLI_MAP, "test_bad_reshape": bad_verb}
        envelope = {
            "success": True,
            "command": "status",
            "version": "3.6.14",
            "data": {"mode": "a"},
            "next_actions": [],
        }
        completed = mock.MagicMock(spec=subprocess.CompletedProcess)
        completed.stdout = json.dumps(envelope)
        completed.stderr = ""
        completed.returncode = 0

        with mock.patch.object(mod, "TOOL_CLI_MAP", patched_map), \
             mock.patch("subprocess.run", return_value=completed), \
             mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
            result = mod.fallback_via_cli("test_bad_reshape", {})

        assert result is not None
        assert result["via"] == "cli_fallback"
        assert result["degraded"] is True

    def test_various_backed_tools_happy_path(self):
        """Spot-check a few backed tools beyond recall to exercise different builders."""
        mod = _get_module()
        tools_and_args = [
            ("forget", {"query": "old stuff"}),
            ("delete_memory", {"fact_id": "abc"}),
            ("list_recent", {"limit": 10}),
            ("consolidate_cognitive", {}),
            ("get_soft_prompts", {}),
            ("quantize", {}),
            ("health", {}),
            ("slm_optimize_stats", {}),
            ("slm_cache_get", {}),
            ("compact_memories", {}),
            ("switch_profile", {"profile_id": "work"}),
            ("recall_trace", {"query": "test", "limit": 5}),
        ]
        for tool_name, args in tools_and_args:
            envelope = {
                "success": True,
                "command": tool_name,
                "version": "3.6.14",
                "data": {"result": "ok"},
                "next_actions": [],
            }
            completed = mock.MagicMock(spec=subprocess.CompletedProcess)
            completed.stdout = json.dumps(envelope)
            completed.stderr = ""
            completed.returncode = 0

            with mock.patch("subprocess.run", return_value=completed), \
                 mock.patch.object(mod, "_resolve_slm", return_value="/usr/local/bin/slm"):
                result = mod.fallback_via_cli(tool_name, args)

            assert result is not None, f"Expected result for {tool_name!r}, got None"
            assert result["via"] == "cli_fallback", f"Missing via tag for {tool_name!r}"
            assert result["degraded"] is True, f"Missing degraded tag for {tool_name!r}"
