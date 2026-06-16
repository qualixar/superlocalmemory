# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# WP-09: MCP→CLI fallback adapter (ships INERT — DQ-2=B)

"""MCP→CLI fallback adapter for SuperLocalMemory.

When an MCP tool call fails, this module re-executes the equivalent
``slm <verb> --json`` command as a subprocess and reshapes the result
into the tool's expected MCP envelope shape.

Key design decisions:
- NEVER raises: every code path returns dict|None.
- shell=False always: argv is always a list.
- Only verified CLI verbs used: see TOOL_CLI_MAP (all argv[0] values are
  confirmed subparsers in cli/main.py).
- Non-backed tools (no CLI equivalent) return {success:False,
  fallback:"unavailable"} immediately without spawning a subprocess.
- Ships INERT (DQ-2=B): nothing calls this module yet. Wiring to
  tools_core.py except-branches is deferred to a future audited diff.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliVerb:
    """Describes how to call ``slm <verb>`` for a specific MCP tool.

    Attributes:
        argv: The base argv after the ``slm`` binary, e.g. ``["recall"]``
              or ``["list", "-n"]``. NEVER includes ``--json`` (added at
              call time) or the ``slm`` binary itself.
        build_args: Callable that takes the MCP tool's ``args`` dict and
                    returns extra argv tokens (positional values, flags).
                    Must return a list; never raises.
        reshape: Callable that takes the ``data`` field from a successful
                 CLI JSON envelope and returns a dict in the MCP tool's
                 return shape. ``via`` and ``degraded`` are added by the
                 caller — do not include them here.
    """

    argv: list[str]
    build_args: Callable[[dict[str, Any]], list[str]]
    reshape: Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper builders (keep anonymous lambdas minimal; use named functions for
# anything non-trivial to aid testing and readability)
# ---------------------------------------------------------------------------


def _no_args(_: dict[str, Any]) -> list[str]:
    """No extra args needed."""
    return []


def _build_recall(args: dict[str, Any]) -> list[str]:
    extra: list[str] = []
    query = args.get("query", "")
    if query:
        extra.append(str(query))
    limit = args.get("limit")
    if limit is not None:
        extra += ["--limit", str(int(limit))]
    return extra


def _reshape_recall(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": data.get("results", []),
        "count": data.get("count", 0),
        "query": data.get("query", ""),
    }


def _build_remember(args: dict[str, Any]) -> list[str]:
    content = args.get("content", "")
    extra: list[str] = [str(content)] if content else []
    tags = args.get("tags", "")
    if tags:
        extra += ["--tags", str(tags)]
    return extra


def _reshape_remember(data: dict[str, Any]) -> dict[str, Any]:
    # Handle both sync {fact_ids, count} and async {queued, pending_id}
    if "fact_ids" in data or "count" in data:
        return {
            "fact_ids": data.get("fact_ids", []),
            "count": data.get("count", 0),
        }
    return {
        "queued": data.get("queued", False),
        "pending_id": data.get("pending_id", ""),
    }


def _build_list_recent(args: dict[str, Any]) -> list[str]:
    limit = args.get("limit", 20)
    return ["-n", str(int(limit))]


def _build_delete_memory(args: dict[str, Any]) -> list[str]:
    fact_id = args.get("fact_id", "")
    return [str(fact_id), "--yes"] if fact_id else ["--yes"]


def _build_update_memory(args: dict[str, Any]) -> list[str]:
    fact_id = args.get("fact_id", "")
    content = args.get("content", "")
    extra: list[str] = []
    if fact_id:
        extra.append(str(fact_id))
    if content:
        extra.append(str(content))
    return extra


def _build_forget(args: dict[str, Any]) -> list[str]:
    query = args.get("query", "")
    return [str(query), "--yes"] if query else ["--yes"]


def _build_consolidate_cognitive(_: dict[str, Any]) -> list[str]:
    return ["--cognitive"]


def _build_set_mode(args: dict[str, Any]) -> list[str]:
    mode = args.get("mode", "")
    return [str(mode)] if mode else []


def _build_profile_switch(args: dict[str, Any]) -> list[str]:
    profile_id = args.get("profile_id", "")
    extra: list[str] = ["switch"]
    if profile_id:
        extra.append(str(profile_id))
    return extra


def _build_trace(args: dict[str, Any]) -> list[str]:
    query = args.get("query", "")
    extra = [str(query)] if query else []
    limit = args.get("limit")
    if limit is not None:
        extra += ["--limit", str(int(limit))]
    return extra


def _passthrough(data: dict[str, Any]) -> dict[str, Any]:
    """Return data unchanged — reshaping not required for this tool."""
    return dict(data)


# ---------------------------------------------------------------------------
# TOOL_CLI_MAP — ONLY verbs verified present in cli/main.py
# Verified via: grep "sub\.add_parser" cli/main.py
# ---------------------------------------------------------------------------

TOOL_CLI_MAP: dict[str, CliVerb] = {
    # recall + search alias (recall has aliases=["search"] in argparse)
    "recall": CliVerb(
        argv=["recall"],
        build_args=_build_recall,
        reshape=_reshape_recall,
    ),
    "search": CliVerb(
        argv=["search"],
        build_args=_build_recall,
        reshape=_reshape_recall,
    ),
    # remember — handles both sync {fact_ids,count} and async {queued,pending_id}
    "remember": CliVerb(
        argv=["remember"],
        build_args=_build_remember,
        reshape=_reshape_remember,
    ),
    # list_recent → slm list -n <limit>
    "list_recent": CliVerb(
        argv=["list"],
        build_args=_build_list_recent,
        reshape=_passthrough,
    ),
    # delete_memory → slm delete <fact_id> --yes
    "delete_memory": CliVerb(
        argv=["delete"],
        build_args=_build_delete_memory,
        reshape=_passthrough,
    ),
    # update_memory → slm update <fact_id> <content>
    "update_memory": CliVerb(
        argv=["update"],
        build_args=_build_update_memory,
        reshape=_passthrough,
    ),
    # get_status → slm status
    "get_status": CliVerb(
        argv=["status"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # forget → slm forget <query> --yes
    "forget": CliVerb(
        argv=["forget"],
        build_args=_build_forget,
        reshape=_passthrough,
    ),
    # consolidate_cognitive → slm consolidate --cognitive
    "consolidate_cognitive": CliVerb(
        argv=["consolidate"],
        build_args=_build_consolidate_cognitive,
        reshape=_passthrough,
    ),
    # get_soft_prompts → slm soft-prompts
    "get_soft_prompts": CliVerb(
        argv=["soft-prompts"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # set_mode → slm mode <m>
    "set_mode": CliVerb(
        argv=["mode"],
        build_args=_build_set_mode,
        reshape=_passthrough,
    ),
    # get_mode → slm mode (no argument = get current)
    "get_mode": CliVerb(
        argv=["mode"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # recall_trace → slm trace <query>
    "recall_trace": CliVerb(
        argv=["trace"],
        build_args=_build_trace,
        reshape=_passthrough,
    ),
    # quantize → slm quantize
    "quantize": CliVerb(
        argv=["quantize"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # health → slm health
    "health": CliVerb(
        argv=["health"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # slm_optimize_stats → slm optimize status (degraded)
    "slm_optimize_stats": CliVerb(
        argv=["optimize", "status"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # slm_cache_get → slm cache status (degraded — no key-level get in CLI)
    "slm_cache_get": CliVerb(
        argv=["cache", "status"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # compact_memories → slm compress status (degraded)
    "compact_memories": CliVerb(
        argv=["compress", "status"],
        build_args=_no_args,
        reshape=_passthrough,
    ),
    # switch_profile → slm profile switch <profile_id>
    "switch_profile": CliVerb(
        argv=["profile"],
        build_args=_build_profile_switch,
        reshape=_passthrough,
    ),
}


# ---------------------------------------------------------------------------
# NON_BACKED_TOOLS — tools with no viable CLI fallback
# Reasons:
#   - fetch: no single-fact CLI fetch verb
#   - code-graph (20): CLI verbs only exist under slm compress (toggles), not query
#   - slm_retrieve/slm_cache_set: cache set/key retrieval not exposed in CLI
#   - observe: lacks --json output (argparse exit or no structured output)
#   - evolve_skill/skill_health/skill_lineage: learning tools, no CLI equivalent
#   - get_version: --version is plain text, not --json envelope
#   - all 8 mesh: P2P daemon-backed, no CLI equivalent
#   - learning/assertion: no CLI equivalent
#   - session_init/close_session/run_maintenance: daemon-implicit lifecycle
#   - report_feedback/report_outcome: no CLI equivalent
#   - get_lifecycle_status/set_retention_policy: no CLI equivalent
#   - get_behavioral_patterns/audit_trail: no CLI equivalent
#   - consistency_check/get_retention_stats: no CLI equivalent
#   - backup_status/memory_used/get_learned_patterns/correct_pattern: no CLI equivalent
#   - get_attribution/build_graph/core_memory/reap_processes: no CLI equivalent
#   - slm_compress: no CLI read equivalent (compress is write-only control)
# ---------------------------------------------------------------------------

NON_BACKED_TOOLS: frozenset[str] = frozenset({
    # Fetch — no single-fact CLI verb
    "fetch",
    # Code-graph tools (20)
    "build_code_graph",
    "update_code_graph",
    "get_blast_radius",
    "get_review_context",
    "query_graph",
    "semantic_search_code",
    "list_graph_stats",
    "find_large_functions",
    "list_flows",
    "get_flow",
    "get_affected_flows",
    "list_communities",
    "get_community",
    "get_architecture_overview",
    "detect_changes",
    "refactor_preview",
    "apply_refactor",
    "code_memory_search",
    "code_entity_history",
    "enrich_blast_radius",
    # Optimize tools without usable CLI read path
    "slm_retrieve",
    "slm_cache_set",
    "slm_compress",
    # observe/evolve lack --json or reliable structured output
    "observe",
    "evolve_skill",
    # Version is plain text
    "get_version",
    # Mesh P2P tools (8)
    "mesh_summary",
    "mesh_peers",
    "mesh_send",
    "mesh_inbox",
    "mesh_state",
    "mesh_lock",
    "mesh_events",
    "mesh_status",
    # Learning / assertion tools
    "log_tool_event",
    "get_assertions",
    "reinforce_assertion",
    "contradict_assertion",
    # Evolution tools
    "skill_health",
    "skill_lineage",
    # Session lifecycle (daemon-implicit)
    "session_init",
    "close_session",
    "run_maintenance",
    # Feedback / reporting
    "report_feedback",
    "report_outcome",
    # V28 tools — no CLI equivalent
    "get_lifecycle_status",
    "set_retention_policy",
    "get_behavioral_patterns",
    "audit_trail",
    # V3 tools without clean CLI fallback
    "consistency_check",
    "get_retention_stats",
    # Core extras without CLI read equivalent
    "backup_status",
    "memory_used",
    "get_learned_patterns",
    "correct_pattern",
    "get_attribution",
    "build_graph",
    "core_memory",
    "reap_processes",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_slm(slm_path: str | None = None) -> str | None:
    """Resolve the ``slm`` CLI binary path.

    Probe order:
    1. Explicit ``slm_path`` argument (from caller).
    2. ``SLM_CLI_PATH`` environment variable.
    3. ``shutil.which("slm")`` — PATH lookup.
    4. venv-adjacent ``bin/slm`` (same Python prefix).

    Returns the first valid path string found, or None if not found.
    Never raises.
    """
    # 1. Explicit override from caller
    if slm_path:
        return slm_path

    # 2. Environment variable
    env_path = os.environ.get("SLM_CLI_PATH", "").strip()
    if env_path:
        return env_path

    # 3. PATH lookup
    which_result = shutil.which("slm")
    if which_result:
        return which_result

    # 4. Venv-adjacent: look for slm next to the current Python interpreter
    python_bin = Path(sys.executable)
    candidate = python_bin.parent / "slm"
    if candidate.exists():
        return str(candidate)

    return None


def _parse_json_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Parse JSON from subprocess stdout, handling banner text prefix.

    The SLM CLI emits JSON on stdout (banner goes to stderr). However,
    some environments or older versions may emit banner text before the
    JSON on stdout. We use brace-scanning to find the first ``{`` and
    attempt to parse from there.

    Returns dict if parsed successfully, None otherwise. Never raises.
    """
    if not stdout:
        return None

    # Fast path: entire stdout is valid JSON
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Brace-scan fallback: find first '{' and try parsing from there
    brace_idx = stdout.find("{")
    if brace_idx == -1:
        return None

    try:
        parsed = json.loads(stdout[brace_idx:])
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _run_slm(
    argv: list[str],
    timeout: float,
) -> dict[str, Any] | None:
    """Run ``slm`` as a subprocess and parse the JSON envelope from stdout.

    Args:
        argv: Full command including the slm binary path, e.g.
              ["/usr/local/bin/slm", "recall", "hello", "--json"].
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed dict from stdout JSON, or None on timeout / OSError /
        non-JSON output. Never raises.
    """
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("slm CLI timed out after %.1fs (argv=%r)", timeout, argv)
        return None
    except OSError as exc:
        logger.warning("slm CLI OSError: %s (argv=%r)", exc, argv)
        return None

    return _parse_json_from_stdout(result.stdout)


# ---------------------------------------------------------------------------
# Internal routing helpers (keep fallback_via_cli under 50 lines)
# ---------------------------------------------------------------------------


def _unavailable(reason: str, tool_name: str) -> dict[str, Any]:
    """Return a clean unavailable envelope."""
    return {
        "success": False,
        "fallback": "unavailable",
        "reason": reason,
        "tool": tool_name,
    }


def _invoke_backed(
    verb: CliVerb,
    safe_args: dict[str, Any],
    tool_name: str,
    slm_bin: str,
    timeout: float,
) -> dict[str, Any] | None:
    """Run the CLI subprocess for a backed tool and return the shaped result.

    Returns shaped dict on success, None if subprocess failed, or an
    error dict if CLI returned success=False.
    """
    try:
        extra = verb.build_args(safe_args)
    except Exception:  # noqa: BLE001
        extra = []

    full_argv: list[str] = [slm_bin, *verb.argv, *extra, "--json"]
    envelope = _run_slm(full_argv, timeout)
    if envelope is None:
        logger.warning("fallback_via_cli: subprocess returned no valid JSON for %r", tool_name)
        return None

    if not envelope.get("success", False):
        return {
            "success": False,
            "error": envelope.get("error", {}),
            "via": "cli_fallback",
            "tool": tool_name,
        }

    data = envelope.get("data", {})
    if not isinstance(data, dict):
        data = {}
    try:
        shaped = verb.reshape(data)
    except Exception:  # noqa: BLE001
        shaped = dict(data)

    shaped["via"] = "cli_fallback"
    shaped["degraded"] = True
    return shaped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fallback_via_cli(
    tool_name: Any,
    args: Any,
    *,
    timeout: float = 20.0,
    slm_path: str | None = None,
) -> dict[str, Any] | None:
    """Attempt to execute an MCP tool via the ``slm`` CLI as a fallback.

    Called from an MCP tool's except branch when the normal code path fails.
    NEVER raises — all exceptions are caught internally.

    Returns:
        - dict (tool MCP shape + ``via:"cli_fallback"`` + ``degraded:True``)
          on success.
        - ``{success:False, fallback:"unavailable"}`` for non-backed / unmapped.
        - ``None`` when backed but subprocess failed (caller surfaces MCP error).
    """
    if not isinstance(tool_name, str):
        return _unavailable("invalid_tool_name", "")
    safe_args: dict[str, Any] = args if isinstance(args, dict) else {}

    if tool_name in NON_BACKED_TOOLS:
        logger.debug("fallback_via_cli: %r is NON_BACKED", tool_name)
        return _unavailable("no_cli_verb", tool_name)

    verb = TOOL_CLI_MAP.get(tool_name)
    if verb is None:
        logger.debug("fallback_via_cli: %r not in TOOL_CLI_MAP", tool_name)
        return _unavailable("unmapped", tool_name)

    slm_bin = _resolve_slm(slm_path)
    if slm_bin is None:
        logger.warning("fallback_via_cli: slm binary not found — cannot fallback %r", tool_name)
        return None

    return _invoke_backed(verb, safe_args, tool_name, slm_bin, timeout)
