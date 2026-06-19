# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — CLI entry point.

Usage: slm <command> [options]

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

# CRITICAL: Set BEFORE any torch/transformers import to prevent Metal/MPS
# GPU memory reservation on Apple Silicon. Without this, macOS Activity
# Monitor shows 3-6 GB for what is actually a 40 MB process.
import os as _os
_os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
_os.environ.setdefault('PYTORCH_MPS_MEM_LIMIT', '0')
_os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
_os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
_os.environ.setdefault('TORCH_DEVICE', 'cpu')

import argparse
import sys

from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT

_HELP_EPILOG = """\
operating modes:
  Mode A  Local Guardian — Zero cloud, zero LLM. All processing stays on
          your machine. Full EU AI Act compliance. Best for privacy-first
          use, air-gapped systems, and regulated environments.
          Retrieval score: 74.8% on LoCoMo benchmark.

  Mode B  Smart Local — Uses a local Ollama LLM for summarization and
          enrichment. Data never leaves your network. EU AI Act compliant.
          Requires: ollama running locally with a model pulled.

  Mode C  Full Power — Uses a cloud LLM (OpenAI, Anthropic, etc.) for
          maximum accuracy. Best retrieval quality, agentic multi-hop.
          Retrieval score: 87.7% on LoCoMo benchmark.

quick start:
  slm setup                   Interactive first-time setup
  slm remember "some fact"    Store a memory
  slm recall "search query"   Semantic search across memories
  slm list -n 20              Show 20 most recent memories
  slm dashboard               Open web dashboard at localhost:8765

ide integration:
  slm mcp                     Start MCP server (used by IDEs)
  slm connect                 Auto-configure all detected IDEs
  slm connect cursor           Configure a specific IDE

examples:
  slm remember "Project X uses PostgreSQL 16" --tags "project-x,db"
  slm recall "which database does project X use"
  slm list -n 50
  slm mode a                  Switch to zero-LLM mode
  slm trace "auth flow"       Recall with per-channel score breakdown
  slm health                  Check math layer status
  slm dashboard --port 9000   Dashboard on custom port
  slm recall "query" --json   Agent-native JSON output (for scripts, CI/CD)

documentation:
  Website:    https://superlocalmemory.com
  GitHub:     https://github.com/qualixar/superlocalmemory
  Paper:      https://arxiv.org/abs/2603.14588
"""


def main() -> None:
    """Parse CLI arguments and dispatch to command handlers."""
    # Fast path: hook invocations bypass argparse entirely (stdlib only, ~30ms)
    if len(sys.argv) >= 3 and sys.argv[1] == "hook":
        from superlocalmemory.hooks.hook_handlers import handle_hook
        handle_hook(sys.argv[2])
        return

    # WP-07: lazy first-run init — runs after hook/mcp fast-paths so stdout
    # is never polluted on those paths (CRIT-3, MCP JSON-RPC purity).
    # Guarded: any failure must not crash the CLI (AC4).
    _is_mcp_cmd = len(sys.argv) >= 2 and sys.argv[1] == "mcp"
    if not _is_mcp_cmd:
        try:
            from superlocalmemory.cli._lazy_init import _ensure_initialized
            _ensure_initialized()
        except Exception:
            pass

    from superlocalmemory.cli.json_output import _get_version
    _ver = _get_version()

    # v3.6.13 (CRITICAL): the `mcp` stdio transport requires stdout to carry
    # ONLY JSON-RPC. The post-upgrade banner + migration notice below must NEVER
    # run on that path, or the first post-upgrade `slm mcp` launch pollutes
    # stdout and the MCP client (Claude Desktop / Cursor) rejects the stream as
    # "not valid JSON". Skip the whole block for `mcp`; the next human `slm`
    # command emits the banner (to stderr) and writes the version marker.
    _is_mcp_stdio = len(sys.argv) >= 2 and sys.argv[1] == "mcp"

    # One-time post-upgrade banner — silent for fresh installs and
    # same-version runs. Guarded against I/O errors internally.
    if not _is_mcp_stdio:
        from superlocalmemory.cli.version_banner import check_and_emit_upgrade_banner
        if check_and_emit_upgrade_banner(_ver):
            # First post-upgrade invocation: apply the data-dir migration if
            # it's safe. When the previous-version daemon is still running
            # we defer — the next daemon start picks it up.
            try:
                import logging as _logging
                from pathlib import Path as _P
                from superlocalmemory.migrations.v3_4_25_to_v3_4_26 import (
                    migrate_if_safe as _migrate_if_safe,
                )
                # WP-07: route through slm_home() so all 3 env aliases are honoured.
                from superlocalmemory.cli._lazy_init import slm_home as _slm_home
                _data = _slm_home()
                _res = _migrate_if_safe(_data)
                if _res.get("status") == "deferred":
                    print(
                        "  note: data migration deferred — the running SLM "
                        "daemon will apply it on its next restart.",
                        file=sys.stderr,
                    )
            except Exception as _mig_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "v3.4.26 migrate_if_safe failed: %s — run `slm doctor`", _mig_exc,
                )
                print(
                    "  note: data migration check failed — run `slm doctor` to diagnose.",
                    file=sys.stderr,
                )

    parser = argparse.ArgumentParser(
        prog="slm",
        description=f"SuperLocalMemory V3 ({_ver}) — AI agent memory with mathematical foundations",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"superlocalmemory {_ver}",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # -- Setup & Config ------------------------------------------------
    init_p = sub.add_parser("init", help="One-command setup: mode + hooks + IDE + warmup")
    init_p.add_argument(
        "--force", action="store_true", help="Re-run full setup even if already configured",
    )
    init_p.add_argument(
        "--gate", action="store_true",
        help="Enable PreToolUse gate (experimental — blocks tools until session_init)",
    )
    # WP-07: non-interactive auto setup (pip post-install, CI, scripts).
    init_p.add_argument(
        "--auto", action="store_true",
        help="Non-interactive setup: mode A + hooks (no TTY required, for CI/scripts)",
    )

    setup_p = sub.add_parser("setup", help="Interactive first-time setup wizard")
    setup_p.add_argument(
        "--auto", action="store_true",
        help="Non-interactive mode: use defaults (for CI/scripts)",
    )

    mode_p = sub.add_parser("mode", help="Get or set operating mode (a/b/c)")
    mode_p.add_argument(
        "value", nargs="?", choices=["a", "b", "c", "A", "B", "C"], help="Mode to set",
    )
    mode_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    provider_p = sub.add_parser("provider", help="Get or set LLM provider for Mode B/C")
    provider_p.add_argument(
        "action", nargs="?", choices=["set"], help="Action",
    )

    connect_p = sub.add_parser("connect", help="Auto-configure IDE integrations (17+ IDEs)")
    connect_p.add_argument("ide", nargs="?", help="Specific IDE to configure (e.g. cursor, codex, continue)")
    connect_p.add_argument(
        "--list", action="store_true", help="List all supported IDEs",
    )
    connect_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")
    # WP-08 CRIT-1: declare missing flags so cmd_connect can read them without getattr fallback
    connect_p.add_argument(
        "--here", action="store_true", default=False,
        help="Write config relative to current working directory (project scope)",
    )
    connect_p.add_argument(
        "--cross-platform", action="store_true", dest="cross_platform", default=False,
        help="Use LLD-05 cross-platform adapter orchestrator",
    )
    connect_p.add_argument(
        "--disable", metavar="ADAPTER", default=None,
        help="Disable a specific cross-platform adapter by name",
    )
    connect_p.add_argument(
        "--profile", metavar="PROFILE", default=None,
        help="Inject SLM_MCP_PROFILE env var into the MCP server block (WP-01)",
    )
    connect_p.add_argument(
        "--dry-run", action="store_true", dest="dry_run", default=False,
        help="Show what would be written without making changes",
    )

    migrate_p = sub.add_parser("migrate", help="Migrate data from V2 to V3 schema")
    migrate_p.add_argument(
        "--rollback", action="store_true", help="Rollback migration",
    )

    # LLD-06 §7.2 — `slm db migrate` wraps LLD-07's additive schema migrations.
    db_p = sub.add_parser("db", help="Database maintenance commands (v3.4.22)")
    db_sub = db_p.add_subparsers(dest="db_command", title="db subcommands")
    db_mig_p = db_sub.add_parser(
        "migrate",
        help="Apply additive schema migrations (LLD-07)",
    )
    db_mig_p.add_argument(
        "--status", action="store_true",
        help="Print migration status from migration_log; no writes",
    )
    db_mig_p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without applying",
    )

    # -- Memory Operations ---------------------------------------------
    remember_p = sub.add_parser("remember", help="Store a memory (extracts facts, builds graph)")
    remember_p.add_argument("content", help="Content to remember")
    remember_p.add_argument("--tags", default="", help="Comma-separated tags")
    remember_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")
    remember_p.add_argument(
        "--sync", dest="sync_mode", action="store_true",
        help="Wait for completion (default: async background processing)",
    )
    remember_p.add_argument(
        "--scope", default=None, choices=("personal", "shared", "global"),
        help="Memory scope: personal, shared, or global. Unset uses the "
             "configured default_scope (personal). Shared memory is opt-in.",
    )
    remember_p.add_argument(
        "--shared-with", default=None,
        help="Comma-separated profile IDs for shared scope",
    )

    # v3.6.12 (parity-3): `search` is an alias of `recall` so the CLI has the
    # same search verb the MCP exposes (handlers dict maps both to cmd_recall).
    recall_p = sub.add_parser("recall", aliases=["search"], help="Semantic search with 4-channel retrieval")
    recall_p.add_argument("query", help="Search query")
    recall_p.add_argument(
        "--limit", type=int, default=CANONICAL_RECALL_LIMIT,
        help=f"Max results (default {CANONICAL_RECALL_LIMIT})",
    )
    recall_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")
    recall_p.add_argument(
        "--fast", action="store_true",
        help="Skip SpreadingActivation 5th channel for sub-second response. "
             "Other 4 channels (semantic, lexical, temporal, structural) still run. "
             "Use when you need recall before a tool call (e.g. before WebSearch).",
    )
    # v3.6.15: shared memory is opt-in. Unset (None) → resolve the configured
    # default (recall_include_global/shared, both False by default). Explicit
    # flags override per-call. default=None on BOTH members of each pair so the
    # store_false's implicit default=True can't sneak back in.
    recall_p.add_argument(
        "--include-global", dest="include_global", action="store_true", default=None,
        help="Include global-scope facts in retrieval (opt-in; default off)",
    )
    recall_p.add_argument(
        "--no-global", dest="include_global", action="store_false", default=None,
        help="Exclude global-scope facts from retrieval",
    )
    recall_p.add_argument(
        "--include-shared", dest="include_shared", action="store_true", default=None,
        help="Include facts shared with this profile (opt-in; default off)",
    )
    recall_p.add_argument(
        "--no-shared", dest="include_shared", action="store_false", default=None,
        help="Exclude shared-scope facts from retrieval",
    )

    forget_p = sub.add_parser("forget", help="Delete memories matching a query (fuzzy)")
    forget_p.add_argument("query", help="Query to match for deletion")
    forget_p.add_argument("--dry-run", action="store_true", default=False, help="Preview matches without deleting")
    forget_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    forget_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    delete_p = sub.add_parser("delete", help="Delete a specific memory by ID (precise)")
    delete_p.add_argument("fact_id", help="Exact fact ID to delete")
    delete_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    delete_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    update_p = sub.add_parser("update", help="Edit the content of a specific memory by ID")
    update_p.add_argument("fact_id", help="Exact fact ID to update")
    update_p.add_argument("content", help="New content for the memory")
    update_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    list_p = sub.add_parser("list", help="List recent memories chronologically (shows IDs for delete/update)")
    list_p.add_argument(
        "--limit", "-n", type=int, default=20, help="Number of entries (default 20)",
    )
    list_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    # -- Diagnostics ---------------------------------------------------
    status_p = sub.add_parser("status", help="System status (mode, profile, DB size)")
    status_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")
    status_p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show extended status: migration log, daemon port, disabled marker, last version",
    )

    health_p = sub.add_parser("health", help="Math layer health (Fisher-Rao, Sheaf, Langevin)")
    health_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    trace_p = sub.add_parser("trace", help="Recall with per-channel score breakdown")
    trace_p.add_argument("query", help="Search query")
    trace_p.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    trace_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    # -- Diagnostics (continued) ----------------------------------------
    doctor_p = sub.add_parser("doctor", help="Pre-flight check: deps, embedding worker, connectivity")
    doctor_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")
    doctor_p.add_argument(
        "--quick", action="store_true",
        help="Run only the fast checks (deps + config); skip daemon/embedding probes",
    )

    # LLD-06 §6.6 — `slm wrap <agent> [args...]` activates the Optimize
    # proxy for a specific agent. Supported agents: claude, claude-settings,
    # codex, aider, cline, generic, etc. See optimize.adapters._agent_registry.
    wrap_p = sub.add_parser(
        "wrap",
        help="Activate Optimize proxy for a specific agent (claude, codex, aider, ...)",
    )
    wrap_p.add_argument(
        "--list", action="store_true",
        help="List all registered agents and their mechanisms",
    )
    wrap_p.add_argument(
        "--persistent", action="store_true",
        help="Persist env vars to the agent's config file (~/.claude/settings.json) instead of launching",
    )
    wrap_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the action that would be taken without executing it",
    )
    wrap_p.add_argument("agent", nargs="?", default=None, help="Agent key (run `slm wrap --list` to see all)")
    wrap_p.add_argument("agent_args", nargs=argparse.REMAINDER, help="Args passed to the agent binary")

    # -- Services ------------------------------------------------------
    sub.add_parser("mcp", help="Start MCP server (stdio transport for IDE integration)")
    sub.add_parser("warmup", help="Pre-download embedding model (~500MB, one-time)")

    dashboard_p = sub.add_parser("dashboard", help="Open 17-tab web dashboard")
    dashboard_p.add_argument(
        "--port", type=int, default=8765, help="Port (default 8765)",
    )

    # V3.3.21: Daemon serve mode
    serve_p = sub.add_parser("serve", help="Start/stop daemon for instant CLI response (~600MB RAM)")
    serve_p.add_argument(
        "action", nargs="?", default="start",
        choices=["start", "stop", "status", "install", "uninstall"],
        help="start (default), stop, status, install (OS service), uninstall",
    )

    # V3.4.9: Full system restart with health verification
    restart_p = sub.add_parser(
        "restart",
        help="Nuclear restart: kill orphans, clean state, start fresh, verify health",
    )
    restart_p.add_argument(
        "--dashboard", action="store_true",
        help="Open dashboard after restart",
    )
    restart_p.add_argument("--json", action="store_true", help="Output structured JSON")

    # -- Profiles ------------------------------------------------------
    profile_p = sub.add_parser("profile", help="Profile management (list/switch/create)")
    profile_p.add_argument(
        "action", choices=["list", "switch", "create"], help="Action",
    )
    profile_p.add_argument("name", nargs="?", help="Profile name")
    profile_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    # -- Active Memory (V3.1) ------------------------------------------
    hooks_p = sub.add_parser("hooks", help="Manage Claude Code hooks for auto memory injection")
    hooks_p.add_argument(
        "action", nargs="?", default="status",
        choices=["install", "remove", "status"], help="Action (default: status)",
    )
    hooks_p.add_argument(
        "--gate", action="store_true",
        help="Enable PreToolUse gate (experimental — blocks tools until session_init)",
    )

    ctx_p = sub.add_parser("session-context", help="Print session context (for hooks)")
    ctx_p.add_argument("query", nargs="?", default="", help="Optional context query")
    ctx_p.add_argument(
        "--max-age-days", type=int, default=30,
        help="Suppress memories older than N days unless score ≥ 0.7 (default: 30). "
             "Set 0 to disable age filter.",
    )
    ctx_p.add_argument(
        "--full", action="store_true",
        help="Use full engine path (slower, requires Ollama). Default is fast SQLite path.",
    )
    ctx_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    # #49: local session open/close (for hooks; no model roundtrip)
    session_p = sub.add_parser(
        "session", help="Open/close a session locally (for hooks; no model roundtrip)"
    )
    session_sub = session_p.add_subparsers(dest="session_command", title="session actions")
    sopen_p = session_sub.add_parser("open", help="Warm session context")
    sopen_p.add_argument("--project-path", default="", help="Project path to derive the warm query")
    sopen_p.add_argument("--query", default="", help="Explicit warm query")
    sopen_p.add_argument("--max-results", type=int, default=10, help="Max memories to warm (default 10)")
    sclose_p = session_sub.add_parser(
        "close", help="Close session, create temporal summaries"
    )
    sclose_p.add_argument(
        "--session-id", default="", help="Session to close (default: most recent)"
    )

    obs_p = sub.add_parser("observe", help="Auto-capture content (pipe or argument)")
    obs_p.add_argument("content", nargs="?", default="", help="Content to evaluate")

    # -- V3.3 Commands -------------------------------------------------
    decay_p = sub.add_parser("decay", help="Run Ebbinghaus forgetting decay cycle")
    decay_p.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Preview without applying (default)",
    )
    decay_p.add_argument(
        "--execute", dest="dry_run", action="store_false",
        help="Apply zone transitions",
    )
    decay_p.add_argument("--profile", default="", help="Target profile")
    decay_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    quantize_p = sub.add_parser("quantize", help="Run EAP embedding quantization cycle")
    quantize_p.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Preview without applying (default)",
    )
    quantize_p.add_argument(
        "--execute", dest="dry_run", action="store_false",
        help="Apply precision changes",
    )
    quantize_p.add_argument("--profile", default="", help="Target profile")
    quantize_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    consolidate_p = sub.add_parser("consolidate", help="Run memory consolidation pipeline")
    consolidate_p.add_argument(
        "--cognitive", action="store_true",
        help="Run CCQ cognitive consolidation",
    )
    consolidate_p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview without applying",
    )
    consolidate_p.add_argument("--profile", default="", help="Target profile")
    consolidate_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    sp_p = sub.add_parser("soft-prompts", help="List active soft prompts (auto-learned patterns)")
    sp_p.add_argument("--profile", default="", help="Target profile")
    sp_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    reap_p = sub.add_parser("reap", help="Find and kill orphaned SLM processes")
    reap_p.add_argument(
        "--force", action="store_true",
        help="Kill orphans (default: dry run only)",
    )
    reap_p.add_argument(
        "--all", action="store_true", dest="all",
        help="Kill ALL slm mcp processes regardless of orphan status (use after IDE switch)",
    )
    reap_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    # V3.4.3: Ingestion adapters
    adapters_p = sub.add_parser(
        "adapters",
        help="Manage ingestion adapters (Gmail, Calendar, Transcript)",
    )
    adapters_p.add_argument(
        "rest", nargs="*", default=[],
        help="Subcommand: list, enable, disable, start, stop, status [name]",
    )

    # V3.4.8: External observation ingestion
    ingest_p = sub.add_parser(
        "ingest",
        help="Import external observations (ECC, JSONL) into SLM learning",
    )
    ingest_p.add_argument(
        "--source", default="ecc",
        choices=["ecc", "jsonl"],
        help="Source type: ecc (Claude Code sessions), jsonl (generic)",
    )
    ingest_p.add_argument(
        "--file", default="",
        help="Specific file to ingest (auto-discovers if not set)",
    )
    ingest_p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Preview without writing",
    )
    ingest_p.add_argument("--json", action="store_true", help="Output structured JSON")

    # V3.4.11: Config get/set (dot-notation)
    config_p = sub.add_parser(
        "config",
        help="Get or set config values (e.g. slm config set evolution.enabled true)",
    )
    config_p.add_argument(
        "action", choices=["get", "set"], help="Action: get or set",
    )
    config_p.add_argument("key", help="Config key in dot notation (e.g. evolution.enabled)")
    config_p.add_argument("value", nargs="?", default=None, help="Value to set (for 'set' action)")
    config_p.add_argument("--json", action="store_true", help="Output structured JSON")

    # V3.4.11: Skill evolution (called from Stop hook, fire-and-forget)
    evolve_p = sub.add_parser(
        "evolve",
        help="Run post-session skill evolution (internal, called by Stop hook)",
    )
    evolve_p.add_argument("--session", default="", help="Session ID to process")
    evolve_p.add_argument("--profile", default="default", help="Profile ID")

    # v3.4.22 — MASTER-PLAN §8 escape hatches (Stage 8 SB-5).
    disable_p = sub.add_parser(
        "disable",
        help="Disable SLM globally (writes ~/.superlocalmemory/.disabled, stops daemon)",
    )
    disable_p.add_argument(
        "--reason", default="",
        help="Optional reason string written into the marker for audit",
    )

    sub.add_parser(
        "enable",
        help="Remove the .disabled marker; print command to start daemon",
    )

    sub.add_parser(
        "clear-cache",
        help="Wipe regenerable caches (memory.db + learning.db are preserved)",
    )

    recon_p = sub.add_parser(
        "reconfigure",
        help="Re-run the interactive postinstall (changes profile/knobs)",
    )
    recon_p.add_argument(
        "extras", nargs="*", default=[],
        help="Extra flags passed to postinstall-interactive.js",
    )

    bench_p = sub.add_parser(
        "benchmark",
        help="Run evo-memory benchmark against an isolated tmp DB (never touches user data)",
    )
    bench_p.add_argument(
        "--json", action="store_true",
        help="Emit JSON result instead of human-readable summary",
    )

    # S-M07: install-token rotation.
    sub.add_parser(
        "rotate-token",
        help="Rotate the SLM install token (run `slm restart` afterwards)",
    )

    # ---- SLM v3.6 Optimize subcommands (additive, never modify above) ----

    # slm optimize status|on|off|savings
    opt_p = sub.add_parser("optimize", help="Optimize module control (cache + compress)")
    opt_sub = opt_p.add_subparsers(dest="opt_command", title="optimize subcommands")
    opt_sub.add_parser("status", help="Show Optimize status")
    opt_sub.add_parser("on", help="Enable cache + compress")
    opt_sub.add_parser("off", help="Disable cache + compress")
    savings_p = opt_sub.add_parser("savings", help="Token/cost savings report")
    savings_p.add_argument("--since", type=int, default=7, help="Days to look back (default 7)")
    savings_p.add_argument("--provider", default=None,
                           choices=["anthropic", "openai", "gemini"],
                           help="Filter by provider")
    for _sp in opt_sub.choices.values():
        if not any(a.option_strings == ["--json"] for a in _sp._actions):
            _sp.add_argument("--json", action="store_true",
                             help="Output structured JSON (agent-native)")

    # slm cache status|clear|invalidate|ttl|semantic
    cache_p = sub.add_parser("cache", help="Cache control (TTL, clear, invalidate, semantic)")
    cache_sub = cache_p.add_subparsers(dest="cache_command", title="cache subcommands")
    cache_sub.add_parser("status", help="Show cache state")
    cache_sub.add_parser("clear", help="Delete all entries for tenant")
    cache_inv_p = cache_sub.add_parser("invalidate", help="Delete entries by tag")
    cache_inv_p.add_argument("--tag", required=True, help="Tag string to match")
    cache_ttl_p = cache_sub.add_parser("ttl", help="Set cache TTL in seconds")
    cache_ttl_p.add_argument("--set", dest="ttl_set", type=int, default=None,
                             help="Exact-cache TTL (seconds, >0)")
    cache_ttl_p.add_argument("--semantic", dest="ttl_semantic", type=int, default=None,
                             help="Semantic-cache TTL (seconds, >0)")
    cache_sem_p = cache_sub.add_parser("semantic", help="Enable/disable semantic cache")
    cache_sem_p.add_argument("semantic_value", choices=["on", "off"], help="on or off")
    for _sp in cache_sub.choices.values():
        if not any(a.option_strings == ["--json"] for a in _sp._actions):
            _sp.add_argument("--json", action="store_true",
                             help="Output structured JSON (agent-native)")
        if not any(a.option_strings == ["--tenant"] for a in _sp._actions):
            _sp.add_argument("--tenant", default="default", help="Tenant ID (default: 'default')")

    # slm compress status|mode|code|prose|ccr
    compress_p = sub.add_parser("compress", help="Compression control (mode, code, prose, CCR)")
    comp_sub = compress_p.add_subparsers(dest="compress_command", title="compress subcommands")
    comp_sub.add_parser("status", help="Show compression state")
    comp_mode_p = comp_sub.add_parser("mode", help="Set compression mode")
    comp_mode_p.add_argument("mode_value", choices=["safe", "aggressive"],
                             help="safe (default) or aggressive")
    comp_code_p = comp_sub.add_parser("code", help="Enable/disable code compression")
    comp_code_p.add_argument("code_value", choices=["on", "off"], help="on or off")
    comp_prose_p = comp_sub.add_parser("prose", help="Enable/disable prose compression")
    comp_prose_p.add_argument("prose_value", choices=["on", "off"], help="on or off")
    comp_ccr_p = comp_sub.add_parser("ccr", help="Enable/disable CCR")
    comp_ccr_p.add_argument("ccr_value", choices=["on", "off"], help="on or off")
    comp_align_p = comp_sub.add_parser("align", help="Enable/disable alignment compression")
    comp_align_p.add_argument("align_value", choices=["on", "off"], help="on or off")
    for _sp in comp_sub.choices.values():
        if not any(a.option_strings == ["--json"] for a in _sp._actions):
            _sp.add_argument("--json", action="store_true",
                             help="Output structured JSON (agent-native)")

    # slm proxy
    proxy_p = sub.add_parser("proxy", help="Start SLM optimization proxy (Anthropic + OpenAI)")
    proxy_p.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    proxy_p.add_argument("--provider", default="anthropic",
                         choices=["anthropic", "openai", "gemini"],
                         help="Target provider (default: anthropic)")
    proxy_p.add_argument("--no-compress", action="store_true", dest="no_compress",
                         help="Disable compression for this session")
    proxy_p.add_argument("--semantic", action="store_true",
                         help="Enable semantic cache for this session")
    proxy_p.add_argument("--json", action="store_true",
                         help="Output structured JSON (agent-native)")

    # slm help-optimize
    help_opt_p = sub.add_parser(
        "help-optimize",
        help="Full Optimize reference: subcommands + agent recipes + safety notes",
    )
    help_opt_p.add_argument(
        "topic", nargs="?", default=None,
        choices=["cache", "compress", "optimize", "proxy", "agents", "safety"],
        help="Topic to display (default: all)",
    )
    help_opt_p.add_argument(
        "--no-pager", action="store_true", dest="no_pager", default=False,
        help="Print to stdout instead of piping through a pager",
    )

    # ---- end SLM v3.6 Optimize subcommands ----

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # V3.3.19: Auto-trigger setup wizard on first use
    from superlocalmemory.cli.setup_wizard import check_first_use
    check_first_use(args.command)

    # V3.4.4: Auto-start daemon for all commands that need it.
    # SLM is always-on — close laptop, reboot, crash: daemon auto-recovers.
    # Cross-platform: macOS + Windows + Linux.
    _NO_DAEMON_COMMANDS = {
        "setup", "mode", "provider", "connect", "migrate", "mcp", "warmup",
        "config", "evolve", "db",
        # v3.4.22 escape hatches — never auto-start the daemon on these.
        "disable", "enable", "clear-cache", "reconfigure", "benchmark",
        "rotate-token",
        # LLD-06 — `slm wrap` may launch the agent binary directly without
        # needing the daemon running (the agent will start the daemon on
        # first LLM call, or the wrap command can be --dry-run).
        "wrap",
        # V3.6 Optimize commands that are config read/write only (no daemon needed)
        "optimize", "cache", "compress", "help-optimize",
        # NOTE: "proxy" NOT here — proxy needs daemon running
    }
    if args.command not in _NO_DAEMON_COMMANDS:
        try:
            from superlocalmemory.cli.daemon import ensure_daemon
            ensure_daemon()  # Starts daemon if not running; no-op if already up
        except Exception:
            pass  # Don't block CLI if daemon start fails — commands have fallbacks

    from superlocalmemory.cli.commands import dispatch

    dispatch(args)


if __name__ == "__main__":
    main()
