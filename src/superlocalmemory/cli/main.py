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

    from superlocalmemory.cli.json_output import _get_version
    _ver = _get_version()

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

    setup_p = sub.add_parser("setup", help="Interactive first-time setup wizard")
    setup_p.add_argument(
        "--auto", action="store_true",
        help="Non-interactive mode: use defaults (for CI/scripts)",
    )

    mode_p = sub.add_parser("mode", help="Get or set operating mode (a/b/c)")
    mode_p.add_argument(
        "value", nargs="?", choices=["a", "b", "c"], help="Mode to set",
    )
    mode_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

    provider_p = sub.add_parser("provider", help="Get or set LLM provider for Mode B/C")
    provider_p.add_argument(
        "action", nargs="?", choices=["set"], help="Action",
    )

    connect_p = sub.add_parser("connect", help="Auto-configure IDE integrations (17+ IDEs)")
    connect_p.add_argument("ide", nargs="?", help="Specific IDE to configure")
    connect_p.add_argument(
        "--list", action="store_true", help="List all supported IDEs",
    )
    connect_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

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

    recall_p = sub.add_parser("recall", help="Semantic search with 4-channel retrieval")
    recall_p.add_argument("query", help="Search query")
    recall_p.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    recall_p.add_argument("--json", action="store_true", help="Output structured JSON (agent-native)")

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
