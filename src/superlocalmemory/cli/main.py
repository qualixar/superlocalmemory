# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
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

documentation:
  Website:    https://superlocalmemory.com
  GitHub:     https://github.com/qualixar/superlocalmemory
  Paper:      https://arxiv.org/abs/2603.14588
"""


def main() -> None:
    """Parse CLI arguments and dispatch to command handlers."""
    try:
        from importlib.metadata import version as _pkg_version
        _ver = _pkg_version("superlocalmemory")
    except Exception:
        _ver = "unknown"

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
    sub.add_parser("setup", help="Interactive first-time setup wizard")

    mode_p = sub.add_parser("mode", help="Get or set operating mode (a/b/c)")
    mode_p.add_argument(
        "value", nargs="?", choices=["a", "b", "c"], help="Mode to set",
    )

    provider_p = sub.add_parser("provider", help="Get or set LLM provider for Mode B/C")
    provider_p.add_argument(
        "action", nargs="?", choices=["set"], help="Action",
    )

    connect_p = sub.add_parser("connect", help="Auto-configure IDE integrations (17+ IDEs)")
    connect_p.add_argument("ide", nargs="?", help="Specific IDE to configure")
    connect_p.add_argument(
        "--list", action="store_true", help="List all supported IDEs",
    )

    migrate_p = sub.add_parser("migrate", help="Migrate data from V2 to V3 schema")
    migrate_p.add_argument(
        "--rollback", action="store_true", help="Rollback migration",
    )

    # -- Memory Operations ---------------------------------------------
    remember_p = sub.add_parser("remember", help="Store a memory (extracts facts, builds graph)")
    remember_p.add_argument("content", help="Content to remember")
    remember_p.add_argument("--tags", default="", help="Comma-separated tags")

    recall_p = sub.add_parser("recall", help="Semantic search with 4-channel retrieval")
    recall_p.add_argument("query", help="Search query")
    recall_p.add_argument("--limit", type=int, default=10, help="Max results (default 10)")

    forget_p = sub.add_parser("forget", help="Delete memories matching a query")
    forget_p.add_argument("query", help="Query to match for deletion")

    list_p = sub.add_parser("list", help="List recent memories chronologically")
    list_p.add_argument(
        "--limit", "-n", type=int, default=20, help="Number of entries (default 20)",
    )

    # -- Diagnostics ---------------------------------------------------
    sub.add_parser("status", help="System status (mode, profile, DB size)")
    sub.add_parser("health", help="Math layer health (Fisher-Rao, Sheaf, Langevin)")

    trace_p = sub.add_parser("trace", help="Recall with per-channel score breakdown")
    trace_p.add_argument("query", help="Search query")

    # -- Services ------------------------------------------------------
    sub.add_parser("mcp", help="Start MCP server (stdio transport for IDE integration)")
    sub.add_parser("warmup", help="Pre-download embedding model (~500MB, one-time)")

    dashboard_p = sub.add_parser("dashboard", help="Open 17-tab web dashboard")
    dashboard_p.add_argument(
        "--port", type=int, default=8765, help="Port (default 8765)",
    )

    # -- Profiles ------------------------------------------------------
    profile_p = sub.add_parser("profile", help="Profile management (list/switch/create)")
    profile_p.add_argument(
        "action", choices=["list", "switch", "create"], help="Action",
    )
    profile_p.add_argument("name", nargs="?", help="Profile name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    from superlocalmemory.cli.commands import dispatch

    dispatch(args)


if __name__ == "__main__":
    main()
