# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Handler for ``slm help-optimize``."""

from __future__ import annotations

import sys
from argparse import Namespace

_HELP_SECTIONS: dict[str, str] = {
    "header": """\
SLM v3.6 Optimize — Developer Reference
========================================
Every feature is configurable via CLI. UI provides the same controls for
non-technical users. All CLI commands write config at runtime — daemon
hot-reloads within 2 seconds. No restart required.
""",
    "optimize": """\
slm optimize — Optimize module control
---------------------------------------
  slm optimize status              Show all Optimize settings
  slm optimize on                  Enable cache + compress
  slm optimize off                 Disable cache + compress (proxy keeps running)
  slm optimize savings [--since N] Token/cost savings report (default: 7 days)
    --since N                      Days to look back (integer, default 7)
    --provider P                   Filter by: anthropic|openai|gemini
  All: --json                      Machine-readable JSON output
""",
    "cache": """\
slm cache — Cache control
--------------------------
  slm cache status                 Entry count, DB size, TTLs, hit rate
  slm cache clear                  Delete all entries for this tenant
  slm cache invalidate --tag <t>   Delete entries whose tag array contains <t>
  slm cache ttl --set <s>          Set exact-cache TTL (seconds)
           --semantic <s>          Set semantic-cache TTL (seconds)
  slm cache semantic on|off        Enable/disable semantic cache
  All: --json  --tenant <id>

  TTL examples:
    slm cache ttl --set 3600        # 1 hour exact cache
    slm cache ttl --set 86400       # 24 hour exact cache (default)
    slm cache ttl --semantic 7200   # 2 hour semantic cache
""",
    "compress": """\
slm compress — Compression control
-----------------------------------
  slm compress status              Show compression mode, per-channel state
  slm compress mode safe|aggressive  Set compression aggressiveness
  slm compress code on|off         Enable/disable code/JSON compression
  slm compress prose on|off        Enable/disable prose compression
  slm compress ccr on|off          Enable/disable Compressed Context Retrieval
  All: --json
""",
    "safety": """\
COMPRESSION SAFETY WARNING
---------------------------
  Safe mode (default):
    - Code/JSON compression: extractive, structure-preserving, lossless.
    - Prose compression: DISABLED in safe mode. Enable manually.
    - CCR: DISABLED by default. Originals stored for reversible retrieval.
    - This mode is production-safe for all use cases.

  Aggressive mode:
    !!  RISK: May reduce output fidelity. Use with caution.  !!
    - Prose compression: LLMLingua-2-style extractive summarization.
    - May omit nuance, hedges, or low-salience context.
    - DO NOT use for:
        - Code generation or code review
        - Legal, compliance, or regulatory text
        - Math, formulas, or structured data generation
        - Any task requiring exact reproduction of input
    - Suitable for:
        - Open-ended brainstorming
        - Summarization of long documents
        - Casual conversation and exploration
    - To revert: slm compress mode safe
""",
    "proxy": """\
slm proxy — Optimization proxy
--------------------------------
  slm proxy [--port P] [--provider anthropic|openai|gemini]
            [--no-compress] [--semantic] [--json]

  Starts the SLM proxy (or reports existing). Proxy intercepts LLM calls,
  applies cache lookup, and optionally compresses context before forwarding.

  Default port: 8765
  Anthropic surface:  http://127.0.0.1:8765
  OpenAI surface:     http://127.0.0.1:8765/v1
""",
    "agents": """\
PER-AGENT SETUP RECIPES
------------------------

--- Claude Code (ANTHROPIC_BASE_URL) ---
  The simplest redirect. Set this env var before launching Claude Code.

  Option A — environment variable (per-session):
    export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
    claude                        # Claude Code picks up the env var

  Option B — slm wrap (one command, recommended):
    slm wrap claude               # starts proxy + sets env + launches claude

  Option C — permanent (add to ~/.zshrc or ~/.bashrc):
    echo 'export ANTHROPIC_BASE_URL=http://127.0.0.1:8765' >> ~/.zshrc
    source ~/.zshrc

  Verify: slm optimize status  ->  look for "proxy: running on :8765"

--- Antigravity (config.toml base_url) ---
  Antigravity reads base_url from its config.toml.
  Default config location: ~/.config/antigravity/config.toml

  Add or update this line under the [api] section:
    [api]
    base_url = "http://127.0.0.1:8765"

  Or use slm wrap (if supported in your version):
    slm wrap antigravity

  Verify: antigravity --debug  ->  look for "base_url: http://127.0.0.1:8765"

--- Generic OpenAI-compatible clients (Cline, Cursor, Aider, OpenCode) ---
  Any client that accepts an OpenAI base_url can use the /v1 surface:
    base_url = http://127.0.0.1:8765/v1
    api_key  = (use your real provider key — proxy forwards it)

  Aider:
    aider --openai-api-base http://127.0.0.1:8765/v1

  Cline (VS Code settings.json):
    "cline.openAiBaseUrl": "http://127.0.0.1:8765/v1"

  Cursor (Settings > Models > Base URL):
    http://127.0.0.1:8765/v1

  OpenCode / other CLI tools:
    OPENAI_BASE_URL=http://127.0.0.1:8765/v1 opencode

  Or use slm wrap:
    slm wrap aider [-- <aider args>]
    slm wrap cursor
    slm wrap opencode

--- SDK adapters (Python) ---
  from superlocalmemory.optimize.adapters.openai_adapter import withSLM
  from openai import OpenAI
  client = withSLM(OpenAI())        # same interface as OpenAI() — zero API change

  from superlocalmemory.optimize.adapters.anthropic_adapter import withSLM
  from anthropic import Anthropic
  client = withSLM(Anthropic())

  NOTE: withSLM() is a pass-through when optimize is OFF.
  No behavioral change until you run: slm optimize on
""",
    "footer": """\
More commands:
  slm help-optimize cache      Cache subcommand reference
  slm help-optimize compress   Compress reference + safety warning
  slm help-optimize agents     Per-agent setup recipes
  slm help-optimize safety     Compression safety warning only

Documentation: https://superlocalmemory.com
GitHub:        https://github.com/qualixar/superlocalmemory
AI Reliability Engineering by @varunPbhardwaj — https://qualixar.com
""",
}


def cmd_help_optimize(args: Namespace) -> None:
    """Print the full slm help-optimize page or a topic-specific section."""
    topic = getattr(args, "topic", None)
    no_pager = getattr(args, "no_pager", False)

    if topic and topic not in _HELP_SECTIONS:
        print(f"Unknown topic: {topic}."
              f" Topics: {' '.join(k for k in _HELP_SECTIONS if k != 'footer')}")
        sys.exit(1)

    if topic:
        text = _HELP_SECTIONS[topic]
    else:
        text = "\n".join(_HELP_SECTIONS.values())

    if no_pager or not sys.stdout.isatty():
        print(text)
    else:
        try:
            import pydoc
            pydoc.pager(text)
        except Exception:
            print(text)
