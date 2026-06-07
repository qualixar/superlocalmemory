"""Agent redirect registry — DATA ONLY, no logic.

PORT: 8765 (INTERFACE-CONTRACT §0). There is NO port 52415 anywhere.
"""

from __future__ import annotations

from typing import Any

AGENT_REGISTRY: dict[str, dict[str, Any]] = {
    "claude": {
        "binary": "claude",
        "mechanism": "env",
        "env_vars": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8765"},
        "settings_path": None,
        "protocol": "anthropic",
        "print_only": False,
        "help_text": (
            "Launches `claude` with ANTHROPIC_BASE_URL pointing at the SLM proxy.\n"
            "All Claude Code calls are intercepted: cache checked, response stored."
        ),
    },
    "claude-settings": {
        "binary": None,
        "mechanism": "settings-file",
        "settings_path": "~/.claude/settings.json",
        "env_vars": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8765"},
        "protocol": "anthropic",
        "print_only": False,
        "help_text": (
            "Writes ANTHROPIC_BASE_URL into ~/.claude/settings.json env block."
        ),
    },
    "antigravity": {
        "binary": "agy",
        "mechanism": "print-only",
        "protocol": "anthropic",
        "print_only": True,
        "help_text": (
            "Antigravity (agy) redirect must be verified at runtime.\n"
            "Run: agy --help | grep -i base\n"
            "Then export the appropriate env var manually."
        ),
    },
    "cline": {
        "binary": None,
        "mechanism": "config-file",
        "config_path": "{vscode_user_dir}/settings.json",
        "config_key": "cline.openAiApiBase",
        "config_value": "http://127.0.0.1:8765/v1",
        "protocol": "openai",
        "print_only": False,
        "help_text": (
            "Edits VS Code user settings to set cline.openAiApiBase."
        ),
    },
    "opencode": {
        "binary": "opencode",
        "mechanism": "print-only",
        "protocol": "openai",
        "print_only": True,
        "help_text": (
            "OpenCode redirect must be verified at runtime.\n"
            "Run: opencode --print-config"
        ),
    },
    "cursor": {
        "binary": "cursor",
        "mechanism": "print-only",
        "protocol": "openai",
        "print_only": True,
        "help_text": (
            "Cursor settings cannot be automated via env vars.\n"
            "Manual: Settings → Models → Override OpenAI Base URL → http://127.0.0.1:8765/v1"
        ),
    },
    "aider": {
        "binary": "aider",
        "mechanism": "env",
        "env_vars": {
            "OPENAI_API_BASE": "http://127.0.0.1:8765/v1",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
        },
        "protocol": "both",
        "print_only": False,
        "help_text": (
            "Launches `aider` with both OpenAI and Anthropic base URLs set."
        ),
    },
    "codex": {
        "binary": "codex",
        "mechanism": "env",
        "env_vars": {"OPENAI_BASE_URL": "http://127.0.0.1:8765/v1"},
        "protocol": "openai",
        "print_only": False,
        "help_text": (
            "Launches `codex` with OPENAI_BASE_URL pointing at the SLM proxy."
        ),
    },
    "copilot": {
        "binary": "copilot",
        "mechanism": "print-only",
        "protocol": "anthropic",
        "print_only": True,
        "help_text": (
            "Copilot redirect must be verified at runtime.\n"
            "Run: gh copilot --help | grep -i provider"
        ),
    },
    "generic": {
        "binary": None,
        "mechanism": "print-only",
        "protocol": "openai",
        "print_only": True,
        "help_text": (
            "Generic OpenAI-compatible client:\n"
            "  export OPENAI_BASE_URL=http://127.0.0.1:8765/v1"
        ),
    },
}
