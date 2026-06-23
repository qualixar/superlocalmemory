# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Claude Code hook integration — hybrid approach (v3.3.6).

CRITICAL PATH (gate + init-done): Shell built-ins only. Cannot crash.
VALUE-ADD (start, checkpoint, stop): Python via `slm hook <name>`,
  wrapped with `2>/dev/null || true` so errors are invisible.

Usage:
    slm hooks install       Install all hooks into Claude Code
    slm hooks remove        Remove SLM hooks from Claude Code
    slm hooks status        Check installation status
    slm init                Full setup including hooks

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
VERSION_DIR = Path.home() / ".superlocalmemory" / "hooks"
VERSION_FILE = VERSION_DIR / ".version"
DISABLED_FILE = VERSION_DIR / ".hooks-disabled"
HOOKS_VERSION = "3.6.18"

# Cross-platform temp dir and marker paths
_TMP = tempfile.gettempdir()
_MARKER = f"{_TMP}/slm-session-initialized"
_START_MARKER = f"{_TMP}/slm-session-start-time"

# Tools that the gate should block (everything except SLM/ToolSearch)
_GATED_TOOLS = "Bash|Read|Write|Edit|Glob|Grep|Agent|WebFetch|WebSearch|NotebookEdit"

# ---------------------------------------------------------------------------
# Platform-specific gate commands (shell built-ins only — CANNOT crash)
# ---------------------------------------------------------------------------

def _gate_cmd() -> str:
    """Gate command: pure shell, no Python, ~1ms.

    Logic: if initialized → allow. If no session started → allow. Else → block.
    Uses specific matcher to exclude SLM tools, so no stdin parsing needed.
    """
    if sys.platform == "win32":
        marker_win = _MARKER.replace("/", "\\")
        start_win = _START_MARKER.replace("/", "\\")
        return (
            f'cmd /c "if exist {marker_win} (exit /b 0)'
            f' else if not exist {start_win} (exit /b 0)'
            f' else (echo [SLM] Call mcp__superlocalmemory__session_init first & exit /b 2)"'
        )
    return (
        f"test -f {_MARKER}"
        f" || test ! -f {_START_MARKER}"
        " || { echo '[SLM] Call mcp__superlocalmemory__session_init first'; exit 2; }"
    )


def _init_done_cmd() -> str:
    """Init-done command: pure shell touch, ~1ms."""
    if sys.platform == "win32":
        return f'cmd /c "echo.>{_MARKER.replace("/", chr(92))}"'
    return f"touch {_MARKER}"


def _wrap_python_cmd(hook_name: str) -> str:
    """Wrap a Python hook with error absorption. Any crash → invisible."""
    if sys.platform == "win32":
        return f'cmd /c "slm hook {hook_name} 2>NUL || exit /b 0"'
    return f"slm hook {hook_name} 2>/dev/null || true"


# ---------------------------------------------------------------------------
# Hook definitions for settings.json
# ---------------------------------------------------------------------------

def _hook_definitions(include_gate: bool = False) -> dict[str, list]:
    """Build Claude Code hook entries.

    Critical path (gate, init-done): Shell built-ins. Cannot crash.
    Value-add (start, checkpoint, stop): Python with error wrapper.
    """
    defs: dict[str, list] = {
        "SessionStart": [
            # v3.6.18: mandate fires FIRST so session_init is never skipped.
            # mcp__superlocalmemory__session_init is DEFERRED — Claude must call
            # ToolSearch before it can invoke the tool. Without this hook,
            # Claude responds before loading the schema → no 6-channel memory.
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("mandate"),
                        "timeout": 5000,
                    }
                ]
            },
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("start"),
                        "timeout": 15000,
                    }
                ]
            },
        ],
        "PostToolUse": [
            {
                "matcher": "Write|Edit",
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("checkpoint"),
                        "timeout": 5000,
                    }
                ],
            },
            # LLD-09 Track A.2 — outcome-population on every tool use.
            # Matches all host tools (SLM MCP tools are excluded via
            # matcher negation below; actual filtering happens in the
            # hook by checking tool_name against SLM's own prefixes).
            {
                "matcher": _GATED_TOOLS,
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("post_tool_outcome"),
                        "timeout": 5000,
                    }
                ],
            },
        ],
        "UserPromptSubmit": [
            # LLD-09 Track A.2 — re-query detection.
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("user_prompt_rehash"),
                        "timeout": 5000,
                    }
                ]
            },
            # v3.4.43 — event-based topic-shift detection. Fires a one-line
            # recall reminder ONLY when the current prompt's content-word set
            # has zero overlap with every prompt in a 5-turn sliding window.
            # Replaces the time-based 15/30-min recall nag previously emitted
            # by _hook_checkpoint. Algorithm + state file are documented in
            # superlocalmemory/hooks/topic_shift_hook.py.
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("topic_shift"),
                        "timeout": 3000,
                    }
                ]
            },
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("stop"),
                        "timeout": 10000,
                    },
                    # LLD-09 Track A.2 — finalize pending outcomes at end.
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("stop_outcome"),
                        "timeout": 10000,
                    },
                ]
            }
        ],
    }

    # v3.4.43 — default PreToolUse entry: pre-web recall on WebSearch/WebFetch.
    # Fires `slm hook before_web` which runs a 4-channel recall on the search
    # query/URL and injects results as a system-reminder BEFORE the web call.
    # Encourages Claude to consider local memories before paying for new web
    # research. Independent of `include_gate` — this is value-add, not gating.
    defs["PreToolUse"] = [
        {
            "matcher": "WebSearch|WebFetch",
            "hooks": [
                {
                    "type": "command",
                    "command": _wrap_python_cmd("before_web"),
                    "timeout": 5000,
                }
            ],
        }
    ]

    if include_gate:
        defs["PreToolUse"].insert(0, {
            "matcher": _GATED_TOOLS,
            "hooks": [
                {
                    "type": "command",
                    "command": _gate_cmd(),
                    "timeout": 500,
                }
            ],
        })
        defs["PostToolUse"].insert(0, {
            "matcher": "mcp__superlocalmemory__session_init",
            "hooks": [
                {
                    "type": "command",
                    "command": _init_done_cmd(),
                    "timeout": 500,
                }
            ],
        })

    return defs


# ---------------------------------------------------------------------------
# Identify SLM hooks in existing settings
# ---------------------------------------------------------------------------

def _is_slm_hook_entry(entry: dict) -> bool:
    """Check if a hook entry belongs to SLM."""
    for hook in entry.get("hooks", []):
        cmd = hook.get("command", "")
        if ("slm hook" in cmd
                or "slm-session" in cmd
                or ".superlocalmemory/hooks/" in cmd
                or "slm-session-initialized" in cmd):
            return True
    return False


# ---------------------------------------------------------------------------
# Safe settings.json merge / removal
# ---------------------------------------------------------------------------

def _merge_hooks(settings: dict, hook_defs: dict) -> dict:
    """Merge SLM hooks into settings, preserving all non-SLM hooks."""
    if "hooks" not in settings:
        settings["hooks"] = {}

    for hook_type, slm_entries in hook_defs.items():
        existing = settings["hooks"].get(hook_type, [])
        cleaned = [e for e in existing if not _is_slm_hook_entry(e)]
        cleaned.extend(slm_entries)
        settings["hooks"][hook_type] = cleaned

    return settings


def _remove_slm_hooks(settings: dict) -> dict:
    """Remove all SLM hook entries, preserve non-SLM hooks."""
    hooks = settings.get("hooks", {})
    for hook_type in list(hooks.keys()):
        cleaned = [e for e in hooks[hook_type] if not _is_slm_hook_entry(e)]
        if cleaned:
            hooks[hook_type] = cleaned
        else:
            del hooks[hook_type]
    if not hooks and "hooks" in settings:
        del settings["hooks"]
    return settings


def _read_settings() -> dict:
    """Read Claude Code settings.json, return empty dict if missing.

    Raises:
        json.JSONDecodeError: propagated as-is when the file exists but is
            malformed, so callers can distinguish 'missing' from 'corrupt'.
    """
    if CLAUDE_SETTINGS.exists():
        return json.loads(CLAUDE_SETTINGS.read_text())
    return {}


def _write_settings(settings: dict) -> None:
    """Write settings.json atomically — tmp file + rename.

    Direct .write_text() would truncate the file on a crash mid-write,
    destroying the user's entire Claude Code configuration. The tmp-then-rename
    pattern is atomic on POSIX (os.replace) and near-atomic on Windows: either
    the full new content lands or the original file is untouched.

    Never overwrites non-SLM settings — _merge_hooks() guarantees that only
    the SLM hooks entries change; all other keys are preserved from the read.
    """
    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(settings, indent=2) + "\n"
    # Write to a sibling tmp file in the same directory so os.replace is atomic
    # (cross-device rename would fail; same-dir rename is guaranteed atomic).
    tmp_path = CLAUDE_SETTINGS.with_suffix(".json.slm_tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, CLAUDE_SETTINGS)
    except Exception:
        # Best-effort cleanup of the tmp file on failure
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def install_hooks(include_gate: bool = False) -> dict:
    """Install SLM hooks into Claude Code settings.json.

    Critical path uses shell built-ins (cannot crash).
    Value-add uses Python with error wrappers (crashes invisible).
    Never overwrites non-SLM hooks.
    Clears .hooks-disabled marker (explicit install = user wants hooks).
    """
    result = {
        "success": False, "errors": [],
        "hooks_added": [], "gate_enabled": include_gate,
    }

    try:
        settings = _read_settings()
        hook_defs = _hook_definitions(include_gate=include_gate)
        settings = _merge_hooks(settings, hook_defs)
        _write_settings(settings)
        result["hooks_added"] = list(hook_defs.keys())
        result["success"] = True
    except Exception as exc:
        result["errors"].append(f"Settings update failed: {exc}")

    try:
        VERSION_DIR.mkdir(parents=True, exist_ok=True)
        VERSION_FILE.write_text(HOOKS_VERSION)
        # Clear disabled marker — explicit install means user wants hooks
        if DISABLED_FILE.exists():
            DISABLED_FILE.unlink()
    except Exception as exc:
        result["errors"].append(f"Version file failed: {exc}")

    return result


def remove_hooks() -> dict:
    """Remove all SLM hooks from Claude Code settings.json.

    Writes a .hooks-disabled marker so auto-install paths respect
    the user's explicit choice. Cleared by explicit `install_hooks()`.
    """
    result = {"success": False, "errors": []}

    try:
        settings = _read_settings()
        settings = _remove_slm_hooks(settings)
        _write_settings(settings)
        result["success"] = True
    except Exception as exc:
        result["errors"].append(f"Settings cleanup failed: {exc}")

    try:
        if VERSION_FILE.exists():
            VERSION_FILE.unlink()
        # Mark as explicitly disabled — auto-install will respect this
        VERSION_DIR.mkdir(parents=True, exist_ok=True)
        DISABLED_FILE.write_text("removed by user\n")
    except Exception:
        pass

    return result


def check_status() -> dict:
    """Check SLM hook installation status."""
    installed_version = ""
    if VERSION_FILE.exists():
        try:
            installed_version = VERSION_FILE.read_text().strip()
        except Exception:
            pass

    hook_types_found: list[str] = []
    has_gate = False
    parse_error: str | None = None

    try:
        settings = _read_settings()
        for hook_type, entries in settings.get("hooks", {}).items():
            if any(_is_slm_hook_entry(e) for e in entries):
                hook_types_found.append(hook_type)
        # v3.4.43: PreToolUse always has the before_web entry by default.
        # `has_gate` should be True only when the _GATED_TOOLS firewall
        # entry is present, NOT merely when any SLM PreToolUse entry exists.
        for entry in settings.get("hooks", {}).get("PreToolUse", []):
            if not _is_slm_hook_entry(entry):
                continue
            for hook in entry.get("hooks", []):
                if "Call mcp__superlocalmemory__session_init first" in hook.get("command", ""):
                    has_gate = True
                    break
            if has_gate:
                break
    except json.JSONDecodeError as exc:
        # File exists but is malformed — report indeterminate state so callers
        # do not conflate 'corrupt' with 'not installed'. Claude Code reads the
        # file independently; hooks may still be firing even though we cannot
        # parse the file here (split-brain scenario).
        parse_error = f"JSON parse error in settings.json: {exc}"
        logger.warning("SLM: %s", parse_error)
    except Exception:
        pass

    # Three-valued installed:
    #   True  — enough SLM hook types found (≥3)
    #   False — file missing or parseable but no SLM hooks
    #   None  — file exists but is corrupt (indeterminate)
    if parse_error is not None:
        installed: bool | None = None
    else:
        installed = len(hook_types_found) >= 3

    result: dict = {
        "installed": installed,
        "version": installed_version,
        "latest_version": HOOKS_VERSION,
        "needs_upgrade": bool(installed_version and installed_version != HOOKS_VERSION),
        "hook_types": hook_types_found,
        "gate_enabled": has_gate,
    }
    if parse_error is not None:
        result["error"] = parse_error
    return result


def upgrade_hooks() -> dict:
    """Upgrade existing hooks to current version. Non-interactive."""
    status = check_status()

    if not status["installed"] and not status["version"]:
        return {"upgraded": False, "reason": "No hooks installed"}

    include_gate = status["gate_enabled"]
    result = install_hooks(include_gate=include_gate)
    result["upgraded"] = result["success"]
    result["from_version"] = status["version"]
    result["to_version"] = HOOKS_VERSION
    return result


def auto_install_if_needed() -> dict | None:
    """Auto-install hooks if not present and not explicitly disabled.

    Called from MCP server startup and npm postinstall.
    Returns install result, or None if skipped.

    Fast path: version file exists and matches → ~0.1ms, returns None.
    """
    try:
        # Respect explicit opt-out
        if DISABLED_FILE.exists():
            return None

        # Already installed and current → skip
        if VERSION_FILE.exists():
            installed = VERSION_FILE.read_text().strip()
            if installed == HOOKS_VERSION:
                return None

        # Install with clear message
        result = install_hooks(include_gate=False)
        if result["success"]:
            logger.info(
                "SLM: Hooks installed into Claude Code (slm hooks remove to undo)"
            )
        return result
    except Exception as exc:
        logger.debug("Auto-install check failed: %s", exc)
        return None


def auto_upgrade_check() -> None:
    """Silent auto-upgrade on version mismatch. ~0.1ms when current."""
    try:
        if not VERSION_FILE.exists():
            legacy_script = VERSION_DIR / "slm-session-start.sh"
            if legacy_script.exists():
                _migrate_legacy_hooks()
            return

        installed = VERSION_FILE.read_text().strip()
        if installed == HOOKS_VERSION:
            return

        result = upgrade_hooks()
        if result.get("upgraded"):
            logger.info("SLM hooks upgraded %s -> %s", installed, HOOKS_VERSION)
    except Exception as exc:
        logger.debug("Hook auto-upgrade failed: %s", exc)


def _migrate_legacy_hooks() -> None:
    """Migrate from bash-script hooks (pre-3.3.6) to hybrid hooks."""
    try:
        settings = _read_settings()
        has_legacy = False
        for entries in settings.get("hooks", {}).values():
            for e in entries:
                for h in e.get("hooks", []):
                    if ".superlocalmemory/hooks/" in h.get("command", ""):
                        has_legacy = True
                        break

        if has_legacy:
            settings = _remove_slm_hooks(settings)
            hook_defs = _hook_definitions(include_gate=False)
            settings = _merge_hooks(settings, hook_defs)
            _write_settings(settings)
            VERSION_DIR.mkdir(parents=True, exist_ok=True)
            VERSION_FILE.write_text(HOOKS_VERSION)
            logger.info("Migrated legacy bash hooks to hybrid hooks (v%s)", HOOKS_VERSION)
    except Exception as exc:
        logger.debug("Legacy hook migration failed: %s", exc)
