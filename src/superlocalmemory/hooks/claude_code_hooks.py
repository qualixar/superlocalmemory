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

import hashlib
import json
import logging
import os
import shlex
import sys
import tempfile
from pathlib import Path

from superlocalmemory import __version__
from superlocalmemory.infra.data_root import canonical_data_root
from superlocalmemory.infra.data_root import state_path as runtime_state_path

logger = logging.getLogger(__name__)

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
_DEFAULT_VERSION_DIR = runtime_state_path("hooks")
_DEFAULT_VERSION_FILE = _DEFAULT_VERSION_DIR / ".version"
_DEFAULT_DISABLED_FILE = _DEFAULT_VERSION_DIR / ".hooks-disabled"
VERSION_DIR = _DEFAULT_VERSION_DIR
VERSION_FILE = _DEFAULT_VERSION_FILE
DISABLED_FILE = _DEFAULT_DISABLED_FILE
HOOKS_VERSION = __version__

# Cross-platform temp dir and backwards-compatible marker overrides. Runtime
# defaults are root-namespaced and resolved when hook definitions are built.
_TMP = tempfile.gettempdir()
_DEFAULT_MARKER = os.path.join(_TMP, "slm-session-initialized")
_DEFAULT_START_MARKER = os.path.join(_TMP, "slm-session-start-time")
_MARKER = _DEFAULT_MARKER
_START_MARKER = _DEFAULT_START_MARKER

# Tools that the gate should block (everything except SLM/ToolSearch)
_GATED_TOOLS = "Bash|Read|Write|Edit|Glob|Grep|Agent|WebFetch|WebSearch|NotebookEdit"

# 4.2.30 cue hook: tool cues before the tool, error cues after. The same matchers
# as plugin-src/hooks/hooks.json (tests/test_hooks/test_host_hook_upgrade.py).
_CUE_BEFORE_TOOLS = "Write|Edit|MultiEdit|Bash"
_CUE_AFTER_TOOLS = "Write|Edit|MultiEdit|NotebookEdit|Bash"


def _version_dir() -> Path:
    """Resolve hook metadata under the selected data root.

    Assigning ``VERSION_DIR`` remains a supported test/embedder override.
    """
    if VERSION_DIR != _DEFAULT_VERSION_DIR:
        return VERSION_DIR
    return runtime_state_path("hooks")


def _version_file() -> Path:
    if VERSION_FILE != _DEFAULT_VERSION_FILE:
        return VERSION_FILE
    return _version_dir() / ".version"


def _disabled_file() -> Path:
    if DISABLED_FILE != _DEFAULT_DISABLED_FILE:
        return DISABLED_FILE
    return _version_dir() / ".hooks-disabled"


def _root_namespace() -> str:
    return hashlib.sha256(
        str(canonical_data_root()).encode("utf-8")
    ).hexdigest()[:16]


def _marker_path() -> str:
    if _MARKER != _DEFAULT_MARKER:
        return _MARKER
    return os.path.join(_TMP, f"slm-session-initialized-{_root_namespace()}")


def _start_marker_path() -> str:
    if _START_MARKER != _DEFAULT_START_MARKER:
        return _START_MARKER
    return os.path.join(_TMP, f"slm-session-start-time-{_root_namespace()}")

# ---------------------------------------------------------------------------
# Platform-specific gate commands (shell built-ins only — CANNOT crash)
# ---------------------------------------------------------------------------

def _gate_cmd() -> str:
    """Gate command: pure shell, no Python, ~1ms.

    Logic: if initialized → allow. If no session started → allow. Else → block.
    Uses specific matcher to exclude SLM tools, so no stdin parsing needed.
    """
    marker = _marker_path()
    start_marker = _start_marker_path()
    if sys.platform == "win32":
        marker_win = marker.replace("/", "\\")
        start_win = start_marker.replace("/", "\\")
        return (
            f'cmd /c "if exist "{marker_win}" (exit /b 0)'
            f' else if not exist "{start_win}" (exit /b 0)'
            f' else (echo [SLM] Call mcp__superlocalmemory__session_init first & exit /b 2)"'
        )
    return (
        f"test -f {shlex.quote(marker)}"
        f" || test ! -f {shlex.quote(start_marker)}"
        " || { echo '[SLM] Call mcp__superlocalmemory__session_init first'; exit 2; }"
    )


def _init_done_cmd() -> str:
    """Init-done command: pure shell touch, ~1ms."""
    marker = _marker_path()
    if sys.platform == "win32":
        marker_win = marker.replace("/", "\\")
        return f'cmd /c "echo.>"{marker_win}""'
    return f"touch {shlex.quote(marker)}"


def _wrap_python_cmd(hook_name: str) -> str:
    """Wrap a Python hook with error absorption. Any crash → invisible."""
    marker = _marker_path()
    start_marker = _start_marker_path()
    if sys.platform == "win32":
        marker_win = marker.replace("/", "\\")
        start_win = start_marker.replace("/", "\\")
        if hook_name == "start":
            return (
                f'cmd /c "slm hook start 2>NUL & echo.>"{start_win}"'
                ' & exit /b 0"'
            )
        if hook_name == "stop":
            return (
                f'cmd /c "slm hook stop 2>NUL & del /q "{marker_win}"'
                f' "{start_win}" 2>NUL & exit /b 0"'
            )
        return f'cmd /c "slm hook {hook_name} 2>NUL || exit /b 0"'

    command = f"slm hook {hook_name} 2>/dev/null || true"
    if hook_name == "start":
        return f"{command}; touch {shlex.quote(start_marker)}"
    if hook_name == "stop":
        return (
            f"{command}; rm -f {shlex.quote(marker)} "
            f"{shlex.quote(start_marker)}"
        )
    return command


# ---------------------------------------------------------------------------
# Hook definitions for settings.json
# ---------------------------------------------------------------------------

def _hook_definitions(include_gate: bool = False) -> dict[str, list]:
    """Build Claude Code hook entries.

    ``timeout`` is in SECONDS (Claude Code hooks spec). Before 4.1.22 these
    were written as milliseconds, so "15000" gave a hook about four hours.

    Critical path (gate, init-done): Shell built-ins. Cannot crash.
    Value-add (start, checkpoint, stop): Python with error wrapper.
    """
    defs: dict[str, list] = {
        "SessionStart": [
            # v3.6.23: advisory SLM session-init hint fires first.
            # mcp__superlocalmemory__session_init may be deferred, so some
            # hosts need ToolSearch before they can invoke the tool.
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("mandate"),
                        "timeout": 5,
                    }
                ]
            },
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("start"),
                        "timeout": 15,
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
                        "timeout": 5,
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
                        "timeout": 5,
                    }
                ],
            },
            # 4.2.30 - error cues after a tool ran (the plugin's hooks.json
            # declares the same entry; a test keeps the two equal).
            {
                "matcher": _CUE_AFTER_TOOLS,
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("cue"),
                        "timeout": 5,
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
                        "timeout": 5,
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
                        "timeout": 3,
                    }
                ]
            },
            # 4.2.30 - per-prompt recall (behind the need gate; one recall per
            # prompt even when the plugin's copy of this hook also runs).
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("auto_recall"),
                        "timeout": 7,
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
                        "timeout": 10,
                    },
                    # LLD-09 Track A.2 — finalize pending outcomes at end.
                    {
                        "type": "command",
                        "command": _wrap_python_cmd("stop_outcome"),
                        "timeout": 10,
                    },
                    # Commit temporal summaries so session decisions survive beyond
                    # the git-state snapshot written by `slm hook stop`.
                    {
                        "type": "command",
                        "command": (
                            'cmd /c "slm session close 2>NUL || exit /b 0"'
                            if sys.platform == "win32"
                            else "slm session close 2>/dev/null || true"
                        ),
                        "timeout": 15,
                    },
                ]
            }
        ],
    }

    # v3.4.43 — default PreToolUse entry: pre-web recall on WebSearch/WebFetch.
    # Fires `slm hook before_web` which runs a recall on the search
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
                    "timeout": 5,
                }
            ],
        },
        # 4.2.30 - tool cues before a tool runs.
        {
            "matcher": _CUE_BEFORE_TOOLS,
            "hooks": [
                {
                    "type": "command",
                    "command": _wrap_python_cmd("cue"),
                    "timeout": 5,
                }
            ],
        },
    ]

    if include_gate:
        defs["PreToolUse"].insert(0, {
            "matcher": _GATED_TOOLS,
            "hooks": [
                {
                    "type": "command",
                    "command": _gate_cmd(),
                    "timeout": 2,
                }
            ],
        })
        defs["PostToolUse"].insert(0, {
            "matcher": "mcp__superlocalmemory__session_init",
            "hooks": [
                {
                    "type": "command",
                    "command": _init_done_cmd(),
                    "timeout": 2,
                }
            ],
        })

    return defs


# ---------------------------------------------------------------------------
# Identify SLM hooks in existing settings
# ---------------------------------------------------------------------------

def _is_slm_command(cmd: object) -> bool:
    """True only for a command some SLM release wrote (exact, after whitespace and temp-path
    normalisation; 4.2.30). A line that merely mentions ``slm hook`` is the person's."""
    from superlocalmemory.hooks._hook_identity import is_known_command

    return is_known_command(cmd)


def _is_slm_hook(hook: object) -> bool:
    """True only for a hook object exactly as SLM wrote it: a known command, no extra keys,
    one of the timeouts SLM wrote. An edited line, or a raised timeout, is the person's."""
    from superlocalmemory.hooks._hook_identity import is_slm_hook

    return is_slm_hook(hook)


def _is_slm_hook_entry(entry: dict) -> bool:
    """Check if a hook entry belongs to SLM."""
    hooks = entry.get("hooks", []) if isinstance(entry, dict) else []
    if not isinstance(hooks, list):
        return False
    return any(_is_slm_hook(hook) for hook in hooks)


# ---------------------------------------------------------------------------
# Safe settings.json merge / removal
# ---------------------------------------------------------------------------

def _merge_hooks(settings: dict, hook_defs: dict) -> dict:
    """Return ``settings`` with SLM's hooks replaced and added, every other entry kept.

    Replacement and addition only (4.2.30): an SLM entry is swapped where it stood,
    a missing one is added, and the user's own hooks, other keys and key order are
    left exactly as they were. Pure: the argument is not changed.
    """
    from superlocalmemory.hooks._hook_merge import replace_owned

    hooks = settings.get("hooks")
    merged_hooks = dict(hooks) if isinstance(hooks, dict) else {}
    for hook_type, slm_entries in hook_defs.items():
        existing = merged_hooks.get(hook_type, [])
        existing = existing if isinstance(existing, list) else []
        merged_hooks[hook_type] = replace_owned(
            existing, _without_edited_copies(existing, slm_entries), _is_slm_command,
            is_owned_hook=_is_slm_hook)
    return {**settings, "hooks": merged_hooks}


def _without_edited_copies(existing: list, slm_entries: list) -> list:
    """SLM's entries minus the commands a person already runs in their own, edited form.

    A line the person changed (an extra variable, a script after it) is theirs and stays; SLM
    must not add a second copy of the same ``slm hook <name>`` beside it, which would fire the
    hook twice. A command counts as theirs only when it is not exactly SLM's.
    """
    from superlocalmemory.hooks._hook_identity import covered_by_edit

    theirs = [str(h.get("command", "")) for e in existing
              if isinstance(e, dict) and isinstance(e.get("hooks"), list)
              for h in e["hooks"] if isinstance(h, dict) and not _is_slm_hook(h)]
    if not theirs:
        return slm_entries
    kept = []
    for entry in slm_entries:
        hooks = [h for h in entry.get("hooks", []) if not covered_by_edit(h.get("command", ""), theirs)]
        if hooks:
            kept.append({**entry, "hooks": hooks})
    return kept


def settings_have_slm_hooks(settings: dict) -> bool:
    """True when ``settings`` already holds at least one SLM hook entry."""
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    if not isinstance(hooks, dict):
        return False
    return any(isinstance(entries, list) and any(_is_slm_hook_entry(e) for e in entries)
               for entries in hooks.values())


def settings_have_gate(settings: dict) -> bool:
    """True when the experimental PreToolUse gate is among the SLM entries."""
    entries = (settings.get("hooks") or {}).get("PreToolUse", [])
    return any(
        _is_slm_hook_entry(e) and any(
            "Call mcp__superlocalmemory__session_init first" in str(h.get("command", ""))
            for h in e.get("hooks", []) if isinstance(h, dict))
        for e in entries if isinstance(e, dict))


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
        return json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))
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

def install_hooks(include_gate: bool = False, *, dry_run: bool = False) -> dict:
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

    if dry_run:
        return _dry_run(lambda s: _merge_hooks(s, _hook_definitions(include_gate=include_gate)),
                        {**result, "hooks_added": list(_hook_definitions(
                            include_gate=include_gate).keys())})
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
        version_dir = _version_dir()
        version_file = _version_file()
        disabled_file = _disabled_file()
        version_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text(HOOKS_VERSION, encoding="utf-8")
        # Clear disabled marker — explicit install means user wants hooks
        if disabled_file.exists():
            disabled_file.unlink()
    except Exception as exc:
        result["errors"].append(f"Version file failed: {exc}")

    return result


def _dry_run(change, result: dict) -> dict:
    """What a change would do to settings.json, with nothing written."""
    import copy

    try:
        before = _read_settings()
        after = change(copy.deepcopy(before))
    except Exception as exc:
        return {**result, "dry_run": True, "errors": [f"Settings read failed: {exc}"]}
    return {**result, "success": True, "dry_run": True, "would_change": after != before,
            "path": str(CLAUDE_SETTINGS)}


def remove_hooks(*, dry_run: bool = False) -> dict:
    """Remove all SLM hooks from Claude Code settings.json.

    Writes a .hooks-disabled marker so auto-install paths respect
    the user's explicit choice. Cleared by explicit `install_hooks()`.
    """
    result = {"success": False, "errors": []}
    if dry_run:
        return _dry_run(_remove_slm_hooks, result)

    try:
        settings = _read_settings()
        settings = _remove_slm_hooks(settings)
        _write_settings(settings)
        result["success"] = True
    except Exception as exc:
        result["errors"].append(f"Settings cleanup failed: {exc}")

    try:
        version_dir = _version_dir()
        version_file = _version_file()
        disabled_file = _disabled_file()
        if version_file.exists():
            version_file.unlink()
        # Mark as explicitly disabled — auto-install will respect this
        version_dir.mkdir(parents=True, exist_ok=True)
        disabled_file.write_text("removed by user\n", encoding="utf-8")
    except Exception:
        pass

    return result


def check_status() -> dict:
    """Check SLM hook installation status."""
    installed_version = ""
    version_file = _version_file()
    if version_file.exists():
        try:
            installed_version = version_file.read_text(encoding="utf-8").strip()
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
        disabled_file = _disabled_file()
        version_file = _version_file()
        # Respect explicit opt-out
        if disabled_file.exists():
            return None

        # Already installed and current → skip
        if version_file.exists():
            installed = version_file.read_text(encoding="utf-8").strip()
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
        version_file = _version_file()
        if not version_file.exists():
            legacy_script = _version_dir() / "slm-session-start.sh"
            if legacy_script.exists():
                _migrate_legacy_hooks()
            return

        installed = version_file.read_text(encoding="utf-8").strip()
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
            _version_dir().mkdir(parents=True, exist_ok=True)
            _version_file().write_text(HOOKS_VERSION, encoding="utf-8")
            logger.info("Migrated legacy bash hooks to hybrid hooks (v%s)", HOOKS_VERSION)
    except Exception as exc:
        logger.debug("Legacy hook migration failed: %s", exc)
