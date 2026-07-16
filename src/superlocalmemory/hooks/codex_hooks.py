"""Additive Codex lifecycle-hook integration.

Codex supports a dedicated ``hooks.json`` beside ``config.toml``.  Keeping
SLM's lifecycle entries there means installing hooks never round-trips or
reformats a user's TOML configuration.  The installer only owns entries marked
with ``SLM_CODEX_HOOK`` and retains every other hook and top-level setting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SLM_MARKER = "SLM_CODEX_HOOK"
DEFAULT_HOOKS_PATH = Path.home() / ".codex" / "hooks.json"
EVENTS = ("SessionStart", "PostToolUse", "UserPromptSubmit", "Stop")


def hook_definitions() -> dict[str, list[dict[str, Any]]]:
    """Return portable, supported Codex lifecycle hooks.

    Commands deliberately resolve the installed ``slm`` executable through
    ``PATH``.  They never embed a developer-specific home directory or Python
    interpreter path.  Codex hook events and the JSON group shape are defined
    in the public Codex hooks specification.
    """
    def entry(command: str, *, matcher: str | None = None, timeout: int = 12,
              status: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "hooks": [{
                "type": "command",
                "command": f"{command} # {SLM_MARKER}",
                "timeout": timeout,
            }],
        }
        if matcher:
            value["matcher"] = matcher
        if status:
            value["hooks"][0]["statusMessage"] = status
        return value

    return {
        "SessionStart": [entry("slm hook codex-start", timeout=15, status="Loading SLM context")],
        "PostToolUse": [entry("slm hook checkpoint", matcher="Edit|Write", timeout=5)],
        "UserPromptSubmit": [entry("slm hook codex-prompt", timeout=5)],
        "Stop": [entry("slm hook codex-stop", timeout=12, status="Saving SLM checkpoint")],
    }


def is_slm_hook_entry(entry: Any) -> bool:
    """Return true only for the SLM-owned Codex hook group."""
    if not isinstance(entry, dict):
        return False
    return any(
        isinstance(hook, dict) and SLM_MARKER in str(hook.get("command", ""))
        for hook in entry.get("hooks", [])
    )


def _is_slm_command(command: Any) -> bool:
    """Recognize only our marker plus the two retired, SLM-specific paths."""
    value = str(command)
    return (
        SLM_MARKER in value
        or ".codex/hooks/auto-recall.py" in value
        or ("universal-hook.py" in value and "--intent slm_" in value)
    )


def _remove_owned_commands(entries: list[Any]) -> tuple[list[Any], bool]:
    """Remove SLM commands while preserving other commands in mixed groups."""
    result: list[Any] = []
    changed = False
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("hooks"), list):
            result.append(entry)
            continue
        retained = [hook for hook in entry["hooks"] if not (
            isinstance(hook, dict) and _is_slm_command(hook.get("command", ""))
        )]
        if len(retained) == len(entry["hooks"]):
            result.append(entry)
            continue
        changed = True
        if retained:
            copy = dict(entry)
            copy["hooks"] = retained
            result.append(copy)
    return result, changed


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("hooks.json must contain a JSON object")
    return data


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".slm_tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_suffix(path.suffix + ".slm.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())


def install_hooks(*, hooks_path: Path = DEFAULT_HOOKS_PATH, dry_run: bool = False) -> dict[str, Any]:
    """Merge portable SLM entries into Codex hooks.json without clobbering it."""
    try:
        data = _read(hooks_path)
        hooks = data.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("hooks.json 'hooks' field must be an object")
        added: list[str] = []
        for event, entries in hook_definitions().items():
            existing = hooks.setdefault(event, [])
            if not isinstance(existing, list):
                raise ValueError(f"hooks.json '{event}' field must be a list")
            # Retire only known obsolete SLM commands, including when they
            # share a matcher group with unrelated user hooks.
            retained, _ = _remove_owned_commands(existing)
            hooks[event] = existing = retained
            if not any(is_slm_hook_entry(entry) for entry in existing):
                existing.extend(entries)
                added.append(event)
        if not dry_run:
            _backup_once(hooks_path)
            _write_atomic(hooks_path, data)
        return {"success": True, "hooks_added": added, "path": str(hooks_path), "dry_run": dry_run}
    except Exception as exc:
        return {"success": False, "errors": [f"Codex hooks update failed: {exc}"], "path": str(hooks_path)}


def remove_hooks(*, hooks_path: Path = DEFAULT_HOOKS_PATH, dry_run: bool = False) -> dict[str, Any]:
    """Remove SLM-owned groups only; never remove user hook definitions."""
    try:
        data = _read(hooks_path)
        hooks = data.get("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("hooks.json 'hooks' field must be an object")
        removed: list[str] = []
        for event in list(hooks):
            entries = hooks[event]
            if not isinstance(entries, list):
                continue
            retained, changed = _remove_owned_commands(entries)
            if changed:
                removed.append(event)
                if retained:
                    hooks[event] = retained
                else:
                    del hooks[event]
        if removed and not dry_run:
            _backup_once(hooks_path)
            _write_atomic(hooks_path, data)
        return {"success": True, "hooks_removed": removed, "path": str(hooks_path), "dry_run": dry_run}
    except Exception as exc:
        return {"success": False, "errors": [f"Codex hooks cleanup failed: {exc}"], "path": str(hooks_path)}


def check_status(*, hooks_path: Path = DEFAULT_HOOKS_PATH) -> dict[str, Any]:
    """Report installed state without mutating the user configuration."""
    try:
        data = _read(hooks_path)
    except Exception as exc:
        return {"installed": None, "hook_types": [], "error": f"JSON parse error: {exc}", "path": str(hooks_path)}
    hooks = data.get("hooks", {})
    if not isinstance(hooks, dict):
        return {"installed": None, "hook_types": [], "error": "hooks field is not an object", "path": str(hooks_path)}
    found = [event for event, entries in hooks.items() if isinstance(entries, list) and any(is_slm_hook_entry(entry) for entry in entries)]
    return {"installed": set(EVENTS).issubset(found), "hook_types": found, "path": str(hooks_path)}
