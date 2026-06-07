"""slm wrap <agent> — table-driven per-agent proxy activation.

PORT: 8765 (INTERFACE-CONTRACT §0). No 52415 anywhere in this file.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from superlocalmemory.optimize.adapters._agent_registry import AGENT_REGISTRY
from superlocalmemory.optimize.proxy.lifecycle import ensure_proxy_running, proxy_port

# Mechanisms that write static config files — need proxy *configured*, not alive.
_STATIC_MECHANISMS = {"settings-file", "config-file", "print-only"}


def _proxy_configured() -> bool:
    """Return True if proxy_enabled=True in optimize.json (no liveness check)."""
    try:
        from superlocalmemory.optimize.config import get_optimize_config
        return get_optimize_config().proxy_enabled
    except Exception:
        return False


def list_agents() -> list[str]:
    """Return all registered agent keys."""
    return list(AGENT_REGISTRY.keys())


def wrap_agent(
    agent_key: str,
    agent_args: list[str],
    *,
    persistent: bool = False,
    dry_run: bool = False,
) -> int:
    """Configure proxy redirection for <agent_key> and optionally launch it.

    Returns:
        0 on success, 1 on error, agent's exit code when subprocess launched.
    """
    if agent_key not in AGENT_REGISTRY:
        known = ", ".join(AGENT_REGISTRY.keys())
        print(
            f"[slm wrap] Unknown agent '{agent_key}'. Known: {known}",
            file=sys.stderr,
        )
        return 1

    port = proxy_port()
    spec = AGENT_REGISTRY[agent_key]
    mechanism = spec.get("mechanism", "print-only")

    # Static mechanisms (settings-file, config-file) only write JSON — proxy
    # doesn't need to be alive yet.  env/subprocess mechanisms inject the proxy
    # URL into a live process, so full liveness is required there.
    # Static mechanisms (settings-file, config-file, print-only) write JSON and
    # dry-run modes only print — neither needs the proxy to be alive right now.
    # Only live subprocess launches (mechanism="env", dry_run=False) require the
    # proxy to be running so the subprocess can actually connect.
    needs_liveness = (mechanism not in _STATIC_MECHANISMS) and not dry_run
    if needs_liveness:
        if not ensure_proxy_running():
            print(
                f"[slm wrap] proxy is not enabled or not running — run "
                f"`slm proxy` to start it, or `slm optimize on` first.",
                file=sys.stderr,
            )
            return 1
    else:
        if not _proxy_configured():
            print(
                f"[slm wrap] proxy is not enabled in optimize.json — set "
                f"`proxy_enabled: true` (port 8765) and re-run, or run "
                f"`slm optimize on` first.",
                file=sys.stderr,
            )
            return 1

    if mechanism == "print-only":
        print(f"[slm wrap] {agent_key}: manual instructions")
        print(spec.get("help_text", ""))
        return 0

    if mechanism == "settings-file":
        # Persist env var to a settings file
        settings_path_str = spec.get("settings_path", "")
        env_vars = spec.get("env_vars", {})
        if not env_vars:
            print(f"[slm wrap] {agent_key}: no env vars to set", file=sys.stderr)
            return 1
        expanded = str(Path(settings_path_str).expanduser())
        path = Path(expanded)
        if dry_run:
            print(f"[slm wrap] would write {path} with env={env_vars}")
            return 0
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError):
                existing = {}
        existing.setdefault("env", {})
        for k, v in env_vars.items():
            existing["env"][k] = v
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps(existing, indent=2))
        except (OSError, ValueError) as exc:
            print(f"[slm wrap] could not write {path}: {exc}", file=sys.stderr)
            return 1
        print(f"[slm wrap] wrote {path}")
        return 0

    if mechanism == "config-file":
        # VS Code settings.json edit. config_path may be:
        # - absolute path → use directly
        # - template with {vscode_user_dir} → expand via _vscode_user_dir()
        # - relative path → resolve against current working directory
        config_value = spec.get("config_value", "")
        config_key = spec.get("config_key", "")
        config_path_str = spec.get("config_path", "")
        if not config_path_str:
            print("[slm wrap] no config_path specified", file=sys.stderr)
            return 1
        if "{vscode_user_dir}" in config_path_str:
            vscode_dir = _vscode_user_dir()
            if vscode_dir is None:
                print("[slm wrap] VS Code user dir not found", file=sys.stderr)
                return 1
            path = Path(config_path_str.replace("{vscode_user_dir}", str(vscode_dir)))
        else:
            path = Path(config_path_str).expanduser()
        if dry_run:
            print(f"[slm wrap] would write {path} key={config_key} value={config_value}")
            return 0
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing[config_key] = config_value
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps(existing, indent=2))
        except OSError as exc:
            print(f"[slm wrap] could not write {path}: {exc}", file=sys.stderr)
            return 1
        print(f"[slm wrap] wrote {path}")
        return 0

    if mechanism == "env":
        # Launch the binary with env vars injected
        binary = spec.get("binary")
        env_vars = spec.get("env_vars", {})
        if not binary:
            print(f"[slm wrap] {agent_key}: no binary specified", file=sys.stderr)
            return 1
        if shutil.which(binary) is None:
            print(
                f"[slm wrap] binary '{binary}' not found in PATH. "
                f"Install {agent_key} or set PATH.",
                file=sys.stderr,
            )
            return 1
        full_env = os.environ.copy()
        for k, v in env_vars.items():
            full_env[k] = v.replace("{port}", str(port))
        if dry_run:
            print(f"[slm wrap] would exec: {binary} {' '.join(agent_args)}")
            print(f"[slm wrap] env: {env_vars}")
            return 0
        try:
            return subprocess.call([binary, *agent_args], env=full_env)
        except FileNotFoundError as exc:
            print(f"[slm wrap] could not launch {binary}: {exc}", file=sys.stderr)
            return 1

    print(f"[slm wrap] {agent_key}: unknown mechanism {mechanism!r}", file=sys.stderr)
    return 1


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _vscode_user_dir() -> Path | None:
    """Return the VS Code user settings directory for the current OS."""
    if sys.platform == "darwin":
        p = Path.home() / "Library" / "Application Support" / "Code" / "User"
        return p if p.exists() or p.parent.exists() else None
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        if appdata:
            p = Path(appdata) / "Code" / "User"
            return p
        return None
    # Linux
    p = Path.home() / ".config" / "Code" / "User"
    return p
