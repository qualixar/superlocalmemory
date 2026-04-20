# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""Global kill-switch for SLM.

Two mechanisms, either disables the entire system cheaply:

1. File-marker: ``~/.superlocalmemory/.disabled`` — persistent across
   reboots, survives daemon restarts, written by ``slm disable``.
2. Environment variable: ``SLM_DISABLE=1`` — per-process, useful for
   CI, sandboxes, or "just for this shell" overrides.

Every hot-path entry point (hooks, MCP tools, recall pipeline, daemon
lifespan) calls :func:`is_disabled` first. Returns ``True`` ⇒ exit
quietly, no side effects.

Backward-compat: unset env + missing marker ⇒ ``False`` ⇒ normal
behaviour. Zero impact on the 18k live users who never touch it.
"""

from __future__ import annotations

import os
from pathlib import Path


_MARKER_NAME = ".disabled"
_ENV_NAME = "SLM_DISABLE"


def _slm_home() -> Path:
    """Return the SLM state directory. Override via ``SLM_HOME`` env."""
    override = os.environ.get("SLM_HOME")
    if override:
        return Path(override)
    return Path.home() / ".superlocalmemory"


def marker_path() -> Path:
    """Where the persistent ``.disabled`` marker lives."""
    return _slm_home() / _MARKER_NAME


def is_disabled() -> bool:
    """Return True iff SLM should no-op everything.

    Precedence: env var first (cheapest check), then file marker. Any
    non-empty, non-"0", non-"false" value in the env counts as disabled.
    """
    env = os.environ.get(_ENV_NAME, "").strip().lower()
    if env and env not in ("0", "false", "no", "off"):
        return True
    try:
        return marker_path().exists()
    except OSError:  # pragma: no cover — defensive against FS errors
        return False


def write_marker(reason: str = "") -> Path:
    """Create the disabled marker. Returns the path."""
    home = _slm_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / _MARKER_NAME
    payload = "disabled"
    if reason:
        payload = f"disabled: {reason}\n"
    path.write_text(payload, encoding="utf-8")
    return path


def remove_marker() -> bool:
    """Remove the disabled marker. Returns True if removed, False if absent."""
    path = marker_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


__all__ = (
    "is_disabled",
    "write_marker",
    "remove_marker",
    "marker_path",
)
