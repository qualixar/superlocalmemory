# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Detect all SuperLocalMemory installations present on this machine.

Supports three install types: pipx, venv (~/.slm-venv), and npm global.
All detection is read-only and fast (< 200 ms). No writes are ever performed.
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Roots for the two Python install types. Patched in tests.
_VENV_ROOT: Path = Path.home() / ".slm-venv"
_PIPX_ROOT: Path = Path.home() / ".local" / "pipx" / "venvs" / "superlocalmemory"


def _npm_global_root() -> Optional[Path]:
    """Return the npm global node_modules root, or None on any failure."""
    try:
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            root = result.stdout.strip()
            if root:
                return Path(root)
    except Exception:  # npm absent, timeout, etc.
        pass
    return None


def _read_python_version(base: Path) -> Optional[str]:
    """Read __version__ from the first matching site-packages layout under base.

    Both layouts are searched. POSIX virtualenvs use
    ``lib/python3.13/site-packages``; Windows uses ``Lib/site-packages`` with no
    version component and a capitalised directory. Searching only the POSIX
    shape made detection silently return None on Windows — which is precisely
    where multi-install divergence between pip and npm is most likely, and where
    the version-mismatch error would then name no installations at all.
    """
    found = _find_package_init(base)
    if found is None:
        return None
    try:
        text = found.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("__version__"):
            # __version__ = "4.1.0"  or  __version__ = '4.1.0'
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip("\"'") or None
    return None


def _find_package_init(base: Path) -> Optional[Path]:
    """Locate the installed ``superlocalmemory/__init__.py`` under ``base``."""
    patterns = [
        str(base / "lib" / "python*" / "site-packages" / "superlocalmemory" / "__init__.py"),
        str(base / "Lib" / "site-packages" / "superlocalmemory" / "__init__.py"),
        str(base / "lib" / "site-packages" / "superlocalmemory" / "__init__.py"),
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    for init_path in sorted(set(matches)):
        return Path(init_path)
    return None


def _resolve_package_dir(base: Path) -> Optional[str]:
    """Return the directory holding the resolved package (4.1.14 #134).

    This is the answer to "which copy actually loads": the venv
    interpreter resolves exactly this ``superlocalmemory/`` directory.
    A stale ``src/`` tree beside an npm install can never shadow it —
    the launcher strips ``PYTHONPATH`` — but naming the authority in
    doctor output ends the confusion class from #128 Bug 2.
    """
    found = _find_package_init(base)
    if found is None:
        return None
    return str(found.parent)


def _read_npm_version(npm_root: Path) -> Optional[str]:
    """Read version from npm global package.json."""
    pkg_json = npm_root / "superlocalmemory" / "package.json"
    try:
        data = json.loads(pkg_json.read_text(encoding="utf-8"))
        return str(data.get("version", "")).strip() or None
    except (OSError, json.JSONDecodeError):
        return None


def _detect_all_installs() -> list[dict]:
    """Return all SuperLocalMemory installs detected on this machine.

    Each entry is a dict with keys:
      - ``path``    (str) — directory of the install
      - ``version`` (str) — version string read from package metadata
      - ``type``    (str) — one of "pipx", "venv", "npm"
      - ``resolved`` (str, optional) — the package directory the install's
        interpreter actually loads (4.1.14 #134: names the single source
        of truth so a stale tree can never silently shadow it).

    Detection is read-only and best-effort. A missing or unreadable install
    produces no entry rather than an error. Subprocess calls are bounded to
    5 seconds total.
    """
    results: list[dict] = []

    # --- pipx ---
    pipx_version = _read_python_version(_PIPX_ROOT)
    if pipx_version is not None:
        results.append({
            "path": str(_PIPX_ROOT) + "/",
            "version": pipx_version,
            "type": "pipx",
        })

    # --- ~/.slm-venv ---
    venv_version = _read_python_version(_VENV_ROOT)
    if venv_version is not None:
        results.append({
            "path": str(_VENV_ROOT) + "/",
            "version": venv_version,
            "type": "venv",
            "resolved": _resolve_package_dir(_VENV_ROOT),
        })

    # --- npm global ---
    npm_root = _npm_global_root()
    if npm_root is not None:
        npm_version = _read_npm_version(npm_root)
        if npm_version is not None:
            npm_venv = npm_root / "superlocalmemory" / ".slm-venv"
            results.append({
                "path": str(npm_root / "superlocalmemory") + "/",
                "version": npm_version,
                "type": "npm",
                "resolved": _resolve_package_dir(npm_venv),
            })

    return results


__all__ = ["_detect_all_installs"]
