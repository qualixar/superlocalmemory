# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""WP-07 — Lazy first-run initialisation (pip cross-install).

Lightweight import only — zero heavy imports, STDOUT-SILENT.

Public API
----------
slm_home() -> Path
    Resolve the SLM data directory from env aliases, falling back to
    ~/.superlocalmemory.  Pure: no mkdir, no I/O beyond os.environ reads.
    Priority: SLM_DATA_DIR -> SL_MEMORY_PATH -> SLM_HOME -> default.

_ensure_initialized() -> None
    Idempotent first-run guard: creates the home dir + a minimal mode-A
    config.json if absent.  Fast-path returns immediately when both dir
    and config exist.  Never raises, never prints to stdout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from superlocalmemory import __version__ as _RUNTIME_VERSION


# ---------------------------------------------------------------------------
# Public: home resolution
# ---------------------------------------------------------------------------


def slm_home() -> Path:
    """Resolve the SLM data directory.

    Resolution order (first non-empty wins):
      1. SLM_DATA_DIR  — canonical env var
      2. SL_MEMORY_PATH — legacy alias (setup_wizard.py)
      3. SLM_HOME      — hook alias (hooks/_outcome_common.py)
      4. ~/.superlocalmemory — hard default

    Pure: no mkdir, no side effects.
    """
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root()


# ---------------------------------------------------------------------------
# Public: idempotent first-run init
# ---------------------------------------------------------------------------

# Minimal mode-A config written on first-run pip install.
# Only mode + base_dir; everything else uses SLMConfig defaults.
_MINIMAL_CONFIG: dict = {
    "mode": "a",
    "version": _RUNTIME_VERSION,
}


def _ensure_initialized() -> None:
    """Idempotent first-run guard.

    Creates ~/.superlocalmemory/ (or env-overridden path) and a minimal
    mode-A config.json if they are absent.

    Contract:
    - NEVER raises (OSError → silent degrade, AC4).
    - NEVER prints to stdout (CRIT-3).
    - NEVER overwrites an existing config.json (AC3).
    - 2nd call is a no-op — mtime of config unchanged (AC2).
    - Does NOT write the .setup-complete sentinel.
    """
    try:
        home = slm_home()
        config = home / "config.json"

        # Fast path: both dir and config exist — nothing to do (AC2).
        if home.is_dir() and config.exists():
            return

        # Ensure the directory exists.
        home.mkdir(parents=True, exist_ok=True)

        # Write config only when absent (AC3 + AC2).
        if not config.exists():
            _write_minimal_config(home, config)

    except OSError:
        # Read-only home or any I/O failure → degrade silently (AC4).
        return


def _write_minimal_config(home: Path, config: Path) -> None:
    """Atomically write a minimal mode-A config.json inside *home*.

    Uses a .tmp file inside the same directory so os.replace() is always
    on the same filesystem (no cross-FS rename).  Never writes to /tmp.
    """
    # PID-namespaced tmp so two `slm` processes first-running concurrently don't
    # clobber each other's tmp (which would make one os.replace fail mid-race).
    tmp_path = home / f".config.json.tmp.{os.getpid()}"
    try:
        payload = json.dumps(_MINIMAL_CONFIG, indent=2)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(str(tmp_path), str(config))
    except OSError:
        # Clean up tmp on failure; ignore errors — degrade silently.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
