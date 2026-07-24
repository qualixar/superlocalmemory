# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Component healer — the repair actions behind the registry (v3.8.2).

Detection lives in :mod:`core.component_registry` (read-only probes). This
module performs the *actions* for components the registry marked
``auto_fixable``: re-download a HuggingFace model, or pip-install a small
pure-python dependency. Both the background self-heal thread
(``server.unified_daemon._self_heal`` Step 0/0.5) and the foreground
``slm doctor --fix`` command call :func:`heal_missing`, so the repair logic
exists in exactly one place.

Safety (Varun's mandate, unchanged):
* NEVER ``sudo``; NEVER auto ``ollama pull`` (surprise network/disk);
  NEVER auto-install multi-GB deps (torch) — those are manual fix commands.
* pip only into a user-writable interpreter (no PEP-668 marker).
* Bounded retries; every action is fail-open — a repair failure never
  raises into the caller and never wedges recall.
"""

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Callable

from superlocalmemory.core import component_registry as cr

# Progress sink: (component_key, human_message) -> None. Defaults to no-op.
ProgressFn = Callable[[str, str], None]


def _noop(_key: str, _msg: str) -> None:
    pass


def _pip_install(package: str, timeout: int = 300) -> tuple[bool, str]:
    """pip-install one package into the current interpreter. Fail-open."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input", package],
            timeout=timeout, capture_output=True, text=True,
        )
        if result.returncode == 0:
            return True, f"installed {package}"
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, (tail[-1] if tail else f"pip exit {result.returncode}")
    except subprocess.TimeoutExpired:
        return False, f"pip install {package} timed out ({timeout}s)"
    except Exception as exc:  # never propagate — heal is fail-open
        return False, f"{type(exc).__name__}: {exc}"


def _heal_embedder() -> tuple[bool, str]:
    from superlocalmemory.cli.setup_wizard import _EMBED_MODEL, _download_model

    ok = _download_model(_EMBED_MODEL, "Embedding model")
    return ok, "embedding model ready" if ok else "download failed"


def _heal_reranker() -> tuple[bool, str]:
    from superlocalmemory.cli.setup_wizard import _RERANKER_MODEL, _download_reranker

    ok = _download_reranker(_RERANKER_MODEL)
    return ok, "reranker ready" if ok else "download failed"


def _heal_sqlite_vec() -> tuple[bool, str]:
    if not cr.pip_is_user_writable():
        return False, "interpreter externally managed (PEP 668) — install manually"
    return _pip_install("sqlite-vec")


# key -> action. Only keys the registry can mark auto_fixable appear here.
_ACTIONS: dict[str, Callable[[], tuple[bool, str]]] = {
    "embedder_model": _heal_embedder,
    "reranker_model": _heal_reranker,
    "sqlite_vec": _heal_sqlite_vec,
}


def heal_missing(
    config: Any = None,
    keys: list[str] | None = None,
    on_progress: ProgressFn | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Repair every auto-fixable missing component (optionally filtered to ``keys``).

    Returns ``{attempted, healed, failed, results}`` where ``results`` is a
    list of ``{key, success, detail}``. Marks each component ``retrying`` in
    the registry's transient overlay while its repair is in flight so the
    dashboard shows live progress; clears the marker on success (a fresh
    probe then confirms ``ok``) or on give-up (probe reports ``missing`` again).
    """
    progress = on_progress or _noop
    targets = cr.auto_fixable_missing(config)
    if keys is not None:
        wanted = set(keys)
        targets = [c for c in targets if c.key in wanted]

    results: list[dict[str, Any]] = []
    healed = failed = 0

    for comp in targets:
        action = _ACTIONS.get(comp.key)
        if action is None:
            # auto_fixable but no registered action — record, do not pretend.
            results.append({"key": comp.key, "success": False,
                            "detail": "no repair action registered"})
            failed += 1
            continue

        cr.mark_transient(comp.key, cr.STATUS_RETRYING, f"repairing {comp.label}…")
        progress(comp.key, f"repairing {comp.label}…")

        success, detail = False, "not attempted"
        for attempt in range(1, max_retries + 1):
            success, detail = action()
            if success:
                break
            progress(comp.key,
                     f"attempt {attempt}/{max_retries} failed: {detail}")
            if attempt < max_retries:
                time.sleep(2)

        cr.clear_transient(comp.key)
        results.append({"key": comp.key, "success": success, "detail": detail})
        if success:
            healed += 1
            progress(comp.key, f"✓ {detail}")
        else:
            failed += 1
            progress(comp.key, f"✗ {detail} (fix manually: {comp.fix_cmd})")

    return {
        "attempted": len(targets),
        "healed": healed,
        "failed": failed,
        "results": results,
    }
