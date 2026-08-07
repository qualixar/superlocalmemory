# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4

"""W1 regression: backend detection must not start LanceDBBackgroundEventLoop.

Proven: `import lancedb` spawns that thread; `find_spec('lancedb')` does not.
Detection under vector_backend='auto' must use find_spec only. Importing the
lancedb_backend *module* must also not start the native runtime.

Runs in a subprocess so other tests that construct LanceDBVectorBackend cannot
pollute the thread list.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = str(REPO_ROOT / "src")


def _run_probe(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": SRC}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=90,
    )


def test_detect_lancedb_does_not_spawn_background_thread():
    """After _detect_lancedb() with vector_backend=auto, no LanceDB thread."""
    code = r"""
import threading
from types import SimpleNamespace
from superlocalmemory.core.backend_orchestrator import BackendOrchestrator

cfg = SimpleNamespace(
    vector_backend="auto",
    graph_backend="auto",
    data_dir="/tmp/slm-w1-detect",
    base_dir="/tmp/slm-w1-detect",
    scale_engine_state="local_core",
)

class _DB:
    def execute(self, *a, **k):
        return []

orch = BackendOrchestrator(cfg, _DB())
available = orch._detect_lancedb()
names = [t.name for t in threading.enumerate()]
lance = [n for n in names if n == "LanceDBBackgroundEventLoop"]
print("available", available)
print("lance_threads", lance)
if lance:
    raise SystemExit(2)
raise SystemExit(0)
"""
    proc = _run_probe(code)
    assert proc.returncode == 0, (
        f"detect spawned LanceDB thread or crashed\n"
        f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-3000:]}"
    )


def test_import_lancedb_backend_module_does_not_spawn_thread():
    """Importing superlocalmemory.vector.lancedb_backend must stay lazy."""
    code = r"""
import threading
import superlocalmemory.vector.lancedb_backend as m
assert m._LANCEDB_AVAILABLE is True or m._LANCEDB_AVAILABLE is False
names = [t.name for t in threading.enumerate()]
lance = [n for n in names if n == "LanceDBBackgroundEventLoop"]
print("available", m._LANCEDB_AVAILABLE)
print("lance_threads", lance)
if lance:
    raise SystemExit(2)
raise SystemExit(0)
"""
    proc = _run_probe(code)
    assert proc.returncode == 0, (
        f"module import spawned LanceDB thread or crashed\n"
        f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-3000:]}"
    )


def test_detect_cozo_uses_find_spec_not_import():
    """_detect_cozo must not import pycozo (same native risk class)."""
    code = r"""
import sys
from types import SimpleNamespace
from superlocalmemory.core.backend_orchestrator import BackendOrchestrator

# Drop pycozo from modules if present so we can prove detect doesn't re-import.
sys.modules.pop("pycozo", None)
sys.modules.pop("pycozo.client", None)

cfg = SimpleNamespace(
    vector_backend="auto",
    graph_backend="auto",
    data_dir="/tmp/slm-w1-cozo",
    base_dir="/tmp/slm-w1-cozo",
    scale_engine_state="local_core",
)

class _DB:
    def execute(self, *a, **k):
        return []

orch = BackendOrchestrator(cfg, _DB())
available = orch._detect_cozo()
# Detection success/fail is fine; the point is pycozo must not be executed.
imported = "pycozo" in sys.modules or "pycozo.client" in sys.modules
print("available", available)
print("imported", imported)
if imported:
    raise SystemExit(2)
raise SystemExit(0)
"""
    proc = _run_probe(code)
    assert proc.returncode == 0, (
        f"_detect_cozo imported pycozo\n"
        f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr[-3000:]}"
    )
