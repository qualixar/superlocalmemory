# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Root conftest — thread safety and path setup ONLY.

RULE: Unit tests MUST NOT touch real models, real DB, real downloads.
All heavy mocking is done in tests/conftest.py (existing) and in
per-directory conftest files, NOT here at root scope.

What belongs here (runs before any import):
  - sys.path so src/ is importable
  - C-library thread count env vars (must precede first import of numpy/torch/lgb)
  - SLM_DATA_DIR default redirect (belt-and-suspenders; tests/conftest.py
    also sets this per-test via monkeypatch)
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATH — src/ on PYTHONPATH before any imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ---------------------------------------------------------------------------
# 2. THREAD SAFETY — must be set before any C extension is first imported.
#
#    Background: SLM depends on torch + scikit-learn + lightgbm, each of
#    which bundles a separate libomp.dylib on macOS ARM.  When all three
#    are loaded in the same process and LightGBM forks OpenMP workers,
#    the parallel workers cross libomp runtime boundaries → SIGSEGV in
#    __kmp_suspend_initialize_thread (macOS ARM, Python crash popup).
#
#    Additionally, LanceDB starts a Rust/tokio background event loop thread
#    on first use.  That thread + LightGBM OpenMP workers = same crash.
#
#    OMP_NUM_THREADS=1 → serial OpenMP execution → no parallel fork →
#    no cross-runtime race → crash eliminated for the entire test session.
#    Production code (SLM MCP server) uses OMP_NUM_THREADS=2 which is safe
#    because LanceDB is opt-in and not co-loaded in the MCP subprocess.
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Intel OpenMP coexistence
os.environ["OMP_NUM_THREADS"] = "1"            # Serial — no parallel fork in tests
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# v3.6.x: TOKENIZERS_PARALLELISM=false prevents HuggingFace tokenizers from
# spawning internal threadpool workers inside xdist worker processes.  Under
# -n8, each worker already owns its own Python GIL; a separate tokenizer
# threadpool inside each worker creates a second layer of native threads that
# races with LanceDB's Rust/tokio background thread, producing the same
# multi-libomp SIGSEGV class as LightGBM alone.  Setting false here at
# collection time (before any HF import) ensures the env var is visible to
# every xdist worker process, which inherits this process's environment.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ORT_DISABLE_COREML"] = "1"         # Prevent ONNX CoreML 3-5 GB alloc on ARM
# v3.6.18: malloc allocator prevents jemalloc GC SIGSEGV on macOS ARM.
# Background: Python's default tcmalloc/jemalloc builds (via CPython extensions)
# can SIGSEGV when fork()ed inside a test under heavy parallel OpenMP + Rust tokio.
# PYTHONMALLOC=malloc forces the system allocator — same crash class as
# OMP_NUM_THREADS=1 but covers a complementary failure path.
os.environ.setdefault("PYTHONMALLOC", "malloc")

# ---------------------------------------------------------------------------
# 3. DATA DIR — belt-and-suspenders default (tests/conftest.py overrides
#    this per-test via monkeypatch; this catches any test that bypasses it)
# ---------------------------------------------------------------------------
_TMP = Path(__file__).parent / ".pytest_tmp_data"
_TMP.mkdir(exist_ok=True)
os.environ.setdefault("SLM_DATA_DIR", str(_TMP))

# ---------------------------------------------------------------------------
# 4. UTILITY FIXTURES — available in ALL test modules
# ---------------------------------------------------------------------------
import pytest  # noqa: E402 — must come after env setup for import ordering


@pytest.fixture()
def free_port() -> int:
    """Return a free ephemeral TCP port on 127.0.0.1.

    Binds to port 0 (kernel assigns a free port), reads back the assigned
    port number, closes the socket, and returns the port.

    Use this fixture anywhere a test needs a local server port without
    colliding with the live SLM daemon (port 8765) or any other fixed port.

    CRIT note — race window: there is a brief period between socket.close()
    and the caller binding/listening on the returned port.  In practice this
    is safe for unit and integration tests because:
    (a) each xdist worker is an isolated process, so port reuse across workers
        is not a concern within a single gate run;
    (b) the OS does not immediately reassign the just-released port to another
        caller on most kernels (TCP TIME_WAIT protection).
    If a test requires the port to remain reserved, it should bind a server
    socket itself before calling free_port() and pass the socket's port down.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _sock:
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _sock.bind(("127.0.0.1", 0))
        return int(_sock.getsockname()[1])
