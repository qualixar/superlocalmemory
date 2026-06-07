"""MCP server constructs a LIGHT MemoryEngine.

End-user contract enforced here:
- `mcp.server.get_engine()` returns a LIGHT engine (no embedder, no LLM,
  no retrieval engine).
- Importing `mcp.server` and constructing the engine does not import the
  ONNX runtime into the process.

These are structural assertions — they don't measure RSS directly, but
together they guarantee the dominant RSS contributor (the ONNX embedder,
~800 MB-1 GB) never loads in the MCP process. Absolute RSS numbers for
Varun's MBP are captured by the companion bench script at
`.backup/plans/slm-3426-unified-queue/baseline/mcp_light_rss.py`.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from superlocalmemory.core.engine_capabilities import Capabilities


def test_get_engine_returns_light_capability():
    # Reset any prior singleton so the test is hermetic.
    from superlocalmemory.mcp import server
    server.reset_engine()

    # V3.5.9: McpEmbedderProxy attaches when the daemon is reachable. The
    # test must not depend on a real daemon being up or down — mock the
    # proxy availability check to False so the LIGHT engine is truly
    # embedder-free regardless of environment.
    import unittest.mock as _mock
    with _mock.patch(
        "superlocalmemory.core.mcp_embedder_proxy.McpEmbedderProxy.is_available",
        return_value=False,
    ):
        engine = server.get_engine()
    try:
        assert engine.capabilities is Capabilities.LIGHT
        assert engine._embedder is None
        assert engine._retrieval_engine is None
        assert engine._llm is None
    finally:
        server.reset_engine()


def test_get_engine_still_exposes_db_layer():
    from superlocalmemory.mcp import server
    server.reset_engine()
    engine = server.get_engine()
    try:
        # DB-only attributes the existing MCP resources depend on.
        assert engine.db is not None
        assert engine.profile_id  # non-empty string
    finally:
        server.reset_engine()


def test_get_engine_concurrent_callers_single_instance():
    """FastMCP dispatches tools from multiple threads. Two threads
    racing on a cold ``get_engine()`` must see the same instance and
    must only construct the engine once."""
    import threading
    from superlocalmemory.mcp import server
    server.reset_engine()

    construct_counter = {"n": 0}
    original_init = None

    def _counting_init(self, config, **kw):
        construct_counter["n"] += 1
        return original_init(self, config, **kw)

    from superlocalmemory.core.engine import MemoryEngine
    original_init = MemoryEngine.__init__
    try:
        MemoryEngine.__init__ = _counting_init  # type: ignore[method-assign]

        engines: list = []
        def _call():
            engines.append(server.get_engine())

        threads = [threading.Thread(target=_call) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert construct_counter["n"] == 1, \
            f"get_engine double-constructed: {construct_counter['n']}x"
        first = engines[0]
        assert all(e is first for e in engines), \
            "threads saw different engine instances"
    finally:
        MemoryEngine.__init__ = original_init  # type: ignore[method-assign]
        server.reset_engine()


def test_subprocess_does_not_import_onnxruntime():
    """Spawn a fresh subprocess that imports mcp.server and calls
    get_engine(). Assert onnxruntime is NOT in sys.modules after init.

    This is the load-bearing contract: LIGHT means the ONNX embedder
    module never loads, which is what keeps multi-IDE RSS bounded.
    """
    script = (
        "import os; "
        # Neuter the warmup side-effects — we only measure get_engine.
        "os.environ['SLM_DISABLE_WARMUP_SIDE_EFFECTS'] = '1'; "
        # Skip the top-level dep check in __init__.py that imports onnxruntime.
        "os.environ['SLM_SKIP_DEP_CHECK'] = '1'; "
        # V3.5.9: McpEmbedderProxy attaches when daemon is reachable. Mock
        # the availability check to False so the subprocess is hermetic
        # regardless of whether the daemon happens to be running.
        "import unittest.mock as _m; "
        "_p = _m.patch('superlocalmemory.core.mcp_embedder_proxy.McpEmbedderProxy.is_available', "
        "             return_value=False); "
        "_p.start(); "
        "from superlocalmemory.mcp import server; "
        "engine = server.get_engine(); "
        "import sys; "
        "loaded = any(m.startswith('onnxruntime') for m in sys.modules); "
        "embedder_none = engine._embedder is None; "
        "caps = engine.capabilities.value; "
        "print(f'ONNX_LOADED={loaded};EMBEDDER_NONE={embedder_none};CAPS={caps}')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
             "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1",
             "SLM_SKIP_DEP_CHECK": "1"},
    )
    if proc.returncode != 0:
        pytest.fail(f"subprocess failed:\nstdout:{proc.stdout}\nstderr:{proc.stderr}")
    out = proc.stdout.strip().splitlines()[-1]
    assert "ONNX_LOADED=False" in out, f"ONNX was loaded into MCP process: {out}"
    assert "EMBEDDER_NONE=True" in out, f"Embedder leaked: {out}"
    assert "CAPS=light" in out, f"Engine not LIGHT: {out}"
