# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Root conftest — shared fixtures for Phase 0 Safety Net.

Provides in-memory DB, mock embedder, Mode A config, and
engine-with-mock-deps fixtures used across all test modules.

V3.3.7: Added session-scoped worker cleanup to prevent orphaned
subprocess workers (reranker_worker, embedding_worker) from leaking
memory across parallel test runs. Each worker consumes 0.5-1.5 GB.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Capture and deny the real user namespace before replacing environment paths.
_REAL_HOME = Path.home().resolve()
_REAL_DATA_ROOT = Path(
    os.environ.get("SLM_DATA_DIR")
    or os.environ.get("SL_MEMORY_PATH")
    or os.environ.get("SLM_HOME")
    or (_REAL_HOME / ".superlocalmemory")
).expanduser().resolve(strict=False)


def _pytest_isolation_audit(event: str, args: tuple) -> None:
    """Deny direct live-state opens and public-daemon socket connections."""
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        candidate = Path(args[0]).expanduser().resolve(strict=False)
        if candidate == _REAL_DATA_ROOT or candidate.is_relative_to(_REAL_DATA_ROOT):
            raise PermissionError(f"pytest denied live SLM state path: {candidate}")
    if event == "socket.connect" and len(args) >= 2:
        address = args[1]
        if (
            isinstance(address, tuple)
            and len(address) >= 2
            and str(address[0]).lower() in {"127.0.0.1", "localhost", "::1"}
            and int(address[1]) in {8765, 8767}
        ):
            raise PermissionError(
                f"pytest denied live SLM daemon port: {address[1]}"
            )


sys.addaudithook(_pytest_isolation_audit)

# Establish a pytest-owned namespace before test modules are imported. Several
# production modules resolve HOME and daemon paths at import time, so a normal
# fixture is too late to protect the user's live installation.
_TEST_ISOLATION_DIR = tempfile.TemporaryDirectory(prefix="slm-pytest-")
_TEST_ROOT = Path(_TEST_ISOLATION_DIR.name).resolve()
_TEST_HOME = _TEST_ROOT / "home"
_TEST_DATA_DIR = _TEST_ROOT / "canonical-data"
_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["SLM_TEST_ISOLATION"] = "1"
os.environ["SLM_TEST_ROOT"] = str(_TEST_ROOT)
os.environ["SLM_TEST_REAL_DATA_ROOT"] = str(_REAL_DATA_ROOT)
os.environ["HOME"] = str(_TEST_HOME)
os.environ["USERPROFILE"] = str(_TEST_HOME)
os.environ["XDG_CONFIG_HOME"] = str(_TEST_HOME / ".config")
os.environ["XDG_CACHE_HOME"] = str(_TEST_HOME / ".cache")
os.environ["XDG_DATA_HOME"] = str(_TEST_HOME / ".local" / "share")
os.environ["SLM_DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["SL_MEMORY_PATH"] = str(_TEST_ROOT / "wrong-legacy-alias")
os.environ["SLM_HOME"] = str(_TEST_ROOT / "wrong-hook-alias")
os.environ["SLM_DAEMON_PORT"] = str(20_000 + secrets.randbelow(40_000))
os.environ["SLM_TEST_INSTANCE_CAPABILITY"] = secrets.token_urlsafe(32)
for _unsafe_env in (
    "SLM_TEST_ALLOW_LIVE_HOME",
    "SLM_HOOK_DAEMON_URL",
    "SLM_MESH_PEER_URL",
    "SLM_MESH_SHARED_SECRET",
    "SLM_MESH_HOST",
    "SLM_MESH_WS_PORT",
    "SLM_MESH_DISCOVERY",
    "SLM_DAEMON_HOST",
    "SLM_HOST",
):
    os.environ.pop(_unsafe_env, None)

import numpy as np  # noqa: E402  (isolation must be installed before imports)
import pytest  # noqa: E402  (isolation must be installed before imports)


@pytest.fixture(autouse=True, scope="function")
def _block_live_slm_home_writes(tmp_path, monkeypatch):
    """Give each test an isolated data root with no live-home escape hatch."""
    data_dir = tmp_path
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy-alias"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook-alias"))


# V3.3.14: Windows CI fix — KeyboardInterrupt during daemon thread teardown.
# On Windows, when pytest exits, daemon threads (reranker warmup, maintenance
# scheduler, parent watchdog) trigger KeyboardInterrupt that kills the process.
# This hook runs BEFORE pytest's thread cleanup and terminates workers cleanly.
def _cancel_test_timer_threads() -> None:
    """Cancel and join timers created inside the pytest process.

    Production owners remain responsible for their own lifecycle. This is the
    final test-harness containment boundary for a failing test or fixture that
    exits before calling its owner's close method.
    """
    import threading

    timers = [
        thread
        for thread in threading.enumerate()
        if isinstance(thread, threading.Timer)
    ]
    for thread in timers:
        thread.cancel()
    for thread in timers:
        thread.join(timeout=0.2)


def pytest_sessionfinish(session, exitstatus):
    """Clean up all SLM subprocess workers before pytest exits."""
    _cancel_test_timer_threads()
    try:
        from superlocalmemory.core.embeddings import _cleanup_all_embedding_services
        _cleanup_all_embedding_services()
    except Exception:
        pass
    try:
        from superlocalmemory.retrieval.reranker import _cleanup_all_rerankers
        _cleanup_all_rerankers()
    except Exception:
        pass
    # Join any SLM daemon threads to prevent Windows KeyboardInterrupt on exit
    import threading
    for t in threading.enumerate():
        if t.daemon and t.name in ("ce-warmup", "ce-init-warmup", "parent-watchdog"):
            try:
                t.join(timeout=2)
            except Exception:
                pass


def pytest_unconfigure(config) -> None:
    """Release the collection-time namespace before interpreter teardown.

    ``TemporaryDirectory`` otherwise waits for its weakref finalizer, which
    emits a ``ResourceWarning`` under ``-X dev`` and hides real fixture leaks
    in the same shutdown window.
    """
    _TEST_ISOLATION_DIR.cleanup()


# ---------------------------------------------------------------------------
# Session-scoped worker cleanup (prevents orphaned subprocess leak)
# ---------------------------------------------------------------------------

def _kill_orphaned_slm_workers() -> None:
    """Never kill workers by machine-wide process-name matching.

    Worker services created by this pytest process are closed through their
    in-process registries. Process-group ownership is required before any real
    daemon subprocess test may run.
    """
    return


@pytest.fixture(autouse=True, scope="session")
def _prevent_heavy_model_loading():
    """Prevent ALL heavy ML model loading during tests.

    V3.4.11: Mock CrossEncoderReranker (spawns 130MB ONNX subprocess)
    and WorkerPool (spawns 930MB embedding subprocess). Without this,
    the full suite takes 20+ minutes. With it: under 2 minutes.

    Tests that explicitly need real models should patch these back.
    """
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = None
    mock_reranker.warmup_sync.return_value = True
    mock_reranker._kill_worker = MagicMock()

    mock_pool = MagicMock()
    mock_pool.store.return_value = {"ok": True, "fact_ids": ["pending:mock"], "count": 1}
    mock_pool.recall.return_value = {"ok": True, "results": [], "count": 0}
    mock_pool.kill.return_value = None

    patches = [
        _patch(
            "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
            return_value=mock_reranker,
        ),
        _patch(
            "superlocalmemory.core.worker_pool.WorkerPool.shared",
            return_value=mock_pool,
        ),
    ]
    for p in patches:
        p.start()

    yield

    for p in patches:
        p.stop()

    try:
        _kill_orphaned_slm_workers()
    except (KeyboardInterrupt, Exception):
        pass


@pytest.fixture(autouse=True)
def cleanup_slm_workers_between_tests():
    """Kill SLM subprocess workers after EACH test to prevent memory pileup.

    V3.3.12: Safety net — if any test bypasses the session mock and creates
    real workers, this cleans them up. Lightweight when no workers exist.
    """
    yield
    _cancel_test_timer_threads()
    try:
        from superlocalmemory.core.embeddings import _cleanup_all_embedding_services
        _cleanup_all_embedding_services()
    except Exception:
        pass
    try:
        from superlocalmemory.retrieval.reranker import _cleanup_all_rerankers
        _cleanup_all_rerankers()
    except Exception:
        pass


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database with full SLM schema.

    Returns a real sqlite3 Connection backed by :memory:.
    Gives real SQL execution without touching disk.
    """
    from superlocalmemory.storage import schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    schema.create_all_tables(conn)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns deterministic 768-dim vectors.

    Uses seeded RNG keyed on input string for reproducibility.
    Implements: embed(), is_available, compute_fisher_params().
    """
    emb = MagicMock()

    def _embed(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    emb.embed.side_effect = _embed
    emb.is_available = True
    emb.compute_fisher_params.return_value = ([0.0] * 768, [1.0] * 768)
    return emb


@pytest.fixture
def mode_a_config(tmp_path):
    """SLMConfig for Mode A using tmp_path as base_dir."""
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.models import Mode

    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    # V3.4.11: Disable cross-encoder in tests — spawning a real reranker
    # subprocess per test adds 3s teardown overhead (kills ONNX worker).
    # 13 engine tests × 3s = 39s wasted. Cross-encoder has its own tests.
    config.retrieval.use_cross_encoder = False
    return config


@pytest.fixture
def engine_with_mock_deps(mode_a_config, mock_embedder, tmp_path):
    """A MemoryEngine with mocked LLM and embedder for fast unit tests.

    Initializes with real DB (on disk in tmp_path) and real schema,
    but mocked embeddings and no LLM. Suitable for testing store/recall
    flow without heavy ML dependencies.
    """
    from superlocalmemory.core.engine import MemoryEngine

    engine = MemoryEngine(mode_a_config)

    # Patch embedder initialization to use our mock
    with patch('superlocalmemory.core.engine_wiring.init_embedder', return_value=mock_embedder):
        engine.initialize()
        engine._embedder = mock_embedder

    yield engine
    engine.close()
