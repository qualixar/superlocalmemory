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

import sqlite3
import sys
from unittest.mock import MagicMock, patch

# Live-root isolation (audit hook, pytest-owned HOME and data root) lives in
# isort: off
# tests/_isolation_plugin.py so it also holds under --noconftest. Importing it
# here is a no-op when pyproject's addopts already loaded it.
from tests._isolation_plugin import (  # noqa: E402,F401  (re-exported for tests)
    _LIVE_ROOT_GUARD,
    ensure_registered as _ensure_isolation_plugin,
)

import numpy as np  # noqa: E402  (isolation must be installed before imports)
import pytest  # noqa: E402  (isolation must be installed before imports)

# A test that leaves a product thread (slm-*, LanceDB's loop) or a child
# process (embedding/reranker worker) running fails by name (audit C-3).
from tests.thread_leak_guard import (  # noqa: E402,F401  (pytest hook registration)
    pytest_runtest_setup,
    pytest_runtest_teardown,
)
# isort: on

pytest_plugins = ["tests.daemon_descriptor_guard"]


def pytest_configure(config) -> None:
    """Fail tests on swallowed live-root refusals even when addopts were overridden."""
    _ensure_isolation_plugin(config)
    from tests.dev_metadata import check_installed_version

    check_installed_version()


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
    try:
        from superlocalmemory.storage.deferred_writes import shutdown_deferred_writes
        shutdown_deferred_writes()
    except Exception:
        pass
    # Stop any config-store hot-reload watchdog daemon threads. A live watchdog
    # racing interpreter finalization intermittently SIGSEGVs on macOS (the
    # watchdog fsevents extension is co-loaded). Signalling + joining them here
    # keeps the harness deterministic; store.py also does this via atexit.
    try:
        from superlocalmemory.optimize.config.store import _stop_all_watchdogs
        _stop_all_watchdogs()
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
    mock_pool.store.return_value = {
        "ok": True,
        "fact_ids": ["fact-mock"],
        "count": 1,
        "operation_id": "operation-mock",
        "pending_id": None,
        "materialization_state": "complete",
    }
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
        # 4.1.18: with the default "auto", a dev Mac that has laya-mlx installed
        # would start a real Laya worker in every engine a test builds.
        _patch(
            "superlocalmemory.core.engine_wiring.init_sufficiency_judge",
            return_value=None,
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
def _machine_not_pressed_unless_a_test_says_so(request, monkeypatch):
    """SLM releases models and refuses to load them while the machine is short of
    memory (``infra.memory_pressure``). A test must not depend on how loaded the
    machine running it is, so every test sees a calm machine; the tests of the
    signal itself are marked ``real_memory_pressure``."""
    if request.node.get_closest_marker("real_memory_pressure"):
        return
    from superlocalmemory.infra import memory_pressure

    calm = memory_pressure.Reading(False, None, 0.0, None, "")
    monkeypatch.setattr(memory_pressure, "reading", lambda: calm)
    monkeypatch.setattr(memory_pressure, "pressed", lambda: False)


@pytest.fixture(autouse=True)
def _withholding_off_unless_a_test_turns_it_on(monkeypatch):
    """4.2.30 D-28 withholds delivered memories at random by design (on by
    default for users), so a test that checks what a session start delivers or
    the order of a recall would fail one run in ten or twenty. Tests of the
    randomiser itself turn it on (``SLM_FEATURE_OUTCOMES_HOLDOUT=on``,
    ``SLM_FEATURE_OUTCOMES_SWAPS=on``) or delete the variable to see the shipped
    default."""
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "off")
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_SWAPS", "off")


@pytest.fixture(autouse=True)
def _stop_leaked_config_watchdogs():
    """Stop any config-store hot-reload watchdog left running by a test.

    Daemon-boot tests call ``ConfigStore.start_watchdog()`` on the shared
    singleton (via unified_daemon) and never stop it, so that daemon thread
    otherwise keeps running for the whole rest of the session. A live watchdog
    thread concurrent with native-extension code (sqlite/numpy/torch) in an
    unrelated later test intermittently SIGSEGVs on macOS. Stopping it at each
    test boundary keeps the watchdog scoped to the test that started it and
    makes full-suite runs deterministic. Cheap no-op when none are running.
    """
    yield
    try:
        from superlocalmemory.optimize.config.store import _stop_all_watchdogs
        _stop_all_watchdogs()
    except Exception:
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
    try:
        from superlocalmemory.storage.deferred_writes import shutdown_deferred_writes
        shutdown_deferred_writes()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def closes_admission_journals(monkeypatch):
    """Close every ``AdmissionJournal`` a test builds, the way shutdown does.

    A journal owns a group-commit writer thread (``slm-journal-writer``) and a
    pool of reader connections. The runtime closes its journal in ``stop()``;
    tests build journals inline in many folders and rarely close them, so the
    writer stayed parked until its idle exit and was reported against the
    test. Every journal built during any test is closed when the test ends,
    so a new test cannot leak one by forgetting to. ``close()`` is not final:
    a journal held by a wider-scoped fixture reopens on its next operation.
    """
    from superlocalmemory.storage.admission_journal import AdmissionJournal

    built: list[AdmissionJournal] = []
    original_init = AdmissionJournal.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(AdmissionJournal, "__init__", _tracking_init)
    yield built
    for journal in built:
        journal.close()


@pytest.fixture(autouse=True)
def _fresh_recall_cache():
    """Each test asks its own questions: kept recall answers and dashboard
    summaries never cross tests."""
    def clear():
        cache = sys.modules.get("superlocalmemory.server.recall_cache")
        if cache is not None:
            cache.clear()
        views = sys.modules.get("superlocalmemory.server.view_cache")
        if views is not None:
            views.drop()

    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def _reset_daemon_enrichment_pool():
    """Give back the daemon's enrichment pool a test created (audit C-5).

    The pool is module-global in ``unified_daemon`` and created on the first
    inline enrichment. Only the app lifespan shuts it down, so a ``TestClient``
    used without its context manager left ``slm-enrich`` threads running for
    the rest of the session. After each test: shut it down, join its threads
    (bounded), and reopen it for the next test -- exactly what a daemon
    restart in one process does.
    """
    yield
    daemon = sys.modules.get("superlocalmemory.server.unified_daemon")
    if daemon is None or getattr(daemon, "_enrichment_pool", None) is None:
        return
    from superlocalmemory.core.thread_join import executor_threads, join_threads

    threads = executor_threads(daemon._enrichment_pool)
    daemon._shutdown_enrichment_pool_if_created()
    daemon._open_enrichment_pool()
    join_threads(threads, owner="test enrichment pool")


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
    # 4.1.25: the answer-support gate has its own tests (test_hard_abstain,
    # test_support_gate, the answerability harness). A mock embedder's random
    # vectors give it nothing to measure, so engine tests that are about
    # something else would abstain on every question.
    config.retrieval.answer_support_gate = False
    return config


def force_sync_enrichment(engine):
    """Route ``engine.store`` through the SYNCHRONOUS full-enrichment path.

    v3.8.2 made ``engine.store()`` queryable-first: it commits a recallable
    fact immediately and defers enrichment (embeddings, graph edges, scenes,
    BM25 tokens, entities, consolidation) to the daemon's background
    materializer. Direct-engine integration tests that assert those artifacts
    exist right after a store must instead drive the complete path
    (``require_complete=True``) — the same full-enrichment entry
    ``canonical_store`` uses by default — so enrichment is materialized before
    the assertions run. Daemon-backed production still enriches via the
    materializer; this only makes the direct-engine test path deterministic.
    """
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    def _sync_store(content, session_id="", session_date=None, speaker="",
                    role="user", metadata=None, *, scope="personal",
                    shared_with=None):
        return canonical_store(
            engine, content, source_type="python-api",
            trusted_actor_id=local_trusted_actor_id("python-api"),
            metadata=metadata, scope=scope, shared_with=shared_with,
            session_id=session_id, session_date=session_date,
            speaker=speaker, role=role, require_complete=True,
        )

    engine.store = _sync_store
    return engine


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
