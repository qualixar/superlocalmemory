# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | Worker idle-timeout configuration tests

"""Regression tests for v3.4.19 — worker idle-timeout defaults + env overrides.

Guards two separate things:

1. **Defaults**: embedding worker and cross-encoder reranker now keep their
   models warm for 30 min (1800 s) by default. Prior to v3.4.19 this was
   120 s, which caused 30-60 s cold-starts on every recall after a short
   pause.

2. **Kill-switch**: ``SLM_EMBED_IDLE_TIMEOUT`` and
   ``SLM_RERANKER_IDLE_TIMEOUT`` must still override the default so users
   on low-RAM machines can flip back to the old aggressive policy without
   a code change or redeploy.

Both constants are read at import time (module-level ``os.environ.get``),
so these tests manipulate the environment and reload the module.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _reload_with_env(module_name: str, env: dict):
    """Reload ``module_name`` with ``env`` overlaid on ``os.environ``.

    Reload the existing module object instead of deleting it from
    ``sys.modules``. Deleting it leaves classes imported during test
    collection bound to a stale globals dictionary, so later patch targets
    silently affect a different module instance.
    """
    saved = {k: os.environ.get(k) for k in env}
    module = importlib.import_module(module_name)
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

        yield importlib.reload(module)
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        importlib.reload(module)


# ---------------------------------------------------------------------------
# Embedding worker idle timeout
# ---------------------------------------------------------------------------


class TestEmbeddingIdleTimeout:
    """Cover the v3.8.1 stability default and the env override."""

    def test_default_is_30_minutes(self):
        """v3.8.1 keeps the worker warm for an interactive working session.

        Five-minute recycling caused repeated 20-30 second cold starts on
        existing-user databases. Low-RAM installs can keep the shorter policy
        through ``SLM_EMBED_IDLE_TIMEOUT``.
        """
        with _reload_with_env(
            "superlocalmemory.core.embeddings",
            {"SLM_EMBED_IDLE_TIMEOUT": None},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 1800, (
                f"v3.8.1 ships a 30-minute default. Got {mod._IDLE_TIMEOUT_SECONDS}s."
            )

    def test_env_var_overrides_default(self):
        """``SLM_EMBED_IDLE_TIMEOUT=120`` reverts to the legacy aggressive policy."""
        with _reload_with_env(
            "superlocalmemory.core.embeddings",
            {"SLM_EMBED_IDLE_TIMEOUT": "120"},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 120, (
                "Kill-switch broken: SLM_EMBED_IDLE_TIMEOUT should restore 120 s."
            )

    def test_env_var_can_set_30_minutes(self):
        """Power users wanting the old 30-min default can opt in via env."""
        with _reload_with_env(
            "superlocalmemory.core.embeddings",
            {"SLM_EMBED_IDLE_TIMEOUT": "1800"},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 1800

    def test_env_var_accepts_zero_for_immediate_kill(self):
        """Edge case: ``0`` means 'kill immediately' — useful for CI/stress tests."""
        with _reload_with_env(
            "superlocalmemory.core.embeddings",
            {"SLM_EMBED_IDLE_TIMEOUT": "0"},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 0


# ---------------------------------------------------------------------------
# Reranker idle timeout
# ---------------------------------------------------------------------------


class TestRerankerIdleTimeout:
    """Same v3.8.1 warm-session contract for the cross-encoder reranker."""

    def test_default_is_30_minutes(self):
        with _reload_with_env(
            "superlocalmemory.retrieval.reranker",
            {"SLM_RERANKER_IDLE_TIMEOUT": None},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 1800

    def test_env_var_overrides_default(self):
        with _reload_with_env(
            "superlocalmemory.retrieval.reranker",
            {"SLM_RERANKER_IDLE_TIMEOUT": "120"},
        ) as mod:
            assert mod._IDLE_TIMEOUT_SECONDS == 120


class TestEmbeddingWorkerRssLimit:
    """Peak guard agrees with the daemon's per-worker memory watchdog."""

    def test_default_is_2500_mb(self):
        with _reload_with_env(
            "superlocalmemory.core.embedding_worker",
            {"SLM_EMBED_WORKER_RSS_LIMIT_MB": None},
        ) as mod:
            assert mod._RSS_LIMIT_MB == 2500

    def test_env_var_can_restore_aggressive_recycling(self):
        with _reload_with_env(
            "superlocalmemory.core.embedding_worker",
            {"SLM_EMBED_WORKER_RSS_LIMIT_MB": "1800"},
        ) as mod:
            assert mod._RSS_LIMIT_MB == 1800


# ---------------------------------------------------------------------------
# Safety: we did NOT bump recall_worker's idle — it holds data caches, not
# model weights, and respawns cheaply. If a future edit accidentally bumps
# it too, flag it.
# ---------------------------------------------------------------------------


class TestRecallWorkerIdleUnchanged:
    """recall_worker should stay short-lived (data caches go stale)."""

    def test_recall_worker_idle_is_still_short(self):
        """Should be ≤ 300 s. Tests catch a well-meaning future bump."""
        with _reload_with_env(
            "superlocalmemory.core.worker_pool",
            {"SLM_RECALL_IDLE_TIMEOUT": None},
        ) as mod:
            assert mod._IDLE_TIMEOUT <= 300, (
                f"recall_worker should stay short-lived; got {mod._IDLE_TIMEOUT}s. "
                "If this was intentional, update the test."
            )
