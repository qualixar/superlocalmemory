# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for v3.4.42 — engine_wiring reranker warmup-status disambiguation (B2b).

The bug:
  Any CLI process (slm health, slm doctor, slm recall) that wires a
  RetrievalEngine while the unified daemon is running could surface an
  alarming fallback warning even though the daemon's reranker was healthy.
  The CLI process's warmup was blocked by the machine-wide singleton
  (correct behavior), but the warning was indistinguishable from a real
  failure and undermined first-run confidence.

The fix:
  After warmup_sync returns False, probe _is_reranker_worker_alive(). If
  another process owns the worker (the legitimate singleton case), log INFO
  describing the situation. If background warmup does not finish, disclose
  fallback scoring at INFO and route diagnostics through ``slm doctor``.
"""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

from superlocalmemory.core.engine_wiring import _log_reranker_warmup_status


class TestRerankerWarmupLogDisambiguation:
    """v3.4.42 fix: distinguish singleton-held (benign) from actual failure."""

    def test_logs_info_when_warmup_sync_returns_true(self, caplog) -> None:
        """Healthy warmup logs INFO 'warm and ready' — unchanged behavior."""
        rr = MagicMock()
        rr.warmup_sync.return_value = True

        with caplog.at_level(logging.INFO, logger="superlocalmemory.core.engine_wiring"):
            _log_reranker_warmup_status(rr)

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("warm and ready" in r.message for r in infos)
        assert warns == []

    def test_logs_info_not_warning_when_singleton_held(self, caplog) -> None:
        """B2b: singleton held by another process → INFO, not WARNING.

        This is the bug fix — before v3.4.42 this case logged a misleading
        WARNING about reranker warmup failure even though the daemon's
        reranker was perfectly fine.
        """
        rr = MagicMock()
        rr.warmup_sync.return_value = False

        with patch(
            "superlocalmemory.retrieval.reranker._is_reranker_worker_alive",
            return_value=True,
        ), caplog.at_level(logging.INFO, logger="superlocalmemory.core.engine_wiring"):
            _log_reranker_warmup_status(rr)

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("held by another process" in r.message for r in infos), (
            "Expected an INFO line explaining the singleton ownership; got: "
            + repr([r.message for r in caplog.records])
        )
        assert warns == [], (
            "Singleton-held case must NOT log WARNING — that was the v3.4.42 false positive."
        )

    def test_logs_info_not_warning_when_background_warmup_does_not_finish(self, caplog) -> None:
        """First-run fallback is disclosed without surfacing a launch-day warning."""
        rr = MagicMock()
        rr.warmup_sync.return_value = False

        with patch(
            "superlocalmemory.retrieval.reranker._is_reranker_worker_alive",
            return_value=False,
        ), caplog.at_level(logging.INFO, logger="superlocalmemory.core.engine_wiring"):
            _log_reranker_warmup_status(rr)

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("fallback scoring" in r.message for r in infos)
        assert warns == []

    def test_probe_exception_still_discloses_fallback_without_warning(self, caplog) -> None:
        """A probe failure cannot turn background warmup into a terminal warning."""
        rr = MagicMock()
        rr.warmup_sync.return_value = False

        with patch(
            "superlocalmemory.retrieval.reranker._is_reranker_worker_alive",
            side_effect=RuntimeError("probe boom"),
        ), caplog.at_level(logging.INFO, logger="superlocalmemory.core.engine_wiring"):
            _log_reranker_warmup_status(rr)

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("fallback scoring" in r.message for r in infos)
        assert warns == []
