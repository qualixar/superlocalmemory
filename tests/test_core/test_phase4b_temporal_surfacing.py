# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Phase 4b — temporal-surfacing regression tests.

Covers:
  - TemporalValidatorConfig exposes event_time_demotion_factor with default 0.5
  - event_time_demotion_factor is overridable
  - MemoryEngine.recall() forwards as_of to run_recall (when set and when None)
  - Default recall call passes as_of=None (byte-identical behavior)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core.config import TemporalValidatorConfig
from superlocalmemory.core.engine import MemoryEngine

# ---------------------------------------------------------------------------
# Config field
# ---------------------------------------------------------------------------


class TestTemporalValidatorConfigEventTimeFactor:
    """TemporalValidatorConfig.event_time_demotion_factor field contract."""

    def test_config_has_event_time_demotion_factor_default(self) -> None:
        """Default value matches the hardcoded constant in temporal_validity_filter."""
        cfg = TemporalValidatorConfig()
        assert cfg.event_time_demotion_factor == 0.5

    def test_config_event_time_factor_overridable(self) -> None:
        """Frozen dataclass allows override at construction time."""
        cfg = TemporalValidatorConfig(event_time_demotion_factor=0.7)
        assert cfg.event_time_demotion_factor == 0.7


# ---------------------------------------------------------------------------
# engine.recall() → run_recall as_of threading
# ---------------------------------------------------------------------------


def _make_stub_response():
    """Minimal duck-typed RecallResponse for run_recall mock."""
    return SimpleNamespace(
        results=[],
        query="stub",
        mode=None,
        query_type="factual",
        channel_weights={},
        total_candidates=0,
        retrieval_time_ms=0.0,
        query_id="stub-qid",
    )


class TestEngineRecallThreadsAsOf:
    """MemoryEngine.recall() must forward as_of to run_recall."""

    def test_recall_threads_as_of_when_set(
        self, engine_with_mock_deps: MemoryEngine, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """as_of is forwarded to run_recall when explicitly provided."""
        captured: dict = {}

        def _fake_run_recall(*args, **kwargs):
            captured.update(kwargs)
            return _make_stub_response()

        monkeypatch.setattr(
            "superlocalmemory.core.recall_pipeline.run_recall",
            _fake_run_recall,
        )

        engine_with_mock_deps.recall("test query", as_of="2026-01-01T00:00:00+00:00")

        assert "as_of" in captured, "run_recall was not called with as_of keyword"
        assert captured["as_of"] == "2026-01-01T00:00:00+00:00"

    def test_recall_threads_as_of_default_none(
        self, engine_with_mock_deps: MemoryEngine, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default recall passes as_of=None — no behavioral change for existing callers."""
        captured: dict = {}

        def _fake_run_recall(*args, **kwargs):
            captured.update(kwargs)
            return _make_stub_response()

        monkeypatch.setattr(
            "superlocalmemory.core.recall_pipeline.run_recall",
            _fake_run_recall,
        )

        engine_with_mock_deps.recall("test query")

        assert "as_of" in captured, "run_recall must always receive as_of kwarg"
        assert captured["as_of"] is None

    def test_default_recall_omits_as_of_value(
        self, engine_with_mock_deps: MemoryEngine, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling recall() without as_of results in run_recall receiving as_of=None."""
        received_as_of: list = []

        def _fake_run_recall(*args, **kwargs):
            received_as_of.append(kwargs.get("as_of", "MISSING"))
            return _make_stub_response()

        monkeypatch.setattr(
            "superlocalmemory.core.recall_pipeline.run_recall",
            _fake_run_recall,
        )

        engine_with_mock_deps.recall("another query")

        assert len(received_as_of) == 1
        assert received_as_of[0] is None, (
            "Default recall must not mutate as_of behavior; expected None"
        )
