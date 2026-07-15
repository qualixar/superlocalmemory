"""Durability and observability regressions for HTTP observation admission."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

from superlocalmemory.server.unified_daemon import ObserveBuffer


def _engine():
    return SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default")


def _decision():
    return SimpleNamespace(
        capture=True,
        category="decision",
        confidence=0.75,
        reason="decision pattern detected",
    )


def test_durable_admission_failure_is_logged_and_not_acknowledged(caplog):
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(_engine())

    with (
        patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto,
        patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
            side_effect=RuntimeError("database unavailable"),
        ),
        caplog.at_level(logging.WARNING, logger="superlocalmemory.unified_daemon"),
    ):
        auto.return_value.evaluate.return_value = _decision()
        result = buf.enqueue("we decided to use sqlite for durable memory")

    assert result["captured"] is False
    assert result["durable"] is False
    assert any("durable admission failed" in record.getMessage() for record in caplog.records)


def test_failed_admission_can_be_retried_immediately():
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(_engine())
    content = "we decided to use sqlite for durable memory"

    with (
        patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto,
        patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
            side_effect=RuntimeError("database unavailable"),
        ),
    ):
        auto.return_value.evaluate.return_value = _decision()
        first = buf.enqueue(content)
        second = buf.enqueue(content)

    assert first["reason"] == "durable admission failed"
    assert second["reason"] == "durable admission failed"


def test_success_returns_durable_operation_receipt():
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(_engine())
    receipt = SimpleNamespace(
        operation_id="op-observe",
        fact_ids=("fact-queryable",),
        state=SimpleNamespace(value="queryable"),
    )

    with (
        patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto,
        patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command"
        ) as build,
    ):
        auto.return_value.evaluate.return_value = _decision()
        build.return_value.submit.return_value = receipt
        result = buf.enqueue("we decided to use sqlite for durable memory")

    assert result == {
        "captured": True,
        "durable": True,
        "queued": True,
        "operation_id": "op-observe",
        "fact_ids": ["fact-queryable"],
        "materialization_state": "queryable",
        "category": "decision",
        "confidence": 0.75,
    }
