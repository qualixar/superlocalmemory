"""Observation admission counts only a durable M018 submission as captured."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from superlocalmemory.server.unified_daemon import ObserveBuffer


def _engine():
    return SimpleNamespace(_config=SimpleNamespace(scope=None), _profile_id="default")


def test_rejected_item_is_not_reported_as_captured():
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(_engine())
    decision = SimpleNamespace(
        capture=False, category="none", confidence=0.0, reason="no patterns matched"
    )

    with patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto:
        auto.return_value.evaluate.return_value = decision
        result = buf.enqueue("ordinary content that is not capture worthy")

    assert result["captured"] is False
    assert result["durable"] is False


def test_accepted_item_requires_submit_receipt():
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(_engine())
    decision = SimpleNamespace(
        capture=True, category="bug", confidence=0.75, reason="bug pattern detected"
    )
    receipt = SimpleNamespace(
        operation_id="op-1",
        fact_ids=("fact-1",),
        state=SimpleNamespace(value="queryable"),
    )

    with (
        patch("superlocalmemory.hooks.auto_capture.AutoCapture") as auto,
        patch(
            "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command"
        ) as build,
    ):
        auto.return_value.evaluate.return_value = decision
        build.return_value.submit.return_value = receipt
        result = buf.enqueue(
            "we fixed the durable ingestion bug in the daemon",
            trusted_actor_id="authenticated:http-observe",
        )

    assert result["captured"] is True
    assert result["durable"] is True
    build.return_value.submit.assert_called_once()
    submitted = build.return_value.submit.call_args.args[0]
    assert submitted.trusted_actor_id == "authenticated:http-observe"
