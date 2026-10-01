# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for SavingsEstimator (optimize/metrics/estimator.py)."""

from datetime import date, timedelta

import pytest

from superlocalmemory.optimize.metrics.estimator import SavingsEstimator
from superlocalmemory.optimize.storage.db import MetricsSnapshot


def test_estimate_basic():
    est = SavingsEstimator()
    snap = MetricsSnapshot(
        tokens_saved_input=1000,
        tokens_saved_output=500,
        tokens_saved_compress=200,
    )
    result = est.estimate(snap, provider="anthropic")
    assert result["usd"] > 0
    assert result["inr"] > 0
    assert result["tokens_saved_total"] == 1700
    assert result["cache_tokens"] == 1500
    assert result["compress_tokens"] == 200
    assert result["pricing_date"] == "2026-06-07"


def test_estimate_zero():
    est = SavingsEstimator()
    snap = MetricsSnapshot()
    result = est.estimate(snap, provider="anthropic")
    assert result["usd"] == 0.0
    assert result["inr"] == 0.0
    assert result["tokens_saved_total"] == 0


def test_estimate_openai():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="openai")
    assert result["usd"] == 2.50


def test_estimate_gemini():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="gemini")
    assert result["usd"] == 1.25


def test_estimate_unknown_provider_falls_back():
    est = SavingsEstimator()
    snap = MetricsSnapshot(tokens_saved_input=1_000_000)
    result = est.estimate(snap, provider="unknown_provider")
    assert result["usd"] == 3.00  # anthropic fallback


@pytest.mark.parametrize("age, expected", [(0, False), (90, False), (91, True)])
def test_is_stale(monkeypatch, age, expected):
    from superlocalmemory.optimize.metrics import estimator

    table_date = date.fromisoformat(SavingsEstimator._PRICING_DATE)

    class ClockDate(date):
        @classmethod
        def today(cls):
            return table_date + timedelta(days=age)

    monkeypatch.setattr(estimator, "date", ClockDate)
    assert SavingsEstimator()._is_stale() is expected
