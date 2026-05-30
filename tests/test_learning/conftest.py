# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory v3.4.58 — test mock infrastructure

"""conftest for tests/test_learning/ — mocks LightGBM C training.

Unit tests in this directory test the LOGIC of the learning pipeline
(signal ingestion, feature extraction, retrain gating, shadow testing,
bandit loops) — NOT the LightGBM C extension itself.

Mocking strategy:
  - ``lightgbm.Dataset`` → MockDataset  (no C calls, no memory alloc)
  - ``lightgbm.train``   → mock_lgb_train (returns MockBooster instantly)

MockBooster.model_to_string() returns a REAL, loadable LightGBM model
string so that DB persistence + lgb.Booster(model_str=...) round-trips
still exercise the real code path, just with pre-computed weights.

Tests that need real LightGBM C training (e.g. benchmarks, CI perf
regression) belong in tests/test_integration/ and are excluded from
the fast unit test suite.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.fixtures.lgb_mock import (
    MockBooster,
    MockDataset,
    mock_lgb_train,
)


@pytest.fixture(autouse=True, scope="session")
def _mock_lgb_training_for_learning_tests():
    """Patch lightgbm.Dataset and lightgbm.train for this test directory.

    Scope is session so the patch is applied once and held for all tests
    in tests/test_learning/. Both the top-level lightgbm package and the
    local aliases used by the retrain modules are patched.
    """
    with (
        patch("lightgbm.Dataset", MockDataset),
        patch("lightgbm.train", mock_lgb_train),
    ):
        yield
