# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Shared setup for the server tests."""

from __future__ import annotations

import pytest

#: Tests that are about what the keyword fallback returns, not how fast its
#: source check runs, get a budget no CI runner can exhaust. The production
#: default (0.5 s) ran out on a busy Linux runner in the 4.2.31 full suite and
#: the fallback, failing closed, returned nothing: ``assert ids == {...}`` saw
#: ``set()``. Tests about the budget itself pass their own value and keep it.
GENEROUS_FIDELITY_BUDGET_S = 10.0


@pytest.fixture(autouse=True)
def _generous_fallback_fidelity_budget(monkeypatch):
    from superlocalmemory.server import recall_fallback

    monkeypatch.setitem(
        recall_fallback.recall_keyword_fallback.__kwdefaults__,
        "fidelity_budget_s", GENEROUS_FIDELITY_BUDGET_S,
    )
