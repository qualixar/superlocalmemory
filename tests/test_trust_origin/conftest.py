"""These tests were written when the 4.2.30 flags shipped dark; each one turns on the flag it is about."""

from __future__ import annotations

import pytest

from tests.helpers.flag_baseline import env_baseline_off


@pytest.fixture(autouse=True)
def _flags_off_baseline(monkeypatch):
    env_baseline_off(monkeypatch)
