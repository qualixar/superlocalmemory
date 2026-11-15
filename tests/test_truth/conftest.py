"""Fixtures for the truth-layer tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def auto_apply_basis(monkeypatch):
    """Lift the W0-18 guard for a test of the retirement path itself (W1 lifts it
    for real when the lawful basis lands)."""
    from superlocalmemory.truth import slot_resolver

    monkeypatch.setattr(slot_resolver, "AUTO_APPLY_BASIS_READY", True)
