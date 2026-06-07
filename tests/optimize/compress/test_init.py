"""Tests for __init__.py — public surface exports."""
from __future__ import annotations


def test_all_exports_present() -> None:
    from superlocalmemory.optimize.compress import (
        CompressRouter,
        CompressTextResult,
        CCRStore,
        CacheAligner,
        AlignResult,
    )
    assert CompressRouter is not None
    assert CompressTextResult is not None
    assert CCRStore is not None
    assert CacheAligner is not None
    assert AlignResult is not None
