# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tests for AutoParameterizeHook (Phase F: The Learning Brain).

TDD RED phase: tests written before implementation.
2 tests per LLD Section 6.6.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from superlocalmemory.core.config import ParameterizationConfig
from superlocalmemory.hooks.auto_parameterize import AutoParameterizeHook
from superlocalmemory.parameterization.pattern_extractor import (
    PatternAssertion,
    PatternCategory,
)


def _make_config(**overrides) -> ParameterizationConfig:
    return ParameterizationConfig(**overrides)


def _make_hook(config=None, patterns=None, prompts=None):
    extractor = MagicMock()
    extractor.extract.return_value = (
        patterns if patterns is not None else [
            PatternAssertion(
                category=PatternCategory.IDENTITY,
                key="role",
                value="Architect",
                confidence=0.8,
                evidence_count=10,
                source="core_memory",
            )
        ]
    )

    generator = MagicMock()
    generator.generate.return_value = (
        prompts if prompts is not None
        else [MagicMock(category="identity")]
    )

    injector = MagicMock()
    injector.store_prompts.return_value = 1

    lifecycle = MagicMock()
    lifecycle.run_lifecycle_review.return_value = {
        "reviewed": 1, "decayed": 0, "removed": 0, "refreshed": 0,
    }

    cfg = config or _make_config()

    return AutoParameterizeHook(
        extractor=extractor,
        generator=generator,
        injector=injector,
        lifecycle=lifecycle,
        config=cfg,
    )


# ---------------------------------------------------------------
# T24: Full pipeline triggered on consolidation
# ---------------------------------------------------------------
def test_on_consolidation_triggers_pipeline():
    """Calling on_consolidation_complete invokes all 4 stages
    and returns status='success'."""
    hook = _make_hook()
    result = hook.on_consolidation_complete("profile_1")

    assert result["status"] == "success"
    assert result["patterns"] == 1
    assert result["prompts"] == 1
    assert "lifecycle" in result


# ---------------------------------------------------------------
# T25: Rate limiting
# ---------------------------------------------------------------
def test_rate_limiting():
    """Calling on_consolidation_complete twice within refresh_interval_hours
    returns status='rate_limited' on second call."""
    hook = _make_hook(config=_make_config(refresh_interval_hours=24))

    # First call: success
    result1 = hook.on_consolidation_complete("profile_1")
    assert result1["status"] == "success"

    # Second call: rate limited
    result2 = hook.on_consolidation_complete("profile_1")
    assert result2["status"] == "rate_limited"


# ---------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------
def test_disabled_config():
    """When config.enabled=False, returns status='disabled'."""
    hook = _make_hook(config=_make_config(enabled=False))
    result = hook.on_consolidation_complete("profile_1")
    assert result["status"] == "disabled"


def test_on_session_end_success():
    """on_session_end with 'success' outcome updates effectiveness."""
    hook = _make_hook()
    # Should not raise
    hook.on_session_end("profile_1", "success")


def test_on_session_end_disabled_tracking():
    """When effectiveness_tracking=False, on_session_end is a no-op."""
    hook = _make_hook(config=_make_config(effectiveness_tracking=False))
    hook.on_session_end("profile_1", "success")


def test_no_patterns():
    """When extractor returns empty patterns, returns status='no_patterns'."""
    hook = _make_hook(patterns=[])
    result = hook.on_consolidation_complete("profile_1")
    assert result["status"] == "no_patterns"


def test_no_prompts_generated():
    """When generator returns empty prompts, returns status='no_prompts'."""
    hook = _make_hook(prompts=[])
    result = hook.on_consolidation_complete("profile_1")
    assert result["status"] == "no_prompts"


def test_on_session_end_failure():
    """on_session_end with 'failure' outcome updates effectiveness."""
    hook = _make_hook()
    hook.on_session_end("profile_1", "failure")


def test_on_session_end_partial():
    """on_session_end with 'partial' outcome updates effectiveness."""
    hook = _make_hook()
    hook.on_session_end("profile_1", "partial")


def test_on_session_end_unknown_outcome():
    """on_session_end with unknown outcome is a no-op."""
    hook = _make_hook()
    hook.on_session_end("profile_1", "unknown_outcome")


def test_rate_limit_invalid_timestamp():
    """Rate limiting with invalid _last_run timestamp skips rate limit."""
    hook = _make_hook()
    hook._last_run = "not-a-valid-iso-date"
    result = hook.on_consolidation_complete("profile_1")
    assert result["status"] == "success"


def test_on_session_end_lifecycle_exception():
    """on_session_end handles exception from lifecycle gracefully."""
    extractor = MagicMock()
    generator = MagicMock()
    injector = MagicMock()
    lifecycle = MagicMock()
    lifecycle.update_effectiveness.side_effect = RuntimeError("DB error")

    hook = AutoParameterizeHook(
        extractor=extractor,
        generator=generator,
        injector=injector,
        lifecycle=lifecycle,
        config=_make_config(),
    )
    # Should not raise
    hook.on_session_end("profile_1", "success")


def test_rate_limit_naive_timestamp():
    """Rate limiting handles naive (no-tzinfo) _last_run timestamp."""
    from datetime import datetime
    hook = _make_hook(config=_make_config(refresh_interval_hours=24))
    # Set a naive timestamp (no timezone)
    hook._last_run = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    result = hook.on_consolidation_complete("profile_1")
    assert result["status"] == "rate_limited"
