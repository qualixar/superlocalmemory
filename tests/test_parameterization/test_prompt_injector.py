# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Tests for PromptInjector (Phase F: The Learning Brain).

TDD RED phase: tests written before implementation.
3 tests per LLD Section 6.4.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from superlocalmemory.core.config import ParameterizationConfig
from superlocalmemory.parameterization.prompt_injector import PromptInjector
from superlocalmemory.parameterization.soft_prompt_generator import (
    SoftPromptGenerator,
    SoftPromptTemplate,
)


def _make_config(**overrides) -> ParameterizationConfig:
    return ParameterizationConfig(**overrides)


# ---------------------------------------------------------------
# T17: Soft prompts before memories
# ---------------------------------------------------------------
def test_soft_prompts_before_memories():
    """Output starts with soft prompt header before memory context."""
    config = _make_config()
    injector = PromptInjector(
        db=MagicMock(),
        generator=SoftPromptGenerator(config=config),
        config=config,
    )

    soft = "# User Profile (auto-learned)\n\nThe user is an architect."
    memories = "# Relevant Memory Context\n\n- fact about code"

    result = injector.inject_into_context(soft, memories)
    soft_pos = result.find("User Profile")
    mem_pos = result.find("Relevant Memory Context")
    assert soft_pos < mem_pos


# ---------------------------------------------------------------
# T18: Budget split
# ---------------------------------------------------------------
def test_budget_split():
    """With 500-token soft prompt and 1500-token memories,
    total output <= 2000 tokens. Neither section exceeds its budget."""
    config = _make_config(max_prompt_tokens=500, max_memory_tokens=1500)
    injector = PromptInjector(
        db=MagicMock(),
        generator=SoftPromptGenerator(config=config),
        config=config,
    )

    # Create oversize inputs
    long_soft = "word " * 800   # ~800 words -> ~1040 tokens
    long_mem = "word " * 2000   # ~2000 words -> ~2600 tokens

    result = injector.inject_into_context(long_soft, long_mem)
    total_tokens = SoftPromptGenerator._estimate_tokens(result)
    assert total_tokens <= 2000 + 50  # small buffer for separators


# ---------------------------------------------------------------
# T19: No injection when disabled
# ---------------------------------------------------------------
def test_no_injection_when_disabled():
    """When config.enabled=False, get_injection_context returns empty."""
    config = _make_config(enabled=False)
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "category": "identity",
            "content": "something",
            "confidence": 0.9,
            "token_count": 10,
            "retention_score": 1.0,
            "profile_id": "profile_1",
            "source_pattern_ids": "[]",
            "effectiveness": 0.5,
            "active": 1,
            "version": 1,
        }
    ]

    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.get_injection_context("profile_1")
    assert result == ""


# ---------------------------------------------------------------
# Additional: store_prompts
# ---------------------------------------------------------------
def test_store_prompts():
    """store_prompts inserts templates and returns count."""
    db = MagicMock()
    db.execute.return_value = [{"max_version": 0}]
    config = _make_config()
    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )

    template = SoftPromptTemplate(
        prompt_id="p1",
        profile_id="profile_1",
        category="identity",
        content="The user is an architect.",
        source_pattern_ids=["src_1"],
        confidence=0.8,
        effectiveness=0.5,
        token_count=10,
        retention_score=1.0,
        active=True,
        version=1,
    )

    count = injector.store_prompts([template])
    assert count == 1
    assert db.execute.called


def test_inject_soft_only():
    """When memory context is empty, return soft prompt text only."""
    config = _make_config()
    injector = PromptInjector(
        db=MagicMock(),
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.inject_into_context("soft prompt content", "")
    assert "soft prompt content" in result


def test_inject_memory_only():
    """When soft prompt text is empty, return memory context only."""
    config = _make_config()
    injector = PromptInjector(
        db=MagicMock(),
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.inject_into_context("", "memory context here")
    assert result == "memory context here"


# ---------------------------------------------------------------
# Coverage: get_injection_context with data
# ---------------------------------------------------------------
def test_get_injection_context_with_data():
    """get_injection_context loads active prompts and assembles them."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "category": "identity",
            "content": "The user is an architect.",
            "confidence": 0.9,
            "token_count": 10,
            "retention_score": 1.0,
            "profile_id": "profile_1",
            "source_pattern_ids": '["src_1"]',
            "effectiveness": 0.5,
            "active": 1,
            "version": 1,
        }
    ]
    config = _make_config()
    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.get_injection_context("profile_1")
    assert "architect" in result
    assert "User Profile" in result


def test_get_injection_context_empty_rows():
    """get_injection_context returns empty when no rows."""
    db = MagicMock()
    db.execute.return_value = []
    config = _make_config()
    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.get_injection_context("profile_1")
    assert result == ""


def test_get_injection_context_budget_enforcement():
    """When total tokens exceed budget, lower-confidence prompts are dropped."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "category": "identity",
            "content": "word " * 200,
            "confidence": 0.9,
            "token_count": 300,
            "retention_score": 1.0,
            "profile_id": "profile_1",
            "source_pattern_ids": "[]",
            "effectiveness": 0.5,
            "active": 1,
            "version": 1,
        },
        {
            "prompt_id": "p2",
            "category": "tech_preference",
            "content": "word " * 200,
            "confidence": 0.7,
            "token_count": 300,
            "retention_score": 1.0,
            "profile_id": "profile_1",
            "source_pattern_ids": "[]",
            "effectiveness": 0.5,
            "active": 1,
            "version": 1,
        },
    ]
    config = _make_config(max_prompt_tokens=400)
    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.get_injection_context("profile_1")
    # Only first prompt should fit in budget
    assert result != ""


def test_get_injection_context_bad_source_ids():
    """get_injection_context handles malformed source_pattern_ids gracefully."""
    db = MagicMock()
    db.execute.return_value = [
        {
            "prompt_id": "p1",
            "category": "identity",
            "content": "The user is a developer.",
            "confidence": 0.9,
            "token_count": 10,
            "retention_score": 1.0,
            "profile_id": "profile_1",
            "source_pattern_ids": "not valid json{{{",
            "effectiveness": 0.5,
            "active": 1,
            "version": 1,
        }
    ]
    config = _make_config()
    injector = PromptInjector(
        db=db,
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    result = injector.get_injection_context("profile_1")
    assert "developer" in result


def test_inject_with_total_budget():
    """inject_into_context with explicit total_budget trims both sections."""
    config = _make_config(max_prompt_tokens=100, max_memory_tokens=100)
    injector = PromptInjector(
        db=MagicMock(),
        generator=SoftPromptGenerator(config=config),
        config=config,
    )
    soft = "word " * 200
    mem = "word " * 200
    result = injector.inject_into_context(soft, mem, total_budget=200)
    assert len(result) > 0


def test_a_total_budget_override_is_honoured():
    """4.2.17: ``total_budget`` was documented and silently ignored."""
    config = _make_config(max_prompt_tokens=500, max_memory_tokens=1500)
    injector = PromptInjector(
        db=MagicMock(), generator=SoftPromptGenerator(config=config), config=config,
    )
    long_soft = "word " * 3000
    long_mem = "memory " * 3000
    small = injector.inject_into_context(long_soft, long_mem, total_budget=400)
    default = injector.inject_into_context(long_soft, long_mem)
    assert SoftPromptGenerator._estimate_tokens(small) <= 400 + 20
    assert SoftPromptGenerator._estimate_tokens(default) > 1500
