# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Soft prompts the store can hold, a review that always runs, and session text
that carries no telemetry.

Found on the author's store (9 Nov 2026): every daily soft-prompt run stored its
``workflow_pattern`` prompt, then tried to store a ``topic_interest`` one, which
the table's CHECK constraint does not allow, and the IntegrityError ended the
run before the lifecycle review. Nothing reported it (debug level), so a
keyword-soup ``tech_preference`` prompt from 23 Aug 2026 ("The user's preferred
technology stack includes: file, session, ended, ...") was never retired and
every session started with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from superlocalmemory.core.config import ParameterizationConfig
from superlocalmemory.hooks.auto_parameterize import AutoParameterizeHook
from superlocalmemory.parameterization import soft_prompt_generator as spg
from superlocalmemory.parameterization.prompt_injector import PromptInjector
from superlocalmemory.parameterization.prompt_lifecycle import PromptLifecycleManager
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

PID = "default"


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _schema_categories() -> set[str]:
    src = (Path(spg.__file__).parents[1] / "storage" / "schema_v32.py").read_text()
    block = src[src.index("soft_prompt_templates"):]
    allowed = re.search(r"category\s+IN\s*\(([^)]*)\)", block).group(1)
    return set(re.findall(r"'([a-z_]+)'", allowed))


def test_the_generator_only_emits_what_the_table_accepts() -> None:
    assert spg.STORABLE_CATEGORIES == _schema_categories()
    assert set(spg.CATEGORY_PRIORITY_ORDER) <= spg.STORABLE_CATEGORIES


def _Pattern(category: str, key: str, value: str):
    from superlocalmemory.parameterization.pattern_extractor import (
        PatternAssertion,
        PatternCategory,
    )

    return PatternAssertion(category=PatternCategory(category), key=key, value=value,
                            confidence=0.9, evidence_count=9, source="test",
                            source_ids=("x",), created_at="2026-11-09T00:00:00+00:00")


class _Extractor:
    def __init__(self, patterns) -> None:
        self._patterns = patterns

    def extract(self, profile_id: str):
        return list(self._patterns)


def _stale_tech_prompt(db) -> None:
    db.execute(
        "INSERT INTO soft_prompt_templates (prompt_id, profile_id, category, content, "
        "source_pattern_ids, confidence, effectiveness, token_count, retention_score, "
        "active, version, created_at, updated_at) VALUES ('old', ?, 'tech_preference', "
        "'The user''s preferred technology stack includes: file, session, ended.', '[]', "
        "1.0, 0.5, 12, 1.0, 1, 16, '2026-08-23 09:43:37', '2026-08-23 09:43:37')", (PID,))


def test_a_run_with_a_topic_pattern_still_reviews_and_retires_stale_prompts(db) -> None:
    _stale_tech_prompt(db)
    config = ParameterizationConfig()
    patterns = [_Pattern("topic_interest", "topics", "user, qualixar, session, ended"),
                _Pattern("workflow_pattern", "query_type", "general workflow")]
    generator = spg.SoftPromptGenerator(config)
    hook = AutoParameterizeHook(_Extractor(patterns), generator,
                                PromptInjector(db, generator, config),
                                PromptLifecycleManager(db, config), config)
    result = hook.on_consolidation_complete(PID)
    assert result.get("lifecycle", {}).get("removed") == 1, result
    active = {r["category"] for r in db.execute(
        "SELECT category FROM soft_prompt_templates WHERE profile_id=? AND active=1", (PID,))}
    assert "tech_preference" not in active


def test_session_text_carries_no_tool_telemetry(db) -> None:
    db.execute(
        "INSERT INTO soft_prompt_templates (prompt_id, profile_id, category, content, "
        "source_pattern_ids, confidence, effectiveness, token_count, retention_score, "
        "active, version, created_at, updated_at) VALUES ('wf', ?, 'workflow_pattern', "
        "'The user typically frequently uses Read (6816 times).', '[]', 1.0, 0.5, 9, 1.0, 1, 1, "
        "datetime('now'), datetime('now'))", (PID,))
    config = ParameterizationConfig()
    injector = PromptInjector(db, spg.SoftPromptGenerator(config), config)
    assert "6816 times" not in injector.get_injection_context(PID)
