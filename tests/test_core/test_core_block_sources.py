# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Core Memory blocks are compiled from memories, not from prompts or hook markers.

Measured on the author's store (9 Nov 2026): of the 15 facts behind the
user_profile, project_context and learned_preferences blocks that every
Claude Code and Codex session start injects as "pinned, high-value", 8 were
fragments of captured prompts ("You need to give me a google flow avatar
prompt.", "Should the user submit it to the Claude form?") and 6 were the hooks'
own markers ("session ended 2026-11-09 13:46", "File changed: x.py"). Only 2
were memories an agent saved.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.core.config import ConsolidationConfig
from superlocalmemory.core.consolidation_engine import ConsolidationEngine
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

PID = "default"


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content: str, fact_type: FactType, *, meta: dict | None = None,
          created: str = "2026-11-01T10:00:00+00:00", confidence: float = 0.9,
          accesses: int = 0) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id=PID, content=content,
                                             metadata=meta or {}))
    fact = AtomicFact(profile_id=PID, memory_id=memory_id, content=content,
                      fact_type=fact_type, created_at=created, confidence=confidence)
    fact_id = db.store_fact(fact)
    for i in range(accesses):
        db.execute("INSERT INTO fact_access_log (log_id, fact_id, profile_id, accessed_at) "
                   "VALUES (?, ?, ?, ?)", (f"{fact_id}-{i}", fact_id, PID,
                                            "2026-11-02T10:00:00+00:00"))
    return fact_id


def _blocks(db) -> dict[str, str]:
    engine = ConsolidationEngine(db, ConsolidationConfig())
    engine.compile_core_blocks_mode_a(PID)
    rows = db.execute("SELECT block_type, content FROM core_memory_blocks WHERE profile_id=?",
                      (PID,))
    return {r["block_type"]: r["content"] for r in rows}


PROMPT = {"source": "auto-capture", "category": "request"}


def test_profile_blocks_skip_captured_prompts_and_hook_markers(db) -> None:
    _save(db, "You need to give me a google flow avatar prompt.", FactType.SEMANTIC,
          meta=PROMPT, accesses=9)
    _save(db, "Should the user submit it to the Claude form?", FactType.OPINION,
          meta=PROMPT, accesses=9)
    _save(db, "[proj] session ended 2026-11-09 13:46", FactType.EPISODIC,
          meta={"tags": "session-end"}, created="2026-11-09T13:46:00+00:00")
    _save(db, "File changed: config.py", FactType.EPISODIC,
          meta={"tags": "hook-file-change"}, created="2026-11-09T13:47:00+00:00")
    _save(db, "Varun prefers prose over bullet lists in reports.", FactType.OPINION,
          meta={"agent_id": "claude"}, accesses=2)
    _save(db, "SLM 4.2.13 shipped on 29 Oct 2026.", FactType.EPISODIC,
          meta={"agent_id": "claude"}, created="2026-10-29T10:00:00+00:00")
    blocks = _blocks(db)
    every = "\n".join(blocks.values())
    for junk in ("google flow avatar", "Claude form", "session ended", "File changed"):
        assert junk not in every, junk
    assert "prose over bullet lists" in blocks["user_profile"]
    assert "prose over bullet lists" in blocks["learned_preferences"]
    assert "4.2.13 shipped" in blocks["project_context"]


def test_a_memory_tagged_by_the_user_is_still_a_source(db) -> None:
    # Tags decide nothing: only a captured prompt or bookkeeping is excluded.
    _save(db, "Decision: releases go out on Tuesdays.", FactType.EPISODIC,
          meta={"tags": "session-end-notes,release"})
    assert "releases go out on Tuesdays" in _blocks(db)["project_context"]


def test_agent_markers_go_but_a_session_summary_with_content_stays(db) -> None:
    _save(db, "session ended (exit) at 2026-11-07 13:49 in config.", FactType.EPISODIC,
          meta={"tags": "agent:agy,project:config,session-end,exit"},
          created="2026-11-07T13:49:00+00:00")
    _save(db, "[api] session ended 2026-11-08 10:00 | branch: fix/4.2.2 | recent: 0545bc9a"
              " fix: calibration reads the workspace", FactType.EPISODIC,
          meta={"tags": "session-end"}, created="2026-11-08T10:00:00+00:00")
    context = _blocks(db)["project_context"]
    assert "(exit)" not in context
    assert "branch: fix/4.2.2" in context


def _session_context(db, tmp_path, monkeypatch, capsys) -> str:
    from superlocalmemory.cli.commands import cmd_session_context

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    cmd_session_context(Namespace(json=False, full=False, query="", max_age_days=0))
    return capsys.readouterr().out


def test_session_start_never_shows_telemetry_or_a_block_built_from_junk(
        db, tmp_path, monkeypatch, capsys) -> None:
    good = _save(db, "Varun prefers prose over bullet lists in reports.", FactType.OPINION,
                 meta={"agent_id": "claude"})
    junk = _save(db, "You need to give me a google flow avatar prompt.", FactType.SEMANTIC,
                 meta=PROMPT)
    rows = [("behavioral_patterns", "time_of_day: hour_21 (confidence: 1.00)", []),
            ("user_profile", "You need to give me a google flow avatar prompt.", [junk]),
            ("learned_preferences", "Varun prefers prose over bullet lists in reports.", [good])]
    for i, (block_type, content, ids) in enumerate(rows):  # as a 4.2.13 compile left them
        db.execute("INSERT INTO core_memory_blocks (block_id, profile_id, block_type, content, "
                   "source_fact_ids, char_count, version, compiled_by, created_at, updated_at) "
                   "VALUES (?, ?, ?, ?, ?, ?, 1, 'rules', '2026-11-09', '2026-11-09')",
                   (f"b{i}", PID, block_type, content, json.dumps(ids), len(content)))
    out = _session_context(db, tmp_path, monkeypatch, capsys)
    assert "time_of_day" not in out
    assert "google flow avatar" not in out
    assert "prose over bullet lists" in out
