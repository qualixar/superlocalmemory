# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A fact the store has retired is never Core Memory.

Found on the author's store (9 Nov 2026): ``remember(replaces=...)`` retired
"Open for 4.2.16: PR #317 ... do not reinstall PyPI 4.2.15" (its
``fact_temporal_validity.system_expired_at`` was set) once 4.2.16 shipped, yet
every Claude Code and Codex session start went on showing it under "Core
Memory (pinned, high-value)". ``get_pinned`` and recall exclude a retired
fact; the session-start fast path (``slm session-context``) and the Core
Memory block compiler read ``atomic_facts`` with their own SQL and never
asked. Both pick their rows through ``core/block_sources``, so that is where
retirement is now checked, together with prompts and bookkeeping.
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
OLD = "Open for 4.2.16: do not reinstall PyPI 4.2.15 on this Mac."
NEW = "SLM 4.2.16 is released and installed on every Mac harness."


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    return mgr


def _save(db, content: str, fact_type: FactType = FactType.OPINION, *,
          importance: float = 0.95, accesses: int = 9) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id=PID, content=content,
                                             metadata={"agent_id": "claude"}))
    fact = AtomicFact(profile_id=PID, memory_id=memory_id, content=content,
                      fact_type=fact_type, confidence=0.95, importance=importance,
                      created_at="2026-11-09T10:00:00+00:00")
    fact_id = db.store_fact(fact)
    for i in range(accesses):
        db.execute("INSERT INTO fact_access_log (log_id, fact_id, profile_id, accessed_at) "
                   "VALUES (?, ?, ?, ?)", (f"{fact_id}-{i}", fact_id, PID,
                                            "2026-11-09T11:00:00+00:00"))
    return fact_id


def _retire(db, fact_id: str, by: str) -> None:
    """What ``remember(replaces=...)`` leaves behind for the old fact."""
    db.execute("INSERT OR IGNORE INTO fact_temporal_validity (fact_id, profile_id) "
               "VALUES (?, ?)", (fact_id, PID))
    db.execute("UPDATE fact_temporal_validity SET system_expired_at = ?, invalidated_by = ? "
               "WHERE fact_id = ? AND profile_id = ?",
               ("2026-11-09T12:00:00+00:00", by, fact_id, PID))


def _session_context(tmp_path, monkeypatch, capsys) -> str:
    from superlocalmemory.cli.commands import cmd_session_context

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    cmd_session_context(Namespace(json=False, full=False, query="", max_age_days=0))
    return capsys.readouterr().out


def _blocks(db) -> str:
    ConsolidationEngine(db, ConsolidationConfig()).compile_core_blocks_mode_a(PID)
    rows = db.execute("SELECT content FROM core_memory_blocks WHERE profile_id=?", (PID,))
    return "\n".join(r["content"] for r in rows)


def test_session_start_never_pins_a_replaced_fact(db, tmp_path, monkeypatch, capsys) -> None:
    old = _save(db, OLD)
    new = _save(db, NEW)
    _retire(db, old, new)
    out = _session_context(tmp_path, monkeypatch, capsys)
    assert "Open for 4.2.16" not in out
    assert "4.2.16 is released" in out


def test_compiled_core_blocks_never_quote_a_replaced_fact(db) -> None:
    for fact_type in (FactType.OPINION, FactType.SEMANTIC, FactType.EPISODIC):
        old = _save(db, f"{OLD} ({fact_type.value})", fact_type)
        new = _save(db, f"{NEW} ({fact_type.value})", fact_type)
        _retire(db, old, new)
    every = _blocks(db)
    assert "Open for 4.2.16" not in every
    assert "4.2.16 is released" in every


def test_a_block_compiled_before_the_fact_was_replaced_is_not_shown(
        db, tmp_path, monkeypatch, capsys) -> None:
    old = _save(db, OLD)
    assert "Open for 4.2.16" in _blocks(db)  # compiled while it was current
    _retire(db, old, _save(db, NEW))
    out = _session_context(tmp_path, monkeypatch, capsys)
    assert "Open for 4.2.16" not in out


def test_a_store_without_the_temporal_table_still_gets_its_context(
        db, tmp_path, monkeypatch, capsys) -> None:
    _save(db, NEW)
    db.execute("DROP TABLE fact_temporal_validity")
    assert "4.2.16 is released" in _blocks(db)
    assert "4.2.16 is released" in _session_context(tmp_path, monkeypatch, capsys)
