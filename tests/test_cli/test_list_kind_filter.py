# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm list --kind`` (LLD/WP8 4.1.19) — same ``list_recent_facts`` helper
the MCP tool uses, so the two surfaces cannot disagree.
"""

from __future__ import annotations

import json
from argparse import Namespace

import pytest

from superlocalmemory.cli import commands
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord
from tests._portable import child_env_base


class _FakeEngine:
    def __init__(self, db, config=None) -> None:
        self._db = db
        self.profile_id = "default"
        self._config = config

    def initialize(self) -> None:
        pass


def _save(db, content, *, kind=None, source=None, confidence=None) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    fact = AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source,
                      memory_kind_confidence=confidence)
    return db.store_fact(fact)


@pytest.fixture()
def fake_engine(tmp_path, monkeypatch):
    from superlocalmemory.core import engine as engine_mod

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    engine = _FakeEngine(db)
    monkeypatch.setattr(engine_mod, "MemoryEngine", lambda *a, **k: engine)
    return engine


@pytest.fixture()
def fake_engine_with_strict_threshold(tmp_path, monkeypatch):
    """Same engine, but with a configured ``display_min_confidence`` (0.50)
    stricter than the module default (0.20) — so a test can tell whether a
    surface reads the CONFIGURED value or silently keeps its own default."""
    from dataclasses import dataclass

    from superlocalmemory.core import engine as engine_mod

    @dataclass
    class _MemoryKindsCfg:
        display_min_confidence: float = 0.50

    @dataclass
    class _Cfg:
        memory_kinds: _MemoryKindsCfg = None
        def __post_init__(self):
            if self.memory_kinds is None:
                self.memory_kinds = _MemoryKindsCfg()

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    engine = _FakeEngine(db, config=_Cfg())
    monkeypatch.setattr(engine_mod, "MemoryEngine", lambda *a, **k: engine)
    return engine


def test_list_kind_filter_keeps_only_matches(fake_engine, capsys) -> None:
    _save(fake_engine._db, "rule one", kind="rule", source="user")
    _save(fake_engine._db, "a decision", kind="decision", source="user")
    commands.cmd_list(Namespace(json=True, limit=10, kind="rule"))
    out = json.loads(capsys.readouterr().out)
    contents = [r["content"] for r in out["data"]["results"]]
    assert contents == ["rule one"]


def test_list_defaults_to_no_kind_filter_when_attribute_absent(fake_engine, capsys) -> None:
    # The existing AST test constructs Namespace(json=True, limit=1) with no
    # `kind` attribute at all — cmd_list must not break on that shape.
    _save(fake_engine._db, "anything")
    commands.cmd_list(Namespace(json=True, limit=1))
    out = json.loads(capsys.readouterr().out)
    assert out["data"]["count"] == 1


def test_list_refuses_an_unknown_kind_before_any_read(fake_engine, capsys) -> None:
    _save(fake_engine._db, "anything")
    with pytest.raises(SystemExit) as exited:
        commands.cmd_list(Namespace(json=True, limit=10, kind="not-a-real-kind"))
    assert exited.value.code == 2


def test_list_text_shows_id_and_kind_label(fake_engine, capsys) -> None:
    """Audit 4.1.20 L1: the help promises IDs; the text view printed none, and
    showed the internal fact type ("semantic") instead of the memory kind."""
    fact_id = _save(fake_engine._db, "We decided to ship on Fridays",
                    kind="decision", source="user")

    commands.cmd_list(Namespace(json=False, limit=10, kind=""))
    out = capsys.readouterr().out

    assert f"id: {fact_id}" in out
    assert "Decision: We decided to ship on Fridays" in out
    assert "(semantic)" not in out
    assert "slm delete <id>" in out


def test_list_displays_the_configured_threshold_not_a_hard_coded_one(
    fake_engine_with_strict_threshold, capsys,
) -> None:
    """4.1.21 #16: ``slm list`` already reads the CONFIGURED
    ``display_min_confidence`` to FILTER by ``--kind`` (``list_recent_facts``
    above) — this pins that the same configured value is also used to LABEL
    each item (``--json`` and the plain-text view), not a 0.20 hard-coded at
    the display call regardless of what the filter call was given."""
    _save(fake_engine_with_strict_threshold._db, "Ship on Fridays only with sign-off.",
         kind="rule", source="model:llm", confidence=0.30)
    commands.cmd_list(Namespace(json=True, limit=10, kind=""))
    out = json.loads(capsys.readouterr().out)
    item = out["data"]["results"][0]
    # 0.30 clears the module default (0.20) but not the configured 0.50: a
    # caller reading the live config must see this demoted to its legacy
    # fact_type mapping, never shown as a "rule" suggestion.
    assert item["memory_kind_state"] == "legacy"
    assert item["memory_kind"] == "semantic"

    commands.cmd_list(Namespace(json=False, limit=10, kind=""))
    text = capsys.readouterr().out
    assert "Fact: Ship on Fridays only with sign-off." in text
    assert "Standing rule" not in text and "(suggested)" not in text


def test_list_help_matches_what_text_mode_shows(tmp_path) -> None:
    import os
    import subprocess
    import sys

    env = dict(os.environ, SLM_DATA_DIR=str(tmp_path / "slm"), **child_env_base(tmp_path))
    out = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "list", "--help"],
        capture_output=True, text=True, timeout=60, env=env,
    ).stdout
    assert "ID" in out and "kind" in out
